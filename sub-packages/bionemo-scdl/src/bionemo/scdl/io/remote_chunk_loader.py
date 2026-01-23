# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote chunk loader with LRU caching for chunked SCDL datasets.

NOTE: This is a simple POC implementation. For production multi-worker/multi-node use:
- Add file locking for shared cache (filelock)
- Add reference counting to prevent evicting in-use chunks
- Use DistributedChunkSampler to shard chunks across nodes
"""

import io
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

import fsspec
import numpy as np

from bionemo.scdl.util.scdl_constants import FileNames


@dataclass
class DownloadStats:
    """Statistics for download timing."""

    total_download_time: float = 0.0  # Cumulative time across all download threads
    total_wait_time: float = 0.0  # Time spent blocked waiting for downloads
    cold_start_time: float = 0.0  # Time to download first window
    wall_clock_time: float = 0.0  # Total wall-clock iteration time
    total_bytes_downloaded: int = 0
    download_count: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_download(self, duration: float, bytes_downloaded: int = 0) -> None:
        """Record a download completion."""
        with self._lock:
            self.total_download_time += duration
            self.total_bytes_downloaded += bytes_downloaded
            self.download_count += 1

    def record_cold_start(self, duration: float) -> None:
        """Record cold start time (first window download)."""
        self.cold_start_time = duration

    def record_wall_clock(self, duration: float) -> None:
        """Record total wall-clock time."""
        self.wall_clock_time = duration

    def record_wait(self, duration: float) -> None:
        """Record time spent waiting for a chunk."""
        with self._lock:
            self.total_wait_time += duration

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.cache_hits += 1

    def summary(self) -> dict:
        """Return a summary of download statistics."""
        total_mb = self.total_bytes_downloaded / 1e6
        effective_throughput = total_mb / self.wall_clock_time if self.wall_clock_time > 0 else 0
        per_thread_throughput = total_mb / self.total_download_time if self.total_download_time > 0 else 0

        return {
            "cold_start_time_sec": round(self.cold_start_time, 2),
            "wall_clock_time_sec": round(self.wall_clock_time, 2),
            "total_wait_time_sec": round(self.total_wait_time, 2),
            "total_bytes_downloaded_mb": round(total_mb, 2),
            "download_count": self.download_count,
            "cache_hits": self.cache_hits,
            "throughput_mbps": round(effective_throughput, 2),
            "per_thread_throughput_mbps": round(per_thread_throughput, 2),
        }


class RemoteChunkLoader:
    """Downloads and caches chunks from remote storage with LRU eviction.

    Args:
        remote_path: Remote path (s3://bucket/path, gs://bucket/path, etc.)
        cache_dir: Local directory for caching chunks. If None, uses temp directory.
        max_cached_chunks: Maximum number of chunks to keep in cache.
        storage_options: Optional dict of options passed to fsspec (e.g., endpoint_url for S3).
    """

    # Files in each chunk directory (uncompressed format)
    CHUNK_FILES: ClassVar[List[str]] = [FileNames.DATA.value, FileNames.ROWPTR.value, FileNames.COLPTR.value]
    # Compressed format (single file)
    COMPRESSED_FILE: ClassVar[str] = "chunk.npz"

    # Type alias for cached chunk data: (file_path, (data, rowptr, colptr) memmaps)
    CacheEntry = Tuple[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]]

    def __init__(
        self,
        remote_path: str,
        cache_dir: Optional[Path] = None,
        max_cached_chunks: int = 2,
        storage_options: Optional[dict] = None,
        batch_download_size: int = 10,
        use_async_downloads: bool = True,
        dtypes: Optional[Dict[str, str]] = None,
    ):
        """Initialize the remote chunk loader.

        Args:
            remote_path: Remote path (s3://bucket/path, gs://bucket/path, etc.)
            cache_dir: Local directory for caching chunks. If None, uses temp directory.
            max_cached_chunks: Maximum number of chunks to keep in cache.
            storage_options: Optional dict of options passed to fsspec.
            batch_download_size: Number of chunks to download at once (unused, for API compat).
            use_async_downloads: Whether to use async downloads (unused, for API compat).
            dtypes: Optional dict mapping file names to dtypes for memmap loading.
        """
        self.remote_path = remote_path.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="scdl_cache_"))
        self.max_cached_chunks = max_cached_chunks
        self.batch_download_size = batch_download_size
        self._use_async = use_async_downloads
        self.dtypes = dtypes or {}
        # Cache stores both path (for cleanup) and memmaps (for access)
        self._cache: OrderedDict[int, "RemoteChunkLoader.CacheEntry"] = OrderedDict()
        self._cache_lock = threading.Lock()  # Protect cache access
        self._downloading: set = set()  # Chunks currently being downloaded
        self._download_complete = threading.Condition(self._cache_lock)
        self._evictable_chunks: set = set()  # Chunks marked as "done" - prefer evicting these

        # Initialize filesystem with optional storage options
        protocol = remote_path.split("://")[0] if "://" in remote_path else "file"
        self._fs = fsspec.filesystem(protocol, **(storage_options or {}))

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Async prefetching
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future: Optional[Future] = None
        self._prefetch_lock = threading.Lock()

        # Download statistics
        self.stats = DownloadStats()

    def set_dtypes(self, dtypes: Dict[str, str]) -> None:
        """Set dtypes for memmap loading (can be called after header is loaded).

        Args:
            dtypes: Dict mapping file names to dtypes.
        """
        self.dtypes = dtypes

    def mark_chunks_done(self, chunk_ids: List[int]) -> None:
        """Mark chunks as done/evictable.

        Call this when you've finished processing a window of chunks and won't
        need them again until the next epoch. Eviction will prefer these chunks.

        Args:
            chunk_ids: List of chunk IDs that can be safely evicted.
        """
        with self._cache_lock:
            self._evictable_chunks.update(chunk_ids)

    def _find_evictable_chunk(self) -> Optional[Tuple[int, "RemoteChunkLoader.CacheEntry"]]:
        """Find a chunk to evict, preferring chunks marked as done.

        Returns:
            Tuple of (chunk_id, (path, memmaps)) or None if no chunks available.
        """
        # First, try to evict chunks marked as "done"
        for chunk_id in list(self._cache.keys()):
            if chunk_id in self._evictable_chunks:
                entry = self._cache.pop(chunk_id)
                self._evictable_chunks.discard(chunk_id)
                return chunk_id, entry

        # Fallback: evict oldest chunk (LRU) - shouldn't happen if sized correctly
        if self._cache:
            return self._cache.popitem(last=False)
        return None

    def _evict_chunk(self, chunk_id: int, cache_entry: "RemoteChunkLoader.CacheEntry") -> None:
        """Evict a chunk from cache, releasing memmaps before deleting files."""
        self._evictable_chunks.discard(chunk_id)
        chunk_path, memmaps = cache_entry
        # Release memmaps first (closes file handles)
        del memmaps
        # Now safe to delete files
        shutil.rmtree(chunk_path, ignore_errors=True)

    def get_chunk(self, chunk_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get chunk as memory-mapped arrays, downloading if needed.

        Args:
            chunk_id: The chunk index to retrieve.

        Returns:
            Tuple of (data, rowptr, colptr) as memory-mapped numpy arrays.
        """
        start_time = time.perf_counter()

        # Collect chunks to evict inside lock, delete outside to avoid blocking
        chunks_to_evict = []

        with self._download_complete:
            # Wait if prefetch is downloading this chunk
            while chunk_id in self._downloading and chunk_id not in self._cache:
                self._download_complete.wait(timeout=0.1)

            # Check cache - return memmaps directly
            if chunk_id in self._cache:
                self._cache.move_to_end(chunk_id)
                self.stats.record_cache_hit()
                _path, memmaps = self._cache[chunk_id]
                return memmaps

            # Evict chunks if at capacity (prefer evicting "done" chunks)
            while len(self._cache) >= self.max_cached_chunks and self._cache:
                evict_result = self._find_evictable_chunk()
                if evict_result is None:
                    break
                old_id, old_entry = evict_result
                chunks_to_evict.append((old_id, old_entry))

            # Mark as downloading
            self._downloading.add(chunk_id)

        # Delete evicted chunks in background thread while we download
        eviction_future = None
        if chunks_to_evict:
            eviction_future = self._prefetch_executor.submit(
                lambda: [self._evict_chunk(old_id, old_entry) for old_id, old_entry in chunks_to_evict]
            )

        try:
            # Download chunk (outside lock)
            local_path = self._download_chunk(chunk_id)

            # Load memmaps
            memmaps = self._load_chunk_memmaps(local_path)

            # Wait for eviction to complete (if any)
            if eviction_future:
                eviction_future.result()

            with self._download_complete:
                self._cache[chunk_id] = (local_path, memmaps)
                self._downloading.discard(chunk_id)
                self._download_complete.notify_all()

            # Record wait time (includes download time for non-prefetched chunks)
            wait_time = time.perf_counter() - start_time
            self.stats.record_wait(wait_time)

            return memmaps
        except Exception:
            with self._download_complete:
                self._downloading.discard(chunk_id)
                self._download_complete.notify_all()
            raise

    def _load_chunk_memmaps(self, chunk_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load memmaps for a single chunk directory.

        Uses np.memmap when dtypes are known (faster), falls back to np.load otherwise.
        """
        if self.dtypes:
            return (
                np.memmap(chunk_path / FileNames.DATA.value, dtype=self.dtypes.get(FileNames.DATA.value), mode="r"),
                np.memmap(
                    chunk_path / FileNames.ROWPTR.value, dtype=self.dtypes.get(FileNames.ROWPTR.value), mode="r"
                ),
                np.memmap(
                    chunk_path / FileNames.COLPTR.value, dtype=self.dtypes.get(FileNames.COLPTR.value), mode="r"
                ),
            )

        else:
            raise ValueError("Dtypes are not set")

    def _download_chunk(self, chunk_id: int) -> Path:
        """Download a chunk from remote storage.

        Supports both compressed (.npz) and uncompressed formats.
        Compressed files are downloaded and extracted to uncompressed format for fast memmap access.
        """
        start_time = time.perf_counter()
        bytes_downloaded = 0

        chunk_name = f"chunk_{chunk_id:05d}"
        print(f"[DOWNLOAD] Starting chunk {chunk_id} ({chunk_name})")
        remote_chunk = f"{self.remote_path}/{chunk_name}"
        local_chunk = self.cache_dir / chunk_name

        # Ensure directory exists
        local_chunk.mkdir(parents=True, exist_ok=True)

        # Check if compressed format exists (single .npz file)
        remote_compressed = f"{remote_chunk}/{self.COMPRESSED_FILE}"
        try:
            # Try compressed format first (1 HTTP request instead of 3)
            with self._fs.open(remote_compressed, "rb") as src:
                compressed_data = src.read()
            bytes_downloaded = len(compressed_data)

            # Load compressed data from memory
            npz = np.load(io.BytesIO(compressed_data))

            # Extract and save as uncompressed files (for fast memmap access)
            np.save(local_chunk / FileNames.DATA.value, npz["data"])
            np.save(local_chunk / FileNames.ROWPTR.value, npz["row_ptr"])
            np.save(local_chunk / FileNames.COLPTR.value, npz["col_ptr"])

            # Record download stats
            duration = time.perf_counter() - start_time
            self.stats.record_download(duration, bytes_downloaded)

            return local_chunk

        except FileNotFoundError:
            # Fall back to uncompressed format (3 files)
            pass
        except Exception:
            # Fall back to uncompressed format (3 files)
            pass

        # Download uncompressed files (original format)
        def download_file(fname: str) -> int:
            remote_file = f"{remote_chunk}/{fname}"
            local_file = local_chunk / fname
            try:
                local_file.parent.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            try:
                with self._fs.open(remote_file, "rb") as src:
                    content = src.read()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Chunk {chunk_id} not found at {remote_file} (compressed also not found at {remote_compressed})"
                )
            with open(local_file, "wb") as dst:
                dst.write(content)
            return len(content)

        with ThreadPoolExecutor(max_workers=3) as executor:
            file_sizes = list(executor.map(download_file, self.CHUNK_FILES))

        # Record download stats
        bytes_downloaded = sum(file_sizes)
        duration = time.perf_counter() - start_time
        self.stats.record_download(duration, bytes_downloaded)

        return local_chunk

    def prefetch_chunks(self, chunk_ids: List[int], max_parallel: int = 16) -> None:
        """Prefetch multiple chunks in parallel (synchronous)."""
        # Filter out already cached chunks
        with self._cache_lock:
            to_download = [cid for cid in chunk_ids if cid not in self._cache and cid not in self._downloading]

        if not to_download:
            return

        # Collect chunks to evict inside lock, delete outside to avoid blocking
        chunks_to_evict = []
        with self._download_complete:
            needed = len(to_download)
            while len(self._cache) + needed > self.max_cached_chunks and self._cache:
                evict_result = self._find_evictable_chunk()
                if evict_result is None:
                    break
                old_id, old_entry = evict_result
                chunks_to_evict.append((old_id, old_entry))

            # Mark all as downloading
            for cid in to_download:
                self._downloading.add(cid)

        try:

            def download_and_load(chunk_id: int) -> Tuple[int, Path, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
                local_path = self._download_chunk(chunk_id)
                memmaps = self._load_chunk_memmaps(local_path)
                return chunk_id, local_path, memmaps

            # Run eviction and downloads in parallel
            with ThreadPoolExecutor(
                max_workers=min(max_parallel, len(to_download) + len(chunks_to_evict))
            ) as executor:
                # Submit eviction tasks (fire and forget)
                for old_id, old_entry in chunks_to_evict:
                    executor.submit(self._evict_chunk, old_id, old_entry)
                # Download all chunks
                results = list(executor.map(download_and_load, to_download))

            # Add to cache
            with self._download_complete:
                for chunk_id, local_path, memmaps in results:
                    self._cache[chunk_id] = (local_path, memmaps)
                    self._downloading.discard(chunk_id)
                self._download_complete.notify_all()

        except Exception:
            with self._download_complete:
                for cid in to_download:
                    self._downloading.discard(cid)
                self._download_complete.notify_all()
            raise

    def prefetch_chunks_async(self, chunk_ids: List[int]) -> None:
        """Start prefetching chunks in background thread."""
        with self._prefetch_lock:
            self._prefetch_future = self._prefetch_executor.submit(self.prefetch_chunks, chunk_ids)

    def wait_for_prefetch(self) -> None:
        """Wait for any ongoing prefetch to complete."""
        with self._prefetch_lock:
            if self._prefetch_future is not None:
                try:
                    self._prefetch_future.result(timeout=300)  # 5 min timeout
                except Exception:
                    pass  # Ignore prefetch errors
                self._prefetch_future = None

    def _remote_exists(self, remote_path: str) -> bool:
        """Check if a remote path exists (uses ls instead of exists for compatibility)."""
        try:
            # Use ls instead of exists() because some S3-compatible storage
            # doesn't support HeadObject which exists() relies on
            parent = "/".join(remote_path.rsplit("/", 1)[:-1])
            name = remote_path.rsplit("/", 1)[-1]
            files = self._fs.ls(parent, detail=False)
            return any(f.endswith(name) for f in files)
        except Exception:
            return False

    def get_metadata(self) -> Path:
        """Download and return path to metadata files (header, features, etc.)."""
        metadata_dir = self.cache_dir / "_metadata"
        if metadata_dir.exists():
            return metadata_dir

        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Download header and feature indices (header.sch is the SCDL header format)
        for name in ["header.sch", "version.json", "metadata.json"]:
            remote_file = f"{self.remote_path}/{name}"
            if self._remote_exists(remote_file):
                self._fs.get(remote_file, str(metadata_dir / name))

        # Download feature directories
        for name in ["var_features", "obs_features"]:
            remote_dir = f"{self.remote_path}/{name}"
            if self._remote_exists(remote_dir):
                local_dir = metadata_dir / name
                self._fs.get(remote_dir, str(local_dir), recursive=True)

        return metadata_dir

    def cleanup(self):
        """Delete all cached data."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
