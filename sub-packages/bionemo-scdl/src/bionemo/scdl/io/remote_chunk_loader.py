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

import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import fsspec


class RemoteChunkLoader:
    """Downloads and caches chunks from remote storage with LRU eviction.

    Args:
        remote_path: Remote path (s3://bucket/path, gs://bucket/path, etc.)
        cache_dir: Local directory for caching chunks. If None, uses temp directory.
        max_cached_chunks: Maximum number of chunks to keep in cache.
        storage_options: Optional dict of options passed to fsspec (e.g., endpoint_url for S3).
    """

    def __init__(
        self,
        remote_path: str,
        cache_dir: Optional[Path] = None,
        max_cached_chunks: int = 2,
        storage_options: Optional[dict] = None,
    ):
        """Initialize the remote chunk loader."""
        self.remote_path = remote_path.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="scdl_cache_"))
        self.max_cached_chunks = max_cached_chunks
        self._cache: OrderedDict[int, Path] = OrderedDict()

        # Initialize filesystem with optional storage options
        protocol = remote_path.split("://")[0] if "://" in remote_path else "file"
        self._fs = fsspec.filesystem(protocol, **(storage_options or {}))

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_chunk(self, chunk_id: int) -> Path:
        """Get local path to chunk, downloading if needed.

        Args:
            chunk_id: The chunk index to retrieve.

        Returns:
            Local path to the chunk directory.
        """
        if chunk_id in self._cache:
            self._cache.move_to_end(chunk_id)
            return self._cache[chunk_id]

        # Evict oldest chunks if at capacity
        while len(self._cache) >= self.max_cached_chunks:
            old_id, old_path = self._cache.popitem(last=False)
            shutil.rmtree(old_path, ignore_errors=True)

        # Download chunk
        local_path = self._download_chunk(chunk_id)
        self._cache[chunk_id] = local_path
        return local_path

    def _download_chunk(self, chunk_id: int) -> Path:
        """Download a chunk from remote storage."""
        chunk_name = f"chunk_{chunk_id:05d}"
        remote_chunk = f"{self.remote_path}/{chunk_name}"
        local_chunk = self.cache_dir / chunk_name

        local_chunk.mkdir(parents=True, exist_ok=True)

        # Download all files in chunk directory
        for remote_file in self._fs.ls(remote_chunk):
            fname = Path(remote_file).name
            self._fs.get(remote_file, str(local_chunk / fname))

        return local_chunk

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

    def __del__(self):
        """Cleanup on garbage collection."""
        self.cleanup()
