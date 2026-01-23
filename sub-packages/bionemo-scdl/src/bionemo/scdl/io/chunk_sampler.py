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

"""Chunk-aware sampler for efficient iteration over chunked SCDL datasets."""

import random
import warnings
from pathlib import Path
from typing import Iterator, Optional

import torch.utils.data
from torch.utils.data import Sampler

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset


def remote_worker_init_fn(worker_id: int) -> None:
    """Initialize per-worker cache directories for remote datasets.

    Use this with DataLoader's worker_init_fn to prevent cache conflicts
    when using multiple workers with remote chunked SCDL:

        DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=remote_worker_init_fn,
            ...
        )

    Each worker gets its own cache directory: {base_cache}_worker_{id}
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    if hasattr(dataset, "_chunk_loader") and dataset._chunk_loader is not None:
        # Create per-worker cache directory
        base_cache = dataset._chunk_loader.cache_dir
        worker_cache = Path(str(base_cache) + f"_worker_{worker_id}")
        worker_cache.mkdir(parents=True, exist_ok=True)
        dataset._chunk_loader.cache_dir = worker_cache
        # Clear cached chunks dict for this worker
        dataset._chunk_loader._cached_chunks.clear()


class ChunkAwareSampler(Sampler[int]):
    """Sampler that iterates by chunks for efficient access patterns.

    This sampler ensures all rows from a chunk window are accessed together
    before moving to the next window. This is optimal for:
    - Local: memory locality (chunk data stays in cache)
    - Remote: prefetching (download chunks once, use all rows)

    Args:
        dataset: A chunked SingleCellMemMapDataset.
        shuffle_chunks: Whether to shuffle chunk order each epoch.
        shuffle_within_window: Whether to shuffle rows within each chunk window.
        chunks_per_window: Number of chunks to load together (more = better randomness).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: SingleCellMemMapDataset,
        shuffle_chunks: bool = True,
        shuffle_within_window: bool = True,
        chunks_per_window: int = 1,
        seed: Optional[int] = None,
    ):
        """Initialize the chunk aware sampler."""
        if not dataset._is_chunked:
            raise ValueError("ChunkAwareSampler requires a chunked dataset")

        self.dataset = dataset
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_window = shuffle_within_window
        self.chunks_per_window = max(1, chunks_per_window)
        self.rng = random.Random(seed)
        self.chunked_info = dataset.header.chunked_info

        # Warn if chunks_per_window exceeds cache size (causes thrashing)
        if dataset._chunk_loader and chunks_per_window > dataset._chunk_loader.max_cached_chunks:
            warnings.warn(
                f"chunks_per_window ({chunks_per_window}) > max_cached_chunks "
                f"({dataset._chunk_loader.max_cached_chunks}). This causes cache thrashing. "
                f"Increase max_cached_chunks or decrease chunks_per_window."
            )
        # Warn if cache too small for effective prefetching
        if dataset._chunk_loader and 2 * chunks_per_window > dataset._chunk_loader.max_cached_chunks:
            warnings.warn(
                f"max_cached_chunks ({dataset._chunk_loader.max_cached_chunks}) < 2 * chunks_per_window "
                f"({2 * chunks_per_window}). Prefetching disabled - no room for next window. "
                f"Set max_cached_chunks >= {2 * chunks_per_window} for prefetching."
            )

    def _preload_memmaps_for_chunks(self, chunk_ids: list) -> None:
        """Preload memmaps for chunks into _remote_chunk_cache.

        This should be called after prefetch completes to avoid creating
        memmaps during iteration. Files are already on disk, we just need
        to create the memmap objects.
        """
        if not hasattr(self.dataset, "_remote_chunk_cache"):
            return

        for chunk_id in chunk_ids:
            if chunk_id not in self.dataset._remote_chunk_cache:
                # get_chunk returns path (chunk already downloaded by prefetch)
                chunk_path = self.dataset._chunk_loader.get_chunk(chunk_id)
                # Load memmaps and cache them
                memmaps = self.dataset._load_chunk_from_path(chunk_path)
                self.dataset._remote_chunk_cache[chunk_id] = memmaps

    def __iter__(self) -> Iterator[int]:
        """Yield row indices, grouped by chunk window.

        In multi-worker mode, each worker handles a disjoint subset of chunks.
        This avoids cache conflicts and ensures efficient parallel loading.
        """
        chunk_ids = list(range(self.chunked_info.num_chunks))

        if self.shuffle_chunks:
            self.rng.shuffle(chunk_ids)

        # Multi-worker support: each worker handles a subset of chunks
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split chunks among workers: worker i gets chunks[i::num_workers]
            chunk_ids = chunk_ids[worker_info.id :: worker_info.num_workers]

        # Build list of windows from this worker's chunks
        windows = []
        for i in range(0, len(chunk_ids), self.chunks_per_window):
            windows.append(chunk_ids[i : i + self.chunks_per_window])

        # Check if we have room for prefetching (need 2x window size in cache)
        can_prefetch = (
            self.dataset._chunk_loader and 2 * self.chunks_per_window <= self.dataset._chunk_loader.max_cached_chunks
        )

        import time as time_module  # Import once outside loop

        iteration_start = time_module.perf_counter()

        # Process windows with pipelined prefetching (1 window ahead)
        for window_idx, window_chunks in enumerate(windows):
            if self.dataset._chunk_loader:
                # For first window, prefetch synchronously since we need it immediately
                if window_idx == 0:
                    cold_start = time_module.perf_counter()
                    self.dataset._chunk_loader.prefetch_chunks(window_chunks)
                    elapsed = time_module.perf_counter() - cold_start
                    self.dataset._chunk_loader.stats.record_cold_start(elapsed)
                else:
                    # Wait for async prefetch started in previous iteration
                    self.dataset._chunk_loader.wait_for_prefetch()

                # Preload memmaps for current window (now that files are downloaded)
                # This avoids creating memmaps during iteration
                self._preload_memmaps_for_chunks(window_chunks)

                # Start async prefetch of NEXT window while we process this one
                if can_prefetch and window_idx + 1 < len(windows):
                    self.dataset._chunk_loader.prefetch_chunks_async(windows[window_idx + 1])

            # Gather all indices from this window
            all_indices = []
            for chunk_id in window_chunks:
                start = chunk_id * self.chunked_info.chunk_size
                end = min(start + self.chunked_info.chunk_size, self.chunked_info.total_rows)
                all_indices.extend(range(start, end))

            if self.shuffle_within_window:
                self.rng.shuffle(all_indices)

            yield from all_indices

            # Mark this window's chunks as safe to evict (prefer these over current/next window)
            if self.dataset._chunk_loader:
                self.dataset._chunk_loader.mark_chunks_done(window_chunks)

        # Record total wall-clock time
        if self.dataset._chunk_loader:
            total_time = time_module.perf_counter() - iteration_start
            self.dataset._chunk_loader.stats.record_wall_clock(total_time)

    def __len__(self) -> int:
        """Return total number of samples.

        Note: In multi-worker mode, each worker sees only its subset,
        but PyTorch DataLoader handles combining results correctly.
        """
        return self.chunked_info.total_rows
