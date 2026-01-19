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
from typing import Iterator, Optional

from torch.utils.data import Sampler

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset


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

    def __iter__(self) -> Iterator[int]:
        """Yield row indices, grouped by chunk window."""
        chunk_ids = list(range(self.chunked_info.num_chunks))

        if self.shuffle_chunks:
            self.rng.shuffle(chunk_ids)

        # Process in windows of N chunks
        for i in range(0, len(chunk_ids), self.chunks_per_window):
            window_chunks = chunk_ids[i : i + self.chunks_per_window]

            # Gather all indices from this window
            all_indices = []
            for chunk_id in window_chunks:
                start = chunk_id * self.chunked_info.chunk_size
                end = min(start + self.chunked_info.chunk_size, self.chunked_info.total_rows)
                all_indices.extend(range(start, end))

            if self.shuffle_within_window:
                self.rng.shuffle(all_indices)

            yield from all_indices

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.chunked_info.total_rows
