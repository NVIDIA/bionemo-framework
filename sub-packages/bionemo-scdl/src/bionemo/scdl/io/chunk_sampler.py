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
from typing import Iterator, Optional

from torch.utils.data import Sampler

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset


class ChunkAwareSampler(Sampler[int]):
    """Sampler that iterates by chunks for efficient access patterns.

    This sampler ensures all rows from a chunk are accessed together before
    moving to the next chunk. This is optimal for:
    - Local: memory locality (chunk data stays in cache)
    - Remote: prefetching (download chunk once, use all rows)

    Args:
        dataset: A chunked SingleCellMemMapDataset.
        shuffle_chunks: Whether to shuffle chunk order each epoch.
        shuffle_within_chunk: Whether to shuffle rows within each chunk.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: SingleCellMemMapDataset,
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize the chunk aware sampler."""
        if not dataset._is_chunked:
            raise ValueError("ChunkAwareSampler requires a chunked dataset")

        self.dataset = dataset
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.rng = random.Random(seed)

        self.chunked_info = dataset.header.chunked_info

    def __iter__(self) -> Iterator[int]:
        """Yield row indices, grouped by chunk."""
        chunk_ids = list(range(self.chunked_info.num_chunks))

        if self.shuffle_chunks:
            self.rng.shuffle(chunk_ids)

        for chunk_id in chunk_ids:
            start = chunk_id * self.chunked_info.chunk_size
            end = min(start + self.chunked_info.chunk_size, self.chunked_info.total_rows)

            if self.shuffle_within_chunk:
                row_indices = list(range(start, end))
                self.rng.shuffle(row_indices)
                yield from row_indices
            else:
                yield from range(start, end)  # Lazy, no list

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.chunked_info.total_rows
