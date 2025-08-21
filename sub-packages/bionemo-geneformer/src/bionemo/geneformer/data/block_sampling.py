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

# MIT License
# Copyright (c) 2025 Davide D'Ascenzo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from functools import lru_cache
from typing import Any, Callable, List, Optional, Union

import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from bionemo.core.data.permute import permute


class MapStyleScDataset(Dataset):
    """Implements an online block sampling scheme to shuffle a dataset in memory with block efficiency.

    Purpose of this class is to evaluate the downstream effects of such a dataset on model performance. Suggest wrapping the ResamplingDataset
    built into BioNeMo, resulting in a structure like:

        UnderlyingDataset -> ResamplingDataset -> MapStyleScDataset -> Model

    This ensures that resampling occurs to support multi-epoch training, while also allowing for 'upsampling' to resolve
    the constraints on scDataset where block_size and batch_size cleanly divide the dataset.

    Performance is not optimal, as it relies on repeated calls to permute, but it does yield the correct results.
    """

    def __init__(
        self,
        dataset: Dataset,
        block_size: int = 1000000,
        batch_size: int = 1,
        fetch_factor: int = 1,
        seed: int = 1,
        use_batch_queries: bool = True,
    ):
        """Implements scDataset block-sampling on __getitem__.

        Every sample is indexed directly, while using the same ordering as the batch-permuted version. Currently only supports datasets that are divisible by block_size and batch_size. Drop last is not supported
        at this time.

        Args:
            dataset (Dataset): Dataset to sample from.
            block_size (int): Size of each sample block. This is used to shuffle the samples.
                              None will be replaced with dataset size.
            batch_size (int): This a determinant in the fetch_size, used for equivalency to the iterable style scDataset.
                This parameter is used to create useful fetches and optimize i/o access.
            fetch_factor (int): Multiplier for fetch size relative to batch size (for compatibility with scDataset).
            seed (int): Seed for the random number generator used for shuffling.
            use_batch_queries (bool): Whether to use batch queries with __getitems__.
        """
        if len(dataset) % block_size != 0:
            raise ValueError("Dataset size must be divisible by block size")
        if len(dataset) % batch_size != 0:
            # raise ValueError("Dataset size must be divisible by batch size")
            print("batch size is not divisible by dataset size")

        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.effective_dataset_size = len(dataset)
        self.use_batch_queries = use_batch_queries
        self.block_size = block_size if block_size is not None else self.dataset_size
        self.batch_size = batch_size
        self.fetch_factor = fetch_factor
        self.fetch_size = self.batch_size * self.fetch_factor
        self.seed = seed
        self._num_blocks = self.dataset_size // self.block_size + 1

        # Calculate number of blocks and fetches for proper scDataset-style shuffling
        self._num_blocks = self.effective_dataset_size // self.block_size
        self._num_fetches = self.effective_dataset_size // self.fetch_size
        self.permute = permute

    def get_fetch_idx(self, idx: int) -> int:
        """Computes the fetch index for a given sample index."""
        block_idx = idx // self.block_size
        fetch_idx = block_idx // (self.fetch_size // self.block_size)
        return fetch_idx

    def get_fetch(self, idx) -> List[int]:
        """Returns a list of indices that are in the same fetch as idx.

        Example:
            block_size = 4, fetch_size = 8
            sample_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            block_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

            get_fetch(block_idx = 7) => [0, 1, 2, 3, 4, 5, 6, 7]
            get_fetch(block_idx = 8) => [8, 9, 10, 11, 12, 13, 14, 15]
        """
        blocks_per_fetch = self.fetch_size // self.block_size
        fetch_id = idx // blocks_per_fetch
        fetch_start_idx = fetch_id * self.fetch_size
        fetch_end_idx = fetch_start_idx + self.fetch_size
        # NOTE that if block_size and dataset_size do not divide, fetch_end_idx will be out of range.
        return list(range(fetch_start_idx, fetch_end_idx))

    def __getitem__(self, idx: int) -> int:  # noqa: D105
        return self.dataset[self.lookup_idx(idx)]

    def __getitems__(self, indices: List[int]) -> Any:
        """Provides data proximal batch lookup by sorting and unsorting indices before performing the lookup.

        Args:
            indices: List of indices to lookup.

        Returns:
            Data corresponding to each index after applying the block sampling algorithm.
        """
        shuffled_ids = np.array([self.lookup_idx(idx) for idx in indices])
        if self.use_batch_queries:
            _sorted_order = np.argsort(shuffled_ids)
            _sorted_idxs = np.sort(shuffled_ids)

            # Turn it back into a list so torch does the right things.
            if hasattr(self.dataset, '__getitems__'):
                sorted_data = self.dataset.__getitems__(_sorted_idxs.tolist())
            else:
                sorted_data = [self.dataset[idx] for idx in _sorted_idxs.tolist()]

            # Reverse the sorting to return the args in the original state.
            data = np.array(sorted_data)[np.argsort(_sorted_order)]
        else:
            data = np.array(self.dataset[shuffled_ids])
        return data

    @lru_cache(maxsize=1024)
    def permuted_block_sample_ids_from_block_idx(self, block_idx: int) -> List[int]:
        """Given a block index, return all the permuted sample ids in the fetch."""
        fetch = self.get_fetch(block_idx)
        permuted_block_sample_ids = [
            self.permute(sample_id // self.block_size, self._num_blocks, self.seed) * self.block_size
            + sample_id % self.block_size
            for sample_id in fetch
        ]
        return permuted_block_sample_ids

    def lookup_idx(self, idx: int) -> int:
        """Given an index, return the block-permuted index.

        Performs the complete block-sampling algorithm online using the repeatable and deterministic permute function.

        Args:
            idx (int): Index to lookup.

        Returns:
            int: Block-permuted index.
        """
        # If the index is out of range, raise IndexError
        if idx >= self.dataset_size:
            raise IndexError("Index out of range")

        # support negative indices
        if idx < 0:
            idx += self.dataset_size

            if idx < 0:
                raise IndexError("Index out of range")

        # find out which block we are in
        block_idx = idx // self.block_size

        # both gets the fetch from block_idx and permutes the block ids.
        # This is the sample ids in the fetch AFTER applying block permutation. The fetch is still in the original idx space.
        permuted_block_sample_ids = self.permuted_block_sample_ids_from_block_idx(block_idx)
        fetch_idx = idx // self.fetch_size

        # range 0..fetch_size-1, these are fetch-level indices.
        permuted_fetch_idx = self.permute(idx % self.fetch_size, self.fetch_size, self.seed + fetch_idx)

        # These are the sample ids after permuting the fetch.
        final_idx = permuted_block_sample_ids[permuted_fetch_idx]

        return final_idx

    def __len__(self) -> int:  # noqa: D105
        return self.dataset_size


class scDataset(IterableDataset):
    """Iterable PyTorch Dataset for single-cell data collections with flexible batching, shuffling, and transformation options.

    Code adapted from https://github.com/scDataset/scDataset. Modifications were made to support the use of bionemo.core.data.permute, which is a hash-based permutation function.
    We do not recommend the use of this code as is- its purpose in BioNeMo-framework is as a comparison point for the
    iterable implementation of scDataset. For use in real-world applications, we suggest using the source code available
    on github: https://github.com/scDataset/scDataset

    Parameters
    ----------
    dataset: object
        The data collection to sample from (e.g., AnnCollection, HuggingFace Dataset, numpy array, etc.).
    batch_size : int
        Number of samples per minibatch.
    block_size : int, default=1
        Number of samples per block for shuffling.
    fetch_factor : int, default=1
        Multiplier for fetch size relative to batch size.
    shuffle : bool, default=False
        Whether to shuffle data before batching.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.
    sort_before_fetch : bool, optional
        Whether to sort indices before fetching.
    shuffle_before_yield : bool, optional
        Whether to shuffle within fetched blocks before yielding batches.
    fetch_transform : Callable, optional
        Function to transform data after fetching.
    batch_transform : Callable, optional
        Function to transform each batch before yielding.
    fetch_callback : Callable, optional
        Custom function to fetch data given indices.
    batch_callback : Callable, optional
        Custom function to fetch batch data given indices.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        block_size: int = 1,
        fetch_factor: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        sort_before_fetch: Optional[bool] = None,
        shuffle_before_yield: Optional[bool] = None,
        fetch_transform: Optional[Callable] = None,
        batch_transform: Optional[Callable] = None,
        fetch_callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        seed: int = 42,
        bionemo_permute: bool = True,
    ):
        """Initialize the scDataset."""
        self.bionemo_permute = bionemo_permute
        if shuffle:
            if sort_before_fetch is None:
                # sort indices before fetching samples
                sort_before_fetch = True
            if shuffle_before_yield is None:
                # Shuffle indices before yielding batches
                shuffle_before_yield = True
        if (not shuffle) and shuffle_before_yield:
            raise ValueError("shuffle_before_yield=True requires shuffle=True")
        if (fetch_factor == 1) and shuffle_before_yield:
            warnings.warn(
                "shuffle_before_yield=True and fetch_factor=1, this will return the same batch unless some downstream logic is applied"
            )
        if (not sort_before_fetch) and shuffle_before_yield:
            warnings.warn(
                "shuffle_before_yield=True and sort_before_fetch=False, this decrease the fetching speed and yield the same result. Consider setting sort_before_fetch=True"
            )
        if shuffle and (not drop_last):
            # Skip this one for the purpose of bionemo, we recommend avoiding drop_last for comparisons.
            pass
            # raise ValueError("shuffle=True requires drop_last=True")
        if (not shuffle) and drop_last:
            raise NotImplementedError("shuffle=False and drop_last=True is not implemented")

        self.batch_size = batch_size
        self.block_size = block_size
        self.fetch_factor = fetch_factor
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sort_before_fetch = sort_before_fetch
        self.shuffle_before_yield = shuffle_before_yield
        self.fetch_size = self.batch_size * self.fetch_factor

        # Store callback functions
        self.fetch_transform = fetch_transform
        self.batch_transform = batch_transform
        self.fetch_callback = fetch_callback
        self.batch_callback = batch_callback

        self.collection = dataset
        self.seed = seed

        self.indices = np.arange(len(self.collection))

    def __len__(self):
        """Return the number of batches in the dataset."""
        n = len(self.indices)
        if self.shuffle and self.drop_last:
            return (n // self.fetch_size * self.fetch_size) // self.batch_size
        if not (not self.shuffle and not self.drop_last):
            warnings.warn(
                "__len__ is only correctly implemented for shuffle=True and drop_last=True or not shuffling and not dropping last batch."
            )
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Yield batches of data according to the current configuration."""
        worker_info = get_worker_info()
        indices = self.indices
        bionemo_permute = self.bionemo_permute
        seed = self.seed
        g = None
        if self.shuffle:
            if worker_info is None:
                g = np.random.default_rng()
            else:
                seed = worker_info.seed - worker_info.id
                g = np.random.default_rng(seed=seed)

            remainder = len(indices) % self.fetch_size

            # Drop randomly selected indices
            if self.drop_last and (remainder != 0):
                remove_indices = g.choice(indices, size=remainder, replace=False)
                mask = ~np.isin(indices, remove_indices)
                indices = indices[mask]

            blocks = indices.reshape(-1, self.block_size)
            if bionemo_permute:
                blocks = np.array([blocks[permute(i, len(blocks), seed)] for i in range(len(blocks))])
            else:
                blocks = g.permutation(blocks, axis=0)
                # permuted over the first axis (n_blocks), same shape
            self.permuted_blocks = blocks
            fetches = blocks.reshape(-1, self.fetch_size)

            if worker_info is not None:
                per_worker = len(fetches) // worker_info.num_workers
                remainder = len(fetches) % worker_info.num_workers

                # Distribute remainder among workers
                if worker_info.id < remainder:
                    # First 'remainder' workers get one extra fetch
                    start = worker_info.id * (per_worker + 1)
                    end = start + per_worker + 1
                else:
                    # Other workers get the base number of fetches
                    start = worker_info.id * per_worker + remainder
                    end = start + per_worker
                fetches = fetches[start:end]

            if self.sort_before_fetch:
                fetches = np.sort(fetches, axis=1)
            self.fetches = fetches
            self.shuffled_fetch_indices = []
            self.shuffled_fetches = []
            for fetch_idx, fetch_ids in enumerate(fetches):
                # Use custom fetch callback if provided, otherwise use default indexing
                if self.fetch_callback is not None:
                    data = self.fetch_callback(self.collection, fetch_ids)
                else:
                    data = list(self.collection[i] for i in fetch_ids)

                if not isinstance(data, np.ndarray):
                    data = np.array(data)

                # Call fetch transform if provided
                if self.fetch_transform is not None:
                    data = self.fetch_transform(data)
                if self.shuffle_before_yield:
                    # Shuffle the indices
                    if bionemo_permute:
                        shuffle_indices = np.array(
                            [permute(i, len(fetch_ids), seed + fetch_idx) for i in range(len(fetch_ids))]
                        )
                    else:
                        shuffle_indices = g.permutation(len(fetch_ids))
                else:
                    shuffle_indices = np.arange(len(fetch_ids))
                self.shuffled_fetch_indices.append(shuffle_indices)

                for i in range(0, len(fetch_ids), self.batch_size):
                    # Use custom batch callback if provided, otherwise use default indexing
                    if self.batch_callback is not None:
                        batch_ids = shuffle_indices[i : i + self.batch_size]
                        batch_data = self.batch_callback(data, batch_ids)
                    else:
                        batch_ids = shuffle_indices[i : i + self.batch_size]
                        batch_data = data[batch_ids]
                        self.shuffled_fetches.append(batch_data)

                    # Call batch transform if provided
                    if self.batch_transform is not None:
                        batch_data = self.batch_transform(batch_data)
                    yield batch_data

        else:  # Not shuffling indices before fetching
            n = len(indices)
            fetch_size = self.fetch_size
            num_fetches = (n + fetch_size - 1) // fetch_size
            fetch_ranges = [(i * fetch_size, min((i + 1) * fetch_size, n)) for i in range(num_fetches)]
            if worker_info is not None:
                per_worker = num_fetches // worker_info.num_workers
                remainder = num_fetches % worker_info.num_workers
                if worker_info.id < remainder:
                    start = worker_info.id * (per_worker + 1)
                    end = start + per_worker + 1
                else:
                    start = worker_info.id * per_worker + remainder
                    end = start + per_worker
                fetch_ranges = fetch_ranges[start:end]
            for fetch_start, fetch_end in fetch_ranges:
                ids = indices[fetch_start:fetch_end]
                # Use custom fetch callback if provided, otherwise use default indexing
                if self.fetch_callback is not None:
                    data = self.fetch_callback(self.collection, ids)
                else:
                    data = self.collection[ids]
                # Call fetch transform if provided
                if self.fetch_transform is not None:
                    data = self.fetch_transform(data)
                # Yield batches
                for j in range(0, len(ids), self.batch_size):
                    # Use custom batch callback if provided, otherwise use default indexing
                    if self.batch_callback is not None:
                        batch_indices = slice(j, j + self.batch_size)
                        batch_data = self.batch_callback(data, batch_indices)
                    else:
                        batch_data = data[j : j + self.batch_size]
                    # Call batch transform if provided
                    if self.batch_transform is not None:
                        batch_data = self.batch_transform(batch_data)
                    yield batch_data

    def set_mode(self, mode):
        """Set dataset mode.

        Args:
            mode (str): One of 'train', 'training', 'eval', 'val', 'evaluation', 'test', 'testing'.

        Raises:
            ValueError: If mode is not recognized.
        """
        mode = mode.lower()
        if mode in ["train", "training"]:
            self.shuffle = True
            self.drop_last = True
            self.sort_before_fetch = True
            self.shuffle_before_yield = True
        elif mode in ["eval", "val", "evaluation", "test", "testing"]:
            self.shuffle = False
            self.drop_last = False
            self.sort_before_fetch = False
            self.shuffle_before_yield = False
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Must be 'train', 'training', 'eval', 'val', 'evaluation', 'test', or 'testing'."
            )

    def subset(self, indices: Union[List[int], np.ndarray]):
        """Subset the dataset to only include the specified indices.

        Parameters
        ----------
        indices : List[int] or numpy.ndarray
            Indices to subset the dataset to.
        """
        if not isinstance(indices, (list, np.ndarray)):
            raise TypeError("indices must be a list of integers or a numpy array")

        if isinstance(indices, list):
            if any(not isinstance(i, int) for i in indices):
                raise TypeError("All elements in indices must be integers")
        elif not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Numpy array must contain integers")

        len_indices = len(self.collection)
        if any(i < 0 or i >= len_indices for i in indices):
            raise IndexError("Indices out of bounds")

        self.indices = np.array(indices)

    def reset_indices(self):
        """Reset the dataset to include all indices."""
        self.indices = np.arange(len(self.collection))
