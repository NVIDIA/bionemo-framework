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


import functools
import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.core import Dataset
from nemo.utils import logging

from bionemo.data.utils import handle_index


__all__ = ["MappedDataset", "SliceDataset", "OnlineSampleMapping", "ResamplingMappedDataset", "FilteredMappedDataset"]


class SliceIndex:
    def __init__(self, dataset, start, end):
        if start < 0:
            raise ValueError(f"start must be > 0: {start}")
        if end < start:
            raise ValueError(f"end must be >= start: {end} not >= {start}")
        if end > len(dataset):
            raise ValueError(f"end must be <= dataset length: {end} not <= {len(dataset)}")

        self.start = start
        self.end = end
        self.length = int(self.end - self.start)

    def __getitem__(self, idx):
        idx = handle_index(self, idx)
        return idx + self.start

    def __len__(self):
        return self.length


class MappedDataset(Dataset, ABC):
    def __init__(self, dataset: Dataset, num_samples: Optional[int] = None, consolidate_sample_mapping: bool = False):
        """
        Produces a remapped version of a `Dataset`.
        Can be used to create a subset of a dataset, or to shuffle it.
        Chainings of `MappedDataset` are supported and are collpased for efficiency.

        Arguments:
            dataset (Dataset): dataset to remap.
            num_samples (Optional[int]): Number of samples the dataset should
                contain. The sampling strategy is based on
                `create_sample_mapping`. `create_sample_mapping` must support
                `num_samples=None` in order for this `num_samples` to be None.
            consolidate_sample_mapping (bool): If True, the sample mapping will flatten any chained MappedDatasets
                                               (default: False since memory consumption will be high for large num_samples)
        """
        self._dataset = dataset

        self.sample_mapping = self.create_sample_mapping(dataset, num_samples)

        # consolidate sample mapping if dataset is MappedDataset
        if consolidate_sample_mapping and isinstance(dataset, MappedDataset):
            self.sample_mapping = [dataset.sample_mapping[i] for i in self.sample_mapping]
            self._dataset = dataset._dataset

    def __len__(self):
        return len(self.sample_mapping)

    def get_idx(self, idx):
        idx = self.sample_mapping[handle_index(self, idx)]
        idx = handle_index(self, idx)
        return idx

    def __getitem__(self, idx):
        idx = self.get_idx(idx)
        return self._dataset[idx]

    @abstractmethod
    def create_sample_mapping(self, dataset: Dataset, num_samples: int) -> Sequence[int]:
        """Sample mapping used to remap a dataset. Implemented by child class.

        Arguments:
            dataset (Dataset): dataset to discretize
            num_samples (int): Number of samples to include in the mapped
                dataset. Child classes may ignore if sampling is not enabled.

        Returns:
            sample_mapping (ArrayLike[int]): If `sample_mapping[i] == j`,
            the `i`th entry in this dataset will be `j`th entry of the original
            dataset.

        """
        raise NotImplementedError()


class SliceDataset(MappedDataset):
    def __init__(self, dataset: Dataset, start: int = 0, end: int = -1):
        """Slices a dataset on the fly.

        Args:
            dataset (Dataset): Dataset to slice
            start (int): First index of slice
            end (int): Last index of slice (exclusive)

        """
        self.start = handle_index(dataset, start)
        self.end = handle_index(dataset, end)
        super().__init__(dataset, None)

    def create_sample_mapping(self, dataset: Dataset, num_samples: Optional[int]) -> Sequence[int]:
        """Creates a sample mapping for trimming the `dataset` to length
        based on the slice arguments.

        Arguments:
            dataset (Dataset): Dataset to slice
            num_samples (Optional[int]): Ignored

        """
        return SliceIndex(dataset, self.start, self.end)


def _infer_kwarg_values(cfg, data_prefix, max_seq_length, seed):
    if data_prefix is None:
        data_prefix = cfg.get("data_prefix")

    if max_seq_length is None:
        max_seq_length = cfg["seq_length"] - 2

    if seed is None:
        seed = cfg["seed"]

    return data_prefix, max_seq_length, seed


class Uniref90ClusterMappingDataset(MappedDataset):
    """
    MappedDataset class with 'create_sample_mapping' implementation. 'create_sample_mapping(dataset)'

    Resulting `sample_mapping` field maps nemo_upsampling_dataset indicies to uniref90_dataset indicies.


    Cluster ID      Cluster Name    Cluster members
    UniRef50_A0A007 Cluster: MoeK5  A0A007
    UniRef50_A0A009F4B0     Cluster: Uncharacterized protein (Fragment)     A0A009F4B0
    UniRef50_A0A009F4U9     Cluster: Bacterial regulatory, tetR family domain protein (Fragment)    A0A009F4U9
    UniRef50_A0A009F6B7     Cluster: Hemagglutinin family protein (Fragment)        A0A009F6B7
    UniRef50_A0A009F6Q0     Cluster: Peptidase C39 family protein (Fragment)        A0A009F6Q0
    UniRef50_A0A009F6T5     Cluster: Concanavalin A-like lectin/glucanases superfamily protein (Fragment)   A0A009F6T5
    UniRef50_A0A009F825     Cluster: Uncharacterized protein        A0A009F825; A0A7X5RNJ4; A0A829RTI8; UPI00112C051E; UPI001580B9E7; UPI00158
    0D784; UPI00070F2CF0; UPI001D187985; UPI00227B0253
    """

    def __init__(
        self,
        uniref50_dataset: "ResamplingMappedDataset",
        uniref90_dataset: Dataset,
        cluster_map_starts_fn: str,
        cluster_map_counts_fn: str,
        data_prefix: str,
        seed=None,
        index_mapping_dir=None,
        name=None,
        buffer_size=int(1e6),
    ):
        self.data_prefix = data_prefix
        self.seed = seed

        self.index_mapping_dir = index_mapping_dir
        self.name = name
        self.buffer_size = buffer_size

        num_samples = 0  # This has no effect on behavior
        logging.info(f"Loading cluster map {cluster_map_counts_fn=}, {cluster_map_starts_fn=}")
        time_start = time.time()
        # Loads the memory maps into a dictionary of 'counts', 'starts'
        cluster_map = self.create_cluster_map_from_files(cluster_map_counts_fn, cluster_map_starts_fn)
        time_end = time.time()
        logging.info(f"Cluster map from json: {time_end - time_start}")
        time_start = time.time()
        sample_map_file = self._sample_mapping_filename(
            index_mapping_dir=index_mapping_dir,
            data_prefix=data_prefix,
            name=name,
            num_samples=len(uniref50_dataset),
            seed=seed,
        )
        # Creates a memory map for the sample mapping
        self.sample_map = self._create_sample_mapping(
            uniref50_dataset, sample_map_file, cluster_map, buffer_size=self.buffer_size
        )
        time_end = time.time()
        logging.info(f"Cluster sample_mapping: {time_end - time_start}")
        super().__init__(uniref90_dataset, num_samples)
        # Now we can free the dictonaries used for creating the cluster mapping
        del cluster_map

    @staticmethod
    def create_cluster_map_from_files(cluster_map_counts_fn, cluster_map_starts_fn, dtype=np.uint64) -> Dict[str, str]:
        """Creates a mapping from cluster_id to cluster_members. This is specifically for mapping samples from
        Uniref50 to Uniref90.

        Json file is expected to be an exact production (meaning, json.loads is sufficient)
        """
        return {
            "counts": np.memmap(cluster_map_counts_fn, dtype=dtype, mode="r"),
            "starts": np.memmap(cluster_map_starts_fn, dtype=dtype, mode="r"),
        }

    def _sample_mapping_filename(
        self,
        index_mapping_dir,
        data_prefix,
        name,
        seed,
        num_samples,
    ):
        # Filename of the index mapping
        if index_mapping_dir is not None:
            indexmap_filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
        else:
            indexmap_filename = data_prefix
        indexmap_filename += "_{}_custersamplemap".format(name)
        indexmap_filename += "_{}mns".format(num_samples)
        indexmap_filename += "_{}s".format(seed)
        indexmap_filename += ".npy"
        return indexmap_filename

    def _create_sample_mapping(self, uniref50_dataset, sample_mapping_file, cluster_map, buffer_size=int(1e6)):
        """
        Create a sample mapping from `uniref50_dataset` using memory-mapped file techniques.
        Utilizes the `uniref50_dataset` to generate a sample map and writes/reads this map to/from a memory-mapped file.

        Args:
            uniref50_dataset (list of str): A dataset where each item is a string in the format of
                ["u50_id0","u50_id1", "u50_id2", ...].
            sample_mapping_file (str): Path to the file where the sample mapping will be stored as a memory-mapped array.
            cluster_map (dict): A dictionary containing mapping information between clusters. Expected to have keys 'counts' and 'starts'.
            buffer_size (int, optional): The size of the buffer used during the mapping creation. Default is 1e6.

        Returns:
            numpy.memmap: A memory-mapped array containing the generated sample mapping.

        Raises:
            RuntimeError: If the torch.distributed process group is not initialized.
        """
        # TODO: promote this to its own ABC (MemMappedDataset)
        n_samples = len(uniref50_dataset)
        rank = torch.distributed.get_rank()
        is_distributed = True
        if rank == 0:
            sample_map = np.memmap(sample_mapping_file, dtype=int, mode="w+", shape=(n_samples,))

            logging.info(f"Creating uf50->uf90 sample mapping for {n_samples} samples on rank {rank}.")
            cluster_counts = cluster_map["counts"]
            buffer_start = 0
            np.random.seed(self.seed)
            while buffer_start < n_samples:
                buffer_end = buffer_start + buffer_size
                cluster_slice = uniref50_dataset.sample_mapping[buffer_start:buffer_end]
                sample_slice = np.random.rand(len(cluster_slice))
                cluster_count_slice = cluster_counts[cluster_slice]
                sampled_relative_indices = np.floor(sample_slice * cluster_count_slice)
                sample_map[buffer_start:buffer_end] = cluster_map["starts"][cluster_slice] + sampled_relative_indices
                buffer_start = buffer_end
            sample_map.flush()

        if is_distributed:
            torch.distributed.barrier()
        logging.info(f"Loading uf50-uf90 sample mapping on rank {rank}.")
        sample_map = np.memmap(sample_mapping_file, dtype=int, mode="r")

        return sample_map

    def create_sample_mapping(self, dataset, num_samples=None) -> Sequence[int]:
        return self.sample_map


class OnlineSampleMapping:
    """
    This class replaces NeMo's get_samples_mapping function which pre-computes.
    It is used to create a sample mapping for certain number of samples, including
    pseudo-random shuffling.
    The sampler allows to down, or upsample a given dataset.
    Shuffling leads to pseudo-random shuffling, where blocks are shuffled,
    and each block is internally shuffled.
    """

    def __init__(
        self,
        dataset_size: int,
        num_samples: int,
        block_size: int = 1000000,
        cache_maxsize: int = 2,
        seed: int = 1,
        shuffle: bool = True,
        truncate_to_block_boundary: bool = False,
    ):
        """
        Args:
            dataset_size (int): Size of the dataset.
            num_samples (int): Number of samples the dataset should contain.
            block_size (int): Size of each sample block. This is used to shuffle the samples.
                              None will be replaced with dataset size.
            cache_maxsize (int): Maximum size of the blocks cache for the get_sample_block function.
            seed (int): Seed for the random number generator used for shuffling.
            shuffle (bool): Whether to shuffle the samples.
            truncate_to_block_boundary (bool): Whether to truncate the last block to the block boundary (could drop samples).
        """
        self.dataset_size = dataset_size
        self.num_samples = num_samples
        self.block_size = block_size if block_size is not None else self.dataset_size
        self.cache_maxsize = cache_maxsize
        self.seed = seed
        self.shuffle = shuffle
        self.truncate_to_block_boundary = truncate_to_block_boundary

        # we need at least num_samples (up-sampling) or dataset_size samples (correct down-sampling)
        self.required_samples = max(self.num_samples, self.dataset_size)
        # block size cannot be larger than dataset size
        self.block_size = min(self.block_size, self.dataset_size)
        # reduce the last block if needed, to match the required number of samples
        last_block_size = self.required_samples % self.block_size
        # store required blocks to cover num_samples samples and dataset_size samples
        self.num_blocks = int(np.ceil(self.required_samples / self.block_size))

        # if required, truncate the last block to the block boundary
        if self.truncate_to_block_boundary and last_block_size:
            # update num_samples to account for truncated last block only if needed
            if self.required_samples == self.num_samples:
                self.num_samples -= last_block_size

            # apdate num_blocks to account for truncated last block
            self.num_blocks -= 1
            self.required_samples -= last_block_size
            last_block_size = 0

        # create a list of blocks (should cover the entire dataset for correct down sampling)
        block_idx_list = np.arange(self.num_blocks)
        # compute the size of each block
        block_size_list = np.full(self.num_blocks, self.block_size)
        if last_block_size:
            block_size_list[-1] = last_block_size
            self.use_digitize = True
        else:
            self.use_digitize = False
        if shuffle:
            local_rng = np.random.RandomState(seed=self.seed)
            idx = local_rng.permutation(np.arange(self.num_blocks))
            block_idx_list = block_idx_list[idx]
            block_size_list = block_size_list[idx]

        # store only required number of blocks
        self.block_idx_list = block_idx_list
        self.block_size_list = block_size_list
        self.block_bins = np.cumsum(block_size_list)

        # NOTE: MAKE get_sample_block A CACHED FUNCTION!!!
        self.get_sample_block = functools.lru_cache(maxsize=cache_maxsize, typed=False)(self.get_sample_block)

    def __str__(self):
        return f"OnlineSampleMapping(dataset_size={self.dataset_size}, num_samples={self.num_samples}, block_size={self.block_size}, cache_maxsize={self.cache_maxsize}, seed={self.seed}, shuffle={self.shuffle}, truncate_to_block_boundary={self.truncate_to_block_boundary})"

    def __getitem__(self, idx: int) -> int:
        # handle slices
        if isinstance(idx, slice):
            slc = idx
            start, stop, step = slc.start, slc.stop, slc.step

            # Handle None values
            start = handle_index(self, start if start is not None else 0)
            if start >= self.num_samples:
                start = self.num_samples
            stop = handle_index(self, stop if stop is not None else self.num_samples)
            if stop >= self.num_samples:
                stop = self.num_samples
            step = step if step is not None else 1
            sample_slice = [self[idx] for idx in range(start, stop, step)]
            return sample_slice
        # handle indices
        else:
            # If the index is out of range, raise IndexError
            if idx >= self.num_samples:
                raise IndexError("Index out of range")

            # support negative indices
            if idx < 0:
                idx += self.num_samples

                if idx < 0:
                    raise IndexError("Index out of range")

            # fetch the block sample index
            if self.use_digitize:
                block_idx = np.digitize(idx, self.block_bins)
            else:
                block_idx = idx // self.block_size
            sample_block = self.get_sample_block(block_idx)

            # use the local index to fetch the sample
            local_idx = idx - self.block_bins[block_idx]
            sample_idx = sample_block[local_idx]

            return sample_idx

    def __len__(self) -> int:
        return self.num_samples

    def __reduce__(self):
        """Add support for pickling. Needed due to functools.lru_cache."""
        # Return a tuple with a callable and arguments to recreate the object
        return (
            self.__class__,
            (
                self.dataset_size,
                self.num_samples,
                self.block_size,
                self.cache_maxsize,
                self.seed,
                self.shuffle,
                self.truncate_to_block_boundary,
            ),
        )

    def __reduce_ex__(self, protocol):
        # Optional method that defines the protocol version
        return self.__reduce__()

    def get_sample_block(self, block_idx: int) -> np.ndarray:
        """
        Returns a block of samples of size self.block_size, shuffled if needed.
        NOTE: This method will be cached using functools.lru_cache for efficiency during construction.
        """
        if block_idx >= self.num_blocks:
            raise IndexError(f"block_idx {block_idx} is out of range. Maximum block_idx is {self.num_blocks-1}")

        # recover index of original block (before shuffling)
        start_idx = self.block_idx_list[block_idx] * self.block_size
        end_idx = start_idx + self.block_size_list[block_idx]
        sample_block = np.arange(start_idx, end_idx)

        # shuffle if needed
        if self.shuffle:
            local_rng = np.random.RandomState(seed=self.seed + block_idx)
            sample_block = local_rng.permutation(sample_block)

        # project indices to the dataset size
        sample_block = sample_block % self.dataset_size

        return sample_block


class ResamplingMappedDataset(MappedDataset):
    """Upsamples / downsamples a dataset to a target length by repeating samples."""

    def __init__(
        self,
        dataset,
        num_samples=None,
        data_prefix=None,
        max_seq_length=None,
        seed=None,
        cfg=None,  # This is usually in model.data subfield
        index_mapping_dir=None,
        name=None,
    ):
        """
        Resamples a dataset to a specified target length by adjusting the sample distribution.
        This can involve either upsampling (repeating samples) or downsampling (reducing samples)
        to match the target sample size. It supports different strategies for resampling, such as
        in-memory mapping or online resampling, based on the provided configuration.

        Parameters:
            dataset (Dataset): The dataset to be resampled.
            num_samples (int, optional): The target number of samples after resampling. If not provided,
                the length of the dataset is used.
            data_prefix (str, optional): A prefix used for data processing or file naming. This can be
                inferred from the configuration if not provided directly. Used in memmap index mapping
                strategy.
            max_seq_length (int, optional): The maximum sequence length for samples in the dataset. This
                can also be inferred from the configuration.
            seed (int, optional): Seed for random number generation, used in shuffling the dataset. This
                can be provided directly or inferred from the configuration.
            index_mapping_dir (str, optional): Directory path where index mapping files are stored or will
                be generated. Used in memmap index mapping strategy.
            name (str, optional): An optional name for the dataset. If not provided, it is inferred from
                the data_prefix or set to None.
            cfg (dict, optional): A configuration dictionary, usually found in model.data, which can
                contain default values for other parameters if they are not explicitly provided. The following keys
                are used by this class:
                - `index_mapping_type`: Strategy for index mapping ("online" or "memmap").
                - `block_size`: Size of each block of samples in online sampling.
                - `cache_maxsize`: Maximum cache size for online sampling.
                - `shuffle`: Whether to shuffle the dataset during resampling.
                - `truncate_to_block_boundary`: Whether to truncate the sample mapping to block boundaries in online sampling.

        Raises:
            ValueError: If num_samples is set to 0, as it's not possible to resample to no samples.
            ValueError: If an unknown index_mapping_type is specified in the configuration.

        This class supports dynamic adjustment of the sample size of a dataset through different
        resampling strategies, enabling flexible dataset manipulation for training or evaluation purposes.
        """
        self.data_prefix, self.max_seq_length, self.seed = _infer_kwarg_values(cfg, data_prefix, max_seq_length, seed)
        self.index_mapping_dir = index_mapping_dir
        self.name = name
        self.cfg = cfg

        super().__init__(dataset, num_samples)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int) -> Sequence[int]:
        if num_samples is None:
            num_samples = len(dataset)

        if num_samples == 0:
            raise ValueError("Number of samples is 0. Cannot be sampled.")

        # TODO: change default to 'online' once validated (and remove support in memmap?)
        index_mapping_type = self.cfg.get("index_mapping_type", "memmap")
        # map dataset samples to the desired num_samples
        if index_mapping_type == "online":
            samples_mapping = OnlineSampleMapping(
                dataset_size=len(dataset),
                num_samples=num_samples,
                block_size=self.cfg.get("block_size", 1000000),
                cache_maxsize=self.cfg.get("cache_maxsize", 2),
                seed=self.seed,
                shuffle=self.cfg.get("shuffle", True),
                truncate_to_block_boundary=self.cfg.get("truncate_to_block_boundary", False),
            )

            # test if samples_mapping.num_samples == num_samples and alert if truncation happened
            if samples_mapping.num_samples != num_samples:
                logging.warning(
                    f"Requested {num_samples} samples, but only {samples_mapping.num_samples} samples were generated due to truncation to block_size boundary."
                )
        elif index_mapping_type == "memmap":
            # TODO: remove support for memmap and use OnlineSampleMapping instead (more efficient)
            samples_mapping = get_samples_mapping(
                indexed_dataset=dataset,
                data_prefix=self.data_prefix,
                num_epochs=None,
                max_num_samples=num_samples,
                # account for <BOS> / <EOS>
                max_seq_length=self.max_seq_length,
                short_seq_prob=0,
                seed=self.seed,
                name=self.data_prefix.split("/")[-1] if self.name is None else self.name,
                binary_head=False,
                index_mapping_dir=self.index_mapping_dir,
            )
            # truncate to max number of num_samples (None is all samples)
            samples_mapping = samples_mapping[:num_samples, 0]
        else:
            raise ValueError(f'Unknown index_mapping_type: {index_mapping_type}, expected "online" or "memmap"')

        return samples_mapping


class FilteredIdxMappedDataset(MappedDataset):
    """Same as FilteredMappedDataset, but operates on sample indices instead of samples. Useful if __getitem__ is not performant."""

    def __init__(self, dataset, criterion_fn, num_samples=None):
        """
        Args:
            dataset (Dataset): Dataset to filter
            critetion_fn (Callable): Function that takes in a sample index and returns True if the sample should be kept
        """
        self.criterion_fn = criterion_fn
        super().__init__(dataset=dataset, num_samples=num_samples)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        samples_mapping = np.where(list(map(self.criterion_fn, np.arange(len(dataset)))))[0]
        return samples_mapping


class FilteredMappedDataset(MappedDataset):
    """Filters samples from a dataset based on a criterion function by mapping the dataset samples."""

    def __init__(self, dataset, criterion_fn, num_samples=None):
        """
        Args:
            dataset (Dataset): Dataset to filter
            critetion_fn (Callable): Function that takes in a sample and returns True if the sample should be kept
        """
        self.criterion_fn = criterion_fn
        super().__init__(dataset=dataset, num_samples=num_samples)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        """Creates a sample mapping for filtering the `dataset` based on the criterion function."""
        samples_mapping = np.where(list(map(self.criterion_fn, dataset)))[0]
        ds_size = len(dataset)
        filtered_ds_size = len(samples_mapping)
        logging.debug(
            f"Filtered out (ignored) {ds_size - filtered_ds_size} samples ( {filtered_ds_size} / {ds_size} )"
        )

        # truncate to max number of num_samples (None is all samples)
        samples_mapping = samples_mapping[:num_samples]
        return samples_mapping


class IndexMappedDataset(MappedDataset):
    """Maps a dataset to a new dataset based on provides indices."""

    def __init__(self, dataset, indices, copy: bool = False):
        """
        Args:
            dataset (Dataset): Dataset to filter
            indices: indices to keep
            copy: Create a deep copy of the underlying dataset (can prevent UAF mistakes in DDP)
        """
        self.sample_mapping = indices
        super().__init__(dataset=deepcopy(dataset), num_samples=None)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        """Creates a sample mapping for filtering the `dataset` based on the criterion function."""
        return self.sample_mapping
