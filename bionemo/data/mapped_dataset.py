# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

import os
import time
from abc import abstractmethod
from typing import Dict, Optional

import numpy as np
import torch
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.core import Dataset
from nemo.utils import logging

from bionemo.data.utils import handle_index


__all__ = ['MappedDataset', 'SliceDataset', 'NeMoUpsampling', 'FilteredMappedDataset']


class SliceIndex:
    def __init__(self, dataset, start, end):
        if start < 0:
            raise ValueError(f'start must be > 0: {start}')
        if end < start:
            raise ValueError(f'end must be >= start: {end} not >= {start}')
        if end > len(dataset):
            raise ValueError(f'end must be <= dataset length: {end} not <= {len(dataset)}')

        self.start = start
        self.end = end
        self.length = int(self.end - self.start)

    def __getitem__(self, idx):
        idx = handle_index(self, idx)
        return idx + self.start

    def __len__(self):
        return self.length


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, num_samples: Optional[int] = None, consolidate_sample_mapping=True):
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
    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
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

    def create_sample_mapping(self, dataset: Dataset, num_samples: Optional[int]):
        """Creates a sample mapping for trimming the `dataset` to length
        based on the slice arguments.

        Arguments:
            dataset (Dataset): Dataset to slice
            num_samples (Optional[int]): Ignored

        """
        return SliceIndex(dataset, self.start, self.end)


def _infer_kwarg_values(cfg, data_prefix, max_seq_length, seed):
    if data_prefix is None:
        data_prefix = cfg.get('data_prefix')

    if max_seq_length is None:
        max_seq_length = cfg['seq_length'] - 2

    if seed is None:
        seed = cfg['seed']

    return data_prefix, max_seq_length, seed


def name_index_file(
    num_epochs, max_num_samples, index_mapping_dir, data_prefix, name, max_seq_length, short_seq_prob, seed
):
    '''This enforces the name index structure to make sure when parameters change, we can still load the same index.
    This is useful for any sample_indexed file.


    This should be used like this:
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
    '''
    import os

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples " "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    if index_mapping_dir is not None:
        indexmap_filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'
    return indexmap_filename


class Uniref90ClusterMappingDataset(MappedDataset):
    '''
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
    '''

    def __init__(
        self,
        uniref50_dataset: 'NeMoUpsampling',
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
        '''Creates a mapping from cluster_id to cluster_members. This is specifically for mapping samples from
        Uniref50 to Uniref90.

        Json file is expected to be an exact production (meaning, json.loads is sufficient)
        '''
        return {
            'counts': np.memmap(cluster_map_counts_fn, dtype=dtype, mode='r'),
            'starts': np.memmap(cluster_map_starts_fn, dtype=dtype, mode='r'),
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
        indexmap_filename += '_{}_custersamplemap'.format(name)
        indexmap_filename += '_{}mns'.format(num_samples)
        indexmap_filename += '_{}s'.format(seed)
        indexmap_filename += '.npy'
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
            sample_map = np.memmap(sample_mapping_file, dtype=int, mode='w+', shape=(n_samples,))

            logging.info(f"Creating uf50->uf90 sample mapping for {n_samples} samples on rank {rank}.")
            cluster_counts = cluster_map['counts']
            buffer_start = 0
            np.random.seed(self.seed)
            while buffer_start < n_samples:
                buffer_end = buffer_start + buffer_size
                cluster_slice = uniref50_dataset.sample_mapping[buffer_start:buffer_end]
                sample_slice = np.random.rand(len(cluster_slice))
                cluster_count_slice = cluster_counts[cluster_slice]
                sampled_relative_indices = np.floor(sample_slice * cluster_count_slice)
                sample_map[buffer_start:buffer_end] = cluster_map['starts'][cluster_slice] + sampled_relative_indices
                buffer_start = buffer_end
            sample_map.flush()

        if is_distributed:
            torch.distributed.barrier()
        logging.info(f"Loading uf50-uf90 sample mapping on rank {rank}.")
        sample_map = np.memmap(sample_mapping_file, dtype=int, mode='r')

        return sample_map

    def create_sample_mapping(self, dataset, num_samples=None) -> np.array:
        return self.sample_map


class NeMoUpsampling(MappedDataset):
    """Upsamples a dataset to a target length by repeating samples."""

    def __init__(
        self,
        dataset,
        num_samples=None,
        data_prefix=None,
        max_seq_length=None,
        seed=None,
        cfg=None,
        index_mapping_dir=None,
        name=None,
    ):
        self.data_prefix, self.max_seq_length, self.seed = _infer_kwarg_values(cfg, data_prefix, max_seq_length, seed)
        self.index_mapping_dir = index_mapping_dir
        self.name = name

        super().__init__(dataset, num_samples)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        if num_samples is None:
            num_samples = len(dataset)

        if num_samples == 0:
            raise ValueError('Number of samples is 0. Cannot be sampled.')

        # map dataset samples to the desired num_samples
        samples_mapping = get_samples_mapping(
            indexed_dataset=dataset,
            data_prefix=self.data_prefix,
            num_epochs=None,
            max_num_samples=num_samples,
            # account for <BOS> / <EOS>
            max_seq_length=self.max_seq_length,
            short_seq_prob=0,
            seed=self.seed,
            name=self.data_prefix.split('/')[-1] if self.name is None else self.name,
            binary_head=False,
            index_mapping_dir=self.index_mapping_dir,
        )

        samples_mapping = samples_mapping[:num_samples, 0]
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

    def __init__(self, dataset, indices):
        """
        Args:
            dataset (Dataset): Dataset to filter
            indices: indices to keep
        """
        self.sample_mapping = indices
        super().__init__(dataset=dataset, num_samples=None)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        """Creates a sample mapping for filtering the `dataset` based on the criterion function."""
        return self.sample_mapping
