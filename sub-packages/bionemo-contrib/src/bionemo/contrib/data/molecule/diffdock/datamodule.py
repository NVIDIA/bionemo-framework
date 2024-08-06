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

from enum import Enum, auto
import glob
import pickle
import random
from typing import Dict, Generator, List, Optional
import lightning as L
from torch_geometric.data.hetero_data import HeteroData
import webdataset as wds

from bionemo.contrib.data.molecule.diffdock.utils import (
    pickles_to_tars
    )


class Split(Enum):
    train = auto()
    val = auto()
    test = auto()


class ScoreModelWDS(L.LightningDataModule):

    """lightning APIs to process score model data and setup dataset and
    dataloader"""

    def __init__(self, dir_heterodata : str, suffix_heterodata : str,
                 prefix_dir_tars_wds : str, names_subset : Dict[Split,
                                                                List[str]],
                 global_batch_size : int, n_workers_dataloader : int,
                 pipeline_wds : Optional[Dict[Split, Generator[HeteroData, None,
                                                               None]]] = None,
                 pipeline_prebatch_wld : Optional[Dict[Split,
                                                       Generator[HeteroData,
                                                                 None, None]]] =
                 None, pin_memory_dataloader : bool = True, prefix_tars_wds :
                 str = "heterographs", n_tars_wds : Optional[int] = None,
                 seed_rng_shfl : int = 0):
        """constructor

        Args:
            dir_heterodata (str): input directory of PyG HeteroData pickled
                files
            suffix_heterodata (str): filename suffix of the input data in
                dir_heterodata. This is also used as the key mapped to the
                tarballed HeteroData object in the webdataset
            prefix_dir_tars_wds (str): directory name prefix to store the output
                webdataset tar files. The actual directories storing the train, val
                and test sets will be suffixed with "train", "val" and "test"
                respectively.
            names_subset (Dict[Split, List[str]]): list of complex names to be
                included in each of the split
            global_batch_size (int): size of batch summing across nodes in Data
                Distributed Parallel, i.e., local_batch_size * n_nodes. NOTE:
                this data module doesn't rely on the input `global_batch_size`
                for batching the samples. The batching is supposed to be done as
                a part of the input `pipeline_prebatch_wld`. `global_batch_size`
                is only used to compute a (pseudo-) epoch length for the data
                loader so that the loader yield approximately n_samples //
                global_batch_size batches
            n_workers_dataloader (int): number of data loading workers (passed
                to pytorch dataloader)
            seed_rng_shfl (int): seed to the random number generators used in
            data loading time for shuffling

        Kwargs:
            pipeline_wds (Optional[Dict[Split, Generator[HeteroData, None,
                None]]]): a dictionary of webdatast composable, i.e., functor
                that maps a generator to another generator that transforms the
                data sample yield from the dataset object, for different splits
            pipeline_prebatch_wld (Optional[Dict[Split, Generator[HeteroData,
                None, None]]]): a dictionary of webloader composable, i.e.,
                functor that maps a generator to another generator that
                transforms the data sample yield from the WebLoader object, for
                different splits. NOTE: this is applied before batching is yield
                from the WebLoader
            pin_memory_dataloader (bool): whether to use pin memory in pytorch
                dataloader
            prefix_tars_wds (str): name prefix to output webdataset tar files
            n_tars_wds (int): attempt to create at least this number of webdataset shards


        """
        super().__init__()

        self._dir_heterodata = dir_heterodata
        self._suffix_heterodata = suffix_heterodata
        self._n_tars_wds = n_tars_wds
        self._prefix_dir_tars_wds = prefix_dir_tars_wds
        self._prefix_tars_wds = prefix_tars_wds

        keys_subset = names_subset.keys()
        if not (Split.train in keys_subset and Split.val in keys_subset):
            raise RuntimeError("Input names_subset must be defined for the "\
                               "train and val splits")

        self._names_subset = names_subset

        self._sizes = {
            split : len(self._names_subset[split]) for split in
            self._names_subset.keys()
            }

        self._dirs_tars_wds = {
            Split.train : f"{self._prefix_dir_tars_wds}train",
            Split.val : f"{self._prefix_dir_tars_wds}val",
            Split.test : f"{self._prefix_dir_tars_wds}test",
            }

        self._pipeline_wds = pipeline_wds
        self._pipeline_prebatch_wld = pipeline_prebatch_wld

        self._global_batch_size = global_batch_size
        self._n_workers_dataloader = n_workers_dataloader
        self._pin_memory_dataloader = pin_memory_dataloader
        self._seed_rng_shfl = seed_rng_shfl

        # to be created later in setup
        self._dataset = dict()


    def _complex_graph_to_tar(self, complex_graph : HeteroData):
        """map input complex graph to webdataset tar file conforming to its
        format requirement

        Args:
            complex_graph (HeteroData): input complex graph

        Returns: webdataset tar file segment (dict)

        """
        return {
            "__key__": complex_graph.name.replace(".", "-"),
            self._suffix_heterodata: pickle.dumps(complex_graph)
            }


    def prepare_data(self) -> None:
        """This is called only by the main process by the Lightning workflow. Do
        not rely on this data module object's state update here as there is no
        way to communicate the state update to other subprocesses

        Returns: None
        """
        for split in self._names_subset.keys():
            # create wds shards (tar files) for train set
            pickles_to_tars(self._dir_heterodata,
                            self._suffix_heterodata,
                            self._names_subset[split],
                            self._dirs_tars_wds[split],
                            self._prefix_tars_wds,
                            self._complex_graph_to_tar,
                            min_num_shards=self._n_tars_wds)


    def _setup_wds(self, split : Split) -> wds.WebDataset:
        """setup webdataset and webloader. This is called by setup()

        Args:
            split (Split): train, val or test split

        Returns: WebDataset

        """
        is_train = split == Split.train
        urls = sorted(glob.glob(
            f"{self._dirs_tars_wds[split]}/{self._prefix_tars_wds}-*.tar")
            )
        dataset = (
            wds.WebDataset(urls, shardshuffle=is_train,
                           nodesplitter=wds.split_by_node,
                           seed=self._seed_rng_shfl)
            .decode()
            .extract_keys(f"*.{self._suffix_heterodata}")
            )
        if (self._pipeline_wds is not None and
                self._pipeline_wds[split] is not None):
            dataset = dataset.compose(self._pipeline_wds[split])
        if is_train:
            dataset = dataset.shuffle(size=16,
                                      rng=random.Random(self._seed_rng_shfl))
        return dataset

    def setup(self, stage: str) -> None:
        """This is called on all Lightning-managed nodes in a multi-node
        training session


        Args:
            stage (str): "fit", "test" or "predict"
        Returns: None
        """
        if stage == "fit":
            self._dataset[Split.train] = self._setup_wds(Split.train)
            self._dataset[Split.val] = self._setup_wds(Split.val)
        elif stage == "test":
            self._dataset[Split.test] = self._setup_wds(Split.test)
        else:
            raise NotImplementedError("Data setup with stage = {stage}\
                                      is not implmented")

    def _setup_dataloader(self, split : Split) -> wds.WebLoader:
        """setup the dataloader for the input dataset split

        Args:
            split (Split): input split type

        Returns: WebLoader object

        """
        dataset = self._dataset[split]
        n_samples = len(self._names_subset[split])
        n_batches = ((n_samples + self._global_batch_size - 1)
                     // self._global_batch_size)
        loader = wds.WebLoader(dataset,
                num_workers=self._n_workers_dataloader,
                pin_memory=self._pin_memory_dataloader,
                batch_size=None
            ).shuffle(5000, rng=random.Random(self._seed_rng_shfl))

        if (self._pipeline_prebatch_wld is not None and
                self._pipeline_prebatch_wld[split] is not None):
            loader = loader.compose(
                self._pipeline_prebatch_wld[split])

        if split == Split.train:
            loader = loader.select(lambda x: len(x) > 1)

        loader = loader.with_epoch(n_batches)

        # strange features required by nemo optimizer lr_scheduler
        loader.dataset = dataset  # seems like only length is used, webloader doesn't have this attr
        loader.batch_size = self._global_batch_size
        loader.drop_last = False
        return loader


    def train_dataloader(self) -> wds.WebLoader:
        assert self._dataset[Split.train] is not None,\
            f"dataset for train has not been setup"
        return self._setup_dataloader(Split.train)


    def val_dataloader(self) -> wds.WebLoader:
        assert self._dataset[Split.val] is not None,\
            f"dataset for val has not been setup"
        return self._setup_dataloader(Split.val)


    def test_dataloader(self) -> wds.WebLoader:
        assert self._dataset[Split.test] is not None,\
            f"dataset for test has not been setup"
        return self._setup_dataloader(Split.test)


    def predict_dataloader(self) -> wds.WebLoader:
        raise NotImplementedError("predict dataloader not implemented")
