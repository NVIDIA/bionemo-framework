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
from functools import partial
import glob
import pickle
import random
from typing import Set, Optional, Tuple
import lightning as L
import torch
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.loader.dataloader import Collater
import webdataset as wds

from bionemo.contrib.data.molecule.diffdock.utils import (
    pickles_to_tars, SizeAwareBatching, estimate_size
    )
from bionemo.contrib.model.molecule.diffdock.utils.diffusion import (
    t_to_sigma, GenerateNoise)


class Split(Enum):
    train = auto()
    val = auto()
    test = auto()


class ScoreModelWDS(L.LightningDataModule):

    """lightning APIs to process score model data and setup dataset and
    dataloader"""

    def __init__(self, dir_heterodata : str, suffix_heterodata : str,
                 prefix_dir_tars_wds : str, names_subset_train : Set[str],
                 names_subset_val : Set[str], local_batch_size : int,
                 global_batch_size : int, n_workers_dataloader : int,
                 tr_sigma_minmax : Tuple[float, float]
                 = (0.1, 19), rot_sigma_minmax : Tuple[float, float] = (0.03,
                                                                        1.55),
                 tor_sigma_minmax : Optional[Tuple[float, float]] = (0.0314,
                                                                     3.14),
                 is_all_atom : bool = False, apply_size_control : Tuple[bool,
                                                                        bool,
                                                                        bool] =
                 (True, False, False), pin_memory_dataloader : bool = True,
                 prefix_tars_wds : str = "heterographs",
                 n_tars_wds : Optional[int] = None, names_subset_test :
                 Optional[Set[str]] = None, seed_rng_shfl : int = 0):
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
            names_subset_train (Set[str]): list of complex names to be included
                in the training data
            names_subset_val (Set[str]): list of complex names to be included
                in the validation data
            local_batch_size (int): size of batch for each node
            global_batch_size (int): size of batch summing across nodes in Data
                Distributed Parallel, i.e., local_batch_size * n_nodes
            n_workers_dataloader (int): number of data loading workers (passed
                to pytorch dataloader)
            seed_rng_shfl (int): seed to the random number generators used in
            data loading time for shuffling

        Kwargs:
            tr_sigma_minmax (Tuple[float, float]): min and max sigma for the
                translational component during diffusion
            rot_sigma_minmax (Tuple[float, float]): min and max sigma for the
                rotational component during diffusion
            tor_sigma_minmax (Optional[Tuple[float, float]]): min and max sigma
                for the torsional component during diffusion
            is_all_atom (bool): whether to treat the data as all-atom system
                during noise transformation
            apply_size_control(Tuple[bool, bool, bool]): whether to use
                SizeAwareBatching for the respective train, val and test data
            pin_memory_dataloader (bool): whether to use pin memory in pytorch
                dataloader
            prefix_tars_wds (str): name prefix to output webdataset tar files
            n_tars_wds (int): attempt to create at least this number of webdataset shards
            names_subset_test (Optional[Set[str]]): list of complex names to be included
                in the test data


        """
        super().__init__()

        self._dir_heterodata = dir_heterodata
        self._suffix_heterodata = suffix_heterodata
        self._n_tars_wds = n_tars_wds
        self._prefix_dir_tars_wds = prefix_dir_tars_wds
        self._prefix_tars_wds = prefix_tars_wds
        self._names_subset_train = names_subset_train
        self._names_subset_val = names_subset_val
        self._names_subset_test = names_subset_test

        self._sizes = {
            Split.train : len(self._names_subset_train),
            Split.val : len(self._names_subset_val),
            Split.test : len(self._names_subset_test) if
            self._names_subset_test is not None else None,
            }

        self._dirs_tars_wds = {
            Split.train : f"{self._prefix_dir_tars_wds}train",
            Split.val : f"{self._prefix_dir_tars_wds}val",
            Split.test : f"{self._prefix_dir_tars_wds}test",
            }

        self._tr_sigma_min, self._tr_sigma_max = tr_sigma_minmax
        self._rot_sigma_min, self._rot_sigma_max = rot_sigma_minmax
        self._tor_sigma_min, self._tor_sigma_max = (None, None)
        self._no_torsion = True
        if tor_sigma_minmax is not None:
            self._tor_sigma_min, self._tor_sigma_max = tor_sigma_minmax
            self._no_torsion = False
        # TODO: the all-atom arg to set_time should be inferred from the
        # complex_graph arg so we don't have to pass it all-the-way down
        self._is_all_atom = is_all_atom

        self._local_batch_size = local_batch_size
        self._global_batch_size = global_batch_size
        self._use_dynamic_batch_size = {
            Split.train : apply_size_control[0],
            Split.val : apply_size_control[1],
            Split.test : apply_size_control[2],
            }
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
        # create wds shards (tar files) for train set
        pickles_to_tars(self._dir_heterodata,
                        self._suffix_heterodata,
                        self._names_subset_train,
                        self._dirs_tars_wds[Split.train],
                        self._prefix_tars_wds,
                        self._complex_graph_to_tar,
                        min_num_shards=self._n_tars_wds)

        # create wds shards (tar files) for val set
        pickles_to_tars(self._dir_heterodata,
                        self._suffix_heterodata,
                        self._names_subset_val,
                        self._dirs_tars_wds[Split.val],
                        self._prefix_tars_wds,
                        self._complex_graph_to_tar,
                        min_num_shards=self._n_tars_wds)

        if self._names_subset_test is not None:
            # create wds shards (tar files) for test set
            pickles_to_tars(self._dir_heterodata,
                            self._suffix_heterodata,
                            self._names_subset_test,
                            self._dirs_tars_wds[Split.test],
                            self._prefix_tars_wds,
                            self._complex_graph_to_tar,
                            min_num_shards=self._n_tars_wds)



    def _setup_wds(self, split : Split) -> wds.WebDataset:
        """setup webdataset and webloader. This is called by setup()

        Args:
            split (Split): train, val or test split

        Returns: WebDataset

        """
        random.seed(self._seed_rng_shfl)
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
        dataset = dataset.compose(
            GenerateNoise(partial(t_to_sigma,
                                  self._tr_sigma_min, self._tr_sigma_max,
                                  self._rot_sigma_min, self._rot_sigma_max,
                                  self._tor_sigma_min,
                                  self._tor_sigma_max),
                          self._no_torsion, self._is_all_atom,
                          copy_ref_pos=(split == Split.val)))
        # sandwiched here to mirror the original DiffDock FW implementation
        size = self._sizes[split]
        # FIXME: remove this with_length since it's overriden later anyway
        dataset = dataset.with_length(size)
        if is_train:
            dataset = dataset.shuffle(size=5000,
                                      rng=random.Random(self._seed_rng_shfl))
        n_batches = ((size + self._global_batch_size - 1)
                     // self._global_batch_size)
        if not self._use_dynamic_batch_size[split]:
            dataset = (
                dataset.batched(self._local_batch_size,
                                collation_fn=Collater(dataset=[],
                                                      follow_batch=None,
                                                      exclude_keys=None))
                .with_epoch(n_batches)
                .with_length(n_batches)
                )
        else:
            f_batching = SizeAwareBatching(
                max_total_size=0.85 * torch.cuda.get_device_properties("cuda:0").total_memory / 2**20,
                size_fn=estimate_size,
                )
            dataset = dataset.compose(f_batching).with_epoch(n_batches)
        if is_train:
            dataset = dataset.select(lambda x: len(x) > 1)

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

    def _setup_dataloader(self, dataset : wds.WebDataset) -> wds.WebLoader:
        """wrap the input dataset into a WebLoader

        Args:
            dataset (wds.WebDataset): input dataset object

        Returns: WebLoader object

        """
        if not hasattr(dataset, "__len__"):
            raise RuntimeError("Input dataset object doesn't have length")
        n_batches = len(dataset)
        loader = wds.WebLoader(dataset,
                num_workers=self._n_workers_dataloader,
                pin_memory=self._pin_memory_dataloader,
                collate_fn=lambda x: x[0],
            ).with_length(n_batches).with_epoch(n_batches)

        # strange features required by nemo optimizer lr_scheduler
        loader.dataset = dataset  # seems like only length is used, webloader doesn't have this attr
        loader.batch_size = self._local_batch_size
        loader.drop_last = False
        return loader


    def train_dataloader(self) -> wds.WebLoader:
        assert self._dataset[Split.train] is not None,\
            f"dataset for train has not been setup"
        return self._setup_dataloader(self._dataset[Split.train])


    def val_dataloader(self) -> wds.WebLoader:
        assert self._dataset[Split.val] is not None,\
            f"dataset for val has not been setup"
        return self._setup_dataloader(self._dataset[Split.val])


    def test_dataloader(self) -> wds.WebLoader:
        assert self._dataset[Split.test] is not None,\
            f"dataset for test has not been setup"
        return self._setup_dataloader(self._dataset[Split.test])


    def predict_dataloader(self) -> wds.WebLoader:
        raise NotImplementedError("predict dataloader not implemented")
