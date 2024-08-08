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
import random
from typing import Any, Dict, Generator, Iterable, List, Optional
import lightning as L
import webdataset as wds

from bionemo.contrib.data.molecule.diffdock.utils import (
    pickles_to_tars
    )


class Split(Enum):
    train = auto()
    val = auto()
    test = auto()


class WDSModule(L.LightningDataModule):

    """lightning data module for using webdataset tar files to setup dataset and
    dataloader. This data module takes a dictionary: Split -> tar file
    directory. In its setup() function, it creates the webdataset object
    chaining up the input `pipeline_wds` workflow. In its
    train/val/test_dataloader(), it creates the WebLoader object chaining up the
    `pipeline_prebatch_wld` workflow"""

    def __init__(self, dirs_tars_wds : Dict[Split, str], n_samples : Dict[Split,
                                                                          int],
                 suffix_keys_wds : Iterable[str], global_batch_size : int,
                 prefix_tars_wds : str = "wdshards",
                 pipeline_wds : Optional[Dict[Split, Generator[Any, None,
                                                               None]]] = None,
                 pipeline_prebatch_wld : Optional[Dict[Split, Generator[Any,
                                                                        None,
                                                                        None]]]
                 = None, seed_rng_shfl : int = 0,
                 kwargs_dl : Optional[Dict[Split, Dict[str,  str]]] = None):
        """constructor

        Args:
            dirs_tars_wds (Dict[Split, str]): input dictionary: Split -> tar file
                directory that contains the webdataset tar files for each split
            n_samples (Dict[Split, int]): input dictionary: Split -> number of
                data samples for each split
            suffix_keys_wds (Iterable): a set of keys each corresponding to a
                data object in the webdataset tar file dictionary. The data
                objects of these keys will be extracted and tupled for each
                sample in the tar files
            global_batch_size (int): size of batch summing across nodes in Data
                Distributed Parallel, i.e., local_batch_size * n_nodes. NOTE:
                this data module doesn't rely on the input `global_batch_size`
                for batching the samples. The batching is supposed to be done as
                a part of the input `pipeline_prebatch_wld`. `global_batch_size`
                is only used to compute a (pseudo-) epoch length for the data
                loader so that the loader yield approximately n_samples //
                global_batch_size batches
        Kwargs:
            prefix_tars_wds (str): name prefix of the input webdataset tar
                files. The input tar files are globbed by
                "{dirs_tars_wds[split]}/{prefix_tars_wds}-*.tar"
            pipeline_wds (Optional[Dict[Split, Generator[Any, None, None]]]): a
                dictionary of webdatast composable, i.e., functor that maps a
                generator to another generator that transforms the data sample
                yield from the dataset object, for different splits. For
                example, this can be used to transform the sample in the worker
                before sending it to the main process of the dataloader
            pipeline_prebatch_wld (Optional[Dict[Split, Generator[Any, None,
                None]]]): a dictionary of webloader composable, i.e., functor
                that maps a generator to another generator that transforms the
                data sample yield from the WebLoader object, for different
                splits. For example, this can be used for batching the samples.
                NOTE: this is applied before batching is yield from the
                WebLoader
            seed_rng_shfl (int): seed to the random number generators used in
                data loading time for shuffling
            kwargs_dl (Optional[Dict[Split, Dict[str,  str]]]): kwargs for data
                loader, e.g., num_workers, of each split


        """
        super().__init__()

        self._dirs_tars_wds = dirs_tars_wds

        keys_subset = self._dirs_tars_wds.keys()
        if not (Split.train in keys_subset and Split.val in keys_subset):
            raise RuntimeError("Input dirs_tars_wds must be defined for the "\
                               "train and val splits")

        if n_samples.keys() != keys_subset:
            raise RuntimeError(f"Input n_samples has different keys than "
                               f"dirs_tars_wds: {n_samples.keys()} vs "
                               f"{keys_subset}"
                               )

        self._n_samples= n_samples

        self._global_batch_size = global_batch_size
        self._suffix_keys_wds = suffix_keys_wds

        self._prefix_tars_wds = prefix_tars_wds
        self._pipeline_wds = pipeline_wds
        self._pipeline_prebatch_wld = pipeline_prebatch_wld

        self._seed_rng_shfl = seed_rng_shfl

        self._kwargs_dl = kwargs_dl

        # to be created later in setup
        self._dataset = dict()


    def prepare_data(self) -> None:
        """This is called only by the main process by the Lightning workflow. Do
        not rely on this data module object's state update here as there is no
        way to communicate the state update to other subprocesses.

        Returns: None
        """
        pass

    def _setup_wds(self, split : Split) -> wds.WebDataset:
        """setup webdataset and webloader. This is called by setup()

        Args:
            split (Split): train, val or test split

        Returns: WebDataset

        """
        if not split in self._dirs_tars_wds.keys():
            raise RuntimeError(f"_setup_wds() is called with {split} "
                               f"split that doesn't have the input tar dir")
        is_train = split == Split.train
        urls = sorted(glob.glob(
            f"{self._dirs_tars_wds[split]}/{self._prefix_tars_wds}-*.tar")
            )
        dataset = (
            wds.WebDataset(urls, shardshuffle=is_train,
                           nodesplitter=wds.split_by_node,
                           seed=self._seed_rng_shfl)
            .decode()
            .extract_keys(f"*.{self._suffix_keys_wds}")
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
        elif stage == "validate":
            self._dataset[Split.val] = self._setup_wds(Split.val)
        elif stage == "test":
            self._dataset[Split.test] = self._setup_wds(Split.test)
        else:
            raise NotImplementedError(f"Data setup with stage = {stage} "\
                                      f"is not implmented")

    def _setup_dataloader(self, split : Split) -> wds.WebLoader:
        """setup the dataloader for the input dataset split

        Args:
            split (Split): input split type

        Returns: WebLoader object

        """
        if self._dataset[split] is None:
            raise RuntimeError(f"_setup_dataloader() is called with {split} "
                               f"split without setting up the corresp. dataset")
        dataset = self._dataset[split]
        n_samples = self._n_samples[split]
        n_batches = ((n_samples + self._global_batch_size - 1)
                     // self._global_batch_size)
        kwargs = self._kwargs_dl[split] if self._kwargs_dl is not None else None
        loader = wds.WebLoader(dataset, batch_size=None,
            **(kwargs if kwargs is not None else {})
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
        return self._setup_dataloader(Split.train)


    def val_dataloader(self) -> wds.WebLoader:
        return self._setup_dataloader(Split.val)


    def test_dataloader(self) -> wds.WebLoader:
        return self._setup_dataloader(Split.test)


    def predict_dataloader(self) -> wds.WebLoader:
        raise NotImplementedError("predict dataloader not implemented")


class PickledDataWDS(WDSModule):

    """lightning APIs to process pickled data into webdataset tar files and
    setup dataset and dataloader. This data module takes a directory of pickled
    data files, data filename prefixes for train/val/test splits, data filename
    suffixes and prepare webdataset tar files by globbing the specific pickeld
    data files {dir_pickled}/{name_subset[split]}.{suffix_pickled} and outputing
    to webdataset tar file with the dict structure: {"__key__" :
    name.replace(".", "-"), suffix_pickled : pickled.dumps(data) }. NOTE: this
    assumes only one pickled file is processed for each sample. In its setup()
    function, it creates the webdataset object chaining up the input
    `pipeline_wds` workflow. In its train/val/test_dataloader(), it creates the
    WebLoader object chaining up the `pipeline_prebatch_wld` workflow"""

    def __init__(self, dir_pickled : str, suffix_pickled : str, names_subset :
                 Dict[Split, List[str]], prefix_dir_tars_wds : str, *args,
                 n_tars_wds : Optional[int] = None, **kwargs):
        """constructor

        Args:
            dir_pickled (str): input directory of pickled data files
            suffix_pickled (str): filename suffix of the input data in
                dir_pickled. This is also used as the key mapped to the
                tarballed pickled object in the webdataset
            names_subset (Dict[Split, List[str]]): list of complex names to be
                included in each of the split
            prefix_dir_tars_wds (str): directory name prefix to store the output
                webdataset tar files. The actual directories storing the train, val
                and test sets will be suffixed with "train", "val" and "test"
                respectively.
            *args: arguments passed to the parent WDSModule

        Kwargs:
            n_tars_wds (int): attempt to create at least this number of
                webdataset shards
            **kwargs: arguments passed to the parent WDSModule


        """
        super().__init__(
            {
                split : f"{prefix_dir_tars_wds}{str(split).split('.')[-1]}"
                for split in names_subset.keys()
                },
            {
                split : len(names_subset[split]) for split in
                names_subset.keys()
            },
            suffix_pickled,
            *args,
             **kwargs
            )

        self._dir_pickled = dir_pickled
        self._suffix_pickled = suffix_pickled
        self._prefix_dir_tars_wds = prefix_dir_tars_wds

        self._names_subset = names_subset

        self._n_tars_wds = n_tars_wds

    def prepare_data(self) -> None:
        """This is called only by the main process by the Lightning workflow. Do
        not rely on this data module object's state update here as there is no
        way to communicate the state update to other subprocesses. The
        `pickles_to_tars` function goes through the data name prefixes in the
        different splits, read the corresponding pickled file and output a
        webdataset tar archive with the dict structure: {"__key__" :
        name.replace(".", "-"), suffix_pickled : pickled.dumps(data) }.

        Returns: None
        """
        for split in self._names_subset.keys():
            # create wds shards (tar files) for train set
            pickles_to_tars(self._dir_pickled,
                            self._suffix_pickled,
                            self._names_subset[split],
                            self._dirs_tars_wds[split],
                            self._prefix_tars_wds,
                            min_num_shards=self._n_tars_wds)
