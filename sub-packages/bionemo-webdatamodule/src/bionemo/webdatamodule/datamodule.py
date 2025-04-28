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


import glob
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, get_args

import lightning as L
import webdataset as wds

from bionemo.webdatamodule.utils import pickles_to_tars


class Split(Enum):
    """Names for each data split."""

    train = auto()
    val = auto()
    test = auto()


class WebDataModule(L.LightningDataModule):
    """A LightningDataModule for using webdataset tar files.

    `WebDataModule` is a `LightningDataModule` for using webdataset tar files to setup PyTorch
    datasets and dataloaders. This data module takes as input a dictionary: Split -> tar file
    directory and vaiours webdataset config settings. In its setup() function, it creates the
    webdataset object chaining up the input `pipeline_wds` workflow. In its train/val/test_dataloader(),
    it creates the WebLoader object chaining up the `pipeline_prebatch_wld` workflow.

    Examples:
    --------
    1. create the data module with input directory to webdataset tar files.
    Depending on which of the downstream Lightning.Trainer methods are called,
    e.g., `Trainer.fit()`, `Trainer.validate()`, `Trainer.test()` or
    `Trainer.predict()`, only a subset of the train, val and test splits need to
    be specified in the various input options to the data module:

    - `Trainer.fit()` requires the `train` and `val` splits
    - `Trainer.validate()` requires the `val` split
    - `Trainer.test()` requires the `test` splits
    - `Trainer.predict()` requires the `test` splits

    Here is an example of constructing the data module for `Trainer.fit()`:
    ```python
    >>> from bionemo.webdatamodule.datamodule import Split, WebDataModule
    >>>
    >>> tar_file_prefix = "shards"
    >>>
    >>> dirs_of_tar_files = {
    >>>     Split.train: "/path/to/train/split/tars",
    >>>     Split.val: "/path/to/val/split/tars",
    >>> }
    >>>
    >>> n_samples {
    >>>     Split.train: 1000,
    >>>     Split.val: 100,
    >>> }
    >>>
    >>> # this is the string to retrieve the corresponding data object from the
    >>> # webdataset file (see
    >>> # https://github.com/webdataset/webdataset?tab=readme-ov-file#the-webdataset-format
    >>> # for details)
    >>> suffix_keys_wds = "tensor.pyd"
    >>>
    >>> seed = 27193781
    >>>
    >>> # Specify the routines to process the samples in the WebDataset object.
    >>> # The routine is a generator of an Iterable of generators that are chained
    >>> # together by nested function calling. The following is equivalent of
    >>> # defining a overall generator of `shuffle(untuple(...))` which
    >>> # untuples the samples and shuffles them. See webdataset's Documentation
    >>> # for details.
    >>> # NOTE: the `untuple` is almost always necessary due to the webdataset's
    >>> # file parsing rule.
    >>>
    >>> untuple = lambda source : (sample for (sample,) in source)
    >>>
    >>> from webdatast import shuffle
    >>> pipeline_wds = {
    >>>     Split.train : [untuple, shuffle(n_samples[Split.train],
    >>>                                     rng=random.Random(seed_rng_shfl))],
    >>>     Split.val: untuple
    >>> }
    >>>
    >>> # Similarly the user can optionally define the processing routine on the
    >>> # WebLoader (the dataloader of webdataset).
    >>> # NOTE: these routines by default take unbatched sample as input so the
    >>> # user can customize their batching routines here
    >>>
    >>> batch = batched(local_batch_size, collation_fn=lambda
                        list_samples : torch.vstack(list_samples))
    >>> pipeline_prebatch_wld = {
            Split.train: [shuffle(n_samples[Split.train],
                                  rng=random.Random(seed_rng_shfl)), batch],
            Split.val : batch,
            Split.test : batch
        }
    >>>
    >>> # the user can optionally specify the kwargs for WebDataset and
    >>> # WebLoader
    >>>
    >>> kwargs_wds = {
    >>>     split : {'shardshuffle' : split == Split.train,
    >>>              'nodesplitter' : wds.split_by_node,
    >>>              'seed' : seed_rng_shfl}
    >>>     for split in Split
    >>>     }
    >>>
    >>> kwargs_wld = {
    >>>     split : {"num_workers": 2} for split in Split
    >>>     }
    >>>
    >>> invoke_wds = {
    >>>     split: [("with_epoch", {"nbatches" : 5})] for split in Split
    >>>     }
    >>>
    >>> invoke_wld = {
    >>>     split: [("with_epoch", {"nbatches" : 5}] for split in Split
    >>>     }
    >>>
    >>> # construct the data module
    >>> data_module = WebDataModule(suffix_keys_wds,
                                    dirs_of_tar_files,
                                    prefix_tars_wds=tar_file_prefix,
                                    pipeline_wds=pipeline_wds,
                                    pipeline_prebatch_wld=pipeline_prebatch_wld,
                                    kwargs_wds=kwargs_wds,
                                    kwargs_wld=kwargs_wld,
                                    invoke_wds=invoke_wds,
                                    invoke_wld=invoke_wld,
                                    )
    ```

    """

    def __init__(
        self,
        suffix_keys_wds: Union[str, Iterable[str]],
        dirs_tars_wds: Dict[Split, str],
        prefix_tars_wds: str = "wdshards",
        pipeline_wds: Optional[Dict[Split, Union[Iterable[Iterable[Any]], Iterable[Any]]]] = None,
        pipeline_prebatch_wld: Optional[Dict[Split, Union[Iterable[Iterable[Any]], Iterable[Any]]]] = None,
        kwargs_wds: Optional[Dict[Split, Dict[str, Any]]] = None,
        kwargs_wld: Optional[Dict[Split, Dict[str, Any]]] = None,
        invoke_wds: Optional[Dict[Split, List[Tuple[str, Dict[str, Any]]]]] = None,
        invoke_wld: Optional[Dict[Split, List[Tuple[str, Dict[str, Any]]]]] = None,
    ):
        """Constructor.

        Args:
            suffix_keys_wds: a set of keys each
                corresponding to a data object in the webdataset tar file
                dictionary. The data objects of these keys will be extracted and
                tupled for each sample in the tar files
            dirs_tars_wds: input dictionary: Split -> tar file
                directory that contains the webdataset tar files for each split
        Kwargs:
            prefix_tars_wds: name prefix of the input webdataset tar
                files. The input tar files are globbed by
                "{dirs_tars_wds[split]}/{prefix_tars_wds}-*.tar"
            pipeline_wds: a dictionary of webdatast composable, i.e.,
                functor that maps a iterator to another iterator that
                transforms the data sample yield from the dataset object, for
                different splits, or an iterable to such a sequence of such
                iterators. For example, this can be used to transform the
                sample in the worker before sending it to the main process of
                the dataloader
            pipeline_prebatch_wld: a dictionary
                of webloader composable, i.e., functor that maps a iterator to
                another iterator that transforms the data sample yield from the
                WebLoader object, for different splits, or an iterable to a
                seuqnence of such iterators. For example, this can be used for
                batching the samples. NOTE: this is applied before batching is
                yield from the WebLoader
            kwargs_wds: kwargs for the WebDataset.__init__()
            kwargs_wld : kwargs for the WebLoader.__init__(), e.g., num_workers, of each split
            invoke_wds: a dictionary of WebDataset methods to be called upon WebDataset
                construction. These methods must return the WebDataset object itself. Examples
                are .with_length() and .with_epoch(). These methods will be applied towards
                the end of returning the WebDataset object, i.e., after the pipline_wds
                have been applied. The inner list of tuples each has its first element as the
                method name and the second element as the corresponding method's kwargs.
            invoke_wld: a dictionary of WebLoader methods to be called upon WebLoader
                construction. These methods must return the WebLoader object itself. Examples
                are .with_length() and .with_epoch(). These methods will be applied towards
                the end of returning the WebLoader object, i.e., after the pipelin_prebatch_wld
                have been applied. The inner list of tuples each has its first element as the
                method name and the second element as the corresponding method's kwargs.
        """
        super().__init__()

        self._dirs_tars_wds = dirs_tars_wds

        if not isinstance(suffix_keys_wds, get_args(Union[str, Iterable])):
            raise TypeError("suffix_keys_wds can only be str or Iterable[str]")

        self._suffix_keys_wds = suffix_keys_wds

        self._prefix_tars_wds = prefix_tars_wds
        self._pipeline_wds = pipeline_wds
        self._pipeline_prebatch_wld = pipeline_prebatch_wld

        self._kwargs_wld = kwargs_wld

        self._kwargs_wds = kwargs_wds

        self._invoke_wds = invoke_wds
        self._invoke_wld = invoke_wld

        # to be created later in setup
        self._dataset = {}

    def prepare_data(self) -> None:
        """This is called only by the main process by the Lightning workflow.

        Do not rely on this data module object's state update here as there is no
        way to communicate the state update to other subprocesses. Is a **no-op**.
        """
        pass

    def _setup_wds(self, split: Split) -> wds.WebDataset:
        """Setup webdataset and webloader. This is called by setup().

        Args:
            split (Split): train, val or test split

        Returns:
            WebDataset

        """
        if split not in self._dirs_tars_wds.keys():
            raise RuntimeError(f"_setup_wds() is called with {split} split that doesn't have the input tar dir")
        urls = sorted(glob.glob(f"{self._dirs_tars_wds[split]}/{self._prefix_tars_wds}-*.tar"))
        kwargs = self._kwargs_wds[split] if self._kwargs_wds is not None else None
        dataset = wds.WebDataset(urls, **(kwargs if kwargs is not None else {})).decode()
        if isinstance(self._suffix_keys_wds, str):
            dataset = dataset.extract_keys(f"*.{self._suffix_keys_wds}")
        else:
            dataset = dataset.extract_keys(*[f"*.{key}" for key in self._suffix_keys_wds])

        if self._pipeline_wds is not None and self._pipeline_wds[split] is not None:
            if isinstance(self._pipeline_wds[split], Iterable):
                dataset = dataset.compose(*self._pipeline_wds[split])
            else:
                dataset = dataset.compose(self._pipeline_wds[split])

        if self._invoke_wds is not None and self._invoke_wds[split] is not None:
            for method in self._invoke_wds[split]:
                name_method, kwargs_method = method
                dataset = getattr(dataset, name_method)(**kwargs_method)
        return dataset

    def setup(self, stage: str) -> None:
        """This is called on all Lightning-managed nodes in a multi-node training session.

        Args:
            stage: "fit", "test" or "predict"
        """
        if stage == "fit":
            self._dataset[Split.train] = self._setup_wds(Split.train)
            self._dataset[Split.val] = self._setup_wds(Split.val)
        elif stage == "validate":
            self._dataset[Split.val] = self._setup_wds(Split.val)
        elif stage == "test":
            self._dataset[Split.test] = self._setup_wds(Split.test)
        elif stage == "predict":
            self._dataset[Split.test] = self._setup_wds(Split.test)
        else:
            raise NotImplementedError(f"Data setup with {stage=} is not implemented.")

    def _setup_dataloader(self, split: Split) -> wds.WebLoader:
        """Setup the dataloader for the input dataset split.

        Args:
            split (Split): input split type

        Returns:
             WebLoader object

        Raises:
            ValueError if `split` doesn't correspond to a known dataset.
        """
        if self._dataset[split] is None:
            raise ValueError(
                f"_setup_dataloader() is called with {split} split without setting up the corresponding dataset."
            )
        dataset = self._dataset[split]
        kwargs = self._kwargs_wld[split] if self._kwargs_wld is not None else None
        loader = wds.WebLoader(dataset, **(kwargs if kwargs is not None else {}))

        if self._pipeline_prebatch_wld is not None and self._pipeline_prebatch_wld[split] is not None:
            if isinstance(self._pipeline_prebatch_wld[split], Iterable):
                loader = loader.compose(*self._pipeline_prebatch_wld[split])
            else:
                loader = loader.compose(self._pipeline_prebatch_wld[split])

        if self._invoke_wld is not None and self._invoke_wld[split] is not None:
            for method in self._invoke_wld[split]:
                name_method, kwargs_method = method
                loader = getattr(loader, name_method)(**kwargs_method)

        return loader

    def train_dataloader(self) -> wds.WebLoader:
        """Webdataset for the training data."""
        return self._setup_dataloader(Split.train)

    def val_dataloader(self) -> wds.WebLoader:
        """Webdataset for the validation data."""
        return self._setup_dataloader(Split.val)

    def test_dataloader(self) -> wds.WebLoader:
        """Webdataset for the test data."""
        return self._setup_dataloader(Split.test)

    def predict_dataloader(self) -> wds.WebLoader:
        """Alias for :func:`test_dataloader`."""
        return self._setup_dataloader(Split.test)


class PickledDataWDS(WebDataModule):
    """A LightningDataModule to process pickled data into webdataset tar files.

    `PickledDataWDS` is a LightningDataModule to process pickled data into webdataset tar files
    and setup dataset and dataloader. This inherits the webdataset setup from its parent module
    `WebDataModule`. This data module takes a directory of pickled data files, data filename
    prefixes for train/val/test splits, data filename suffixes and prepare webdataset tar files
    by globbing the specific pickle data files `{dir_pickles}/{name_subset[split]}.{suffix_pickles}`
    and outputing to webdataset tar file with the dict structure:
    ```
        {"__key__" : name.replace(".", "-"),
         suffix_pickles : pickled.dumps(data) }
    ```
    NOTE: this assumes only one pickled file is processed for each sample. In
    its setup() function, it creates the webdataset object chaining up the input
    `pipeline_wds` workflow. In its train/val/test_dataloader(), it creates the
    WebLoader object chaining up the `pipeline_prebatch_wld` workflow.

    Examples:
    --------
    1. create the data module with a directory of pickle files and the file name
    prefix thereof for different splits to used by `Lightning.Trainer.fit()`

    ```python
    >>> from bionemo.core.data.datamodule import Split, PickledDataWDS

    >>> dir_pickles = "/path/to/my/pickles/dir"

    >>> # the following will use `sample1.mydata.pt` and `sample2.mydata.pt` as the
    >>> # training dataset and `sample4.mydata.pt` and `sample5.mydata.pt` as the
    >>> # validation dataset

    >>> suffix_pickles = "mydata.pt"

    >>> names_subset = {
    >>>     Split.train: [sample1, sample2],
    >>>     Split.val: [sample4, sample5],
    >>> }

    >>> # the following setting will attempt to create at least 5 tar files in
    >>> # `/path/to/output/tars/dir/myshards-00000{0-5}.tar`

    >>> n_tars_wds = 5
    >>> prefix_tars_wds = "myshards"
    >>> output_dir_tar_files = {
            Split.train : "/path/to/output/tars/dir-train",
            Split.val : "/path/to/output/tars/dir-val",
            Split.test : "/path/to/output/tars/dir-test",
        }

    >>> # user can optionally customize the data processing routines and kwargs used
    >>> # in the WebDataset and WebLoader (see the examples in `WebDataModule`)

    >>> pipeline_wds = { Split.train: ... }

    >>> pipeline_prebatch_wld = { Split.train: ... }

    >>> kwargs_wds = { Split.train: ..., Split.val: ... }

    >>> kwargs_wld = { Split.train: ..., Split.val: ... }

    >>> invoke_wds = { Split.train: ..., Split.val: ... }

    >>> invoke_wld = { Split.train: ..., Split.val: ... }

    >>> # create the data module
    >>> data_module = PickledDataWDS(
    >>>     dir_pickles,
    >>>     names_subset,
    >>>     suffix_pickles, # `WebDataModule` args
    >>>     output_dir_tar_files, # `WebDataModule` args
    >>>     n_tars_wds=n_tars_wds,
    >>>     prefix_tars_wds=prefix_tars_wds, # `WebDataModule` kwargs
    >>>     pipeline_wds=pipeline_wds, # `WebDataModule` kwargs
    >>>     pipeline_prebatch_wld=pipelines_wdl_batch, # `WebDataModule` kwargs
    >>>     kwargs_wds=kwargs_wds, # `WebDataModule` kwargs
    >>>     kwargs_wld=kwargs_wld, # `WebDataModule` kwargs
    >>>     invoke_wds=invoke_wds, # `WebDataModule` kwargs
    >>>     invoke_wld=invoke_wld, # `WebDataModule` kwargs
    >>> )
    ```
    """

    def __init__(
        self,
        dir_pickles: str,
        names_subset: Dict[Split, List[str]],
        *args,
        n_tars_wds: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            dir_pickles: input directory of pickled data files
            names_subset: list of filename prefix of
                the data samples to be loaded in the dataset and dataloader for
                each of the split
            *args: arguments passed to the parent WebDataModule
            n_tars_wds: attempt to create at least this number of
                webdataset shards
            **kwargs: arguments passed to the parent WebDataModule
        """
        super().__init__(
            *args,
            **kwargs,
        )

        self._dir_pickles = dir_pickles

        self._names_subset = names_subset

        self._n_tars_wds = n_tars_wds

    def prepare_data(self) -> None:
        """This is called only by the main process by the Lightning workflow.

        Do not rely on this data module object's state update here as there is no
        way to communicate the state update to other subprocesses. The nesting
        `pickles_to_tars` function goes through the data name prefixes in the
        different splits, read the corresponding pickled file and output a
        webdataset tar archive with the dict structure: {"__key__" :
        name.replace(".", "-"), suffix_pickles : pickled.dumps(data) }.
        """
        for split in self._names_subset.keys():
            # create wds shards (tar files) for train set
            pickles_to_tars(
                self._dir_pickles,
                self._names_subset[split],
                self._suffix_keys_wds,
                self._dirs_tars_wds[split],
                self._prefix_tars_wds,
                min_num_shards=self._n_tars_wds,
            )
