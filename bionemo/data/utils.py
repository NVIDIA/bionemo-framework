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


import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import braceexpand
import torch
from nemo.utils import logging
from omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from bionemo.data.molecule import MoleculeCsvDataset


__all__ = [
    "DatasetTypes",
    "expand_dataset_paths",
    "create_dataset",
    "build_train_valid_test_datasets",
    "handle_index",
]


class DatasetTypes(Enum):
    zinc_csv = 0


def expand_dataset_paths(filepath: str, ext: str) -> List[str]:
    """Expand dataset paths from braces"""
    if ext and not filepath.endswith(ext):
        filepath = filepath + ext

    # TODO this should eventually be moved to a Nemo fileutils module or similar
    filepath = re.sub(r"""\(|\[|\<|_OP_""", "{", filepath)  # replaces '(', '[', '<' and '_OP_' with '{'
    filepath = re.sub(r"""\)|\]|\>|_CL_""", "}", filepath)  # replaces ')', ']', '>' and '_CL_' with '}'
    dataset_paths = list(braceexpand.braceexpand(filepath))
    return dataset_paths


def check_paths_exist(dataset_paths):
    """Check that the expanded dataset paths are valid and they exist."""
    errors = []
    for filepath in dataset_paths:
        if not os.path.exists(filepath):
            errors.append(filepath)
    return errors


def gunzip(i: str, o: str, exist_ok: bool = False):
    """Unzips a gzipped file

    Args:
        i (str): Filepath of file to unzip
        o (str): Filepath to save unzipped file to
        exist_ok (bool, optional): If True, `i` will be unzipped to `o` even if
            `o` already exists. If False, and error is raised. Defaults to
            False.

    Raises:
        FileExistsError: If `o` already exists and `exist_ok` is False.
        FileNotFoundError: If `i` does not exist.
    """
    if not exist_ok and os.path.exists(o):
        raise FileExistsError(f"Unzipping {i} to already existing path: {o}")
    if os.path.exists(i):
        with gzip.open(i, "rb") as f_in:
            with open(o, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise FileNotFoundError(i)


class DatasetBuilderSpec:
    """
    Abstract class for building datasets.

    Constructing a dataset from a list of files is a complex task.

    It involves:
    1. Parsing the string that delineates the files to be included, e.g.,
        "x001.csv,x002.csv" such that it corresponds to a list of files.
    2. Ensuring that the listed files actually exist on the filesystem.
    3. Instantiating a dataset from the given files.

    In each of these steps, there are ways that different types of files
    that could be used, different parsing schemes, and different PyTorch
    datasets that may be used.

    A DatasetBuilder encapsulates each of these steps such that the dataset
    instantiation can be performed by any class that implements the methods
    here.

    """

    def __init__(self, options: Dict[str, Any]):
        """
        Initializes a dataset builder

        Arguments:
            options (Dict[str, Any]): A dictionary that stores attributes that
                can arbitrarily be accessed by the builder.

        Attributes:
            options (Dict[str, Any]): Stores attributes that can be accessed
                by the builder
            dataset_paths (List[str]): Corresponds to the files to include
                in the dataset
            dataset (Optional[Dataset]): This will be the dataset that has been
                constructed by the builder, if it has been built yet.
        """
        self.options = options
        self.dataset_paths = []
        self.dataset = None

    def initialize_dataset(self):
        """
        Initializes the dataset

        Returns:
            Dataset: The initialized dataset
        """
        self.format_dataset_paths()
        self.check_dataset_paths_exist()
        return self.create_dataset()

    def check_dataset_paths_exist(self):
        """
        Checks whether the dataset's paths exist.

        Raises:
            AssertionError: If there are any files in the dataset's paths
                that do not exist.

        """
        errors = []
        for filepath in self.dataset_paths:
            file_does_not_exist = self.check_path(filepath)
            if file_does_not_exist:
                errors.append(file_does_not_exist)
        if len(errors) != 0:
            raise ValueError(f"Following files do not exist {' '.join(errors)}")

    def check_path(self, filepath):
        """
        Checks whether an individual file exists. Because Dataset's may perform
        their own modifcations on filepath names to read/write data, the
        logic to check whether the file exists may differ between the concrete
        dataset classes.

        Arguments:
            filepath (str): a string that can be used to identify the filepath

        Returns:
            Optional[str]: If the file exists, this returns None, otherwise
                it returns the on the filepath.
        """
        if not os.path.exists(filepath):
            return filepath

    def format_dataset_paths(self):
        """
        Modifies the objects dataset_paths attribute so that it contains
        the paths of all of the files to include in the dataset.

        This function will essentially allow for custom parsing of arbitrary
        input text.

        Examples:
        * 'chr1,chr2' -> ['chr1.fa', 'chr2.fa']
        * 'x001..x003' -> ['x001.csv', 'x002'csv', 'x003.csv']

        """
        raise NotImplementedError()

    def create_dataset(self):
        """
        This instantiates a concrete dataset class and sets it to this object's
        `dataset` attribute.
        """
        raise NotImplementedError()


def get_filepath(options):
    filepath = cfg_get_key(options["cfg"], "dataset_path", default="")
    filepath = os.path.join(filepath, options["name"], options["dataset"])

    return filepath


class CSVDatasetBuilder(DatasetBuilderSpec):
    """
    Builds instances MoleculeCsvDataset

    Examples:
    >>> dataset_builder = CSVDatasetBuilder(options=...)
    >>> csv_dataset_instance = dataset_builder.initialize_dataset()

    """

    def __init__(self, options: Dict[str, Any]):
        """
        Initializes a dataset builder

        Arguments:
            options (Dict[str, Any]): A dictionary that stores attributes that
                can arbitrarily be accessed by the builder. Requires keys:
                ['filepath', 'cfg'].

        Attributes:
            options (Dict[str, Any]): Stores attributes that can be accessed
                by the builder
            dataset_paths (List[str]): Corresponds to the files to include
                in the dataset
            dataset (Optional[Dataset]): This will be the dataset that has been
                constructed by the builder, if it has been built yet.
        """
        super().__init__(options)

    def format_dataset_paths(self):
        """
        Parses CSV paths.

        """
        filepath = get_filepath(self.options)
        # Get datasets and load data
        logging.info(f"Loading data from {filepath}")
        self.dataset_paths = expand_dataset_paths(filepath, ".csv")

    def check_path(self, filepath):
        """
        Checks whether a CSV exists.

        Arguments:
            filepath (str): a string that can be used to identify the filepath

        Returns:
            Optional[str]: If the file exists, this returns None, otherwise
                it returns the on the filepath.

        """
        if not os.path.exists(filepath):
            return filepath

    def create_dataset(self):
        """
        Instantiates a MoleculeCsvDataset.

        Returns:
            Dataset: Dataset instantiated from paths.
        """
        cfg = self.options["cfg"]
        self.dataset = MoleculeCsvDataset(dataset_paths=self.dataset_paths, cfg=cfg)
        return self.dataset


class DatasetFactorySpec:
    def create_dataset(self, options) -> DatasetBuilderSpec:
        """
        Creates a dataset with the given options.
        """
        raise NotImplementedError()


class FormattedDatasetFactory(DatasetFactorySpec):
    """
    Abstract class for constructing datasets builders.

    Attributes:
        formats (Dict[str, DatasetBuilderSpec]): Keys are formats that this
            factory supports, values are the classes of the builders.

    """

    formats: Dict[str, DatasetBuilderSpec]

    def create_dataset(self, options) -> DatasetBuilderSpec:
        """
        Creates a dataset with the given options.

        Arguments:
            options (Dict[str, Any]): Requires key 'cfg' to have a key for
                'dataset_format', which refers to a format to use from
                this objects `formats` attribute. Additional key
                requirements will vary by the dataset being built.

        Returns:
            Dataset: the initialized dataset to use.

        Raises:
            ValueError: if 'dataset_format' from options['cfg'] is not
                recognized.

        """

        dataset_format = cfg_get_key(options["cfg"], "dataset_format")

        if dataset_format in self.formats:
            builder_cls = self.formats[dataset_format]
            builder = builder_cls(options)
        else:
            raise ValueError(f"Unrecognized data format." f" Expected one of: {self.keys()}")
        return builder.initialize_dataset()

    @classmethod
    def keys(cls):
        """Lists available format values.

        Returns:
            List[str]: Allowable format values.
        """
        return list(cls.formats.keys())


class DefaultDatasetFactory(FormattedDatasetFactory):
    """
    Examples:
    >>> dataset_factory = DefaultDatasetFactory()
    >>> dataset = dataset_factory.create_dataset(options={'datset_format': 'csv', ...})
    """

    def __init__(self):
        """
        Initializes a dataset factory for handling csv and bin formats.
        """
        self.formats = {
            "csv": CSVDatasetBuilder,
        }


def get_default_builder_factory():
    """
    Returns:
        DatasetFactory: A factory method for producing datasets.
    """
    return DefaultDatasetFactory()


def create_dataset(
    cfg: DictConfig,
    num_samples: int,
    name: str,
    dataset: str,
    dataset_factory: Optional[DatasetFactorySpec] = None,
):
    if dataset_factory is None:
        dataset_factory = get_default_builder_factory()

    options = {
        "cfg": cfg,
        "num_samples": num_samples,
        "name": name,
        "dataset": dataset,
    }

    dataset = dataset_factory.create_dataset(options)

    return dataset


def build_train_valid_test_datasets(
    cfg: DictConfig,
    trainer: Trainer,
    train_valid_test_num_samples: List[int],
    dataset_factory: Optional[FormattedDatasetFactory] = None,
):
    ds_train = cfg.dataset.train
    ds_val = cfg.dataset.val
    ds_test = cfg.dataset.test

    # Build individual datasets.
    train_dataset = create_dataset(
        cfg,
        train_valid_test_num_samples[0],
        "train",
        ds_train,
        dataset_factory,
    )

    validation_dataset = create_dataset(
        cfg,
        train_valid_test_num_samples[1],
        "val",
        ds_val,
        dataset_factory,
    )

    test_dataset = create_dataset(
        cfg,
        train_valid_test_num_samples[2],
        "test",
        ds_test,
        dataset_factory,
    )

    return (train_dataset, validation_dataset, test_dataset)


def cfg_get_key(cfg, key, default=None):
    with open_dict(cfg):
        return cfg.get(key, default)


def pad_token_ids(
    token_ids: Union[List[int], List[torch.Tensor]],
    padding_value: int = 0,
    padding_len: int = None,
    pad_size_divisible_by: int = 1,
    **convert_to_kwargs,
):
    """
    Pads token ids with padding value, and return the padded tokens and
    the corresponding mask.

    Args:
        token_ids: List of token ids or tensors
        padding_value: Value to pad with. Defaults to 0.
        padding_len: Max length of the padded token ids. Defaults to None.
        pad_size_divisible_by: Pad the length of the token ids to be divisible by this number. Defaults to 1.
        **convert_to_kwargs: Passed directly to tensor.to(**kwargs) if provided

    Returns:
        Tuple[List[int], List[int]]: Padded token ids and mask
    """
    lengths = torch.tensor([len(s) for s in token_ids])
    if padding_len is None:
        padding_len = lengths.max()

    # make padding divisible by pad_size_divisible_by
    if pad_size_divisible_by > 1:
        padding_len = int(math.ceil(padding_len / pad_size_divisible_by) * pad_size_divisible_by)

    # build mask
    mask = torch.arange(padding_len)[None, :] < lengths[:, None]

    # make sure all sequences are pytorch tensors
    token_ids = [torch.tensor(s) if not torch.is_tensor(s) else s for s in token_ids]
    # pad sequences
    masked_token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=padding_value)

    # convert to desired device
    if len(convert_to_kwargs):
        mask = mask.to(**convert_to_kwargs)
        masked_token_ids = masked_token_ids.to(**convert_to_kwargs)

    # Further pad the sequences to the fixed maximum length, if necessary
    if masked_token_ids.size(1) < padding_len:
        padding_size = padding_len - masked_token_ids.size(1)
        masked_token_ids = torch.nn.functional.pad(masked_token_ids, [0, padding_size], value=padding_value)

    return (masked_token_ids, mask)


def handle_index(dataset, idx):
    """
    Remaps negative indices and handles numpy int indices.

    Arguments:
        dataset (Dataset): dataset to index into
        idx (int): Index. Can include negative indices.
    Returns:
        int: Remapped and fully qualified index.

    Raises:
        IndexError: If a negative index is out of range.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from torch.utils.data import TensorDataset
        >>> from nemo_chem.data.fasta_dataset import handle_index
        >>> dataset = TensorDataset(torch.tensor(-np.arange(5)))
        >>> handle_index(dataset, 1)
        1
        >>> handle_index(dataset, -2)
        3

    """
    if idx < 0 and idx > -len(dataset) - 1:
        idx = len(dataset) + idx
    elif idx < 0:
        raise IndexError(f"Index out of range: {idx}")
    return idx


def verify_checksum_matches(file_path: str, expected_checksum: str) -> bool:
    """Verify that file checksum matches expected value

    Args:
        file_path (str): Path to file
        expected_checksum (str): Checksum value

    Returns:
        bool: True if checksum matches else false
    """

    file_hash = hashlib.md5(open(file_path, "rb").read()).hexdigest()
    if file_hash == expected_checksum:
        matches = True
    else:
        matches = False
        logging.info(f"Checksum verification failed. Expected {expected_checksum} but got {file_hash}.")
    return matches


def get_ngc_registry_file_list(
    ngc_registry_target: str, ngc_registry_version: str, ngc_org: str, ngc_team: Optional[str] = None
) -> List[str]:
    """
    Get listing of NGC registry files:
    https://docs.ngc.nvidia.com/cli/cmd_registry.html
    NGC CLI install: https://ngc.nvidia.com/setup/installers/cli
    Args:
        ngc_registry_target: NGC registry target name for dataset
        ngc_registry_version: NGC registry version for dataset
        ngc_org: NGC organization
        ngc_team: NGC team, Optional and defaults to None
    Returns:
        list of file names
    """
    filelist_cmd = "ngc registry resource info --format_type json --files "
    filelist_cmd += f"--org {ngc_org} "

    target = ngc_org
    if ngc_team:
        filelist_cmd += f"--team {ngc_team} "
        target += "/" + ngc_team

    target += "/" + ngc_registry_target
    filelist_cmd += f" {target}:{ngc_registry_version}"

    try:
        result = subprocess.run(filelist_cmd, capture_output=True, shell=True, check=True)
        if result.stderr:
            logging.warning(result.stderr.decode())
        json_output = result.stdout.strip()
        file_list = json.loads(json_output)["file_list"]
        file_list = [x["path"] for x in file_list]
    except subprocess.CalledProcessError as e:
        logging.error(f"File list retrival failed: {e}")
        file_list = []
    except Exception:
        logging.error(f"File list retrival failed for command '{filelist_cmd}' and output '{json_output}'")
        file_list = []

    return file_list


def download_registry_from_ngc(
    ngc_registry_target: str,
    ngc_registry_version: str,
    ngc_org: str,
    ngc_team: Optional[str] = None,
    dest: Optional[str] = ".",
    exclude: Optional[str] = None,
    expected_checksum: Optional[str] = None,
    file: Optional[str] = None,
    format_type: Optional[str] = None,
    debug: bool = False,
) -> str:
    """
    Downloads data from NGC registry. Please refer to the documentation for more details:
    https://docs.ngc.nvidia.com/cli/cmd_registry.html
    NGC CLI install: https://ngc.nvidia.com/setup/installers/cli
    Args:
        ngc_registry_target: NGC registry target name for dataset
        ngc_registry_version: NGC registry version for dataset
        ngc_org: NGC organization
        ngc_team: NGC team, Optional and defaults to None
        dest: path to store the downloaded files. If None, default is set to  "."
        exclude: exclude files or directories from the downloaded dataset. Supports standard Unix shell-style wildcards
        expected_checksum: expected checksum value
        file: specify individual files to download from the dataset. Supports standard Unix shell-style wildcards
        format_type: specify the output format type, possible choices: ascii, csv, json. if None, default is set to ascii
        debug: enable debug mode
    Returns:
        path to the folder where dataset is downloaded
    """

    download_cmd = "ngc registry resource download-version "

    if dest:
        download_cmd += f"--dest {dest} "
    if exclude:
        download_cmd += f"--exclude {exclude} "
    if file:
        download_cmd += f"--file {file} "
    if format_type:
        download_cmd += f"--format_type {format_type} "
    if debug:
        download_cmd += "--debug "

    download_cmd += f"--org {ngc_org} "

    target = ngc_org
    if ngc_team:
        download_cmd += f"--team {ngc_team} "
        target += "/" + ngc_team

    target += "/" + ngc_registry_target
    download_cmd += f" {target}:{ngc_registry_version}"

    # Determine if file is already downloaded
    file_list = get_ngc_registry_file_list(ngc_registry_target, ngc_registry_version, ngc_org, ngc_team)

    download_file = True
    if expected_checksum and file_list:
        if len(file_list) > 1:
            logging.info("Checksum verification not supported if resource contains more than one file.")
        else:
            file_name = file_list[0]
            download_dir = os.path.join(dest, f"{ngc_registry_target}_v{ngc_registry_version}")
            if os.path.exists(os.path.join(download_dir, file_name)):
                hash_path = os.path.join(download_dir, file_name)
                download_file = False if verify_checksum_matches(hash_path, expected_checksum) else True

    if download_file:
        os.system(download_cmd)
    else:
        logging.info(f"Download of {target} to {hash_path} skipped because file exists and MD5 checksums match.")

    downloaded_file_list = os.listdir(download_dir)

    # TODO update logic if there is more than one downloaded file present
    assert len(downloaded_file_list) == 1, AssertionError(
        f"Expected only one downloaded file got {len(downloaded_file_list)}."
    )
    file_path = os.path.join(download_dir, downloaded_file_list[0])
    return file_path


def download_dataset_from_ngc(
    ngc_dataset_id: int,
    dest: Optional[str] = None,
    dir: Optional[str] = None,
    exclude: Optional[str] = None,
    file: Optional[str] = None,
    format_type: Optional[str] = None,
    resume: Optional[str] = None,
    dry_run: bool = False,
    debug: bool = False,
    compress_file: bool = False,
) -> str:
    """
    Downloads dataset from NGC. Please refer to the documentation for more details:
    https://docs.ngc.nvidia.com/cli/cmd_dataset.html
    NGC CLI install: https://ngc.nvidia.com/setup/installers/cli
    Args:
        ngc_dataset_id: NGC id of the dataset
        dest: path to store the downloaded files. If None, default is set to  "."
        dir: directories to download from dataset. Supports standard Unix shell-style wildcards
        exclude: exclude files or directories from the downloaded dataset. Supports standard Unix shell-style wildcards
        file: specify individual files to download from the dataset. Supports standard Unix shell-style wildcards
        format_type: specify the output format type, possible choices: ascii, csv, json. if None, default is set to ascii
        resume: resume the download for the dataset. Specify the file name saved by the download. Files will be downloaded to the directory of the file name.
        dry_run: list total size of the download without performing the download
        debug: enable debug mode
        compress_file: download the entire dataset directory as a zip file
    Returns:
        path to the folder where dataset is downloaded
    """
    cmd = "ngc dataset download "
    if dest:
        cmd += f"--dest {dest} "
    if dir:
        cmd += f"--dir {dir} "
    if exclude:
        cmd += f"--exclude {exclude} "
    if file:
        cmd += f"--file {file} "
    if format_type:
        cmd += f"--format_type {format_type} "
    if resume:
        cmd += f"--resume {resume} "
    if debug:
        cmd += "--debug "
    if dry_run:
        cmd += "--dry-run "
    if compress_file:
        cmd += "--zip "

    os.system(f"{cmd} {ngc_dataset_id}")
    return os.path.join(dest, str(ngc_dataset_id))
