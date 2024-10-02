# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import fcntl
import gc
import hashlib
import logging
import os
import pickle as pkl
from pathlib import Path
from typing import List, Optional, Union

import torch
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVFieldsMemmapDataset, CSVMemMapDataset
from nemo.core import Dataset
from omegaconf import OmegaConf

from bionemo.data.mapped_dataset import ResamplingMappedDataset
from bionemo.data.memmap_fasta_fields_dataset import FASTAFieldsMemmapDataset
from bionemo.data.utils import check_paths_exist, expand_dataset_paths


_CSV_FIELDS_MMAP_TYPE = "csv_fields_mmap"
_CSV_MMAP_TYPE = "csv_mmap"
_FASTA_FIELDS_MMAP_TYPE = "fasta_fields_mmap"

_DATA_IMPL_TYPE_CLS = {
    _CSV_FIELDS_MMAP_TYPE: CSVFieldsMemmapDataset,
    _CSV_MMAP_TYPE: CSVMemMapDataset,
    _FASTA_FIELDS_MMAP_TYPE: FASTAFieldsMemmapDataset,
}


def add_hash_to_metadata(index_folder: str, dataset_paths: List[str]) -> None:
    """Computes SHA256 hash for each dataset files and adds it
     to the metadata of the corresponding dataset files.
    Name of metadata file is inferred from name of dataset file
    Args:
        index_folder: Path to directory where index files are present
        dataset_paths: A list of paths of dataset files
    Returns:
        None
    """
    for dataset_path in dataset_paths:
        # compute hash
        with open(dataset_path, "rb") as f:
            filehash = hashlib.sha256(f.read()).hexdigest()

        # Write hash to metadata file
        # lstrip needed here to ensure that dataset_path is not an absolute path
        # if it is absolute, then index_folder is ignored
        if index_folder is None:
            index_path = dataset_path
        else:
            index_path = os.path.join(index_folder, dataset_path.lstrip("/"))
        metadata_fn = f"{index_path}.idx.info"  # TODO generalize .idx
        print(f"opening {metadata_fn}")
        with open(metadata_fn, "rb") as fh:
            metadata = pkl.load(fh)

        # Do not overwrite if it already contains hash
        if "sha256" in metadata.keys():
            prev_hash = metadata["sha256"]
            if prev_hash == filehash:
                # No need to overwrite the same file hash
                return 0

        metadata["sha256"] = filehash
        with open(metadata_fn, "wb") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            pkl.dump(metadata, f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return 0


def has_file_changed(index_folder, file: str) -> bool:
    """Checks if file has changed based on previosuly computed SHA256 hash
    in its corresponding metadata file. Returns True if file has changed
    otherwise returns False
    Args:
        index_folder: Path to directory where index files are present
        file: Name of data file
    Returns:
        bool
    """
    if index_folder is None:
        metadata_fn = file + ".idx.info"
    else:
        metadata_fn = os.path.join(index_folder, file + ".idx.info")
    if os.path.exists(metadata_fn):
        with open(metadata_fn, "rb") as metadata_fh:
            metadata_dict = pkl.load(metadata_fh)
            # Case where .info file was created by a previous version
            if "sha256" not in metadata_dict.keys():
                return True
            else:
                prev_hash = metadata_dict["sha256"]
        with open(file, "rb") as f:
            curr_hash = hashlib.sha256(f.read()).hexdigest()
        return not (prev_hash == curr_hash)
    else:
        return False


def delete_stale_resampled_file(path: str) -> None:
    """Finds and deletes resampled index files if it exists.
    Name of resampled index file is inferred in the function.
    Args:
        path: Name of index folder or path of dataset files
    Returns:
        None
    """
    if os.path.isdir(path):
        all_npy_files = Path(path).glob("*.npy")
    else:
        all_npy_files = Path(os.path.dirname(path)).glob("*.npy")

    # Find name of index file
    match_substrings = ("indexmap", "mns", "msl", "ssp")  # these substrings should be present in index file
    idx_files = []
    for file in all_npy_files:
        if all(s in str(file) for s in match_substrings):
            idx_files.append(file)

    for idx_file in idx_files:
        logging.info(f"Removing stale resampled idx file: {idx_file}")
        os.remove(idx_file)


# TODO (@sichu): raise exception if cannot read the dataset properly
def build_typed_dataset(
    dataset_paths: Union[str, List[str]],
    data_impl: str,
    use_upsampling: bool,
    cfg: OmegaConf,
    num_samples: Optional[int] = None,
    limit_batches_scale_factor: Optional[float] = None,
) -> Dataset:
    """
    Builds dataset based on preferred implementation given provided paths to the files with data and
    optionally down/upsamples it to num_samples.
    Args:
        dataset_paths: local path or list of paths to the files with data
        data_impl: dataset implementation type specified as key in _DATA_IMPL_TYPE_CLS
        cfg: config to be passed to a dataset constructor
        num_samples: down/upsample size of the dataset, if applicable. If None, then the num_samples equals len(dataset)
        limit_batches_scale_factor: Reduces the number of samples to be `limit_batches_scale_factor` * len(dataset) samples instead of all samples.
    Returns:
        Dataset
    """

    assert (
        data_impl in _DATA_IMPL_TYPE_CLS.keys()
    ), f'Argument data_impl must be set to: {", ".join(_DATA_IMPL_TYPE_CLS.keys())}'
    dataset_cls = _DATA_IMPL_TYPE_CLS[data_impl]

    assert "data_impl_kwargs" in cfg, (
        f"Config 'cfg' should contain 'data_impl_kwargs.{data_impl}' key being "
        f"a dictionary with arguments to the constructor of {dataset_cls.__name__}"
    )

    data_impl_kwargs = cfg.data_impl_kwargs.get(data_impl, {})
    if data_impl_kwargs == {}:
        logging.info(f"Default values of the arguments are used to initialize dataset {dataset_cls.__name__}")

    if data_impl == _FASTA_FIELDS_MMAP_TYPE:
        ext = ".fasta"
    else:
        ext = ".csv"

    if isinstance(dataset_paths, list):
        dataset_paths: List[str] = [
            path for dataset_path in dataset_paths for path in expand_dataset_paths(dataset_path, ext=ext)
        ]
    elif isinstance(dataset_paths, str):
        dataset_paths: List[str] = expand_dataset_paths(dataset_paths, ext=ext)
    else:
        raise ValueError("Argument dataset_paths should be a str or list of str corresponding to paths to data")

    errors = check_paths_exist(dataset_paths)
    assert len(errors) == 0, "Following files do not exist %s" % " ".join(errors)
    logging.info(f'Loading data from {", ".join(dataset_paths)}')

    index_mapping_dir = cfg.get("index_mapping_dir", os.path.dirname(dataset_paths[0]))
    # Delete stale index and metadata files

    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not is_distributed or (is_distributed and torch.distributed.get_rank() == 0):
        file_changed_list = [has_file_changed(index_mapping_dir, dataset_path) for dataset_path in dataset_paths]

        for file_changed, dataset_path in zip(file_changed_list, dataset_paths):
            if file_changed:
                if index_mapping_dir is None:
                    metadata_fn = f"{dataset_path}.idx.info"
                    idx_fn = f"{dataset_path}.idx.npy"
                else:
                    metadata_fn = os.path.join(index_mapping_dir, dataset_path.lstrip("/") + ".idx.info")
                    idx_fn = os.path.join(index_mapping_dir, dataset_path.lstrip("/") + ".idx.npy")
                logging.info(f"Deleting stale files: {metadata_fn} {idx_fn} if exist")
                if os.path.exists(metadata_fn):
                    os.remove(metadata_fn)
                if os.path.exists(idx_fn):
                    os.remove(idx_fn)

    dataset: Dataset = dataset_cls(
        dataset_paths=dataset_paths, index_mapping_dir=index_mapping_dir, **data_impl_kwargs
    )
    gc.collect()

    # Calculate and add has to metadata files (ie., *.idx.info files)

    if not is_distributed or (is_distributed and torch.distributed.get_rank() == 0):
        add_hash_to_metadata(index_mapping_dir, dataset_paths)

    if use_upsampling:
        assert num_samples is not None, (
            'To enable upsampling, "num_samples" need to be specified as '
            "the number of samples in the upsampled dataset"
        )
        # We save the scaling down for here.
        if limit_batches_scale_factor is not None:
            num_samples = int(num_samples * (limit_batches_scale_factor * len(dataset)))
        data_prefix = cfg.get("data_prefix", None)
        if data_prefix is None:
            data_prefix = os.path.commonprefix(dataset_paths)

        # Delete previous resampled index file (like small.csv_small.csv_indexmap_19200mns_128msl_0.00ssp_42s.npy) if stale
        if not is_distributed or (is_distributed and torch.distributed.get_rank() == 0):
            if any(file_changed_list):
                delete_stale_resampled_file(index_mapping_dir if index_mapping_dir else data_prefix)
            else:
                logging.info("No .idx.info files have changed. Deleting resampled index file is not necessary")

        # small.csv_small.csv_indexmap_19200mns_128msl_0.00ssp_42s.npy gets built here
        dataset = ResamplingMappedDataset(
            dataset,
            num_samples=num_samples,
            cfg=cfg,
            data_prefix=data_prefix,
            index_mapping_dir=index_mapping_dir,
            max_seq_length=cfg.max_seq_length,
            seed=cfg.seed,
        )
    return dataset
