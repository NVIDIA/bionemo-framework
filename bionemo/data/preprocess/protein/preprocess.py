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
import os
import pathlib
import shutil
import tempfile
from multiprocessing import Pool
from typing import Optional, Tuple, Union

import numpy as np
import pyfastx
import requests
import tqdm
from nemo.utils import logging

from bionemo.data.utils import download_registry_from_ngc, get_ngc_registry_file_list, verify_checksum_matches


__all__ = ["FastaPreprocess", "UniRef50Preprocess", "ESM2Preprocess"]

UNIREF50_ROOT_DIR = "/tmp/uniref50"
UNIREF50_MD5_CHECKSUM = "e619d3689749562d743f8ecf29a7a7c2"


class PreprocessMixin:
    def prepare_dataset(self, *args, **kwargs):
        """Prepare dataset with the following requirement(s).

        1. (Optional) download dataset
        2. Preprocess dataset in pyfastx.Fasta
        3. Split samples into train/validate/test portions
        """
        raise NotImplementedError

    def _download_and_extract_fasta_gz(self, url, download_dir):
        """Download fasta.gz from url and unzip

        Args:
            url (str): URL for fasta.gz location.
            download_dir (str): Download directory

        Returns:
            str: Path to unzipped FASTA file
        """
        assert url.endswith(".fasta.gz"), AssertionError(f"Expected URL to end with `.fasta.gz`, got {url}..")

        filename, gz_ext = os.path.splitext(os.path.split(url)[-1])  # gz extension
        filename, fasta_ext = os.path.splitext(filename)  # fasta extension

        file_path = os.path.join(download_dir, filename + fasta_ext)
        tmp_file_path = os.path.join(download_dir, filename + "_tmp" + fasta_ext)

        gz_file_path = file_path + gz_ext
        tmp_gz_file_path = tmp_file_path + gz_ext

        if os.path.exists(file_path):
            logging.info(f"{url} already exists at {file_path}...")
            return file_path

        logging.info(f"Downloading file to {gz_file_path}...")
        try:
            if not os.path.exists(gz_file_path):
                # Download gzipped file from url
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(tmp_gz_file_path, "wb") as f:
                        for chunk in r.raw.stream(1024, decode_content=False):
                            if chunk:
                                f.write(chunk)

                os.rename(tmp_gz_file_path, gz_file_path)

            # Extract gzipped file and clean up
            logging.info(f"Extracting file to {file_path}...")
            with gzip.open(gz_file_path, "rb") as f_in:
                with open(tmp_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.rename(tmp_file_path, file_path)
            os.remove(gz_file_path)

            return file_path

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.error(f"{url} Not found")
                return
            else:
                logging.error(f"Could not download file {url}: {e.response.status_code}")
                raise e

    @staticmethod
    def _index_fasta_data(fasta_indexer, val_size, test_size, random_seed):
        """Create index lists for train, validation, and test splits

        Args:
            fasta_indexer (pyfastx): Memory mapped index of FASTA file
            val_size (int): Number of protein sequences to put in validation set.
            test_size (int): Numter of protein sequences to put in test set.
            random_seed (int): Random seed.
            ordered_splits (bool): sorts the resulting array of samples.

        Returns:
            List of indexes: list of train, validation, test indexes
        """
        sample_list = np.arange(len(fasta_indexer))

        rng = np.random.default_rng(random_seed)
        rng.shuffle(sample_list)

        val_samples = sample_list[:val_size]
        test_samples = sample_list[val_size : val_size + test_size]
        train_samples = sample_list[val_size + test_size :]
        assert len(val_samples) == val_size, AssertionError("Validation dataset is not the correct size.")
        assert len(test_samples) == test_size, AssertionError("Test dataset is not the correct size.")
        assert len(fasta_indexer) - len(val_samples) - len(test_samples) == len(train_samples), AssertionError(
            "Train dataset is not the correct size."
        )
        return train_samples, val_samples, test_samples

    @staticmethod
    def _protein_sequence_filewriter_map(args):
        """enables p.map"""
        ordered_args = (
            args["fasta_indexer"],
            args["record_id_list"],
            args["file_index"],
            args["split_name"],
            args["output_dir"],
            args["delimiter"],
        )
        return PreprocessMixin._protein_sequence_filewriter(*ordered_args)

    @staticmethod
    def _protein_sequence_filewriter(fasta_indexer, record_id_list, file_index, split_name, output_dir, delimiter=","):
        """CSV file writer for FASTA data

        Args:
            fasta_indexer (Union[pyfastx, str]): Memory mapped index of FASTA file or name of fasta file to open.
                if intended to be use with multiprocessing.Pool, pass in a filename.
            record_id_list (Numpy array): array of file indexes for the splits
            file_index (int): Index number of the filename.
            split_name (str): Directory name for the split -- "train", "val", "test"
            output_dir (str): Output directory for CSV data.
            delimiter (str, optional): CSV delimiter. Defaults to ','.
        """

        split_path = os.path.join(output_dir, split_name)
        pathlib.Path(split_path).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(split_path, f"x{str(file_index).zfill(3)}.csv")

        if isinstance(fasta_indexer, str):
            # NOTE pass a string if you want to use with Pool.map
            _idx = pyfastx.Fasta(
                fasta_indexer,
                build_index=True,
                uppercase=True,
                key_func=lambda x: x.split()[0][len("UniRef90_") :],  # TODO (sichu): clean up UniRef90_
            )
            fasta_indexer = _idx

        with open(file_name, "w") as fh:
            header_str = delimiter.join(["record_id", "record_name", "sequence_length", "sequence"])
            fh.write(header_str + "\n")
            for record_id in record_id_list:
                record = fasta_indexer[record_id]
                output = delimiter.join([str(record.id), record.name, str(len(record.seq)), record.seq])
                fh.write(output + "\n")
        return

    def train_val_test_split(self, train_samples, val_samples, test_samples, num_csv_files, fasta_indexer, output_dir):
        """Create CSV files for train, val, test data

        Args:
            train_samples (numpy array): Array of index numbers for training data.
            val_samples (numpy array): Array of index numbers for validation data
            test_samples (numpy array): Array of index numbers for test data
            num_csv_files (int): Number of CSV files to create for each train/val/test split.
            fasta_indexer (pyfastx): Memory mapped index of FASTA file
            output_dir (str): Output directory for CSV data.
        """

        for split_name, record_id_list in zip(["train", "val", "test"], [train_samples, val_samples, test_samples]):
            logging.info(f"Creating {split_name} split...")

            for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files)):
                logging.debug(f"Writing file number {file_index}...")
                self._protein_sequence_filewriter(
                    record_id_list=record_id_split,
                    file_index=file_index,
                    split_name=split_name,
                    fasta_indexer=fasta_indexer,
                    output_dir=output_dir,
                )

        if not os.path.isdir(output_dir):
            raise ValueError(
                f"Attempted to create a dataset output directory: {output_dir} but it failed and was not created."
            )


class FastaPreprocess(PreprocessMixin):
    def __init__(self, root_directory: Union[str, pathlib.Path]) -> None:
        """Prepocess fasta file for pre-training.

        Args:
            root_directory (Union[str, pathlib.Path]): Directory for download.

        The split data can be found in root_directory.
        """
        super().__init__()
        self.root_directory = pathlib.Path(root_directory)

    def prepare_dataset(
        self,
        fasta_path: Union[str, pathlib.Path],
        mode: str,
        num_csv_files: int = 50,
        output_dir: Union[str, pathlib.Path, None] = None,
    ) -> None:
        """
        Prepares and splits the dataset into train/test/validation subsets, converts the fasta file to CSV format.

        Args:
            fasta_path (Union[str, pathlib.Path]): Single fasta for pretraining.
            mode (str): Either "train", "val" and "test" to create dataset for specific split.
            num_csv_files (int): Number of csv broken down from fasta dataset.
            output_dir (Union[str, pathlib.Path, None]): Output directory of processed dataset. Default: root_directory / "processed"
            random_seed (int): Random seed used in train-val-test split.
        """
        if output_dir is None:
            output_dir = self.root_directory
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Indexing custom dataset from {fasta_path}.")

        fasta_path = str(fasta_path)
        fasta_indexer = pyfastx.Fasta(fasta_path, build_index=True, uppercase=True)
        record_id_list = np.arange(len(fasta_indexer))

        for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files)):
            logging.debug(f"Writing file number {file_index}...")
            self._protein_sequence_filewriter(
                record_id_list=record_id_split,
                file_index=file_index,
                split_name=mode,
                fasta_indexer=fasta_path,  # pass fasta_path to support Pool.map
                output_dir=output_dir,
            )


class UniRef50Preprocess(PreprocessMixin):
    def __init__(
        self, root_directory: str = UNIREF50_ROOT_DIR, checksum: Optional[str] = UNIREF50_MD5_CHECKSUM
    ) -> None:
        """Prepocess UniRef50 data for pre-training.

        Args:
            root_directory (str): Directory for download. Defaults to /tmp/uniref50.
            checksum (Optional[str]): Checksum for file


        Data are downloaded to root_directory/raw (/tmp/uniref50/raw). The split data can be found in
        root_directory/processed.
        """
        self.root_directory = pathlib.Path(root_directory)
        self.checksum = checksum

    def process_files_uniprot(self, url, download_dir):
        """Download the UniRef50 fasta file and decompress it.

        Parameters:
            url (str): URL for UniRef50 location.
            download_dir (str): Download directory for UniRef50 file.
        """

        logging.info("Data processing can take an hour or more depending on system resources.")

        logging.info(f"Downloading file from {url}...")

        os.makedirs(download_dir, exist_ok=True)
        file_path = self._download_and_extract_fasta_gz(url=url, download_dir=download_dir)
        return file_path

    @staticmethod
    def process_files_ngc(ngc_registry_target, ngc_registry_version, download_dir, output_dir, checksum):
        assert os.environ.get("NGC_CLI_API_KEY", False), AssertionError(
            """NGC API key not defined as environment variable "NGC_CLI_API_KEY".
                                                                           Aborting resource download."""
        )
        ngc_org = os.environ.get("NGC_CLI_ORG", None)
        assert ngc_org, AssertionError("NGC org must be defined by the environment variable NGC_CLI_ORG")
        ngc_team = os.environ.get("NGC_CLI_TEAM", None)

        # Check if resource already exists at final destination
        file_list = get_ngc_registry_file_list(ngc_registry_target, ngc_registry_version, ngc_org, ngc_team)
        file_exists = False
        if len(file_list) > 1:
            logging.info("Checksum verification not supported if resource contains more than one file.")
        else:
            file_name = file_list[0]
            output_path = os.path.join(output_dir, file_name)
            if os.path.exists(output_path):
                file_exists = True if verify_checksum_matches(output_path, checksum) else False

        # Download resource and copy if needed
        if not file_exists:
            os.makedirs(download_dir, exist_ok=True)
            tmp_download_path = download_registry_from_ngc(
                ngc_registry_target=ngc_registry_target,
                ngc_registry_version=ngc_registry_version,
                ngc_org=ngc_org,
                ngc_team=ngc_team,
                dest=download_dir,
                expected_checksum=checksum,
            )

            # Move to destination directory and clean up
            file_name = os.path.basename(tmp_download_path)
            output_path = os.path.join(output_dir, file_name)  # Ensures output_path is defined when file is downloaded
            shutil.copyfile(tmp_download_path, output_path)
            logging.info(f"Download complete at {output_path}.")
        else:
            logging.info(f"File download skipped because file exists at {output_path} and has expected checksum.")

        return output_path

    def prepare_dataset(
        self,
        output_dir: Optional[pathlib.Path] = None,
        ngc_registry_target=None,
        ngc_registry_version=None,
        url: str = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
        source: str = "ngc",
        num_csv_files: int = 50,
        val_size: int = 5000,
        test_size: int = 1000000,
        random_seed: int = 0,
    ):
        """Download UniRef50 dataset and split into train, valid, and test sets.

        Args:
            url (str): URL for UniRef50 location.
            num_csv_files (int): Number of CSV files to create for each train/val/test split.
            val_size (int): Number of protein sequences to put in validation set.
            test_size (int): Number of protein sequences to put in test set.
            random_seed (int): Random seed.
        """

        logging.info(
            "Download and preprocess of UniRef50 data does not currently use GPU. Workstation or CPU-only instance recommended."
        )

        download_dir = self.root_directory.joinpath("raw")
        if output_dir is None:
            output_dir = self.root_directory.joinpath("processed")
        os.makedirs(output_dir, exist_ok=True)
        if source == "ngc":
            assert ngc_registry_target is not None
            assert ngc_registry_version is not None
            file_path = self.process_files_ngc(
                ngc_registry_target=ngc_registry_target,
                ngc_registry_version=ngc_registry_version,
                download_dir=download_dir,
                output_dir=output_dir,
                checksum=self.checksum,
            )

        elif source == "uniprot":
            file_path = self.process_files_uniprot(url=url, download_dir=download_dir)

        logging.info("UniRef50 data processing complete.")

        logging.info("Indexing UniRef50 dataset.")
        fasta_indexer = pyfastx.Fasta(file_path, build_index=True, uppercase=True)
        train_samples, val_samples, test_samples = self._index_fasta_data(
            fasta_indexer=fasta_indexer, val_size=val_size, test_size=test_size, random_seed=random_seed
        )

        logging.info(f"Writing processed dataset files to {output_dir}...")
        self.train_val_test_split(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            num_csv_files=num_csv_files,
            fasta_indexer=fasta_indexer,
            output_dir=output_dir,
        )


class ESM2Preprocess(UniRef50Preprocess):
    def prepare_dataset(
        self,
        uf50_datapath: pathlib.Path,
        uf50_output_dir: pathlib.Path,
        cluster_mapping_tsv: Optional[pathlib.Path] = None,
        uf90_datapath: Optional[pathlib.Path] = None,
        uf90_output_dir: Optional[pathlib.Path] = None,
        num_csv_files: int = 50,
        sort_fastas: bool = False,
        mode: str = "train",
        num_preprocess_workers: int = 16,
    ):
        """
        Prepares and splits the dataset into train/test/validation subsets, converts the fasta files to CSV format,
        and constructs both counts and start memmaps for each dataset. These are used for the underlying cluster mapping
        and are *required.*

        Args:
            uf50_datapath: Path to the raw fasta file for UniRef50. The data is divided into train/test/validation
                subsets and is utilized to decide which clusters to sample.
            uf90_datapath: Path to the raw fasta file for UniRef90. The data is processed into CSV format similar to
                uf50 but isn't split into train/test/validation. The sequences are ultimately used during training.
            cluster_mapping_tsv: Path to the TSV file where the first column represents the cluster ID (fasta header in uf50)
                and the second column lists the members separated by commas. The members correspond to entries in the uf90 fasta file.
            uf50_output_dir: Directory where the processed CSVs for uf50 are saved. This directory will have subdirectories
                'train', 'test', and 'val'.
            uf90_output_dir: Directory where the processed CSVs for uf90 are saved. A child directory named 'uf90_csvs' is
                created inside this directory for storing the CSVs.
            num_csv_files (int, optional): Number of files to divide each fasta file into after preprocessing. Defaults to 50.
            sort_fastas: Booleal flag indicating whether or not to sort the fasta files. Default is False which means we do not sort.
            mode: A string indicating whether we want one of "train", "val" or "test.
            num_preprocess_workers: Int indicating how many workers we want to use for our _protein_sequence_filewriter_map

        Returns:
            None

        Note:
            The method constructs 'cluster_map.json' inside the `uf90_output_dir` which is vital for subsequent steps.
            The structure of the output directories is essential for the YAML configuration file.
        """

        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Mode {mode} is not one of train, val, test.")

        if mode != "train" and sort_fastas is True:
            raise ValueError("Cannot have val/test mode with sort_fastas True")

        if mode == "train" and sort_fastas is False:
            logging.warning("Mode is train, but sort_fastas is false. This might have unintended behavior.")

        if mode != "train" and uf90_datapath is not None:
            raise ValueError("Currently, only training mode utilizes the uf90 datapaths.")

        if mode in ["train", "val", "test"] and not os.path.exists(uf50_datapath):
            raise FileNotFoundError(f"input argument uf50_datapath: {uf50_datapath} is not found in mode {mode}.")
        if mode in ["train"] and not os.path.exists(uf90_datapath):
            raise FileNotFoundError(f"input argument uf90_datapath: {uf90_datapath} is not found in mode {mode}.")
        if mode in ["train"] and not os.path.exists(cluster_mapping_tsv):
            raise FileNotFoundError(
                f"input argument cluster_mapping_tsv: {cluster_mapping_tsv} is not found in mode {mode}."
            )

        logging.info("Indexing UniRef50 dataset.")
        uf50_fasta_indexer = pyfastx.Fasta(uf50_datapath, build_index=True, uppercase=True)

        os.makedirs(uf50_output_dir, exist_ok=True)
        if uf90_output_dir is not None:
            os.makedirs(uf90_output_dir, exist_ok=True)

        if uf90_datapath is not None:
            logging.info("Indexing UniRef90 dataset.")
            uf90_fasta_indexer = pyfastx.Fasta(uf90_datapath, build_index=True, uppercase=True)

        logging.info("Creating cluster mapping")
        if cluster_mapping_tsv is not None:
            global_starts, global_counts, all_cids, all_cmembers = self._load_cluster_mapping(cluster_mapping_tsv)

        if sort_fastas:
            uf50_datapath, uf90_datapath = self._sort_fastas(
                uf50_fasta_indexer=uf50_fasta_indexer,
                uf90_fasta_indexer=uf90_fasta_indexer,
                all_cids=all_cids,
                all_cmembers=all_cmembers,
            )
            new_uf90_fasta_indexer = pyfastx.Fasta(uf90_datapath, build_index=True)

        logging.info("Loading sorted fasta files")
        new_uf50_fasta_indexer = pyfastx.Fasta(uf50_datapath, build_index=True)

        record_id_list = np.sort(np.arange(len(new_uf50_fasta_indexer)))

        split_path = os.path.join(uf50_output_dir, mode)
        pathlib.Path(split_path).mkdir(parents=True, exist_ok=True)
        # Create the memmap files.
        counts_fn = os.path.join(split_path, "counts.mmap")
        starts_fn = os.path.join(split_path, "starts.mmap")

        # Create the CSV files
        logging.info(f"Writing processed uf50 {mode} dataset files to {uf50_output_dir}...")
        for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files)):
            logging.debug(f"Writing file number {file_index}...")
            self._protein_sequence_filewriter(
                record_id_list=record_id_split,
                file_index=file_index,
                split_name=mode,
                fasta_indexer=new_uf50_fasta_indexer,
                output_dir=uf50_output_dir,
            )

        # force the new cluster order for uf90
        if mode == "train":
            _, _ = self._make_local_memmaps(
                record_id_list, global_starts, global_counts, counts_mmap_fn=counts_fn, starts_mmap_fn=starts_fn
            )

            record_id_list = np.arange(len(new_uf90_fasta_indexer))
            split_name = "uf90_csvs"
            with Pool(num_preprocess_workers) as p:
                p.map(
                    UniRef50Preprocess._protein_sequence_filewriter_map,
                    [
                        {
                            "record_id_list": record_id_split,
                            "file_index": file_index,
                            "split_name": split_name,
                            "fasta_indexer": uf90_datapath,
                            "output_dir": uf90_output_dir,
                            "delimiter": ",",
                        }
                        for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files))
                    ],
                )

        if not os.path.isdir(split_path):
            raise ValueError(
                f"Attempted to create a dataset output directory: {split_path} in mode {mode} but it failed and was not created."
            )

    @staticmethod
    def _make_local_memmaps(
        samples_arr, starts_global, counts_global, counts_mmap_fn, starts_mmap_fn, memmap_dtype=np.uint64
    ):
        """Constructs memmaps using only the locally available samples. starts and counts remain the same, but we use the
        sample_arr to find them in the global counts and global starts maps.

        Returns - counts memmap and starts memmap
        """
        counts_local_mm = np.memmap(counts_mmap_fn, dtype=memmap_dtype, mode="w+", shape=(len(samples_arr),))
        starts_local_mm = np.memmap(starts_mmap_fn, dtype=memmap_dtype, mode="w+", shape=(len(samples_arr),))
        for i, global_sample_idx in enumerate(samples_arr):
            start = starts_global[global_sample_idx]
            counts = counts_global[global_sample_idx]
            starts_local_mm[i] = start
            counts_local_mm[i] = counts
        counts_local_mm.flush()
        starts_local_mm.flush()

        return counts_local_mm, starts_local_mm

    @staticmethod
    def _sort_fastas(
        uf50_fasta_indexer, uf90_fasta_indexer, all_cids, all_cmembers
    ) -> Tuple[pathlib.Path, pathlib.Path]:
        """
        Sorta fasta from the sampling pool for train.

        This function is required for the memmap dependency. For memmap, we want all samples
        sorted into the order in which we are going to use them for training.

        Args:
            uf50_fasta_indexer: The fasta indexer generated by pyfastx, from the uniref50 fasta file.
            uf90_fasta_indexer: The fasta indexer generated by pyfastx, from the uniref90 fasta file.
            all_cids: All uniref50 training samples.
            all_cmembers: All the members that can be sampled from uniref90, which
                can also be a 50, since they are sampled with replacement.

        Returns:
            A filename for the UF50 files.
            A filename for the UF90 files.
        """
        new_uf50_fn = tempfile.NamedTemporaryFile().name
        new_uf90_fn = tempfile.NamedTemporaryFile().name
        logging.info(f"Sorting fasta files in temporary file: {new_uf50_fn=} {new_uf90_fn=}")
        with open(new_uf50_fn, "w") as uf50_fa_out, open(new_uf90_fn, "w") as uf90_fa_out:
            for cid, members in tqdm.tqdm(zip(all_cids, all_cmembers)):
                uf50_entry = uf50_fasta_indexer[cid]
                uf50_fa_out.write(f">{uf50_entry.name}\n")
                uf50_fa_out.write(f"{uf50_entry.seq}\n")
                # Update new ordered fastas
                for member in members:
                    uf90_entry = uf90_fasta_indexer[member]
                    uf90_fa_out.write(f">{uf90_entry.name}\n")
                    uf90_fa_out.write(f"{uf90_entry.seq}\n")
        return new_uf50_fn, new_uf90_fn

    @staticmethod
    def _load_cluster_mapping(cluster_mapping_tsv):
        """Loads the cluster map into two arrays, counts and sizes. As a side effect, creates new
        temp fasta files that are in the same sort order as cluster_mapping_tsv. This is required for
        csv creation to match the indexing structure in the cluster map.
        """
        with open(cluster_mapping_tsv, "r") as fd:
            pos = 0
            all_cids, all_cmembers = [], []
            # Parse fasta

            for i, line in enumerate(tqdm.tqdm(fd)):
                if i == 0:
                    continue  # skip header
                cid, cmembers, *_ = line.strip().split("\t")
                members = cmembers.split(",")
                all_cids.append(cid)
                all_cmembers.append(members)

            starts_global = np.zeros(shape=(len(all_cmembers)), dtype=np.int64)
            counts_global = np.zeros(shape=(len(all_cmembers)), dtype=np.int64)
            for i, (cid, members) in enumerate(zip(all_cids, all_cmembers)):
                starts_global[i] = pos
                counts_global[i] = len(members)
                pos += len(members)

                # This has to return the right fasta filenames for this all to work out.
        return starts_global, counts_global, all_cids, all_cmembers
