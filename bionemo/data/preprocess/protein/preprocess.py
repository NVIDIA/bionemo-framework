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

import gzip
import os
import pathlib
import shutil
import tempfile
from multiprocessing import Pool
from typing import Optional, Tuple

import numpy as np
import pyfastx
import requests
import tqdm
from nemo.utils import logging

from bionemo.data.utils import download_registry_from_ngc, get_ngc_registry_file_list, verify_checksum_matches


__all__ = ['UniRef50Preprocess']

ROOT_DIR = '/tmp/uniref50'
MD5_CHECKSUM = 'e619d3689749562d743f8ecf29a7a7c2'


class UniRef50Preprocess(object):
    def __init__(self, root_directory: Optional[str] = ROOT_DIR, checksum: Optional[str] = MD5_CHECKSUM) -> None:
        """Prepocess UniRef50 data for pre-training.

        Args:
            root_directory (Optional[str]): Directory for download. Defaults to /tmp/uniref50.
            checksum (Optional[str]): Checksum for file


        Data are downloaded to root_directory/raw (/tmp/uniref50/raw). The split data can be found in
        root_directory/processed.
        """
        super().__init__()
        self.root_directory = pathlib.Path(root_directory)
        self.checksum = checksum

    def _process_file(self, url, download_dir):
        """Download UniRef50 file and unzip

        Args:
            url (str): URL for UniRef50 location.
            download_dir (str): Download directory for UniRef50 file.

        Returns:
            str: Path to UniRef50 FASTA file
        """
        assert url.endswith('.fasta.gz'), AssertionError(f'Expected URL to end with `.fasta.gz`, got {url}..')

        filename, gz_ext = os.path.splitext(os.path.split(url)[-1])  # gz extension
        filename, fasta_ext = os.path.splitext(filename)  # fasta extension

        file_path = os.path.join(download_dir, filename + fasta_ext)
        tmp_file_path = os.path.join(download_dir, filename + '_tmp' + fasta_ext)

        gz_file_path = file_path + gz_ext
        tmp_gz_file_path = tmp_file_path + gz_ext

        if os.path.exists(file_path):
            logging.info(f'{url} already exists at {file_path}...')
            return file_path

        logging.info(f'Downloading file to {gz_file_path}...')
        try:
            if not os.path.exists(gz_file_path):
                # Download gzipped file from url
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(tmp_gz_file_path, 'wb') as f:
                        for chunk in r.raw.stream(1024, decode_content=False):
                            if chunk:
                                f.write(chunk)

                os.rename(tmp_gz_file_path, gz_file_path)

            # Extract gzipped file and clean up
            logging.info(f'Extracting file to {file_path}...')
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(tmp_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.rename(tmp_file_path, file_path)
            os.remove(gz_file_path)

            return file_path

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.error(f'{url} Not found')
                return
            else:
                logging.error(f'Could not download file {url}: {e.response.status_code}')
                raise e

    def process_files_uniprot(self, url, download_dir):
        """Download the UniRef50 fasta file and decompress it.

        Parameters:
            url (str): URL for UniRef50 location.
            download_dir (str): Download directory for UniRef50 file.
        """

        logging.info('Data processing can take an hour or more depending on system resources.')

        logging.info(f'Downloading file from {url}...')

        os.makedirs(download_dir, exist_ok=True)
        file_path = self._process_file(url=url, download_dir=download_dir)
        return file_path

    @staticmethod
    def process_files_ngc(ngc_registry_target, ngc_registry_version, download_dir, output_dir, checksum):
        assert os.environ.get('NGC_CLI_API_KEY', False), AssertionError(
            """NGC API key not defined as environment variable "NGC_CLI_API_KEY".
                                                                           Aborting resource download."""
        )
        ngc_org = os.environ.get('NGC_CLI_ORG', None)
        assert ngc_org, AssertionError('NGC org must be defined by the environment variable NGC_CLI_ORG')
        ngc_team = os.environ.get('NGC_CLI_TEAM', None)

        # Check if resource already exists at final destination
        file_list = get_ngc_registry_file_list(ngc_registry_target, ngc_registry_version, ngc_org, ngc_team)
        file_exists = False
        if len(file_list) > 1:
            logging.info('Checksum verification not supported if resource contains more than one file.')
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
            logging.info(f'Download complete at {output_path}.')
        else:
            logging.info(f'File download skipped because file exists at {output_path} and has expected checksum.')

        return output_path

    @staticmethod
    def _index_fasta_data(fasta_indexer, val_size, test_size, random_seed):
        """Create index lists for train, validation, and test splits

        Args:
            fasta_indexer (pyfastx): Memory mapped index of UniRef50 FASTA file
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
        assert len(val_samples) == val_size, AssertionError('Validation dataset is not the correct size.')
        assert len(test_samples) == test_size, AssertionError('Test dataset is not the correct size.')
        assert len(fasta_indexer) - len(val_samples) - len(test_samples) == len(train_samples), AssertionError(
            'Train dataset is not the correct size.'
        )
        return train_samples, val_samples, test_samples

    @staticmethod
    def _protein_sequence_filewriter_map(args):
        '''enables p.map'''
        ordered_args = (
            args['fasta_indexer'],
            args['record_id_list'],
            args['file_index'],
            args['split_name'],
            args['output_dir'],
            args['delimiter'],
        )
        return UniRef50Preprocess._protein_sequence_filewriter(*ordered_args)

    @staticmethod
    def _protein_sequence_filewriter(fasta_indexer, record_id_list, file_index, split_name, output_dir, delimiter=','):
        """CSV file writer for FASTA data

        Args:
            fasta_indexer (Union[pyfastx, str]): Memory mapped index of UniRef50 FASTA file or name of fasta file to open.
                if intended to be use with multiprocessing.Pool, pass in a filename.
            record_id_list (Numpy array): array of file indexes for the splits
            file_index (int): Index number of the filename.
            split_name (str): Directory name for the split -- "train", "val", "test"
            output_dir (str): Output directory for CSV data.
            delimiter (str, optional): CSV delimiter. Defaults to ','.
        """

        split_path = os.path.join(output_dir, split_name)
        pathlib.Path(split_path).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(split_path, f'x{str(file_index).zfill(3)}.csv')

        if isinstance(fasta_indexer, str):
            # NOTE pass a string if you want to use with Pool.map
            _idx = pyfastx.Fasta(
                fasta_indexer, build_index=True, uppercase=True, key_func=lambda x: x.split()[0][len("UniRef90_") :]
            )
            fasta_indexer = _idx

        with open(file_name, 'w') as fh:
            header_str = delimiter.join(['record_id', 'record_name', 'sequence_length', 'sequence'])
            fh.write(header_str + '\n')
            for record_id in record_id_list:
                record = fasta_indexer[record_id]
                output = delimiter.join([str(record.id), record.name, str(len(record.seq)), record.seq])
                fh.write(output + '\n')
        return

    def train_val_test_split(self, train_samples, val_samples, test_samples, num_csv_files, fasta_indexer, output_dir):
        """Create CSV files for train, val, test data

        Args:
            train_samples (numpy array): Array of index numbers for training data.
            val_samples (numpy array): Array of index numbers for validation data
            test_samples (numpy array): Array of index numbers for test data
            num_csv_files (int): Number of CSV files to create for each train/val/test split.
            fasta_indexer (pyfastx): Memory mapped index of UniRef50 FASTA file
            output_dir (str): Output directory for CSV data.
        """

        for split_name, record_id_list in zip(['train', 'val', 'test'], [train_samples, val_samples, test_samples]):
            logging.info(f'Creating {split_name} split...')

            for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files)):
                logging.debug(f'Writing file number {file_index}...')
                self._protein_sequence_filewriter(
                    record_id_list=record_id_split,
                    file_index=file_index,
                    split_name=split_name,
                    fasta_indexer=fasta_indexer,
                    output_dir=output_dir,
                )

    def prepare_dataset(
        self,
        ngc_registry_target=None,
        ngc_registry_version=None,
        url='https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz',
        source='ngc',
        output_dir=None,
        num_csv_files=50,
        val_size=5000,
        test_size=1000000,
        random_seed=0,
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
            'Download and preprocess of UniRef50 data does not currently use GPU. Workstation or CPU-only instance recommended.'
        )

        download_dir = self.root_directory.joinpath('raw')
        if output_dir is None:
            output_dir = self.root_directory.joinpath('processed')
        os.makedirs(output_dir, exist_ok=True)
        if source == 'ngc':
            assert ngc_registry_target is not None
            assert ngc_registry_version is not None
            file_path = self.process_files_ngc(
                ngc_registry_target=ngc_registry_target,
                ngc_registry_version=ngc_registry_version,
                download_dir=download_dir,
                output_dir=output_dir,
                checksum=self.checksum,
            )

        elif source == 'uniprot':
            file_path = self.process_files_uniprot(url=url, download_dir=download_dir)

        logging.info('UniRef50 data processing complete.')

        logging.info('Indexing UniRef50 dataset.')
        fasta_indexer = pyfastx.Fasta(file_path, build_index=True, uppercase=True)
        train_samples, val_samples, test_samples = self._index_fasta_data(
            fasta_indexer=fasta_indexer, val_size=val_size, test_size=test_size, random_seed=random_seed
        )

        logging.info(f'Writing processed dataset files to {output_dir}...')
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
        uf50_datapath,
        uf90_datapath,
        cluster_mapping_tsv,
        uf50_output_dir,
        uf90_output_dir,
        num_csv_files=50,
        val_size=5000,
        test_size=1000000,
        random_seed=0,
        sort_fastas=True,
    ):
        """
        Prepares and splits the dataset into train/test/validation subsets, converts the fasta files to CSV format,
        and constructs both counts and start memmaps for each dataset. These are used for the underlying cluster mapping
        and are *required.*

        Args:
            uf50_datapath (str): Path to the raw fasta file for UniRef50. The data is divided into train/test/validation
                subsets and is utilized to decide which clusters to sample.
            uf90_datapath (str): Path to the raw fasta file for UniRef90. The data is processed into CSV format similar to
                uf50 but isn't split into train/test/validation. The sequences are ultimately used during training.
            cluster_mapping_tsv (str): Path to the TSV file where the first column represents the cluster ID (fasta header in uf50)
                and the second column lists the members separated by commas. The members correspond to entries in the uf90 fasta file.
            uf50_output_dir (str): Directory where the processed CSVs for uf50 are saved. This directory will have subdirectories
                'train', 'test', and 'val'.
            uf90_output_dir (str): Directory where the processed CSVs for uf90 are saved. A child directory named 'uf90_csvs' is
                created inside this directory for storing the CSVs.
            num_csv_files (int, optional): Number of files to divide each fasta file into after preprocessing. Defaults to 50.
            val_size (int, optional): Number of samples designated for the validation set.
            test_size (int, optional): Number of samples designated for the test set. The training size is inferred from
                test_size and val_size.
            random_seed (int, optional): Seed for randomization when splitting samples for train/test/validation. Defaults to 0.

        Returns:
            None

        Note:
            The method constructs 'cluster_map.json' inside the `uf90_output_dir` which is vital for subsequent steps.
            The structure of the output directories is essential for the YAML configuration file.
        """

        os.makedirs(uf50_output_dir, exist_ok=True)
        os.makedirs(uf90_output_dir, exist_ok=True)

        # Do this for uf50, uf90
        logging.info('Indexing UniRef50 dataset.')
        uf50_fasta_indexer = pyfastx.Fasta(uf50_datapath, build_index=True, uppercase=True)
        logging.info('Indexing UniRef90 dataset.')
        # TODO: this is the thing thats causing us issues with some IDs (why didnt this cause problems before?)
        uf90_fasta_indexer = pyfastx.Fasta(uf90_datapath, build_index=True, uppercase=True)

        logging.info('Creating cluster mapping')
        # cluster_map_resources = self._sort_fastas_load_cluster_mapping(uf50_fasta_indexer, uf90_fasta_indexer, cluster_mapping_tsv, sort_fastas)
        global_starts, global_counts, all_cids, all_cmembers = self._load_cluster_mapping(cluster_mapping_tsv)

        if sort_fastas:
            new_uf50_fn, new_uf90_fn = self._sort_fastas(
                uf50_fasta_indexer=uf50_fasta_indexer,
                uf90_fasta_indexer=uf90_fasta_indexer,
                all_cids=all_cids,
                all_cmembers=all_cmembers,
            )
        else:
            new_uf50_fn, new_uf90_fn = uf50_datapath, uf90_datapath

        # new_uf50_fn, new_uf90_fn, global_starts, global_counts = cluster_map_resources['uf50_fn'], cluster_map_resources['uf90_fn'], cluster_map_resources['starts'], cluster_map_resources['counts']

        logging.info('Loading sorted fasta files')
        new_uf50_fasta_indexer = pyfastx.Fasta(new_uf50_fn, build_index=True)
        new_uf90_fasta_indexer = pyfastx.Fasta(new_uf90_fn, build_index=True)

        # Undo the shuffling that occurs when splitting with sort.
        train_samples, val_samples, test_samples = map(
            np.sort,
            self._index_fasta_data(
                fasta_indexer=new_uf50_fasta_indexer, val_size=val_size, test_size=test_size, random_seed=random_seed
            ),
        )

        for split_name, record_id_list in zip(['train', 'val', 'test'], [train_samples, val_samples, test_samples]):
            logging.info(f"Making cluster memmap for {split_name}")
            split_path = os.path.join(uf50_output_dir, split_name)
            pathlib.Path(split_path).mkdir(parents=True, exist_ok=True)

            counts_fn = os.path.join(split_path, 'counts.mmap')
            starts_fn = os.path.join(split_path, 'starts.mmap')

            logging.warning("Not tracking mmap dtype- should intantiate this")
            _counts, _starts = self._make_local_memmaps(
                record_id_list, global_starts, global_counts, counts_mmap_fn=counts_fn, starts_mmap_fn=starts_fn
            )

        logging.info(f'Writing processed uf50 dataset files to {uf50_output_dir}...')
        self.train_val_test_split(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            num_csv_files=num_csv_files,
            fasta_indexer=new_uf50_fasta_indexer,
            output_dir=uf50_output_dir,
        )

        # force the new cluster order for uf90
        new_uf90_fasta_indexer = pyfastx.Fasta(new_uf90_fn, build_index=True, uppercase=True)  # Duplicate for testing
        record_id_list = np.arange(len(new_uf90_fasta_indexer))
        split_name = 'uf90_csvs'
        with Pool(16) as p:
            p.map(
                UniRef50Preprocess._protein_sequence_filewriter_map,
                [
                    {
                        'record_id_list': record_id_split,
                        'file_index': file_index,
                        'split_name': split_name,
                        'fasta_indexer': new_uf90_fn,
                        'output_dir': uf90_output_dir,
                        'delimiter': ',',
                    }
                    for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files))
                ],
            )

    @staticmethod
    def _make_local_memmaps(
        samples_arr, starts_global, counts_global, counts_mmap_fn, starts_mmap_fn, memmap_dtype=np.uint64
    ):
        '''Constructs memmaps using only the locally available samples. starts and counts remain the same, but we use the
        sample_arr to find them in the global counts and global starts maps.

        Returns - counts memmap and starts memmap
        '''
        counts_local_mm = np.memmap(counts_mmap_fn, dtype=memmap_dtype, mode='w+', shape=(len(samples_arr),))
        starts_local_mm = np.memmap(starts_mmap_fn, dtype=memmap_dtype, mode='w+', shape=(len(samples_arr),))
        for i, global_sample_idx in enumerate(samples_arr):
            start = starts_global[global_sample_idx]
            counts = counts_global[global_sample_idx]
            starts_local_mm[i] = start
            counts_local_mm[i] = counts
        counts_local_mm.flush()
        starts_local_mm.flush()

        return counts_local_mm, starts_local_mm

    @staticmethod
    def _sort_fastas(uf50_fasta_indexer, uf90_fasta_indexer, all_cids, all_cmembers) -> Tuple[str, str]:
        new_uf50_fn = tempfile.NamedTemporaryFile().name
        new_uf90_fn = tempfile.NamedTemporaryFile().name
        logging.info(f"Sorting fasta files in temporary file: {new_uf50_fn=} {new_uf90_fn=}")
        with open(new_uf50_fn, 'w') as uf50_fa_out, open(new_uf90_fn, 'w') as uf90_fa_out:
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
        '''Loads the cluster map into two arrays, counts and sizes. As a side effect, creates new
        temp fasta files that are in the same sort order as cluster_mapping_tsv. This is required for
        csv creation to match the indexing structure in the cluster map.
        '''
        with (open(cluster_mapping_tsv, 'r') as fd,):
            pos = 0
            all_cids, all_cmembers = [], []
            # Parse fasta
            import time

            start = time.time()
            for i, line in enumerate(tqdm.tqdm(fd)):
                if i == 0:
                    continue  # skip header
                cid, cmembers, *_ = line.strip().split("\t")
                members = cmembers.split(',')
                all_cids.append(cid)
                all_cmembers.append(members)
            end = time.time()
            logging.info('finished sorting in ', end - start)

            starts_global = np.zeros(shape=(len(all_cmembers)), dtype=np.int64)
            counts_global = np.zeros(shape=(len(all_cmembers)), dtype=np.int64)
            for i, (cid, members) in enumerate(zip(all_cids, all_cmembers)):
                starts_global[i] = pos
                counts_global[i] = len(members)
                pos += len(members)

                # This has to return the right fasta filenames for this all to work out.
        return starts_global, counts_global, all_cids, all_cmembers
