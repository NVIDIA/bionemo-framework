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
import gzip
import json
import os
import pathlib
import shutil
import subprocess
import tarfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Literal

import numpy as np
import pandas as pd
import webdataset as wds
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from tfrecord.torch.dataset import TFRecordDataset

from bionemo.data.dna.splice_site_dataset import ChrSpliceSitesDataset
from bionemo.data.fasta_dataset import ConcatDataset
from bionemo.data.preprocess import ResourcePreprocessor
from bionemo.data.utils import expand_dataset_paths, gunzip
from bionemo.tokenizer.dna_tokenizer import KmerTokenizer
from bionemo.utils.fasta import FastaSplitNs
from bionemo.utils.gff import build_donor_acceptors_midpoints, parse_gff3
from bionemo.utils.remote import FTPRemoteResource, RemoteResource


@dataclass
class GRCh38p13_ResourcePreprocessor(ResourcePreprocessor):
    """ResourcePreprocessor for the human genome assembly produced by encode.
    GRCh38.p13, all primary chromosomes are downloaded. Since this resource does not create an archive for contigs,
    we must create a remote resource for each file. Preprocessing therefore requires working on sets of files.
    """

    dest_directory: str = "GRCh38.p13"

    def get_remote_resources(self) -> List[RemoteResource]:
        checksums = {
            "chr1.fna.gz": "43aab6c472a690d9c29c62fb0b47a3f6",
            "chr2.fna.gz": "9e54b2ba05e6866a72b9971a7354353b",
            "chr3.fna.gz": "41931af8120dd459c738f1baec318e6f",
            "chr4.fna.gz": "2c54d3c6ba306720991f739926c6f21e",
            "chr5.fna.gz": "8e0d2b7564f12365dbfa4a8a0d5a853f",
            "chr6.fna.gz": "c6b5c2b2e2ca35ca344f0effde41a8ef",
            "chr7.fna.gz": "4a1783cdaaa2eabaf519d4af64b28a97",
            "chr8.fna.gz": "9eb00c85eea5bd8c3cd657c5222e11f8",
            "chr9.fna.gz": "20f430cf235d488731b22feb94761cb8",
            "chr10.fna.gz": "bb89574740611a25a39e701c6d325af8",
            "chr11.fna.gz": "e734e9db468dc00e3f66750125afb95c",
            "chr12.fna.gz": "ee240c28c4b1a3d8c8629584cd74a1dc",
            "chr13.fna.gz": "0d34e0e2ee8b4c9f6ce9ff1d9da45ebc",
            "chr14.fna.gz": "62dc673400b14e3358bfadbe1a3eeb1e",
            "chr15.fna.gz": "67fc014176fdd16dee78f7ada13442bf",
            "chr16.fna.gz": "5bbf216e70f852b92086873ff57a75f0",
            "chr17.fna.gz": "bec08127a5a09e7a5ebb4b0d7935da71",
            "chr18.fna.gz": "cd7fba61549c548a9cf0933a75ad0f8d",
            "chr19.fna.gz": "74f27da749d5413748750004ae5f41aa",
            "chr20.fna.gz": "72f06b446f0419457f5db73ba5952686",
            "chr21.fna.gz": "73a7361a0ec60c88fc8a19dd4e071201",
            "chr22.fna.gz": "af2eb2257f7a4f42802f53e360d86400",
            "chrX.fna.gz": "c4b5470fe49284db1f98a94c997179e9",
            "chrY.fna.gz": "e32e7eac0f2f17f070c0244ecaa60a46",
        }

        basename = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/"
        resources = []

        # Check the remote for the filename structure, one for each autosome.
        for contig in list(range(1, 23)) + ["X", "Y"]:
            filename = f"chr{contig}.fna.gz"
            url = basename + filename
            resource = RemoteResource(
                dest_directory=self.dest_directory,
                dest_filename=filename,
                root_directory=self.root_directory,
                checksum=checksums.get(filename),
                url=url,
            )
            resources.append(resource)
        return resources

    def prepare_resource(self, resource: RemoteResource) -> str:
        """Logs and downloads the passed resource.

        resource: RemoteResource - Resource to be prepared.

        Returns - the absolute destination path for the downloaded resource
        """
        logging.info(f"Downloading {resource.url}")
        return resource.download_resource()

    def prepare(self):
        return [self.prepare_resource(resource) for resource in self.get_remote_resources()]


class Hg38chromResourcePreprocessor(ResourcePreprocessor):
    """Prepackaged object for downloading hg38 chroms from UCSC. Returns the GenomeResource object associated with it.
    Methods like these should be tightly coupled with the data, and should NOT be very reusable. They specify a specific way
    to download and prepare a specific dataset. We should chose from a predefine set of pipelines.
    """

    def get_remote_resources(self) -> List[RemoteResource]:
        dest_directory = "hg38"

        checksum = "a5aa5da14ccf3d259c4308f7b2c18cb0"
        url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chromFa.tar.gz"

        obj = RemoteResource(
            dest_directory=dest_directory,
            dest_filename="hg38.chromFa.tar.gz",
            root_directory=self.root_directory,
            checksum=checksum,
            url=url,
        )
        return [obj]

    def prepare(self) -> List[str]:
        """hg38 prepare method:

        Download the remote tarball to a local temp file
        Unpack the tarball
        (optionally) split all sequences on 'N's

        Returns the fully qualified path of the prepared filenames.
        """
        logging.info("Downloading")
        [resource] = self.get_remote_resources()
        resource.download_resource(overwrite=False)

        # For hg38 we expect there to be tar.gz
        with tarfile.open(resource.fully_qualified_dest_filename) as file:
            logging.info("Extracting hg38")
            file.extractall(resource.fully_qualified_dest_folder)
        basename = os.path.join(resource.fully_qualified_dest_folder, "chroms")
        return [str(filepath) for filepath in pathlib.Path(basename).glob("*.fa")]


@dataclass
class GRCh38Ensembl99FastaResourcePreprocessor(ResourcePreprocessor):
    dest_directory = "GRCh38.ensembl.99"

    def get_remote_resources(self) -> List[RemoteResource]:
        fasta_checksums = {
            "Homo_sapiens.GRCh38.dna.chromosome.1.fa.gz": "6f157cbdaed6fcc811dfdef69f22b505",
            "Homo_sapiens.GRCh38.dna.chromosome.10.fa.gz": "24d6c88bf2a518a0ca386122bd568b78",
            "Homo_sapiens.GRCh38.dna.chromosome.11.fa.gz": "8cd66df217b4e872a65525db39a9b702",
            "Homo_sapiens.GRCh38.dna.chromosome.12.fa.gz": "96cdb417f9b049b75e3005cbf03027d9",
            "Homo_sapiens.GRCh38.dna.chromosome.13.fa.gz": "9049a7bcafadb55ad236052969a43500",
            "Homo_sapiens.GRCh38.dna.chromosome.14.fa.gz": "bd4f6eac70a32aaa64aeb05a59047cd8",
            "Homo_sapiens.GRCh38.dna.chromosome.15.fa.gz": "92d35108cfd2c0352116eed6c5d07dcf",
            "Homo_sapiens.GRCh38.dna.chromosome.16.fa.gz": "aa06df6a5350030370a5a5a7be87f2db",
            "Homo_sapiens.GRCh38.dna.chromosome.17.fa.gz": "f063898e374644bdc2502ba49c2cf7f1",
            "Homo_sapiens.GRCh38.dna.chromosome.18.fa.gz": "d8e1fd676b8527e7d0be5e8bca1d6b7d",
            "Homo_sapiens.GRCh38.dna.chromosome.19.fa.gz": "ca4f1a9927913fe5a62585a492d13c1a",
            "Homo_sapiens.GRCh38.dna.chromosome.2.fa.gz": "83420da4ab136c529d69a97784cc06e0",
            "Homo_sapiens.GRCh38.dna.chromosome.20.fa.gz": "09f9d3715f2f96dd7cb655e3e70c70f5",
            "Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz": "91d0fdc8e16fd9d6418765365bc5f50e",
            "Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz": "eb321ecd19ebe81ea0c8b88a969cd5fd",
            "Homo_sapiens.GRCh38.dna.chromosome.3.fa.gz": "2c81cc8b83ae869025eed0652c177955",
            "Homo_sapiens.GRCh38.dna.chromosome.4.fa.gz": "8522a6ed38b3d304654d8a43d73cec91",
            "Homo_sapiens.GRCh38.dna.chromosome.5.fa.gz": "6619cd2c8a4ff5152225b4342bfeccd2",
            "Homo_sapiens.GRCh38.dna.chromosome.6.fa.gz": "5f619230e8ce6e2282106d51bb0ab1b3",
            "Homo_sapiens.GRCh38.dna.chromosome.7.fa.gz": "6595478d50e702f16376afd6561c7efc",
            "Homo_sapiens.GRCh38.dna.chromosome.8.fa.gz": "e50080ae3c5169906499a6ade6cabe16",
            "Homo_sapiens.GRCh38.dna.chromosome.9.fa.gz": "69cdf198b7d535617285cb049b2701d1",
            "Homo_sapiens.GRCh38.dna.chromosome.X.fa.gz": "974059b6a79eeed96ccc8ee9d36d9e8e",
            "Homo_sapiens.GRCh38.dna.chromosome.Y.fa.gz": "c4d00789488f974a30f9bdf80718940b",
        }

        fasta_basename = "http://ftp.ensembl.org/pub/release-99/fasta/homo_sapiens/dna/"
        resources = []

        # Check the remote for the filename structure, one for each autosome.
        for contig in list(range(1, 23)) + ["X", "Y"]:
            fasta_filename = f"Homo_sapiens.GRCh38.dna.chromosome.{contig}.fa.gz"
            fasta_url = fasta_basename + fasta_filename
            fasta_resource = RemoteResource(
                dest_directory=self.dest_directory,
                dest_filename=fasta_filename,
                root_directory=self.root_directory,
                checksum=fasta_checksums.get(fasta_filename),
                url=fasta_url,
            )
            resources.append(fasta_resource)
        return resources

    def prepare_resource(self, resource: RemoteResource) -> str:
        logging.info(f"Downloading {resource.url}")
        resource.download_resource()
        return resource.fully_qualified_dest_filename

    def prepare(self):
        return [self.prepare_resource(resource) for resource in self.get_remote_resources()]


@dataclass
class GRCh38Ensembl99GFF3ResourcePreprocessor(ResourcePreprocessor):
    """Downloads the Ensembl datasets required for SpliceSite prediction in the DNABERT publication.
    We download the gff files as well, as we dont currently have another usecase for this reference.
    """

    train_perc: float = 0.8
    val_perc: float = 0.1
    test_perc: float = 1 - train_perc - val_perc
    size: int = 30000
    dest_directory: str = "GRCh38.ensembl.99"

    def get_remote_resources(self) -> List[RemoteResource]:
        gff_checksums = {
            "Homo_sapiens.GRCh38.99.chromosome.1.gff3.gz": "d25f02ad6ae1a4282d36712ba18ffd7e",
            "Homo_sapiens.GRCh38.99.chromosome.10.gff3.gz": "12d89463b97578cd7d707220840b9cc8",
            "Homo_sapiens.GRCh38.99.chromosome.11.gff3.gz": "4fae3f3ed60cfacbb11d0e80e41426ca",
            "Homo_sapiens.GRCh38.99.chromosome.12.gff3.gz": "5f4afb1b18a4217486f1c02fae940ee0",
            "Homo_sapiens.GRCh38.99.chromosome.13.gff3.gz": "a30128f4721169fb02e39bd9491bc235",
            "Homo_sapiens.GRCh38.99.chromosome.14.gff3.gz": "1ea87c335645a44ca6199a18fbbd1889",
            "Homo_sapiens.GRCh38.99.chromosome.15.gff3.gz": "5a4403ea419263f645e9afdf7870dcc6",
            "Homo_sapiens.GRCh38.99.chromosome.16.gff3.gz": "a5a71da50a8c670f7aeb45d9a8161c49",
            "Homo_sapiens.GRCh38.99.chromosome.17.gff3.gz": "c05033a6feda2d7556ebcc7d9ecb09a8",
            "Homo_sapiens.GRCh38.99.chromosome.18.gff3.gz": "de29c4f9c5517a16b5a94215285cf544",
            "Homo_sapiens.GRCh38.99.chromosome.19.gff3.gz": "80eb251e613b677e68f378f0ce7d93dd",
            "Homo_sapiens.GRCh38.99.chromosome.2.gff3.gz": "9ade0d0ceea30d09ac0fa78b17b2bcd8",
            "Homo_sapiens.GRCh38.99.chromosome.20.gff3.gz": "045fc9d7c92684a265763732e42f8836",
            "Homo_sapiens.GRCh38.99.chromosome.21.gff3.gz": "ef107abe0babacc5b1bebed6c30c22de",
            "Homo_sapiens.GRCh38.99.chromosome.22.gff3.gz": "cdaae16026f02d8de6d299e15370d8e3",
            "Homo_sapiens.GRCh38.99.chromosome.3.gff3.gz": "49a9ee40a63dbe69b74a183c7419d9bc",
            "Homo_sapiens.GRCh38.99.chromosome.4.gff3.gz": "e55b4b344a88595f3db27dad307592e8",
            "Homo_sapiens.GRCh38.99.chromosome.5.gff3.gz": "5824c261df8161eb6494ae168f43ecf1",
            "Homo_sapiens.GRCh38.99.chromosome.6.gff3.gz": "416ab751d58299ea38067b40a7d49e46",
            "Homo_sapiens.GRCh38.99.chromosome.7.gff3.gz": "33532bca0f8d1f898494daf7d5217aa2",
            "Homo_sapiens.GRCh38.99.chromosome.8.gff3.gz": "7e04f12425dd01fe77fa66a38498c44b",
            "Homo_sapiens.GRCh38.99.chromosome.9.gff3.gz": "17d121d9f6aa0c6fe35912850cb900fe",
            "Homo_sapiens.GRCh38.99.chromosome.X.gff3.gz": "4bbbb922322e57399bea4b9247834849",
            "Homo_sapiens.GRCh38.99.chromosome.Y.gff3.gz": "549fee6028a3b992637f6d8d775160fb",
        }

        resources = []
        gff_basename = "http://ftp.ensembl.org/pub/release-99/gff3/homo_sapiens/"

        # Check the remote for the filename structure, one for each autosome.
        for contig in list(range(1, 23)) + ["X", "Y"]:
            gff_filename = f"Homo_sapiens.GRCh38.99.chromosome.{contig}.gff3.gz"
            gff_url = gff_basename + gff_filename
            gff_resource = RemoteResource(
                dest_directory=self.dest_directory,
                dest_filename=gff_filename,
                root_directory=self.root_directory,
                checksum=gff_checksums.get(gff_filename),
                url=gff_url,
            )
            resources.append(gff_resource)
        return resources

    def prepare_resource(self, resource: RemoteResource) -> str:
        logging.info(f"Downloading {resource.url}")
        resource.download_resource()
        return resource.fully_qualified_dest_filename

    @staticmethod
    def _get_chr_from_filename(filename):
        """Built for the GRCh38.99 Ensembl filenames.

        "prefix/Homo_sapiens.GRCh38.99.chromosome.Y.gff3.gz" -> 'Y'
        """
        before, after = filename.split("chromosome")
        _, _chr, gff3, gz = after.split(".")
        return _chr

    def prepare(self) -> List[str]:
        # download the resources
        gff_gzs = [self.prepare_resource(resource) for resource in self.get_remote_resources()]
        chrs = [self._get_chr_from_filename(filename) for filename in gff_gzs]
        gffs = [filename[:-3] for filename in gff_gzs]

        # Gunzip.
        for gff_gz, gff in zip(gff_gzs, gffs):
            gunzip(gff_gz, gff, exist_ok=True)

        # Assemble datasets
        datasets = []
        for _chr, gff_filename in zip(chrs, gffs):
            # Filter non-autosomes
            if str(_chr) not in set(map(str, range(1, 23))):
                continue
            gff_contents = parse_gff3(gff_filename)
            donor_acceptor_midpoints = build_donor_acceptors_midpoints(gff_contents)
            datasets.append(ChrSpliceSitesDataset(donor_acceptor_midpoints, _chr))
        all_sites = ConcatDataset(datasets)

        # Choose cuts and make final splits.
        df = pd.DataFrame([all_sites[i] for i in range(len(all_sites))])

        # NOTE: this is the only thign that has to be decoupled to have a generic preprocessor.
        splits = self.write_train_val_test_split(df)

        return splits

    def write_train_val_test_split(self, df):
        """Performs a train/val/test split of the splice site dataframe

        Args:
            df (pd.DataFrame): Dataframe to split
        """
        # Lifted out of constructor
        sizes = [0, 0, 0]
        for i in range(self.size):
            sizes[i % 3] += 1

        df0, df1, df2 = df.loc[df.kind == 0], df.loc[df.kind == 1], df.loc[df.kind == 2]
        np.random.seed(724)
        # randomly downsample from each class
        indices0 = np.random.choice(len(df0), size=sizes[0], replace=False)
        indices1 = np.random.choice(len(df1), size=sizes[1], replace=False)
        indices2 = np.random.choice(len(df2), size=sizes[2], replace=False)
        sites_sample_df = pd.concat([df0.iloc[indices0], df1.iloc[indices1], df2.iloc[indices2]])

        # train val test split dataset sizes
        n_total = len(df)
        n_sampled = len(sites_sample_df)
        train_ub = int(self.train_perc * n_sampled)
        val_ub = int(self.val_perc * n_sampled) + train_ub
        test_ub = n_sampled

        np.random.seed(825)
        shuffled_indices = np.arange(n_total)
        np.random.shuffle(shuffled_indices)
        train_df = df.iloc[shuffled_indices[:train_ub]]
        val_df = df.iloc[shuffled_indices[train_ub:val_ub]]
        test_df = df.iloc[shuffled_indices[val_ub:test_ub]]
        # save train val test split
        datadir = Path(self.root_directory) / self.dest_directory
        os.makedirs(datadir, exist_ok=True)

        # Here its prepared and we are ready for whatever comes next.
        train_df.to_csv(datadir / "train.csv")
        val_df.to_csv(datadir / "val.csv")
        test_df.to_csv(datadir / "test.csv")
        return (datadir / "train.csv", datadir / "val.csv", datadir / "test.csv")


# This abstract dataclass helps solve the problem where we want to use
# dataclass to define the constructor for DNABERTPreprocess but also want to
# extend the constructor.
@dataclass
class DNABERTPreprocessorDataClass:
    genome_dir: str
    tokenizer_model_path: str
    tokenizer_vocab_path: str
    tokenizer_k: int
    dataset_conf: OmegaConf


class DNABERTPreprocess(DNABERTPreprocessorDataClass):
    root_directory = "/"

    def __init__(self, *args, **kwargs):
        """Downloads and preprocesses GRCh38p13 reference genome and constructs
        the tokenizer for DNABERT

        genome_dir (str): Directory to store the reference genome in
        tokenizer_model_path (str): Filepath to store the tokenzier parameters
        tokenizer_vocab_path (str): Filepath to store the tokenizer vocab
        tokenizer_k (int): k-mer size for the tokenizer
        dataset_conf (OmegaConf): has 'train', 'val', 'test' keys containing
            the names of preprocessed train/val/test files to use for training.
        """
        super().__init__(*args, **kwargs)
        self._validate_tokenizer_args(
            self.tokenizer_model_path,
            self.tokenizer_vocab_path,
        )

    def build_tokenizer(self, model_output_name, vocab_output_name, k):
        """Builds a tokenizer for a given k

        Args:
            model_output_name (str): Filepath to store the tokenizer parameters
            vocab_output_name (str): Filepath to store the tokenizer vocab
            k (int): k-mer size for the tokenizer
        """
        tokenizer = KmerTokenizer(k=k)
        tokenizer.build_vocab_from_k()
        tokenizer.save_vocab(
            model_file=model_output_name,
            vocab_file=vocab_output_name,
        )
        return tokenizer

    def _validate_tokenizer_args(self, model_output_name, vocab_output_name):
        model_exists = os.path.exists(model_output_name)
        vocab_exists = os.path.exists(vocab_output_name)
        if model_exists and vocab_exists:
            logging.warning(
                f"Tokenizer model file: {model_output_name} and tokenizer "
                f"vocab name: {vocab_output_name} already exist. Skipping "
                f"tokenizer building stage."
            )
            return
        elif model_exists:
            raise ValueError(
                f"Tokenizer model file: {model_output_name} already "
                f"exists, but vocab file: {vocab_output_name} does not."
            )
        elif vocab_exists:
            raise ValueError(
                f"Tokenizer vocab file: {vocab_output_name} already "
                f"exists, but model file: {model_output_name} does not."
            )

    def split_train_val_test_chrs(self, preprocessed_files):
        """Splits the preprocessed files into train/val/test dirs

        Args:
            preprocessed_files (List[str]): List of preprocessed files

        Raises:
            ValueError: If the a requested train/val/test file does not exist
                in the preprocessed files
        """
        splits = ["train", "val", "test"]
        preprocessed_filenames = [os.path.basename(f) for f in preprocessed_files]
        for split in splits:
            split_dir = os.path.join(self.genome_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            pattern = self.dataset_conf[split]
            files_from_config = expand_dataset_paths(pattern, "")
            for fa_file in files_from_config:
                # if a preprocessed file matches the pattern in the config
                # for train/val/test, copy it from the genome directory to the
                # appropriate folder for the split, e.g., train
                # Files will not be overwritten if they already exist.
                # this design is so that we can split training by chromosome
                # and can update the training chromosomes along the way.

                try:
                    preprocessed_index = preprocessed_filenames.index(fa_file)
                except ValueError:
                    raise ValueError(
                        f"File: {fa_file} from {split} config: {pattern} not " f"found in {preprocessed_filenames}."
                    )

                file_exists_in_split_dir = os.path.exists(os.path.join(split_dir, fa_file))
                if file_exists_in_split_dir:
                    logging.warning(
                        f"File: {fa_file} not copied to {split} split" f" directory because it already exists."
                    )
                else:
                    logging.info(f"Copying file: {fa_file} to {split_dir}")
                    shutil.copy(preprocessed_files[preprocessed_index], split_dir)

    def preprocess(self):
        """Preprocesses for the DNABERT model"""
        # TODO WARN!!!! ultimately we should let our config choose a valid resource.

        filenames = GRCh38p13_ResourcePreprocessor(
            dest_directory=self.genome_dir, root_directory=self.root_directory
        ).prepare()

        preprocessed_files = self.preprocess_fastas(filenames)

        logging.info("Creating train/val/test split.")
        self.split_train_val_test_chrs(preprocessed_files)

        self.build_tokenizer(
            self.tokenizer_model_path,
            self.tokenizer_vocab_path,
            self.tokenizer_k,
        )

    def preprocess_fastas(self, filenames):
        """Splits fasta files into contigs on N's, which is need for training

        Args:
            filenames (List[str]): List of files to preprocess

        Returns:
            List[str]: List of preprocessed files
        """
        logging.info("Splitting fasta files...")
        # TODO another place that could be configurable. For now we only have one.
        fasta_preprocessor = FastaSplitNs(filenames)
        preprocessed_files = []
        for fasta in fasta_preprocessor.get_elements():
            preprocessed_file = fasta_preprocessor.get_chunked_name(fasta)
            if os.path.exists(preprocessed_file):
                logging.warning(f"Splitting skipped: already processed " f"{preprocessed_file}")
            else:
                fasta_preprocessor.apply(fasta)
            preprocessed_files.append(preprocessed_file)
        return preprocessed_files


class CorePromoterResourcePreparer(ResourcePreprocessor):
    """This class is responsible for downloading the appropriate files for core promoter prediction.

    this comes from the HPDnew database, and is tightly coupled to (which reference?)

    """

    dest_directory = "GRCh38.ensembl.99"

    def get_remote_resources(self) -> List[RemoteResource]:
        resource_prom_checksum = "3c4915c7fa367f1dd3d9e86b47efc0eb"  # Downloaded and manually computed.
        resource_tata_checksum = "340f62c2162e44523be6328618313868"

        resource_prom_url = "ftp://ccg.epfl.ch:21/epdnew/H_sapiens/006/Hs_EPDnew_006_hg38.bed"

        resource_tata_url = "ftp://ccg.epfl.ch:21/epdnew/H_sapiens/006/db/promoter_motifs.txt"

        resource_prom = FTPRemoteResource(
            dest_directory=self.dest_directory,
            dest_filename="Hs_EPDnew_006_hg38.bed",  # Retain the existing filename
            root_directory=self.root_directory,
            checksum=resource_prom_checksum,
            url=resource_prom_url,
        )
        resource_tata = FTPRemoteResource(
            dest_directory=self.dest_directory,
            dest_filename="promoter_motifs.txt",  # Retain the existing filename
            root_directory=self.root_directory,
            checksum=resource_tata_checksum,
            url=resource_tata_url,
        )
        return [resource_prom, resource_tata]

    def prepare(self) -> List:
        # Just download? Anything else?
        resources = self.get_remote_resources()
        logging.info("Downloading promoter resources")
        return [r.download_resource() for r in resources]


Organism = Literal["human", "mouse"]
Subset = Literal["train", "valid", "test"]


class BasenjiDatasetPreprocessor:
    """
    Downloads Basenji2 dataset in original TFRecord format
    (https://github.com/calico/basenji/tree/master/manuscripts/cross2020),
    converts it to WebDataset format and reorganizes metadata

    Constructor config required fields:
     - tfdata_path: where original dataset will be downloaded
     - webdataset_path: where preprocessed dataset will be placed
     - compress: if compression should be applied to WebDataset shards
     - bucket_name: GCP bucket name where TFRecords can be pulled from
    """

    def __init__(self, dataset_cfg: OmegaConf):
        self.cfg = dataset_cfg

    def _src_pth(self, organism: Organism):
        return os.path.join(self.cfg.tfdata_path, "data", organism)

    def _dst_pth(self, organism: Organism):
        return os.path.join(self.cfg.webdataset_path, organism)

    def _get_tfrecords_iterator(self, organism: Organism, subset: Subset) -> Iterator[Any]:
        tfrecords = sorted(
            glob.glob(os.path.join(self._src_pth(organism), "tfrecords", f"{subset}-*.tfr")),
            key=lambda x: int(x.split("-")[-1].split(".")[0]),
        )
        for tfrecord_path in tfrecords:
            # store in float16 as original data is stored in float16
            # casting to 32bit precision should be done through dataloaders
            tfrecords_dataset = TFRecordDataset(
                tfrecord_path, index_path=None, description={"target": "float16"}, compression_type="zlib"
            )
            yield from tfrecords_dataset

    def _write_single_split(self, organism: Organism, subset: Subset):
        metadata = json.load(open(os.path.join(self._src_pth(organism), "statistics.json")))
        os.makedirs(self._dst_pth(organism), exist_ok=True)
        wd_subset_pattern = os.path.join(self._dst_pth(organism), f"{subset}-%04d.tar")
        tfrecords_iterator = self._get_tfrecords_iterator(organism, subset)

        processed_count = 0
        with wds.ShardWriter(pattern=wd_subset_pattern, maxsize=5e8, compress=self.cfg.compress, mode=0o777) as sink:
            for example in tfrecords_iterator:
                sample = {
                    "__key__": str(processed_count),
                    "target.pth": example["target"].reshape(metadata["target_length"], metadata["num_targets"]),
                }
                sink.write(sample)
                processed_count += 1

        if processed_count != metadata[f"{subset}_seqs"]:
            warnings.warn(
                f"Read {processed_count} examples from TFrecords but \
                          manifest says there should be {metadata[f'{subset}_seqs']}. \
                          Your dataset might be incomplete.",
                BytesWarning,
            )

    def _download(self) -> subprocess.CompletedProcess:
        # despite gsutil available via Python API, it does not offer multiprocessing download
        os.makedirs(self.cfg.tfdata_path, exist_ok=True)
        cmd = ["gsutil", "-m", "cp", "-n", "-r", f"gs://{self.cfg.bucket_name}/*", self.cfg.tfdata_path]
        return subprocess.run(cmd)

    def _decompress_atlas(self, organism: Organism):
        from bionemo.data.dna.enformer.basenji_dataset import ATLAS_NAMES

        org_atlas_name = ATLAS_NAMES[organism]
        atlas_src_path = os.path.join(self.cfg.tfdata_path, f"{org_atlas_name}.gz")
        atlas_dst_path = os.path.join(self._dst_pth(organism), org_atlas_name)

        with gzip.open(atlas_src_path, "r") as f_in, open(atlas_dst_path, "wb") as f_out:
            logging.info(f"Decompressing {atlas_src_path} to {atlas_dst_path}")
            shutil.copyfileobj(f_in, f_out)

    def _move_metadata(self, organism: Organism):
        org_dest = self._dst_pth(organism)
        os.makedirs(org_dest, exist_ok=True)
        for file in glob.glob(os.path.join(self._src_pth(organism), "*.*")):
            logging.info(f"Copying {file} to {org_dest}")
            shutil.copy2(file, org_dest)

    def process(self):
        download_result = self._download()
        if download_result.returncode == 0:
            for organism in ["human", "mouse"]:
                self._move_metadata(organism)
                self._decompress_atlas(organism)
                for subset in ["train", "valid", "test"]:
                    self._write_single_split(organism=organism, subset=subset)

        else:
            msg = (
                " Requester pays bucket requires authenthication and a project the transfer fee should be billed to. \
                   Please make sure you these set up. \
                   For more information please visit https://cloud.google.com/storage/docs/using-requester-pays#using "
            )
            raise RuntimeError(msg)
