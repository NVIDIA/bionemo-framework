# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal
from zipfile import ZipFile

import numpy as np
import scanpy
from nemo.utils import logging
from scanpy import AnnData

from bionemo.data.preprocess import ResourcePreprocessor
from bionemo.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.utils.remote import RemoteResource


@dataclass
class GeneformerResourcePreprocessor(ResourcePreprocessor):
    """ResourcePreprocessor for the Geneformer model. Downloads the gene_name_id_dict.pkl and gene_median_dictionary.pkl files."""

    dest_directory: str = "geneformer"

    def get_remote_resources(self) -> List[RemoteResource]:
        url_fn = {
            "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl?download=true": "gene_name_id_dict.pkl",
            "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl?download=true": "gene_median_dictionary.pkl",
        }

        resources = []
        for url, filename in url_fn.items():
            resource = RemoteResource(
                dest_directory=self.dest_directory,
                dest_filename=filename,
                root_directory=self.root_directory,
                checksum=None,
                url=url,
            )
            resources.append(resource)
        return resources

    def prepare_resource(self, resource: RemoteResource) -> str:
        """Logs and downloads the passed resource.

        resource: RemoteResource - Resource to be prepared.

        Returns - the absolute destination path for the downloaded resource
        """
        return resource.download_resource()

    def prepare(self):
        return [self.prepare_resource(resource) for resource in self.get_remote_resources()]


class GeneformerPreprocess(ABC):
    def __init__(self, download_directory: Path, medians_file_path: Path, tokenizer_vocab_path: Path):
        """Downloads HGNC symbols

        preproc_dir (str): Directory to store the reference preproc in
        tokenizer_vocab_path (str): Filepath to store the tokenizer vocab
        dataset_conf (OmegaConf): has 'train', 'val', 'test' keys containing
            the names of preprocessed train/val/test files to use for training.
        """
        self.download_directory = download_directory
        self.medians_file_path = medians_file_path
        self.tokenizer_vocab_path = tokenizer_vocab_path
        self._validate_tokenizer_args(
            self.tokenizer_vocab_path,
        )

    @abstractmethod
    def get_median_filenames(self):
        raise NotImplementedError("Subclasses must implement get_median_filenames methods")

    def build_and_save_tokenizer(self, median_dict, gene_to_ens, vocab_output_name):
        """Builds the GeneTokenizer using the median dictionary
        then serializes and saves the dictionary to disk.
        """
        tokenizer = GeneTokenizer.from_medians_and_genes_dicts(median_dict, gene_to_ens)
        tokenizer.save_vocab(vocab_output_name)
        return tokenizer

    def _validate_tokenizer_args(self, vocab_output_name):
        vocab_exists = os.path.exists(vocab_output_name)
        if vocab_exists:
            logging.warning(f"Tokenizer vocab file: {vocab_output_name} already exists. Overwriting...")

    def preprocess(self) -> dict[Literal["tokenizer", "median_dict"], Any]:
        """Preprocesses for the Geneformer model"""
        gene_name_dict_fn, gene_median_dict_fn = self.get_median_filenames()
        # Load artifacts
        with open(gene_name_dict_fn, "rb") as fd:
            gene_ens = pickle.load(fd)

        with open(gene_median_dict_fn, "rb") as fd:
            median_dict = pickle.load(fd)

        # Save converted artifacts to JSON to prevent pickle issues.
        medians_dir = os.path.dirname(self.medians_file_path)
        if not os.path.exists(medians_dir):
            os.makedirs(medians_dir, exist_ok=True)  # ensure the dir exists but be ok with race conditions.
        with open(self.medians_file_path, "w") as fp:
            json.dump(median_dict, fp)

        if self.tokenizer_vocab_path is not None:
            tokenizer = self.build_and_save_tokenizer(
                median_dict,
                gene_ens,
                self.tokenizer_vocab_path,
            )
        else:
            tokenizer = None

        return {"tokenizer": tokenizer, "median_dict": median_dict}


class GeneformerPreprocessPaperMedians(GeneformerPreprocess):
    def get_median_filenames(self):
        gene_name_dict_fn, gene_median_dict_fn = GeneformerResourcePreprocessor(
            dest_directory=self.download_directory,
        ).prepare()
        return gene_name_dict_fn, gene_median_dict_fn


class AdamsonResources(ResourcePreprocessor):
    """Downloads and unpacks the resources for the Adamson et al 2020 dataset."""

    dest_directory: str = "adamson"

    def get_remote_resources(self) -> List[RemoteResource]:
        data_url = {
            "adamson.zip": "https://dataverse.harvard.edu/api/access/datafile/6154417",
            "gene2go_all.pkl": "https://dataverse.harvard.edu/api/access/datafile/6153417",
            "all_pert_genes": "https://dataverse.harvard.edu/api/access/datafile/6934320",
        }

        resources = [
            RemoteResource(
                dest_directory=self.dest_directory,
                dest_filename="gene2go_all.pkl",
                root_directory=self.root_directory,
                checksum="77c9af0c61c30ea4d7a85680f4d122dc",
                url=data_url["gene2go_all.pkl"],
            ),
            RemoteResource(
                dest_directory=self.dest_directory,
                dest_filename="adamson.zip",
                root_directory=self.root_directory,
                checksum="0bde631bae60ee8c105991ff0e0d4a20",
                url=data_url["adamson.zip"],
            ),
            RemoteResource(
                dest_directory=self.dest_directory,
                dest_filename="all_pert_genes.pkl",
                root_directory=self.root_directory,
                checksum=None,
                url=data_url["all_pert_genes"],
            ),
        ]
        return resources

    def prepare(self) -> List[str]:
        """
        Downloads the pickle file and zip file associated with the adamson dataset and
        unzips the zip file.

        Returns (in this order): [gene2go_all.pkl, go.csv, perturbed_processed.hdf5, all_pert_genes.pkl]
        """
        artifacts = []
        for resource in self.get_remote_resources():
            if resource.dest_filename == "adamson.zip":
                with ZipFile(resource.download_resource(), "r") as zip_ref:
                    zip_ref.extractall(self.dest_directory)
                dest = resource.fully_qualified_dest_folder
                artifacts.extend([f"{dest}/adamson/go.csv", f"{dest}/adamson/perturb_processed.h5ad"])
            else:
                artifacts.append(resource.download_resource())
        return artifacts  # gene2go_all.pkl, go.csv, perturbed_processed.hdf5, all_pert_genes.pkl

    def prepare_annotated(self) -> dict:
        """same method as `prepare` but annotates the outputs in a dictionary."""
        gene2go, go, perturbed, pert_genes = self.prepare()
        return {"gene2go_pkl": gene2go, "go_csv": go, "perturbed_h5ad": perturbed, "pert_genes_pkl": pert_genes}


def preprocess_adamson(
    adamson_perturbed_processed_fn: str,
    gene2go_pkl_fn: str,
    all_pert_genes_pkl_fn: str,
    dest_preprocessed_anndata_fn: str = "perturb_preprocessed.hdf5",
    dest_target_gep_fn: str = "target_gep.npy",
):
    """
    Preprocesses the Adamson perturbed data. This is done in two steps:
        - Opens up the PERTURB-seq ann data object,
        - samples one control sample for each PERTURBed sample
        - writes both to disk.
        - Result is a 3 x N set of of, control gene expression, pertubation targets, target (perturbed) gene expression

    Args:
        adamson_perturbed_processed_fn (str): File path to the processed Adamson perturbed data.
        gene2go_pkl_fn (str): File path to the gene-to-GO mapping pickle file.
        all_pert_genes_pkl_fn (str): File path to the pickle file containing all perturbed genes.
        dest_preprocessed_anndata_fn (str): Destination file name for the preprocessed AnnData object.
        dest_target_gep_fn (str): Destination file name for the target gene expression profile.

    Returns:
        tuple: A tuple containing the file path to the preprocessed AnnData object and the file path to the target gene expression profile.
    """
    data = scanpy.read_h5ad(adamson_perturbed_processed_fn)

    with open(gene2go_pkl_fn, "rb") as f:
        gene2go = pickle.load(f)

    with open(all_pert_genes_pkl_fn, "rb") as f:
        pert_names = pickle.load(f)

    # TODO: lift process_dataset into this namespace and out of the Adamson namespace.
    preprocessed_anndata, target_gep = process_adamson_dataset(data, gene2go, pert_names)

    preprocessed_anndata.write(dest_preprocessed_anndata_fn)
    with open(dest_target_gep_fn, "wb") as f:
        np.save(f, target_gep)
    return dest_preprocessed_anndata_fn, dest_target_gep_fn


def process_adamson_dataset(
    data: AnnData,
    gene2go: dict[str, str],
    pert_genes: dict[str, str],
) -> tuple[AnnData, np.ndarray]:
    """
    Preprocesses the dataset by performing the following steps:
    1. Filters genes (columns) from the dataset if they are not included in our collection of go terms.
    2. Normalizes the data.
    3. Creates and returns the newly filtered AnnData object.


    Args:
        data (AnnData): AnnData object containing a column in `obs` called 'items', which contains a '+' joined list of pertubation gene targets, or 'ctrl' for a control.
        gene2go (dict): Dictionary mapping gene names to GO terms.
        pert_genes (dict): Dictionary containing perturbation genes.

    Returns:
        AnnData: AnnData where each row is a randomly sampled control (unperturbed) gene expression profile, and the obs column contains a set of pertubations associated with the sample.
        np.ndarray: The target gene expression profile of the AnnData object. Each row contains the 'original' (perturbed) gene expression profile corresponding to same row in the AnnData object.
    """

    gene2go = {pg: gene2go[pg] for pg in pert_genes if pg in gene2go}

    conds = data.obs["condition"]
    pert_names = set(gene2go.keys())

    genes_in_go = np.unique([c for c in conds if _filter_pert_in_go(c, pert_names)])
    filt_go = data.obs[data.obs.condition.isin(genes_in_go)].index
    data = data[filt_go, :]
    scanpy.pp.normalize_total(data, target_sum=10, inplace=True)

    # - create new AnnData with targets (controls)
    ctrl_samples = data.X[data.obs.condition == "ctrl"]

    # Target gene expression profile
    target_gep = data.X.copy().toarray()

    input = ctrl_samples[np.random.choice(np.arange(ctrl_samples.shape[0]), size=len(data), replace=True), :].copy()

    # NOTE: `data` is a randomly sampled control gene expression profile with the original set of marked pertubations
    # NOTE: target_gep is the original (unmutated, but filtered) gene expression profile.
    data = AnnData(X=input.toarray(), obs=data.obs, var=data.var)
    # NOTE: there is likely a cleaner way to return these, three arrays should be enough, but we pack two into one array (data), and leave the other separate
    #           adamson is a SINGLE example of PERTURB-seq, we should in the future define what these formats look like.
    return data, target_gep


def _filter_pert_in_go(condition: str, pert_names: set[str]) -> bool:
    """Filters perturbations in gene ontology (GO) based on the condition and perturbation names.

    Args:
        condition (str): The condition to filter perturbations. Possible values are "ctrl" or a combination of perturbations separated by '+'.
        pert_names (Set[str]): The set of perturbation names.

    Returns:
        bool: True if the perturbations is in the filter, or the sample is a ctrl, False otherwise.
    """
    if condition == "ctrl":
        return True
    else:
        splits = condition.split("+")
        num_perts = (splits[0] in pert_names) + (splits[1] in pert_names)
        num_ctrl = (splits[0] == "ctrl") + (splits[1] == "ctrl")

        if num_perts + num_ctrl == 2:
            return True
        else:
            return False
