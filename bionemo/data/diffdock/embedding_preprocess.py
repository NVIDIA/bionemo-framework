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


import multiprocessing
import os
from multiprocessing import Pool
from threading import Lock
from typing import List, Tuple, Union

import pandas as pd
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from esm import FastaBatchedDataset, pretrained
from nemo.utils import logging
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class ThreadSafeList:
    # constructor
    def __init__(self):
        # initialize the list
        self._list = []
        # initialize the lock
        self._lock = Lock()

    # add a value to the list
    def append(self, value):
        # acquire the lock
        with self._lock:
            # append the value
            self._list.append(value)

    def extend(self, value):
        # acquire the lock
        with self._lock:
            self._list.extend(value)

    # remove and return the last value from the list
    def pop(self):
        # acquire the lock
        with self._lock:
            # pop a value from the list
            return self._list.pop()

    # read a value from the list at an index
    def get(self, index):
        # acquire the lock
        with self._lock:
            # read a value at the index
            return self._list[index]

    # return the number of items in the list
    def length(self):
        # acquire the lock
        with self._lock:
            return len(self._list)


amino_acid_seq_three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SER": "S",
    "SEC": "U",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "XAA": "X",
    "XLE": "J",
}


def get_sequences_from_pdbfile(protein_data: Union[List, Tuple], to_fasta: bool = True) -> Union[List, str]:
    """Extract protein sequence from protein pdb file

    Args:
        protein_data (Union[List, Tuple]): list/tuple with 2 elements: (complex name, path to the protein pdb file).
            complex name is used as key for sequence. Required if to_fasta is True.
        to_fasta (bool, optional): if save to fasta file. Defaults to True.

    Returns:
        Union[List, str]: List of Bio.SeqRecord.SeqRecord if to_fasta is True, else string of sequence
    """

    complex_name, protein_path = protein_data

    if to_fasta:
        records = []
    else:
        sequence = None
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure("random_id", protein_path)
    structure = structure[0]
    for i, chain in enumerate(structure):
        seq = ""
        for _, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
            if c_alpha is not None and n is not None and c is not None:
                try:
                    seq += amino_acid_seq_three_to_one[residue.get_resname()]
                except KeyError:
                    seq += "-"
                    logging.warning(
                        f"encountered unknown AA: {residue.get_resname()} in the complex {protein_path}. Replacing it with a dash - ."
                    )
        if to_fasta:
            index = f"{complex_name}_chain_{i}"
            record = SeqRecord(Seq(seq), str(index))
            record.description = ""
            records.append(record)
        else:
            if sequence is None:
                sequence = seq
            else:
                sequence += ":" + seq
    if to_fasta:
        return records
    else:
        return sequence


def compute_ESM_embeddings(model, alphabet, dataset, store_path=None):
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    truncation_seq_length = toks_per_batch * 2

    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    if store_path is None:
        embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                value = representations[33][i, 1 : truncate_len + 1].clone()
                if store_path is None:
                    embeddings[label] = value
                else:
                    torch.save(value, os.path.join(store_path, f"{label}.pt"))

    if store_path is None:
        return embeddings


class DataPreprocess(object):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.three_to_one = amino_acid_seq_three_to_one
        self.num_cores = cfg.get("num_workers")
        self.esm_embeddings_path = cfg.esm_embeddings_path
        os.makedirs(self.esm_embeddings_path, exist_ok=True)

        if self.num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
            logging.warning(
                f"using all available {self.num_cores} cpu cores for multiprocessing the pdb to fasta conversion"
            )

    def get_pdb_files(self) -> List[str]:
        """load all the pdb files specified in the embedding_preprocess.yaml

        Returns:
            List[str]: path to individual *.pdb files
        """
        file_paths = []
        df = pd.read_csv(self.cfg.protein_ligand_csv)

        for complex_name, protein_path in df[["complex_name", "protein_path"]].values.tolist():
            file_paths.append((complex_name, os.path.join(self.cfg.protein_data_dir, protein_path)))

        return file_paths

    def pdb2fasta(self) -> None:
        """Convert pdb files to fasta files, will be multithreaded if possible"""
        file_paths = self.get_pdb_files()
        record_pieces = ThreadSafeList()

        with Pool(self.num_cores) as p:
            map_fn = p.imap_unordered if self.num_cores > 1 else map
            for records in map_fn(get_sequences_from_pdbfile, file_paths):
                record_pieces.extend(records)

        if self.cfg.output_fasta_file is not None:
            logging.info(f"writing records to file at {self.cfg.output_fasta_file}")
            SeqIO.write(record_pieces._list, self.cfg.output_fasta_file, "fasta")

    def fasta2esm(self) -> None:
        """convert fasta files to esm files, this operation runs on GPU"""
        model_location = "esm2_t33_650M_UR50D"
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        dataset = FastaBatchedDataset.from_file(self.cfg.output_fasta_file)
        compute_ESM_embeddings(model, alphabet, dataset, self.esm_embeddings_path)
        del model
        torch.cuda.empty_cache()


def prep_embedding(cfg: DictConfig) -> None:
    logging.info("\n\n********* Preprocess protein structures ******")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    dpp = DataPreprocess(cfg)

    logging.info("Converting pdb files to fasta")
    dpp.pdb2fasta()

    logging.info("Converting fasta files to ESM2")
    dpp.fasta2esm()

    logging.info("ESM preparation complete.")
