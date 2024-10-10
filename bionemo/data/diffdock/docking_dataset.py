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


import copy
import glob
import math
import multiprocessing
import os
import pickle
import random
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from multiprocessing import Pool, get_start_method, set_start_method
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform

from bionemo.data.diffdock.process_mols import (
    extract_receptor_structure,
    get_lig_graph_with_matching,
    get_rec_graph,
    parse_pdb_from_path,
    read_molecule,
)
from bionemo.data.diffdock.webdataset_utils import pickles_to_tars
from bionemo.model.molecule.diffdock.utils import so3, torus
from bionemo.model.molecule.diffdock.utils.ddp import get_rank
from bionemo.model.molecule.diffdock.utils.diffusion import modify_conformer, set_time


@dataclass(kw_only=True)
class HeteroGraphDataConfig:
    """Protein-Ligand complex hetero graph data config data class"""

    data_dir: Optional[os.PathLike] = None  # path to the protein-ligand complex structures
    protein_ligand_csv: Optional[os.PathLike] = None  # csv file with complex_names, protein and ligand paths
    # refer to example/molecule/diffdock/conf/embedding_preprocess.yaml
    # for more details, set to None if use cached data
    cache_path: os.PathLike  # path to save cached data
    split_path: os.PathLike = None  # path to the split file.
    esm_embeddings_path: Optional[os.PathLike] = None  # path to the esm embedding results

    all_atoms: bool  # all atom or coarse grained/residue for protein
    limit_complexes: int = 0  # if choose a subset of samples, set 0 to ignore
    max_lig_size: Optional[int] = None  # maximal ligand size, set to None to ignore
    remove_hs: bool  # if remove hydrogen in ligands
    receptor_radius: float  # receptor graph cutoff
    c_alpha_max_neighbors: Optional[int] = None  # C-alpha/residue maximal neighbors, set to None to ignore
    atom_radius: float  # all atom receptor graph cutoff
    atom_max_neighbors: Optional[int] = None  # all atom graph maximal neighbors, set to None to ignore
    matching_popsize: int = 20  # A multiplier for setting the total population size
    # when optimizing the generated conformer for matching
    matching_maxiter: int = 20  # The maximum number of generations over which the entire population is evolved
    # when optimizing the generated conformer for matching
    no_torsion: bool = False  # whether not considering torsion
    matching: Optional[bool] = None  # if use RDKit to generate matching conformers

    generate_conformer_max_iterations: Optional[int] = (
        10  # maximal number of iterations for RDkit to generate conformers.
    )
    # if failed, start with random coordinates. default to 10
    generate_conformer_enforce_chirality: Optional[bool] = False
    # whether keep enforcing chirality if failed with `generate_conformer_max_iterations`` iterations for RDkit to generate conformers.
    # Default to False
    keep_original: Optional[bool] = True  # if keep original ligand positions. Default to True.
    num_conformers: Optional[int] = 1  # number of reference conformers to generate from RDKit. Default to 1.

    num_chunks: Optional[int] = 1  # number of chunks to group the whole data in the split
    seed: Optional[int] = None  # random seed

    min_num_shards: Optional[int] = (
        0  # minimal number of shard tar files to create when using webdataset. set to None to ignore
    )

    @classmethod
    def init_from_hydra_config(cls, data_config: DictConfig):
        """initialize data class from hydra dict config

        Args:
            data_config (DictConfig): Data config from hydra

        Returns:
            HeteroGraphDataConfig: data config in data class HeteroGraphDataConfig
        """
        inputs = {}
        for field in fields(HeteroGraphDataConfig):
            if hasattr(data_config, field.name):
                inputs[field.name] = data_config.get(field.name)
        if "no_torsion" in inputs and ("matching" not in inputs or inputs["matching"] is None):
            inputs["matching"] = not inputs["no_torsion"]
        return HeteroGraphDataConfig(**inputs)


def get_heterograph_path_from_data_config(data_config: HeteroGraphDataConfig) -> os.PathLike:
    """Get the heterograph data cache path for given data config

    Args:
        data_config (HeteroGraphDataConfig): Protein-Ligand complex hetero graph data config data class

    Returns:
        os.PathLike: absolute full cache path for the split in the given data config
    """
    if data_config.all_atoms:
        cache_prefix = "allatoms"
    else:
        cache_prefix = "torsion"
    full_cache_path = os.path.join(
        data_config.cache_path,
        f"{cache_prefix}_limit{data_config.limit_complexes}"
        # f"_INDEX{os.path.splitext(os.path.basename(split_path))[0]}"
        f"_maxLigSize{data_config.max_lig_size}_H{int(not data_config.remove_hs)}"
        f"_recRad{data_config.receptor_radius}_recMax{data_config.c_alpha_max_neighbors}"
        + (
            ""
            if not data_config.all_atoms
            else f"_atomRad{data_config.atom_radius}_atomMax{data_config.atom_max_neighbors}"
        )
        + (
            ""
            if not data_config.matching or data_config.num_conformers == 1
            else f"_confs{data_config.num_conformers}"
        )
        + "_esmEmbeddings",
    )
    return full_cache_path


def read_strings_from_txt(path: os.PathLike) -> List:
    """read lines from a txt file

    Args:
        path (os.PathLike): path to a text file

    Returns:
        List: list of each entry in the text file
    """
    # every line will be one element of the returned list
    return np.genfromtxt(path, dtype=str).tolist()


class NoiseTransform(BaseTransform):
    """Apply forward diffusion on the ligand

    Args:
        t_to_sigma (Callable): Callable to embed time
        no_torsion (bool): if not to perturb ligand torsion degrees
        all_atom (bool): # all atom or coarse grained/residue for protein
    """

    def __init__(self, t_to_sigma: Callable, no_torsion: bool, all_atom: bool):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom

    def __call__(self, data):
        t = np.random.uniform()
        t_tr, t_rot, t_tor = t, t, t
        return self.apply_noise(data, t_tr, t_rot, t_tor)

    def apply_noise(self, data, t_tr, t_rot, t_tor, tr_update=None, rot_update=None, torsion_updates=None):
        if not torch.is_tensor(data["ligand"].pos):
            data["ligand"].pos = random.choice(data["ligand"].pos)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
        set_time(data, t_tr, t_rot, t_tor, 1, self.all_atom, device=None)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = (
            np.random.normal(loc=0.0, scale=tor_sigma, size=data["ligand"].edge_mask.sum())
            if torsion_updates is None
            else torsion_updates
        )
        torsion_updates = None if self.no_torsion else torsion_updates
        modify_conformer(
            data,
            tr_update,
            torch.from_numpy(rot_update).float(),
            None if data["ligand"].edge_mask.sum() == 0 else torsion_updates,
        )

        data.tr_score = -tr_update / tr_sigma**2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data["ligand"].edge_mask.sum()) * tor_sigma
        return data

    def apply_noise_iter(self, source, keep_pos=False):
        for (data,) in source:
            if keep_pos:
                data["ligand"].aligned_pos = deepcopy(data["ligand"].pos)
            yield self.__call__(data)


@contextmanager
def ForkingBehavior(
    *,
    start_method: Literal["spawn", "fork", "forkserver"],
    force: bool,
):
    """Contextmanager to set the method for starting child processes in multiprocessing.
    Refer to https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method

    Args:
        start_method (Literal['spawn', 'fork', 'forkserver']): multiprocessing start method to use in the context
        force (bool): Raises RuntimeError if the start method has already been set and force is not True.
                      If start_method is None and force is True then the start method is set to None.
                      If start_method is None and force is False then the context is set to the default context.
    """
    prev_start_method = get_start_method()
    set_start_method(start_method, force=force)
    try:
        yield
    finally:
        set_start_method(prev_start_method, force=True)


class ProteinLigandDockingDataset(Dataset):
    def __init__(
        self,
        data_config: HeteroGraphDataConfig,
        transform: Optional[Callable] = None,
        num_workers: int = 1,
    ):
        """Protein ligand complex graph dataset

        Args:
            data_config (HeteroGraphDataConfig): Protein-Ligand complex hetero graph data config data class, refer to HeteroGraphDataConfig for more details
            transform (Callable, optional): transformation applied to the data
            num_workers (int, optional): number of workers to do data preprocessing. Defaults to 1.
        """
        super(ProteinLigandDockingDataset, self).__init__()
        self.transform = transform

        self.data_config = data_config
        self.split_path = self.data_config.split_path
        self.num_workers = num_workers

        self.full_cache_path = get_heterograph_path_from_data_config(self.data_config)
        self.split_cache_path = self.full_cache_path + (
            f"_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}" if self.split_path is not None else ""
        )

        if not (os.path.exists(self.split_cache_path) and os.listdir(self.split_cache_path)):
            logging.info(
                f"Complex graphs split webdataset tar files do not exist yet: {self.split_cache_path}. "
                "Use load_complex_graphs() to build"
            )

        self.webdataset_urls: Optional[List] = None
        self.webdataset_fname_suffix_in_tar = "heterodata.pyd"

    def build_complex_graphs(self):
        local_rank = get_rank()
        if local_rank == 0:
            os.makedirs(self.full_cache_path, exist_ok=True)
        self.preprocessing()

    def load_complex_graphs(self):
        if not (os.path.exists(self.split_cache_path) and os.listdir(self.split_cache_path)):
            if os.path.exists(self.full_cache_path) and os.listdir(self.full_cache_path):
                os.makedirs(self.split_cache_path, exist_ok=True)
                complex_names = set(read_strings_from_txt(self.split_path))
                pickles_to_tars(
                    self.full_cache_path,
                    "HeteroData.pyd",
                    complex_names,
                    self.split_cache_path,
                    "heterographs",
                    lambda complex_graph: {
                        "__key__": complex_graph.name.replace(".", "-"),
                        self.webdataset_fname_suffix_in_tar: pickle.dumps(complex_graph),
                    },
                    self.data_config.min_num_shards,
                )
            else:
                raise RuntimeError(
                    f"Can not load processed complex graph pickle files from {self.full_cache_path}, "
                    f"which are required to create WebDataset tar files. "
                    f"Use build_complex_graphs() to build."
                )

        self.webdataset_urls = glob.glob(os.path.join(self.split_cache_path, "heterographs-*.tar"))
        if len(self.webdataset_urls) == 0:
            raise RuntimeError(f"{self.split_cache_path} is empty")

    @lru_cache(maxsize=None)
    def len(self):
        return len(read_strings_from_txt(self.split_path))

    def get(self, idx):
        raise NotImplementedError("Using webdataset as backend which does not support indexing")

    def preprocessing(self):
        with ForkingBehavior(start_method="spawn", force=True):
            logging.info(
                f"[preprocessing] processing complexes from [{self.data_config.data_dir}] and saving them to [{self.full_cache_path}]"
            )
            logging.info(f"[preprocessing] reading complexes from {self.data_config.data_dir}")

            # skip the preprocessed complexes saved in the folder.
            processed_names = {
                filename[: -len(".HeteroData.pyd")]
                for filename in os.listdir(self.full_cache_path)
                if filename.endswith(".HeteroData.pyd")
                and os.path.getsize(os.path.join(self.full_cache_path, filename)) > 0
            }
            num_processed_names = len(processed_names)
            if num_processed_names > 0:
                logging.info(
                    f"{num_processed_names} complexes have been processed in {self.full_cache_path}, skipping them"
                )

            split_complex_names = set(read_strings_from_txt(self.split_path))
            complex_names = split_complex_names - processed_names  # complexes for preprocessing in the split

            if (
                os.path.isfile(self.data_config.protein_ligand_csv)
                and os.stat(self.data_config.protein_ligand_csv).st_size > 0
            ):
                df = pd.read_csv(self.data_config.protein_ligand_csv)
            else:
                logging.warning(
                    f"The protein-ligand complex csv file does not exist : {self.data_config.protein_ligand_csv}. skipping"
                )
                return
            if len(df) == 0:
                logging.warning(
                    f"The protein-ligand complex csv file in empty : {self.data_config.protein_ligand_csv}. skipping"
                )
                return

            complexes_all = df[df["complex_name"].str.slice().isin(complex_names)][
                ["complex_name", "protein_path", "ligand_paths"]
            ].values.tolist()

            if self.data_config.limit_complexes is not None and self.data_config.limit_complexes != 0:
                complexes_all = complexes_all[
                    : max(0, self.data_config.limit_complexes - len(processed_names & split_complex_names))
                ]

            if len(complexes_all) == 0:
                logging.info(
                    f"All complexes have been processed in {self.split_path} and saved in {self.full_cache_path}, skipping"
                )
                return
            complex_names_all = list(zip(*complexes_all))[0]

            num_cores = multiprocessing.cpu_count()
            if self.num_workers < num_cores:
                logging.info(f"num_workers < num_cores: {self.num_workers} < {num_cores}")

            chunks = self.data_config.num_chunks
            chunk_size = math.ceil(len(complexes_all) / chunks)
            complex_chunks = [complexes_all[chunk_size * i : chunk_size * (i + 1)] for i in range(chunks)]

            with Pool(self.num_workers) as p:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                logging.info(f"Computing for {len(complex_names_all)} complexes...")
                total = 0
                for lig_cnt, _ in map_fn(self.get_complex, complex_chunks):
                    total += lig_cnt
                logging.info(f"Total processed complexes: {total}")

    def get_complex(self, complexes: List) -> Tuple[int, int]:
        total_complexes = 0
        total_ligands = 0

        for i in range(len(complexes)):
            name, protein_path, ligand_paths = complexes[i]

            lm_embedding_chain = [
                torch.load(file)
                for file in sorted(glob.glob(os.path.join(self.data_config.esm_embeddings_path, f"{name}_chain_*.pt")))
            ]
            if len(lm_embedding_chain) == 0:
                logging.warning(
                    f"ESM embedding is missing for {name} in folder {self.data_config.esm_embeddings_path}, skipping"
                )
                continue

            try:
                rec_model = parse_pdb_from_path(os.path.join(self.data_config.data_dir, protein_path))
            except Exception as e:
                logging.error(f"Skipping {name} because of the error:")
                logging.error(e)
                continue

            lig = None
            for ligand_path in ligand_paths.split(","):
                try:
                    lig = read_molecule(
                        os.path.join(self.data_config.data_dir, ligand_path), remove_hs=False, sanitize=True
                    )
                    break
                except Exception as e:
                    logging.error(f"Skipping ligand file {ligand_path} because of the error:")
                    logging.error(e)
                    continue
            if lig is None:
                logging.warning(f"Fail to read ligand molecule {name} in {ligand_paths}, skipping")
                continue

            if self.data_config.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.data_config.max_lig_size:
                logging.warning(
                    f"Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.data_config.max_lig_size}. Not including {name} in preprocessed data."
                )
                continue
            complex_graph = HeteroData()
            complex_graph["name"] = name
            try:
                get_lig_graph_with_matching(
                    lig,
                    complex_graph,
                    self.data_config.matching_popsize,
                    self.data_config.matching_maxiter,
                    self.data_config.matching,
                    self.data_config.keep_original,
                    self.data_config.num_conformers,
                    remove_hs=self.data_config.remove_hs,
                    seed=self.data_config.seed,
                    generate_conformer_max_iterations=self.data_config.generate_conformer_max_iterations,
                    generate_conformer_enforce_chirality=self.data_config.generate_conformer_enforce_chirality,
                )
                rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(
                    copy.deepcopy(rec_model), lig, lm_embedding_chains=lm_embedding_chain
                )
                if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                    logging.warning(
                        f"LM embeddings for complex {name} did not have the right length for the protein {len(c_alpha_coords)} != {len(lm_embeddings)}. Skipping {name}."
                    )
                    continue

                get_rec_graph(
                    rec,
                    rec_coords,
                    c_alpha_coords,
                    n_coords,
                    c_coords,
                    complex_graph,
                    rec_radius=self.data_config.receptor_radius,
                    c_alpha_max_neighbors=self.data_config.c_alpha_max_neighbors,
                    all_atoms=self.data_config.all_atoms,
                    atom_radius=self.data_config.atom_radius,
                    atom_max_neighbors=self.data_config.atom_max_neighbors,
                    remove_hs=self.data_config.remove_hs,
                    lm_embeddings=lm_embeddings,
                )

            except Exception as e:
                logging.error(f"Skipping {name} because of the error: {e}")
                continue

            protein_center = torch.mean(complex_graph["receptor"].pos, dim=0, keepdim=True)
            complex_graph["receptor"].pos -= protein_center
            if self.data_config.all_atoms:
                complex_graph["atom"].pos -= protein_center

            if (not self.data_config.matching) or self.data_config.num_conformers == 1:
                complex_graph["ligand"].pos -= protein_center
            else:
                for p in complex_graph["ligand"].pos:
                    p -= protein_center

            complex_graph.original_center = protein_center
            complex_graph.mol = lig
            with open(os.path.join(self.full_cache_path, f"{name}.HeteroData.pyd"), "wb") as f:
                pickle.dump(complex_graph, f)

            total_complexes += 1
            total_ligands += 1

        return total_ligands, total_complexes


def read_mol(protein_dir, name, remove_hs=False, ligand_file_name_suffix="_ligand.sdf"):
    filename = name + ligand_file_name_suffix
    lig = read_molecule(os.path.join(protein_dir, name, filename), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(
            os.path.join(protein_dir, name, filename[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True
        )
    return lig


def read_mols(protein_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(protein_dir, name)):
        if file.endswith(".sdf") and "rdkit" not in file:
            lig = read_molecule(os.path.join(protein_dir, name, file), remove_hs=remove_hs, sanitize=True)
            # read mol2 file if sdf file cannot be sanitized
            if lig is None and os.path.exists(os.path.join(protein_dir, name, file[:-4] + ".mol2")):
                logging.warning(
                    "Using the .sdf file failed. We found a .mol2 file instead and are trying to use that."
                )
                lig = read_molecule(
                    os.path.join(protein_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True
                )
            if lig is not None:
                ligs.append(lig)
    return ligs


class DataSplit(Enum):
    train = "train"
    validation = "validation"
    test = "test"


def diffdock_build_dataset(
    data_config: DictConfig,
    split_config: DictConfig,
    t_to_sigma: Callable,
    _num_conformers: bool = True,
    mode: DataSplit = DataSplit("train"),
) -> ProteinLigandDockingDataset:
    """Build heterograph dataset for protein-ligand complexes for DiffDock Score model training

    Args:
        data_config (DictConfig): hydra config cfg.data section
        split_config (DictConfig): hydra config cfg.model.[train_ds, validation_ds, test_ds] section
        t_to_sigma (Callable): function to embed diffusion time
        _num_conformers (bool, optional): whether to generate multiple conformers from RDKit rather than default 1 conformer. Defaults to True.
        mode (DataSplit, optional): mode of the dataset, could be DataSplit("train"), DataSplit("validation") or DataSplit("test"). Defaults to DataSplit("train").

    Returns:
        ProteinLigandDockingDataset: Protein Ligand Docking Dataset
    """

    if t_to_sigma is not None:
        transform = NoiseTransform(
            t_to_sigma=t_to_sigma, no_torsion=data_config.no_torsion, all_atom=data_config.all_atoms
        )
    else:
        transform = None

    config = HeteroGraphDataConfig.init_from_hydra_config(data_config)
    config.split_path = split_config.split_val if "val" in mode.name else split_config.get(f"split_{mode.name}")
    config.num_conformers = split_config.num_conformers if _num_conformers else 1
    config.min_num_shards = split_config.get("min_num_shards")

    return ProteinLigandDockingDataset(
        data_config=config,
        transform=transform,
        num_workers=split_config.num_workers,
    )
