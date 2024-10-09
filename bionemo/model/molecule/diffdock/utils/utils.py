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


import os
import signal
import subprocess
import warnings
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
import yaml
from nemo.utils import logging
from rdkit import Chem
from rdkit.Chem import MolToPDBFile, RemoveHs
from spyrmsd import molecule, rmsd


def get_obrmsd(mol1_path, mol2_path, cache_name=None):
    cache_name = datetime.now().strftime("date%d-%m_time%H-%M-%S.%f") if cache_name is None else cache_name
    os.makedirs(".openbabel_cache", exist_ok=True)
    if not isinstance(mol1_path, str):
        MolToPDBFile(mol1_path, ".openbabel_cache/obrmsd_mol1_cache.pdb")
        mol1_path = ".openbabel_cache/obrmsd_mol1_cache.pdb"
    if not isinstance(mol2_path, str):
        MolToPDBFile(mol2_path, ".openbabel_cache/obrmsd_mol2_cache.pdb")
        mol2_path = ".openbabel_cache/obrmsd_mol2_cache.pdb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return_code = subprocess.run(
            f"obrms {mol1_path} {mol2_path} > .openbabel_cache/obrmsd_{cache_name}.rmsd", shell=True
        )
        print(return_code)
    obrms_output = read_strings_from_txt(f".openbabel_cache/obrmsd_{cache_name}.rmsd")
    rmsds = [line.split(" ")[-1] for line in obrms_output]
    return np.array(rmsds, dtype=np.float)


def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def save_yaml_file(path, content):
    assert isinstance(path, str), f"path must be a string, got {path} which is a {type(path)}"
    content = yaml.dump(data=content)
    if "/" in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content)


def get_optimizer_and_scheduler(cfg, model, scheduler_mode="min"):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.trainer.lr,
        weight_decay=cfg.model.optim.weight_decay,
    )

    if cfg.model.optim.sched.name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=0.7,
            patience=cfg.model.optim.sched.patience,
            min_lr=cfg.trainer.lr / 100,
        )
    else:
        logging.debug("No scheduler")
        scheduler = None

    return optimizer, scheduler


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
        )
        return RMSD


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def estimate_memory_usage(data, num_cross_edges, estimate_memory_usage, bias=True):
    # bias is from the memory of model, so when estimate the upper bound for size aware batch sampler, we don't need this
    total_memory = (
        estimate_memory_usage.coeff_ligand_num_nodes * data["ligand"].num_nodes
        + estimate_memory_usage.coeff_ligand_num_edges * data["ligand", "ligand"].num_edges
        + estimate_memory_usage.coeff_receptor_num_nodes * data["receptor"].num_nodes
        + estimate_memory_usage.coeff_receptor_num_edges * data["receptor", "receptor"].num_edges
        + estimate_memory_usage.coeff_num_cross_edges * num_cross_edges
    )
    if bias:
        return total_memory + estimate_memory_usage.bias
    else:
        return total_memory
