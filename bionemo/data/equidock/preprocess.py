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


#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pickle
import random
import warnings

import pandas as pd
from biopandas.pdb import PandasPdb
from dgl import save_graphs
from joblib import Parallel, cpu_count, delayed
from omegaconf import OmegaConf

from bionemo.data.equidock.protein_utils import (
    preprocess_unbound_bound,
    preprocess_unbound_bound_dips,
    protein_to_graph_unbound_bound,
)


warnings.filterwarnings("ignore", category=FutureWarning)


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, **kwargs):
    """
    Extends dgllife pmap function.

    Parallel map using joblib.

    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.

    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(pickleable_fn)(*d, **kwargs) for i, d in enumerate(data)
    )

    return results


def get_residues_db5(pdb_filename):
    df = PandasPdb().read_pdb(pdb_filename).df["ATOM"]
    df.rename(
        columns={
            "chain_id": "chain",
            "residue_number": "residue",
            "residue_name": "resname",
            "x_coord": "x",
            "y_coord": "y",
            "z_coord": "z",
            "element_symbol": "element",
        },
        inplace=True,
    )
    # Not the same as sequence order !
    residues = list(df.groupby(["chain", "residue", "resname"]))
    return residues


def get_residues_DIPS(dill_filename):
    x = pd.read_pickle(dill_filename)
    df0 = x.df0
    df0.rename(
        columns={
            "chain_id": "chain",
            "residue_number": "residue",
            "residue_name": "resname",
            "x_coord": "x",
            "y_coord": "y",
            "z_coord": "z",
            "element_symbol": "element",
        },
        inplace=True,
    )
    # Not the same as sequence order !
    residues0 = list(df0.groupby(["chain", "residue", "resname"]))
    df1 = x.df1
    df1.rename(
        columns={
            "chain_id": "chain",
            "residue_number": "residue",
            "residue_name": "resname",
            "x_coord": "x",
            "y_coord": "y",
            "z_coord": "z",
            "element_symbol": "element",
        },
        inplace=True,
    )
    # Not the same as sequence order !
    residues1 = list(df1.groupby(["chain", "residue", "resname"]))
    return residues0, residues1


def preprocess(cfg: OmegaConf):
    raw_data_path = cfg.raw_data_path
    split_files_path = cfg.split_files_path
    reload_mode = cfg.reload_mode  # ['train', 'val', 'test']

    cache_path = os.path.join(
        cfg.cache_path,
        cfg.data_name
        + "_"
        + cfg.graph_nodes
        + "_maxneighbor_"
        + str(cfg.graph_max_neighbor)
        + "_cutoff_"
        + str(cfg.graph_cutoff)
        + "_pocketCut_"
        + str(cfg.pocket_cutoff)
        + "/cv_"
        + str(cfg.split),
    )

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    if raw_data_path is None or split_files_path is None:
        raise ValueError(f"raw_data_path ({raw_data_path}) or split_files_path {split_files_path} is None!")

    frac_str = ""

    if reload_mode == "train" and cfg.data_name == "dips":
        frac_str = "frac_" + str(cfg.data_fraction) + "_"

    label_filename = os.path.join(cache_path, "label_" + frac_str + reload_mode + ".pkl")

    if os.path.exists(label_filename):
        print(f"Not recreating {label_filename}, because data already exists! \n\n\n")
        print(f"Delete {label_filename} and run again! \n\n\n")
        return

    ligand_graph_filename = os.path.join(cache_path, "ligand_graph_" + frac_str + reload_mode + ".bin")

    receptor_graph_filename = os.path.join(cache_path, "receptor_graph_" + frac_str + reload_mode + ".bin")

    if cfg.data_name == "db5":
        if cfg.data_fraction is not None:
            raise ValueError(f"DB5 requires data_fraction of None, but it is {cfg.data_fraction}")

        onlyfiles = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]

        code_set = {file.split("_")[0] for file in onlyfiles}
        split_code_set = set()

        with open(os.path.join(split_files_path, reload_mode + ".txt"), "r") as f:
            for line in f.readlines():
                split_code_set.add(line.rstrip())

        code_set = code_set & split_code_set
        code_list = list(code_set)

        bound_ligand_residues_list = [
            get_residues_db5(os.path.join(raw_data_path, code + "_l_b.pdb")) for code in code_list
        ]
        bound_receptor_residues_list = [
            get_residues_db5(os.path.join(raw_data_path, code + "_r_b.pdb")) for code in code_list
        ]

        input_residues_lists = [
            (bound_ligand_residues_list[i], bound_receptor_residues_list[i])
            for i in range(len(bound_ligand_residues_list))
        ]
        print("Start preprocess_unbound_bound")
        preprocess_result = pmap_multi(
            preprocess_unbound_bound,
            input_residues_lists,
            n_jobs=cfg.n_jobs,
            graph_nodes=cfg.graph_nodes,
            pos_cutoff=cfg.pocket_cutoff,
            inference=False,
        )

        print("Done preprocess_unbound_bound\n\n")

    elif cfg.data_name == "dips":
        if reload_mode != "train":
            data_fraction = 1.0
        else:
            data_fraction = cfg.data_fraction

        dill_filenames_list = []
        with open(os.path.join(split_files_path, "pairs-postprocessed-" + reload_mode + ".txt"), "r") as f:
            for line in f.readlines():
                dill_filenames_list.append(line.rstrip())

        random.shuffle(dill_filenames_list)
        dill_filenames_list = dill_filenames_list[: int(data_fraction * len(dill_filenames_list))]

        print("Num of pairs in ", reload_mode, " = ", len(dill_filenames_list))

        def get_raw_path(x):
            return (os.path.join(raw_data_path, x),)

        all_paths = list(map(get_raw_path, dill_filenames_list))

        print("Start preprocess_unbound_bound")
        preprocess_result = pmap_multi(
            preprocess_unbound_bound_dips,
            all_paths,
            n_jobs=cfg.n_jobs,
            graph_nodes=cfg.graph_nodes,
            pos_cutoff=cfg.pocket_cutoff,
            inference=False,
        )

        print("Done preprocess_unbound_bound\n\n")
    else:
        raise NotImplementedError(f"data_name={cfg.data_name} is not implemented!")

    unbound_predic_ligand_list, unbound_predic_receptor_list = [], []
    bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list = [], []
    pocket_coors_list = []
    for result in preprocess_result:
        (
            unbound_predic_ligand,
            unbound_predic_receptor,
            bound_ligand_repres_nodes_loc_array,
            bound_receptor_repres_nodes_loc_array,
            pocket_coors,
        ) = result
        if pocket_coors is not None:
            unbound_predic_ligand_list.append(unbound_predic_ligand)
            unbound_predic_receptor_list.append(unbound_predic_receptor)
            bound_ligand_repres_nodes_loc_array_list.append(bound_ligand_repres_nodes_loc_array)
            bound_receptor_repres_nodes_loc_array_list.append(bound_receptor_repres_nodes_loc_array)
            pocket_coors_list.append(pocket_coors)

    del preprocess_result

    label = {
        "pocket_coors_list": pocket_coors_list,
        "bound_ligand_repres_nodes_loc_array_list": bound_ligand_repres_nodes_loc_array_list,
        "bound_receptor_repres_nodes_loc_array_list": bound_receptor_repres_nodes_loc_array_list,
    }

    with open(label_filename, "wb") as outfile:
        pickle.dump(label, outfile, pickle.HIGHEST_PROTOCOL)

    protein_to_graph_input = [
        (
            unbound_predic_ligand_list[i],
            unbound_predic_receptor_list[i],
            bound_ligand_repres_nodes_loc_array_list[i],
            bound_receptor_repres_nodes_loc_array_list[i],
        )
        for i in range(len(unbound_predic_ligand_list))
    ]
    print("Start protein_to_graph_unbound_bound")

    both_proteins_to_graph_pair_list = pmap_multi(
        protein_to_graph_unbound_bound,
        protein_to_graph_input,
        n_jobs=cfg.n_jobs,
        graph_nodes=cfg.graph_nodes,
        cutoff=cfg.graph_cutoff,
        max_neighbor=cfg.graph_max_neighbor,
        one_hot=False,
        residue_loc_is_alphaC=cfg.graph_residue_loc_is_alphaC,
    )
    print("Done protein_to_graph_unbound_bound")

    ligand_graph_list, receptor_graph_list = [], []
    for result in both_proteins_to_graph_pair_list:
        ligand_graph, receptor_graph = result
        ligand_graph_list.append(ligand_graph)
        receptor_graph_list.append(receptor_graph)

    save_graphs(ligand_graph_filename, ligand_graph_list)
    save_graphs(receptor_graph_filename, receptor_graph_list)
