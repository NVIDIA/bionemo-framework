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
import pickle

import numpy as np
import torch
from dgl import load_graphs
from torch.utils.data import Dataset

from bionemo.data.equidock.protein_utils import UniformRotation_Translation


__all__ = ["Unbound_Bound_Data", "nemo_get_dataset"]


class Unbound_Bound_Data(Dataset):
    def __init__(self, cfg, if_swap=True, reload_mode="train"):
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

        self.cfg = cfg
        self.reload_mode = reload_mode
        self.if_swap = if_swap

        frac_str = ""
        if reload_mode == "train" and cfg["data_name"] == "dips":
            frac_str = "frac_" + str(cfg.data_fraction) + "_"

        label_filename = os.path.join(cache_path, "label_" + frac_str + reload_mode + ".pkl")
        self.ligand_graph_filename = os.path.join(cache_path, "ligand_graph_" + frac_str + reload_mode + ".bin")
        self.receptor_graph_filename = os.path.join(cache_path, "receptor_graph_" + frac_str + reload_mode + ".bin")

        with open(label_filename, "rb") as infile:
            label = pickle.load(infile)

        self.pocket_coors_list = label["pocket_coors_list"]
        self.bound_ligand_repres_nodes_loc_array_list = label["bound_ligand_repres_nodes_loc_array_list"]
        self.bound_receptor_repres_nodes_loc_array_list = label["bound_receptor_repres_nodes_loc_array_list"]
        self.rot_T, self.rot_b = [], []
        infile.close()

    def __len__(self):
        return len(self.pocket_coors_list)

    def __getitem__(self, idx):
        ligand_graph_list, _ = load_graphs(self.ligand_graph_filename, [idx])
        receptor_graph_list, _ = load_graphs(self.receptor_graph_filename, [idx])

        swap = False
        if self.if_swap:
            rnd = np.random.uniform(low=0.0, high=1.0)
            if rnd > 0.5:
                swap = True

        if swap:  # Just as a sanity check, but our model anyway is invariant to such swapping.
            bound_ligand_repres_nodes_loc_array = self.bound_receptor_repres_nodes_loc_array_list[idx]
            bound_receptor_repres_nodes_loc_array = self.bound_ligand_repres_nodes_loc_array_list[idx]
            ligand_graph = receptor_graph_list[0]
            receptor_graph = ligand_graph_list[0]
        else:
            bound_ligand_repres_nodes_loc_array = self.bound_ligand_repres_nodes_loc_array_list[idx]
            bound_receptor_repres_nodes_loc_array = self.bound_receptor_repres_nodes_loc_array_list[idx]
            ligand_graph = ligand_graph_list[0]
            receptor_graph = receptor_graph_list[0]

        pocket_coors_ligand = self.pocket_coors_list[idx]
        pocket_coors_receptor = self.pocket_coors_list[idx]

        # Randomly rotate and translate the ligand.
        if len(self.rot_T) != 0:
            rot_T, rot_b = self.rot_T[idx], self.rot_b[idx]
        else:
            rot_T, rot_b = UniformRotation_Translation(translation_interval=self.cfg["translation_interval"])

        ligand_original_loc = ligand_graph.ndata["x"].detach().numpy()

        mean_to_remove = ligand_original_loc.mean(axis=0, keepdims=True)

        pocket_coors_ligand = (rot_T @ (pocket_coors_ligand - mean_to_remove).T).T + rot_b
        ligand_new_loc = (rot_T @ (ligand_original_loc - mean_to_remove).T).T + rot_b

        ligand_graph.ndata["new_x"] = torch.from_numpy(ligand_new_loc.astype(np.float32))

        return (
            ligand_graph,
            receptor_graph,
            torch.from_numpy(bound_ligand_repres_nodes_loc_array.astype(np.float32)),
            torch.from_numpy(bound_receptor_repres_nodes_loc_array.astype(np.float32)),
            torch.from_numpy(pocket_coors_ligand.astype(np.float32)),
            torch.from_numpy(pocket_coors_receptor.astype(np.float32)),
        )


def nemo_get_dataset(args, reload):
    if reload not in ["train", "val", "test"]:
        raise ValueError(f"reload(={reload}) is not valid, choose from train, val, and test!")
    if_swap = True if reload == "train" else False

    data_set = Unbound_Bound_Data(args, if_swap=if_swap, reload_mode=reload)

    return data_set
