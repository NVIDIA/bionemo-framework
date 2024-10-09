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


from typing import Tuple

import dgl
import numpy as np
from biopandas.pdb import PandasPdb
from pytorch_lightning.core import LightningModule

from bionemo.data.equidock.protein_utils import (
    get_residues,
    preprocess_unbound_bound,
    protein_to_graph_unbound_bound,
)
from bionemo.model.protein.equidock.equidock_model import EquiDock
from bionemo.model.protein.equidock.utils.train_utils import batchify_and_create_hetero_graphs_inference
from bionemo.model.utils import _reconfigure_microbatch_calculator, initialize_model_parallel, restore_model


class EquiDockInference(LightningModule):
    """
    Base class for inference.
    """

    def __init__(self, cfg, interactive: bool = False):
        super().__init__()
        self.cfg = cfg
        self.interactive = interactive
        self.model = self.load_model(cfg)
        self._trainer = self.model.trainer

    def load_model(self, cfg):
        """Load saved model checkpoint
        Params:
            checkpoint_path: path to nemo checkpoint
        Returns:
            Loaded model
        """
        # load model class from config which is required to load the .nemo file
        model = restore_model(
            restore_path=cfg.model.restore_from_path, cfg=cfg, model_cls=EquiDock, interactive=self.interactive
        )

        # move self to same device as loaded model
        self.to(model.device)

        # check whether the DDP is initialized
        initialize_model_parallel(model, interactive=self.interactive)

        if not self.interactive:
            # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
            _reconfigure_microbatch_calculator(
                rank=0,  # This doesn't matter since it is only used for logging
                rampup_batch_size=None,
                global_batch_size=1,
                # Make sure that there is no "grad acc" while decoding.
                micro_batch_size=1,
                # We check above to make sure that dataparallel size is always 1 at inference.
                data_parallel_size=1,
            )

        model.freeze()
        self.model = model
        return model

    def create_ligand_receptor_graphs_arrays(
        self, ligand_filename: str, receptor_filename: str
    ) -> Tuple[dgl.graph, dgl.graph, np.ndarray, np.ndarray]:
        """Creates ligand and receptor graphs and arrays (coordinates) from ligand and receptor files"""

        # Preprocess ligand and receptor
        (
            unbound_predic_ligand,
            unbound_predic_receptor,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
        ) = preprocess_unbound_bound(
            get_residues(ligand_filename),
            get_residues(receptor_filename),
            graph_nodes=self.cfg.model.graph_nodes,
            pos_cutoff=self.cfg.data.pocket_cutoff,
            inference=True,
        )

        # Make graphs
        ligand_graph, receptor_graph = protein_to_graph_unbound_bound(
            unbound_predic_ligand,
            unbound_predic_receptor,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
            graph_nodes=self.cfg.model.graph_nodes,
            cutoff=self.cfg.data.graph_cutoff,
            max_neighbor=self.cfg.data.graph_max_neighbor,
            one_hot=False,
            residue_loc_is_alphaC=self.cfg.model.graph_residue_loc_is_alphaC,
        )

        if self.cfg.model.input_edge_feats_dim < 0:
            self.cfg.model.input_edge_feats_dim = ligand_graph.edata["he"].shape[1]

        ligand_graph.ndata["new_x"] = ligand_graph.ndata["x"]

        assert (
            np.linalg.norm(bound_ligand_repres_nodes_loc_clean_array - ligand_graph.ndata["x"].detach().cpu().numpy())
            < 1e-1
        )

        return (
            ligand_graph,
            receptor_graph,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
        )

    def forward(self, batch):
        """Forward pass of the model"""
        return self.model.net(batch)

    def infer(self, ligand_filename, receptor_filename, data_cfg):
        """Run inference"""
        # Create ligand and receptor graphs and arrays
        (
            ligand_graph,
            receptor_graph,
            _,
            _,
        ) = self.model.create_ligand_receptor_graphs_arrays(ligand_filename, receptor_filename, data_cfg)

        # Create a batch of a single DGL graph
        batch_hetero_graph = batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph)

        batch_hetero_graph = batch_hetero_graph.to(self.model.device)
        (
            _,
            _,
            _,
            all_rotation_list,
            all_translation_list,
        ) = self.forward(batch_hetero_graph)

        # Get rotation and translation
        rotation = all_rotation_list[0].detach().cpu().numpy()
        translation = all_translation_list[0].detach().cpu().numpy()

        # Get initial ligand position
        ppdb_ligand = PandasPdb().read_pdb(ligand_filename)
        unbound_ligand_all_atoms_pre_pos = (
            ppdb_ligand.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy().squeeze().astype(np.float32)
        )
        unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T + translation

        ppdb_ligand.df["ATOM"][["x_coord", "y_coord", "z_coord"]] = unbound_ligand_new_pos  # unbound_ligand_new_pos

        return ppdb_ligand
