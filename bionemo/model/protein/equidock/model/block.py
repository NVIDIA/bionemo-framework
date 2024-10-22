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

import math
import sys
import warnings

import dgl
import torch
from nemo.core import NeuralModule
from nemo.utils import logging
from torch import nn
from torch.cuda.amp import autocast

from bionemo.model.protein.equidock.model.layers import IEGMN_Layer
from bionemo.model.protein.equidock.model.utils import get_non_lin


class IEGMN(NeuralModule):
    def __init__(self, args, n_lays, fine_tune):
        """
        Independent E(3)-Equivariant Graph Matching layers
        """
        super(IEGMN, self).__init__()

        self.debug = args["debug"]
        self.graph_nodes = args["graph_nodes"]

        self.rot_model = args["rot_model"]

        self.noise_decay_rate = args["noise_decay_rate"]
        self.noise_initial = args["noise_initial"]

        self.use_edge_features_in_gmn = args["use_edge_features_in_gmn"]

        self.use_mean_node_features = args["use_mean_node_features"]

        # 21 types of amino-acid types
        # if we want to use ESM llm model we should replace it here!
        self.residue_emb_layer = nn.Embedding(num_embeddings=21, embedding_dim=args["residue_emb_dim"])

        assert self.graph_nodes == "residues"
        input_node_feats_dim = args["residue_emb_dim"]  # One residue type

        if self.use_mean_node_features:
            input_node_feats_dim += 5  # Additional features from mu_r_norm

        self.iegmn_layers = nn.ModuleList()

        self.iegmn_layers.append(
            IEGMN_Layer(
                orig_h_feats_dim=input_node_feats_dim,
                h_feats_dim=input_node_feats_dim,
                out_feats_dim=args["iegmn_lay_hid_dim"],
                fine_tune=fine_tune,
                args=args,
            )
        )

        if args["shared_layers"]:
            interm_lay = IEGMN_Layer(
                orig_h_feats_dim=input_node_feats_dim,
                h_feats_dim=args["iegmn_lay_hid_dim"],
                out_feats_dim=args["iegmn_lay_hid_dim"],
                args=args,
                fine_tune=fine_tune,
            )
            for layer_idx in range(1, n_lays):
                self.iegmn_layers.append(interm_lay)

        else:
            for layer_idx in range(1, n_lays):
                self.iegmn_layers.append(
                    IEGMN_Layer(
                        orig_h_feats_dim=input_node_feats_dim,
                        h_feats_dim=args["iegmn_lay_hid_dim"],
                        out_feats_dim=args["iegmn_lay_hid_dim"],
                        args=args,
                        fine_tune=fine_tune,
                    )
                )

        assert args["rot_model"] == "kb_att"

        # Attention layers
        self.num_att_heads = args["num_att_heads"]
        self.out_feats_dim = args["iegmn_lay_hid_dim"]

        self.att_mlp_key_ROT = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False),
        )
        self.att_mlp_query_ROT = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.num_att_heads * self.out_feats_dim, bias=False),
        )

        self.mlp_h_mean_ROT = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(args["dropout"]),
            get_non_lin(args["nonlin"], args["leakyrelu_neg_slope"]),
        )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.0)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, batch_hetero_graph):
        device = batch_hetero_graph.device
        orig_coors_ligand = batch_hetero_graph.nodes["ligand"].data["new_x"]
        orig_coors_receptor = batch_hetero_graph.nodes["receptor"].data["x"]

        coors_ligand = batch_hetero_graph.nodes["ligand"].data["new_x"]
        coors_receptor = batch_hetero_graph.nodes["receptor"].data["x"]

        # Embed residue types with a lookup table.
        h_feats_ligand = self.residue_emb_layer(
            batch_hetero_graph.nodes["ligand"].data["res_feat"].view(-1).long()
        )  # (N_res, emb_dim)
        h_feats_receptor = self.residue_emb_layer(
            batch_hetero_graph.nodes["receptor"].data["res_feat"].view(-1).long()
        )  # (N_res, emb_dim)

        if self.debug:
            logging.info(torch.max(h_feats_ligand), "h_feats_ligand before layers ")

        if self.use_mean_node_features:
            h_feats_ligand = torch.cat(
                [h_feats_ligand, torch.log(batch_hetero_graph.nodes["ligand"].data["mu_r_norm"])], dim=1
            )
            h_feats_receptor = torch.cat(
                [h_feats_receptor, torch.log(batch_hetero_graph.nodes["receptor"].data["mu_r_norm"])], dim=1
            )

        if self.debug:
            logging.info(
                torch.max(h_feats_ligand),
                torch.norm(h_feats_ligand),
                "h_feats_ligand before layers but after mu_r_norm",
            )

        original_ligand_node_features = h_feats_ligand
        original_receptor_node_features = h_feats_receptor

        original_edge_feats_ligand = batch_hetero_graph.edges["ll"].data["he"] * self.use_edge_features_in_gmn
        original_edge_feats_receptor = batch_hetero_graph.edges["rr"].data["he"] * self.use_edge_features_in_gmn

        for i, layer in enumerate(self.iegmn_layers):
            if self.debug:
                logging.info("layer ", i)

            coors_ligand, h_feats_ligand, coors_receptor, h_feats_receptor = layer(
                hetero_graph=batch_hetero_graph,
                coors_ligand=coors_ligand,
                h_feats_ligand=h_feats_ligand,
                original_ligand_node_features=original_ligand_node_features,
                original_edge_feats_ligand=original_edge_feats_ligand,
                orig_coors_ligand=orig_coors_ligand,
                coors_receptor=coors_receptor,
                h_feats_receptor=h_feats_receptor,
                original_receptor_node_features=original_receptor_node_features,
                original_edge_feats_receptor=original_edge_feats_receptor,
                orig_coors_receptor=orig_coors_receptor,
            )

        if self.debug:
            logging.info(torch.max(h_feats_ligand), "h_feats_ligand after MPNN")
            logging.info(torch.max(coors_ligand), "coors_ligand before after MPNN")

        batch_hetero_graph.nodes["ligand"].data["x_iegmn_out"] = coors_ligand
        batch_hetero_graph.nodes["receptor"].data["x_iegmn_out"] = coors_receptor
        batch_hetero_graph.nodes["ligand"].data["hv_iegmn_out"] = h_feats_ligand
        batch_hetero_graph.nodes["receptor"].data["hv_iegmn_out"] = h_feats_receptor

        list_hetero_graph = dgl.unbatch(batch_hetero_graph)

        all_T_align_list = []
        all_b_align_list = []
        all_Y_receptor_att_ROT_list = []
        all_Y_ligand_att_ROT_list = []

        # TODO: run SVD in batches, if possible
        for the_idx, hetero_graph in enumerate(list_hetero_graph):
            # Get H vectors
            # (m, d)
            self.device = hetero_graph.device

            H_receptor_feats = hetero_graph.nodes["receptor"].data["hv_iegmn_out"]
            H_receptor_feats_att_mean_ROT = torch.mean(
                self.mlp_h_mean_ROT(H_receptor_feats), dim=0, keepdim=True
            )  # (1, d)

            # (n, d)
            H_ligand_feats = hetero_graph.nodes["ligand"].data["hv_iegmn_out"]
            H_ligand_feats_att_mean_ROT = torch.mean(
                self.mlp_h_mean_ROT(H_ligand_feats), dim=0, keepdim=True
            )  # (1, d)

            d = H_ligand_feats.shape[1]
            assert d == self.out_feats_dim

            # Z coordinates
            Z_receptor_coors = hetero_graph.nodes["receptor"].data["x_iegmn_out"]

            Z_ligand_coors = hetero_graph.nodes["ligand"].data["x_iegmn_out"]

            #################### AP 1: compute two point clouds of K_heads points each, then do Kabsch  #########################
            # Att weights to compute the receptor centroid. They query is the average_h_ligand. Keys are each h_receptor_j
            att_weights_receptor_ROT = torch.softmax(
                # (K_heads, m_rec, d)
                self.att_mlp_key_ROT(H_receptor_feats).view(-1, self.num_att_heads, d).transpose(0, 1)
                @
                # (K_heads, d, 1)
                self.att_mlp_query_ROT(H_ligand_feats_att_mean_ROT)
                .view(1, self.num_att_heads, d)
                .transpose(0, 1)
                .transpose(1, 2)
                / math.sqrt(d),  # (K_heads, m_receptor, 1)
                dim=1,
            ).view(self.num_att_heads, -1)

            Y_receptor_att_ROT = att_weights_receptor_ROT @ Z_receptor_coors  # K_heads, 3
            all_Y_receptor_att_ROT_list.append(Y_receptor_att_ROT)

            # Att weights to compute the ligand centroid. They query is the average_h_receptor. Keys are each h_ligand_i
            att_weights_ligand_ROT = torch.softmax(
                self.att_mlp_key_ROT(H_ligand_feats).view(-1, self.num_att_heads, d).transpose(0, 1)
                @ self.att_mlp_query_ROT(H_receptor_feats_att_mean_ROT)
                .view(1, self.num_att_heads, d)
                .transpose(0, 1)
                .transpose(1, 2)
                / math.sqrt(d),  # (K_heads, n_ligand, 1)
                dim=1,
            ).view(self.num_att_heads, -1)

            Y_ligand_att_ROT = att_weights_ligand_ROT @ Z_ligand_coors  # K_heads, 3
            all_Y_ligand_att_ROT_list.append(Y_ligand_att_ROT)

            # Apply Kabsch algorithm
            Y_receptor_att_ROT_mean = Y_receptor_att_ROT.mean(dim=0, keepdim=True)  # (1,3)
            Y_ligand_att_ROT_mean = Y_ligand_att_ROT.mean(dim=0, keepdim=True)  # (1,3)

            A = (Y_receptor_att_ROT - Y_receptor_att_ROT_mean).transpose(0, 1) @ (
                Y_ligand_att_ROT - Y_ligand_att_ROT_mean
            )  # 3, 3

            assert not torch.isnan(A).any()
            with autocast(enabled=False):
                U, S, Vt = torch.linalg.svd(A.float())

            num_it = 0
            while (
                torch.min(S) < 1e-3
                or torch.min(torch.abs((S**2).view(1, 3) - (S**2).view(3, 1) + torch.eye(3).to(device))) < 1e-2
            ):
                if self.debug:
                    logging.info("S inside loop ", num_it, " is ", S, " and A = ", A)

                A = A + torch.rand(3, 3).to(device) * torch.eye(3).to(device)
                U, S, Vt = torch.linalg.svd(A)
                num_it += 1

                if num_it > 10:
                    message = "SVD consistently numerically unstable! Exitting ... "
                    warnings.warn(message)
                    logging.info(message)
                    sys.exit(1)
            with autocast(enabled=False):
                corr_mat = torch.diag(torch.Tensor([1, 1, torch.sign(torch.det(A.float()))])).to(device)
            T_align = (U @ corr_mat) @ Vt

            b_align = Y_receptor_att_ROT_mean - torch.t(T_align @ Y_ligand_att_ROT_mean.t())  # (1,3)

            #################### end AP 1 #########################

            if self.debug:
                logging.info("DEBUG: Y_receptor_att_ROT_mean", Y_receptor_att_ROT_mean)
                logging.info("DEBUG: Y_ligand_att_ROT_mean", Y_ligand_att_ROT_mean)

            all_T_align_list.append(T_align)
            all_b_align_list.append(b_align)

        return [all_T_align_list, all_b_align_list, all_Y_ligand_att_ROT_list, all_Y_receptor_att_ROT_list]
