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

import torch
from dgl import function as fn
from nemo.core import NeuralModule
from nemo.utils import logging
from torch import nn

from bionemo.model.protein.equidock.model.utils import (
    apply_final_h_layer_norm,
    compute_cross_attention,
    get_final_h_layer_norm,
    get_layer_norm,
    get_mask,
    get_non_lin,
)


class Gaussian(nn.Module):
    def __init__(self, offsets: torch.Tensor, etas: torch.Tensor, trainable: bool = False) -> None:
        super(Gaussian, self).__init__()
        self.trainable = trainable
        if self.trainable:
            self.offsets = nn.Parameter(offsets)
            self.etas = nn.Parameter(etas)
        else:
            self.register_buffer("offsets", offsets)
            self.register_buffer("etas", etas)

    def forward(self, x):
        x = x.norm(dim=-1, keepdim=True)
        dx = x - self.offsets
        return (-self.etas * dx**2).exp().detach()


class SinusoidsEmbedding(nn.Module):
    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class IEGMN_Layer(NeuralModule):
    def __init__(self, orig_h_feats_dim, h_feats_dim, out_feats_dim, fine_tune, args):  # in dim of h  # out dim of h
        """
        A single layer of Independent E(3)-Equivariant Graph Matching Network.

        m(ji)  = phi_e(hl(i), hl(j), exp(-|xl(i) - xl(j)|^2/sigma), f(ji)) \forall e(ji) in E1 U E2
        mu(ji) = a(ji) W hl(j), \forall i \\in V1 & j \\in V2 || i \\in V2 & j \\in V1
        m(i) = mean(j \\in N(i), m(ji)), \forall i \\in V1 U V2
        mu(i \\in V_l ) = sum(j \\in V_!l, mu(ji))
        xlp(i) = \\eta x0(i) + (1-\\eta) xl(i) + sum( j \\in N(i), (xl(i) - xl(j)) phi_x(m(ji)) ) \forall i \\in V1 U V2
        hlp(i) = (1-\beta) hl(i) + \beta * phih(hl(i), m(i), mu(i), f(i))

        a(ji) = exp(< ksiq(hl(i)), ksik(hl(j)) >)/ sum(j', exp(< ksiq(hl(i)), ksik(hl(j')) >))

        Z1.shape (3, n), Z2.shape (3, m)
        H1.shape (d, n), H2.shape (d, m)
        Z1, H1, Z2, H2 = IEGMN(X1, F1, X2, F2)

        """
        super(IEGMN_Layer, self).__init__()

        input_edge_feats_dim = args["input_edge_feats_dim"]
        dropout = args["dropout"]
        nonlin = args["nonlin"]
        self.cross_msgs = args["cross_msgs"]
        layer_norm = args["layer_norm"]
        layer_norm_coors = args["layer_norm_coors"]
        self.final_h_layer_norm = args["final_h_layer_norm"]
        self.use_dist_in_layers = args["use_dist_in_layers"]
        self.skip_weight_h = args["skip_weight_h"]
        self.x_connection_init = args["x_connection_init"]
        leakyrelu_neg_slope = args["leakyrelu_neg_slope"]

        self.fine_tune = fine_tune

        self.debug = args["debug"]

        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim

        self.all_sigmas_dist = torch.as_tensor([1.5**x for x in range(15)])
        # self.radial_featurization = Gaussian(self.all_sigmas_dist, torch.as_tensor([1.0]))

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear((h_feats_dim * 2) + input_edge_feats_dim + len(self.all_sigmas_dist), self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, self.out_feats_dim),
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
        )

        # NODES
        self.node_norm = nn.Identity()  # nn.LayerNorm(h_feats_dim)

        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, h_feats_dim),
            nn.Linear(h_feats_dim, out_feats_dim),
        )

        self.final_h_layernorm_layer = get_final_h_layer_norm(self.final_h_layer_norm, out_feats_dim)

        # The scalar weight to be multiplied by (x_i - x_j)
        self.coors_mlp = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm_coors, self.out_feats_dim),
            nn.Linear(self.out_feats_dim, 1),
        )

        if self.fine_tune:
            self.att_mlp_cross_coors_Q = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_K = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim, bias=False),
                get_non_lin(nonlin, leakyrelu_neg_slope),
            )
            self.att_mlp_cross_coors_V = nn.Sequential(
                nn.Linear(h_feats_dim, h_feats_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(h_feats_dim, 1),
            )
        # self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.0)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges1(self, edges):
        return {"cat_feat": torch.cat([edges.src["feat"], edges.dst["feat"]], dim=1)}

    def forward(
        self,
        hetero_graph,
        coors_ligand,
        h_feats_ligand,
        original_ligand_node_features,
        original_edge_feats_ligand,
        orig_coors_ligand,
        coors_receptor,
        h_feats_receptor,
        original_receptor_node_features,
        original_edge_feats_receptor,
        orig_coors_receptor,
    ):
        device = hetero_graph.device
        with hetero_graph.local_scope():
            hetero_graph.nodes["ligand"].data["x_now"] = coors_ligand
            hetero_graph.nodes["receptor"].data["x_now"] = coors_receptor
            # first time set here
            hetero_graph.nodes["ligand"].data["feat"] = h_feats_ligand
            hetero_graph.nodes["receptor"].data["feat"] = h_feats_receptor

            if self.debug:
                logging.info(torch.max(hetero_graph.nodes["ligand"].data["x_now"]), "x_now : x_i at layer entrance")
                logging.info(
                    torch.max(hetero_graph.nodes["ligand"].data["feat"]), "data[feat] = h_i at layer entrance"
                )

            hetero_graph.apply_edges(
                fn.u_sub_v("x_now", "x_now", "x_rel"), etype=("ligand", "ll", "ligand")
            )  # x_i - x_j
            hetero_graph.apply_edges(fn.u_sub_v("x_now", "x_now", "x_rel"), etype=("receptor", "rr", "receptor"))

            x_rel_mag_ligand = hetero_graph.edges[("ligand", "ll", "ligand")].data["x_rel"] ** 2
            # ||x_i - x_j||^2 : (N_res, 1)
            x_rel_mag_ligand = torch.sum(x_rel_mag_ligand, dim=1, keepdim=True)
            x_rel_mag_ligand = torch.cat(
                [torch.exp(-x_rel_mag_ligand / sigma) for sigma in self.all_sigmas_dist], dim=-1
            )

            x_rel_mag_receptor = hetero_graph.edges[("receptor", "rr", "receptor")].data["x_rel"] ** 2
            x_rel_mag_receptor = torch.sum(x_rel_mag_receptor, dim=1, keepdim=True)
            x_rel_mag_receptor = torch.cat(
                [torch.exp(-x_rel_mag_receptor / sigma) for sigma in self.all_sigmas_dist], dim=-1
            )

            if not self.use_dist_in_layers:
                x_rel_mag_ligand = x_rel_mag_ligand * 0.0
                x_rel_mag_receptor = x_rel_mag_receptor * 0.0

            if self.debug:
                logging.info(
                    torch.max(hetero_graph.edges[("ligand", "ll", "ligand")].data["x_rel"]), "x_rel : x_i - x_j"
                )
                logging.info(
                    torch.max(x_rel_mag_ligand, dim=0).values,
                    "x_rel_mag_ligand = [exp(-||x_i - x_j||^2 / sigma) for sigma = 1.5 ** x, x = [0, 15]]",
                )

            hetero_graph.apply_edges(self.apply_edges1, etype=("ligand", "ll", "ligand"))  # i->j edge:  [h_i h_j]
            hetero_graph.apply_edges(self.apply_edges1, etype=("receptor", "rr", "receptor"))

            cat_input_for_msg_ligand = torch.cat(
                (hetero_graph.edges["ll"].data["cat_feat"], original_edge_feats_ligand, x_rel_mag_ligand),  # [h_i h_j]
                dim=-1,
            )
            cat_input_for_msg_receptor = torch.cat(
                (hetero_graph.edges["rr"].data["cat_feat"], original_edge_feats_receptor, x_rel_mag_receptor), dim=-1
            )

            hetero_graph.edges["ll"].data["msg"] = self.edge_mlp(cat_input_for_msg_ligand)  # m_{i->j}
            hetero_graph.edges["rr"].data["msg"] = self.edge_mlp(cat_input_for_msg_receptor)

            if self.debug:
                logging.info(
                    torch.max(hetero_graph.edges["ll"].data["msg"]),
                    "data[msg] = m_{i->j} = phi^e(h_i, h_j, f_{i,j}, x_rel_mag_ligand)",
                )

            mask = get_mask(hetero_graph.batch_num_nodes("ligand"), hetero_graph.batch_num_nodes("receptor"), device)

            # \mu_i
            hetero_graph.nodes["ligand"].data["aggr_cross_msg"] = compute_cross_attention(
                self.att_mlp_Q(h_feats_ligand),
                self.att_mlp_K(h_feats_receptor),
                self.att_mlp_V(h_feats_receptor),
                mask,
                self.cross_msgs,
            )

            hetero_graph.nodes["receptor"].data["aggr_cross_msg"] = compute_cross_attention(
                self.att_mlp_Q(h_feats_receptor),
                self.att_mlp_K(h_feats_ligand),
                self.att_mlp_V(h_feats_ligand),
                mask.transpose(0, 1),
                self.cross_msgs,
            )

            if self.debug:
                logging.info(
                    torch.max(hetero_graph.nodes["ligand"].data["aggr_cross_msg"]),
                    "aggr_cross_msg(i) = sum_j a_{i,j} * h_j",
                )

            edge_coef_ligand = self.coors_mlp(hetero_graph.edges["ll"].data["msg"])  # \phi^x(m_{i->j})
            # (x_i - x_j) * \phi^x(m_{i->j})
            hetero_graph.edges["ll"].data["x_moment"] = hetero_graph.edges["ll"].data["x_rel"] * edge_coef_ligand
            edge_coef_receptor = self.coors_mlp(hetero_graph.edges["rr"].data["msg"])
            hetero_graph.edges["rr"].data["x_moment"] = hetero_graph.edges["rr"].data["x_rel"] * edge_coef_receptor

            if self.debug:
                logging.info(torch.max(edge_coef_ligand), r"edge_coef_ligand : \phi^x(m_{i->j})")
                logging.info(
                    torch.max(hetero_graph.edges["ll"].data["x_moment"]),
                    r"data[x_moment] = (x_i - x_j) * \phi^x(m_{i->j})",
                )

            hetero_graph.update_all(
                fn.copy_e("x_moment", "m"), fn.mean("m", "x_update"), etype=("ligand", "ll", "ligand")
            )

            hetero_graph.update_all(
                fn.copy_e("x_moment", "m"), fn.mean("m", "x_update"), etype=("receptor", "rr", "receptor")
            )

            hetero_graph.update_all(fn.copy_e("msg", "m"), fn.mean("m", "aggr_msg"), etype=("ligand", "ll", "ligand"))

            hetero_graph.update_all(
                fn.copy_e("msg", "m"), fn.mean("m", "aggr_msg"), etype=("receptor", "rr", "receptor")
            )

            x_final_ligand = (
                self.x_connection_init * orig_coors_ligand
                + (1.0 - self.x_connection_init) * hetero_graph.nodes["ligand"].data["x_now"]
                + hetero_graph.nodes["ligand"].data["x_update"]
            )

            x_final_receptor = (
                self.x_connection_init * orig_coors_receptor
                + (1.0 - self.x_connection_init) * hetero_graph.nodes["receptor"].data["x_now"]
                + hetero_graph.nodes["receptor"].data["x_update"]
            )

            if self.fine_tune:
                x_final_ligand = x_final_ligand + self.att_mlp_cross_coors_V(h_feats_ligand) * (
                    hetero_graph.nodes["ligand"].data["x_now"]
                    - compute_cross_attention(
                        self.att_mlp_cross_coors_Q(h_feats_ligand),
                        self.att_mlp_cross_coors_K(h_feats_receptor),
                        hetero_graph.nodes["receptor"].data["x_now"],
                        mask,
                        self.cross_msgs,
                    )
                )
                x_final_receptor = x_final_receptor + self.att_mlp_cross_coors_V(h_feats_receptor) * (
                    hetero_graph.nodes["receptor"].data["x_now"]
                    - compute_cross_attention(
                        self.att_mlp_cross_coors_Q(h_feats_receptor),
                        self.att_mlp_cross_coors_K(h_feats_ligand),
                        hetero_graph.nodes["ligand"].data["x_now"],
                        mask.transpose(0, 1),
                        self.cross_msgs,
                    )
                )

            if self.debug:
                logging.info(
                    torch.max(hetero_graph.nodes["ligand"].data["aggr_msg"]), r"data[aggr_msg]: \sum_j m_{i->j} "
                )
                logging.info(
                    torch.max(hetero_graph.nodes["ligand"].data["x_update"]),
                    r"data[x_update] : \sum_j (x_i - x_j) * \phi^x(m_{i->j})",
                )
                logging.info(torch.max(x_final_ligand), "x_i new = x_final_ligand : x_i + data[x_update]")

            input_node_upd_ligand = torch.cat(
                (
                    self.node_norm(hetero_graph.nodes["ligand"].data["feat"]),
                    hetero_graph.nodes["ligand"].data["aggr_msg"],
                    hetero_graph.nodes["ligand"].data["aggr_cross_msg"],
                    original_ligand_node_features,
                ),
                dim=-1,
            )

            input_node_upd_receptor = torch.cat(
                (
                    self.node_norm(hetero_graph.nodes["receptor"].data["feat"]),
                    hetero_graph.nodes["receptor"].data["aggr_msg"],
                    hetero_graph.nodes["receptor"].data["aggr_cross_msg"],
                    original_receptor_node_features,
                ),
                dim=-1,
            )

            # Skip connections
            if self.h_feats_dim == self.out_feats_dim:
                node_upd_ligand = (
                    self.skip_weight_h * self.node_mlp(input_node_upd_ligand)
                    + (1.0 - self.skip_weight_h) * h_feats_ligand
                )
                node_upd_receptor = (
                    self.skip_weight_h * self.node_mlp(input_node_upd_receptor)
                    + (1.0 - self.skip_weight_h) * h_feats_receptor
                )
            else:
                node_upd_ligand = self.node_mlp(input_node_upd_ligand)
                node_upd_receptor = self.node_mlp(input_node_upd_receptor)

            if self.debug:
                logging.info("node_mlp params")
                for p in self.node_mlp.parameters():
                    print(p)
                logging.info(torch.max(input_node_upd_ligand), "concat(h_i, aggr_msg, aggr_cross_msg)")
                logging.info(torch.max(node_upd_ligand), "h_i new = h_i + MLP(h_i, aggr_msg, aggr_cross_msg)")

            node_upd_ligand = apply_final_h_layer_norm(
                hetero_graph, node_upd_ligand, "ligand", self.final_h_layer_norm, self.final_h_layernorm_layer
            )
            node_upd_receptor = apply_final_h_layer_norm(
                hetero_graph, node_upd_receptor, "receptor", self.final_h_layer_norm, self.final_h_layernorm_layer
            )

            return x_final_ligand, node_upd_ligand, x_final_receptor, node_upd_receptor
