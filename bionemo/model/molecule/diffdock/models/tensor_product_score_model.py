# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import partial

import torch
from e3nn import o3
from nemo.utils import logging
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter_mean

from bionemo.data.diffdock.process_mols import (
    lig_feature_dims,
    rec_residue_feature_dims,
)
from bionemo.model.molecule.diffdock.models.common_blocks import (
    AtomEncoder,
    GaussianSmearing,
    TensorProductConvLayer,
)
from bionemo.model.molecule.diffdock.utils import (
    so3,
    torus,
)
from bionemo.model.molecule.diffdock.utils.diffusion import (
    t_to_sigma as t_to_sigma_compl,
)
from bionemo.model.molecule.diffdock.utils.diffusion import (
    timestep_embedding,
)
from bionemo.model.molecule.diffdock.utils.utils import estimate_memory_usage


class TensorProductScoreModel(torch.nn.Module):
    """
    Tensor Product Score Model.
    """

    def __init__(self, cfg: OmegaConf):
        super(TensorProductScoreModel, self).__init__()

        timestep_emb_func = timestep_embedding(
            embedding_type=cfg.embedding_type,
            embedding_dim=cfg.sigma_embed_dim,
            embedding_scale=cfg.embedding_scale,
        )
        sh_lmax = 2
        confidence_no_batchnorm = False
        confidence_dropout = 0
        batch_norm = not cfg.no_batch_norm
        batch_norm_with_shift = cfg.get('batch_norm_with_shift', True)
        dropout = cfg.dropout
        tensor_product_type = cfg.tensor_product.type
        use_second_order_repr = cfg.tensor_product.use_second_order_repr
        tp_dtype = cfg.tensor_product.get('dtype', torch.float32)
        if tp_dtype in [16, '16']:
            tp_dtype = torch.float16
        elif tp_dtype == 'bf16':
            tp_dtype = torch.bfloat16
        else:
            tp_dtype = torch.float32
        lm_embedding_type = None
        if cfg.esm_embeddings_path is not None:
            lm_embedding_type = "esm"
        num_confidence_outputs = (
            len(cfg.rmsd_classification_cutoff) + 1
            if "rmsd_classification_cutoff" in cfg and isinstance(cfg.rmsd_classification_cutoff, ListConfig)
            else 1
        )
        self.t_to_sigma = partial(t_to_sigma_compl, cfg=cfg)
        self.in_lig_edge_features = 4
        self.sigma_embed_dim = cfg.sigma_embed_dim
        self.lig_max_radius = cfg.max_radius
        self.rec_max_radius = 30
        self.cross_max_distance = cfg.cross_max_distance
        self.dynamic_max_cross = cfg.dynamic_max_cross
        self.center_max_distance = 30
        self.distance_embed_dim = cfg.distance_embed_dim
        self.cross_distance_embed_dim = cfg.cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        ns, nv = cfg.ns, cfg.nv
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = cfg.scale_by_sigma
        self.no_torsion = cfg.diffusion.no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = cfg.confidence_mode
        self.num_conv_layers = cfg.num_conv_layers
        self.clamp = cfg.get('clamp', False)
        self.estimate_memory_usage = cfg.get('estimate_memory_usage', None)
        if self.estimate_memory_usage is not None and self.estimate_memory_usage.maximal is None:
            self.estimate_memory_usage.maximal = (
                0.75 * torch.cuda.get_device_properties('cuda:0').total_memory / 2**20
            )

        # embedding layers
        self.lig_node_embedding = AtomEncoder(
            emb_dim=self.ns,
            feature_dims=lig_feature_dims,
            sigma_embed_dim=self.sigma_embed_dim,
        )
        self.lig_edge_embedding = nn.Sequential(
            nn.Linear(
                self.in_lig_edge_features + self.sigma_embed_dim + self.distance_embed_dim,
                self.ns,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ns, self.ns),
        )

        self.rec_node_embedding = AtomEncoder(
            emb_dim=self.ns,
            feature_dims=rec_residue_feature_dims,
            sigma_embed_dim=self.sigma_embed_dim,
            lm_embedding_type=lm_embedding_type,
        )
        self.rec_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.distance_embed_dim, self.ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ns, self.ns),
        )

        self.cross_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.cross_distance_embed_dim, self.ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ns, self.ns),
        )

        self.lig_distance_expansion = GaussianSmearing(0.0, self.lig_max_radius, self.distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, self.rec_max_radius, self.distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, self.cross_max_distance, self.cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o + {nv}x2e",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
            ]
        else:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o",
                f"{ns}x0e + {nv}x1o + {nv}x1e",
                f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
            ]

        (
            lig_conv_layers,
            rec_conv_layers,
            lig_to_rec_conv_layers,
            rec_to_lig_conv_layers,
        ) = ([], [], [], [])

        for i in range(self.num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]

            if tensor_product_type == "e3nn" or use_second_order_repr:
                tp_type = "e3nn"
                tp_param = -1
            elif tensor_product_type == "marta":
                tp_type = "marta"
                tp_param = min(i, 3)

            parameters = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "n_edge_features": 3 * ns,
                "hidden_features": 3 * ns,
                "residual": False,
                "batch_norm": batch_norm,
                "batch_norm_with_shift": batch_norm_with_shift,
                "dropout": dropout,
                "tp_param": tp_param,
                "tp_type": tp_type,
                "dtype": tp_dtype,
            }

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)

        if self.confidence_mode:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if self.num_conv_layers >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs),
            )
        else:
            # center of mass translation and rotation components
            self.center_distance_expansion = GaussianSmearing(0.0, self.center_max_distance, self.distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(self.distance_embed_dim + self.sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns),
            )

            self.final_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps="2x1o + 2x1e",
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm,
                batch_norm_with_shift=batch_norm_with_shift,
                tp_type='e3nn',
            )
            self.tr_final_layer = nn.Sequential(
                nn.Linear(1 + self.sigma_embed_dim, ns),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ns, 1),
            )
            self.rot_final_layer = nn.Sequential(
                nn.Linear(1 + self.sigma_embed_dim, ns),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ns, 1),
            )

            if not self.no_torsion:
                # torsion angles components
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(self.distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns),
                )
                self.final_tp_tor = o3.FullTensorProduct(
                    self.sh_irreps, "2e", _optimize_einsums=cfg.get("optimize_einsums", None)
                )
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.lig_conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f"{ns}x0o + {ns}x0e",
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    batch_norm_with_shift=batch_norm_with_shift,
                    tp_type='e3nn',
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False),
                )
        self.torus_score_norm = torus.TorusScoreNorm()

    def forward(self, data):
        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(
                *[data.complex_t[noise_type] for noise_type in ["tr", "rot", "tor"]]
            )
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ["tr", "rot", "tor"]]

        # build ligand graph
        (
            lig_node_attr,
            lig_edge_index,
            lig_edge_attr,
            lig_edge_sh,
        ) = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        (
            rec_node_attr,
            rec_edge_index,
            rec_edge_attr,
            rec_edge_sh,
        ) = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build cross graph
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        if cross_edge_index.numel() == 0 and not self.confidence_mode:
            return (
                0.0 * lig_node_attr[: len(data.name), :3],
                0.0 * lig_node_attr[: len(data.name), :3],
                0.0 * lig_edge_attr.reshape(-1)[: data['ligand'].edge_mask.sum().item()],
            )
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        # estimate the memory of forward pass
        if self.estimate_memory_usage is not None and not self.confidence_mode and self.training:
            total_memory = estimate_memory_usage(data, cross_edge_attr.shape[0], self.estimate_memory_usage)
            if total_memory > self.estimate_memory_usage.maximal:
                logging.warning(
                    f'Estimated memory {total_memory} exceeds maximal = {self.estimate_memory_usage.maximal} '
                    f' for batch: {data.name} with {cross_edge_attr.shape[0]} cross edges, skipping'
                )
                return (
                    0.0 * lig_node_attr[: len(data.name), :3],
                    0.0 * lig_node_attr[: len(data.name), :3],
                    0.0 * lig_edge_attr.reshape(-1)[: data['ligand'].edge_mask.sum().item()],
                )

        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing
            lig_edge_attr_ = torch.cat(
                [
                    lig_edge_attr,
                    lig_node_attr[lig_src, : self.ns],
                    lig_node_attr[lig_dst, : self.ns],
                ],
                -1,
            )
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat(
                [
                    cross_edge_attr,
                    lig_node_attr[cross_lig, : self.ns],
                    rec_node_attr[cross_rec, : self.ns],
                ],
                -1,
            )
            lig_inter_update = self.rec_to_lig_conv_layers[l](
                rec_node_attr,
                cross_edge_index,
                rec_to_lig_edge_attr_,
                cross_edge_sh,
                out_nodes=lig_node_attr.shape[0],
            )

            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat(
                    [
                        rec_edge_attr,
                        rec_node_attr[rec_src, : self.ns],
                        rec_node_attr[rec_dst, : self.ns],
                    ],
                    -1,
                )
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

                lig_to_rec_edge_attr_ = torch.cat(
                    [
                        cross_edge_attr,
                        lig_node_attr[cross_lig, : self.ns],
                        rec_node_attr[cross_rec, : self.ns],
                    ],
                    -1,
                )
                rec_inter_update = self.lig_to_rec_conv_layers[l](
                    lig_node_attr,
                    torch.flip(cross_edge_index, dims=[0]),
                    lig_to_rec_edge_attr_,
                    cross_edge_sh,
                    out_nodes=rec_node_attr.shape[0],
                )

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(
                    rec_node_attr,
                    (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]),
                )
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        # compute confidence score
        if self.confidence_mode:
            scalar_lig_attr = (
                torch.cat([lig_node_attr[:, : self.ns], lig_node_attr[:, -self.ns :]], dim=1)
                if self.num_conv_layers >= 3
                else lig_node_attr[:, : self.ns]
            )
            confidence = self.confidence_predictor(scatter_mean(scalar_lig_attr, data["ligand"].batch, dim=0)).squeeze(
                dim=-1
            )
            return confidence

        # compute translational and rotational score vectors
        (
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
        ) = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[0], : self.ns]], -1)

        lig_node_attr = self.clamp_value(lig_node_attr)
        center_edge_attr = self.clamp_value(center_edge_attr)

        global_pred = self.final_conv(
            lig_node_attr,
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
            out_nodes=data.num_graphs,
        )

        global_pred = self.clamp_value(global_pred, add_noise=True)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t["tr"])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data["ligand"].x.device)

        if self.no_torsion or data["ligand"].edge_mask.sum() == 0:
            return tr_pred, rot_pred, torch.empty(0, device=data["ligand"].x.device)

        # torsional components
        (
            tor_bonds,
            tor_edge_index,
            tor_edge_attr,
            tor_edge_sh,
        ) = self.build_bond_conv_graph(data)
        tor_bond_vec = data["ligand"].pos[tor_bonds[1]] - data["ligand"].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization="component")
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat(
            [
                tor_edge_attr,
                lig_node_attr[tor_edge_index[1], : self.ns],
                tor_bond_attr[tor_edge_index[0], : self.ns],
            ],
            -1,
        )

        lig_node_attr = self.clamp_value(lig_node_attr)
        tor_edge_attr = self.clamp_value(tor_edge_attr)

        tor_pred = self.tor_bond_conv(
            lig_node_attr,
            tor_edge_index,
            tor_edge_attr,
            tor_edge_sh,
            out_nodes=data["ligand"].edge_mask.sum(),
            reduce="mean",
        )
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data["ligand"].batch][data["ligand", "ligand"].edge_index[0]][data["ligand"].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(
                torch.tensor(self.torus_score_norm(edge_sigma.cpu().numpy())).float().to(data["ligand"].x.device)
            )

        tr_pred = self.clamp_value(tr_pred, add_noise=True)
        rot_pred = self.clamp_value(rot_pred, add_noise=True)

        return tr_pred, rot_pred, tor_pred

    def clamp_value(self, x, add_noise=False):
        if self.clamp and ((self.training and self.clamp.train) or ((not self.training) and self.clamp.eval)):
            x = torch.clamp(x, -self.clamp.max, self.clamp.max)
            if add_noise:
                x = x + self.clamp.random_scale * torch.where(x.abs() == self.clamp.max, torch.randn_like(x), 0.0)

        return x

    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features
        data["ligand"].node_sigma_emb = self.timestep_emb_func(data["ligand"].node_t["tr"])

        # compute edges
        radius_edges = radius_graph(data["ligand"].pos, self.lig_max_radius, data["ligand"].batch)
        edge_index = torch.cat([data["ligand", "ligand"].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat(
            [
                data["ligand", "ligand"].edge_attr,
                torch.zeros(
                    radius_edges.shape[-1],
                    self.in_lig_edge_features,
                    device=data["ligand"].x.device,
                ),
            ],
            0,
        )

        # compute initial features
        edge_sigma_emb = data["ligand"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data["ligand"].x, data["ligand"].node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data["ligand"].pos[dst.long()] - data["ligand"].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # builds the receptor initial node and edge embeddings
        data["receptor"].node_sigma_emb = self.timestep_emb_func(
            data["receptor"].node_t["tr"]
        )  # tr rot and tor noise is all the same
        node_attr = torch.cat([data["receptor"].x, data["receptor"].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data["receptor", "receptor"].edge_index
        src, dst = edge_index
        edge_vec = data["receptor"].pos[dst.long()] - data["receptor"].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["receptor"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(
                data["receptor"].pos / cross_distance_cutoff[data["receptor"].batch],
                data["ligand"].pos / cross_distance_cutoff[data["ligand"].batch],
                1,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )
        else:
            edge_index = radius(
                data["receptor"].pos,
                data["ligand"].pos,
                cross_distance_cutoff,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )

        src, dst = edge_index
        edge_vec = data["receptor"].pos[dst.long()] - data["ligand"].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["ligand"].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat(
            [
                data["ligand"].batch.unsqueeze(0),
                torch.arange(len(data["ligand"].batch)).to(data["ligand"].x.device).unsqueeze(0),
            ],
            dim=0,
        )

        center_pos = torch.zeros((data.num_graphs, 3)).to(data["ligand"].x.device)

        center_pos.index_add_(0, index=data["ligand"].batch, source=data["ligand"].pos)
        center_pos = center_pos / torch.bincount(data["ligand"].batch).unsqueeze(1)

        edge_vec = data["ligand"].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["ligand"].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data["ligand", "ligand"].edge_index[:, data["ligand"].edge_mask].long()
        bond_pos = (data["ligand"].pos[bonds[0]] + data["ligand"].pos[bonds[1]]) / 2
        bond_batch = data["ligand"].batch[bonds[0]]
        edge_index = radius(
            data["ligand"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=data["ligand"].batch,
            batch_y=bond_batch,
        )

        edge_vec = data["ligand"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        return bonds, edge_index, edge_attr, edge_sh
