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


from functools import partial

import torch
from cugraph_equivariant.nn import FullyConnectedTensorProductConv
from e3nn import o3
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter_mean

from bionemo.data.diffdock.process_mols import (
    lig_feature_dims,
    rec_atom_feature_dims,
    rec_residue_feature_dims,
)
from bionemo.model.molecule.diffdock.models.common_blocks import AtomEncoder, GaussianSmearing
from bionemo.model.molecule.diffdock.utils import (
    so3,
    torus,
)
from bionemo.model.molecule.diffdock.utils.batchnorm import BatchNorm
from bionemo.model.molecule.diffdock.utils.diffusion import (
    get_timestep_embedding,
)
from bionemo.model.molecule.diffdock.utils.diffusion import (
    t_to_sigma as t_to_sigma_compl,
)


class TensorProductScoreModelAllAtom(torch.nn.Module):
    def __init__(self, cfg: OmegaConf):
        super(TensorProductScoreModelAllAtom, self).__init__()

        timestep_emb_func = get_timestep_embedding(
            embedding_type=cfg.embedding_type,
            embedding_dim=cfg.sigma_embed_dim,
            embedding_scale=cfg.embedding_scale,
        )
        sh_lmax = 2
        confidence_no_batchnorm = False
        confidence_dropout = 0
        batch_norm = not cfg.no_batch_norm
        dropout = cfg.dropout
        use_second_order_repr = cfg.tensor_product.use_second_order_repr
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
            nn.Linear(ns, ns),
        )

        self.rec_node_embedding = AtomEncoder(
            emb_dim=self.ns,
            feature_dims=rec_residue_feature_dims,
            sigma_embed_dim=self.sigma_embed_dim,
            lm_embedding_type=lm_embedding_type,
        )
        self.rec_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        # for all-atoms only.
        self.atom_node_embedding = AtomEncoder(
            emb_dim=ns,
            feature_dims=rec_atom_feature_dims,
            sigma_embed_dim=self.sigma_embed_dim,
        )
        self.atom_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.lr_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.cross_distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.ar_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.la_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.cross_distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
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

        # convolutional layers
        conv_layers = []
        for i in range(self.num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                "in_irreps": o3.Irreps(in_irreps),
                "sh_irreps": self.sh_irreps,
                "out_irreps": o3.Irreps(out_irreps),
                "batch_norm": False,
                "mlp_channels": [3 * ns, 3 * ns],
                "mlp_activation": nn.Sequential(nn.ReLU(), nn.Dropout(dropout)),
                "e3nn_compat_mode": True,
            }

            for _ in range(9):  # 3 intra & 6 inter per each layer
                conv_layers.append(FullyConnectedTensorProductConv(**parameters))

        self.conv_layers = nn.ModuleList(conv_layers)

        if batch_norm:
            batch_norm_layers = []
            for i in range(len(conv_layers)):
                batch_norm_layers.append(BatchNorm(conv_layers[i].out_irreps, with_shift=True))
            self.batch_norm_layers = nn.ModuleList(batch_norm_layers)
        else:
            self.batch_norm_layers = None

        # confidence and affinity prediction layers
        if self.confidence_mode:
            output_confidence_dim = num_confidence_outputs

            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if self.num_conv_layers >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, output_confidence_dim),
            )

        else:
            # convolution for translational and rotational scores
            self.center_distance_expansion = GaussianSmearing(0.0, self.center_max_distance, self.distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(self.distance_embed_dim + self.sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns),
            )

            self.final_conv = FullyConnectedTensorProductConv(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=o3.Irreps("2x1o + 2x1e"),
                batch_norm=False,
                mlp_channels=[2 * ns, 2 * ns],
                mlp_activation=nn.Sequential(nn.ReLU(), nn.Dropout(dropout)),
                e3nn_compat_mode=True,
            )
            if batch_norm:
                self.final_batch_norm = BatchNorm(self.final_conv.out_irreps, with_shift=True)
            else:
                self.final_batch_norm = None

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
                # convolution for torsional score
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(self.distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns),
                )
                self.final_tp_tor = o3.FullTensorProduct(
                    self.sh_irreps, "2e", _optimize_einsums=cfg.get("optimize_einsums", None)
                )
                self.tor_bond_conv = FullyConnectedTensorProductConv(
                    in_irreps=self.conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=o3.Irreps(f"{ns}x0o + {ns}x0e"),
                    batch_norm=False,
                    mlp_channels=[3 * ns, 3 * ns],
                    mlp_activation=nn.Sequential(nn.ReLU(), nn.Dropout(dropout)),
                    e3nn_compat_mode=True,
                )
                if batch_norm:
                    self.tor_bond_batch_norm = BatchNorm(self.tor_bond_conv.out_irreps, with_shift=True)
                else:
                    self.tor_bond_batch_norm = None
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False),
                )

            self.torus_score_norm = torus.TorusScoreNorm(cfg.seed)

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
        lig_src_ids, lig_dst_ids = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_node_attr_scalars = lig_node_attr[:, : self.ns]
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        (
            rec_node_attr,
            rec_edge_index,
            rec_edge_attr,
            rec_edge_sh,
        ) = self.build_rec_conv_graph(data)
        rec_src_ids, rec_dst_ids = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_node_attr_scalars = rec_node_attr[:, : self.ns]
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build atom graph
        (
            atom_node_attr,
            atom_edge_index,
            atom_edge_attr,
            atom_edge_sh,
        ) = self.build_atom_conv_graph(data)
        atom_src_ids, atom_dst_ids = atom_edge_index
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_node_attr_scalars = atom_node_attr[:, : self.ns]
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

        # build cross graph
        cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1) if self.dynamic_max_cross else self.cross_max_distance
        (
            lr_edge_index,
            lr_edge_attr,
            lr_edge_sh,
            la_edge_index,
            la_edge_attr,
            la_edge_sh,
            ar_edge_index,
            ar_edge_attr,
            ar_edge_sh,
        ) = self.build_cross_conv_graph(data, cross_cutoff)
        atom_ar_ids, rec_ar_ids = ar_edge_index
        lig_lr_ids, rec_lr_ids = lr_edge_index
        lig_la_ids, atom_la_ids = la_edge_index
        ra_edge_index = ar_edge_index.flip(dims=(0,))
        rl_edge_index = lr_edge_index.flip(dims=(0,))
        al_edge_index = la_edge_index.flip(dims=(0,))

        lr_edge_attr = self.lr_edge_embedding(lr_edge_attr)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)
        ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)

        for l in range(self.num_conv_layers):
            # LIGAND updates
            lig_edge_node_attr = torch.hstack(
                (lig_edge_attr, lig_node_attr_scalars[lig_src_ids], lig_node_attr_scalars[lig_dst_ids])
            )
            lig_update = self.conv_layers[9 * l](
                lig_node_attr,
                lig_edge_sh,
                lig_edge_node_attr,
                (lig_edge_index, (lig_node_attr.shape[0], lig_node_attr.shape[0])),
            )
            if self.batch_norm_layers is not None:
                lig_update = self.batch_norm_layers[9 * l](lig_update)

            rl_edge_node_attr = torch.hstack(
                (lr_edge_attr, rec_node_attr_scalars[rec_lr_ids], lig_node_attr_scalars[lig_lr_ids])
            )
            lr_update = self.conv_layers[9 * l + 1](
                rec_node_attr,
                lr_edge_sh,
                rl_edge_node_attr,
                (rl_edge_index, (rec_node_attr.shape[0], lig_node_attr.shape[0])),
            )
            if self.batch_norm_layers is not None:
                lr_update = self.batch_norm_layers[9 * l + 1](lr_update)

            al_edge_node_attr = torch.hstack(
                (la_edge_attr, atom_node_attr_scalars[atom_la_ids], lig_node_attr_scalars[lig_la_ids])
            )
            la_update = self.conv_layers[9 * l + 2](
                atom_node_attr,
                la_edge_sh,
                al_edge_node_attr,
                (al_edge_index, (atom_node_attr.shape[0], lig_node_attr.shape[0])),
            )
            if self.batch_norm_layers is not None:
                la_update = self.batch_norm_layers[9 * l + 2](la_update)

            if l != self.num_conv_layers - 1:  # last layer optimisation
                # ATOM UPDATES
                atom_edge_node_attr = torch.hstack(
                    (atom_edge_attr, atom_node_attr_scalars[atom_src_ids], atom_node_attr_scalars[atom_dst_ids])
                )
                atom_update = self.conv_layers[9 * l + 3](
                    atom_node_attr,
                    atom_edge_sh,
                    atom_edge_node_attr,
                    (atom_edge_index, (atom_node_attr.shape[0], atom_node_attr.shape[0])),
                )
                if self.batch_norm_layers is not None:
                    atom_update = self.batch_norm_layers[9 * l + 3](atom_update)

                la_edge_node_attr = torch.hstack(
                    (la_edge_attr, lig_node_attr_scalars[lig_la_ids], atom_node_attr_scalars[atom_la_ids])
                )
                al_update = self.conv_layers[9 * l + 4](
                    lig_node_attr,
                    la_edge_sh,
                    la_edge_node_attr,
                    (la_edge_index, (lig_node_attr.shape[0], atom_node_attr.shape[0])),
                )
                if self.batch_norm_layers is not None:
                    al_update = self.batch_norm_layers[9 * l + 4](al_update)

                ra_edge_node_attr = torch.hstack(
                    (ar_edge_attr, rec_node_attr_scalars[rec_ar_ids], atom_node_attr_scalars[atom_ar_ids])
                )
                ar_update = self.conv_layers[9 * l + 5](
                    rec_node_attr,
                    ar_edge_sh,
                    ra_edge_node_attr,
                    (ra_edge_index, (rec_node_attr.shape[0], atom_node_attr.shape[0])),
                )
                if self.batch_norm_layers is not None:
                    ar_update = self.batch_norm_layers[9 * l + 5](ar_update)

                # RECEPTOR updates
                rec_edge_node_attr = torch.hstack(
                    (rec_edge_attr, rec_node_attr_scalars[rec_src_ids], rec_node_attr_scalars[rec_dst_ids])
                )
                rec_update = self.conv_layers[9 * l + 6](
                    rec_node_attr,
                    rec_edge_sh,
                    rec_edge_node_attr,
                    (rec_edge_index, (rec_node_attr.shape[0], rec_node_attr.shape[0])),
                )
                if self.batch_norm_layers is not None:
                    rec_update = self.batch_norm_layers[9 * l + 6](rec_update)

                lr_edge_node_attr = torch.hstack(
                    (lr_edge_attr, lig_node_attr_scalars[lig_lr_ids], rec_node_attr_scalars[rec_lr_ids])
                )
                rl_update = self.conv_layers[9 * l + 7](
                    lig_node_attr,
                    lr_edge_sh,
                    lr_edge_node_attr,
                    (lr_edge_index, (lig_node_attr.shape[0], rec_node_attr.shape[0])),
                )
                if self.batch_norm_layers is not None:
                    rl_update = self.batch_norm_layers[9 * l + 7](rl_update)

                ar_edge_node_attr = torch.hstack(
                    (ar_edge_attr, atom_node_attr_scalars[atom_ar_ids], rec_node_attr_scalars[rec_ar_ids])
                )
                ra_update = self.conv_layers[9 * l + 8](
                    atom_node_attr,
                    ar_edge_sh,
                    ar_edge_node_attr,
                    (ar_edge_index, (atom_node_attr.shape[0], rec_node_attr.shape[0])),
                )
                if self.batch_norm_layers is not None:
                    ra_update = self.batch_norm_layers[9 * l + 8](ra_update)

            # padding original features and update features with residual updates
            lig_node_attr = F.pad(lig_node_attr, (0, lig_update.shape[-1] - lig_node_attr.shape[-1]))
            lig_node_attr = lig_node_attr + lig_update + la_update + lr_update
            lig_node_attr_scalars = lig_node_attr[:, : self.ns]

            if l != self.num_conv_layers - 1:  # last layer optimisation
                atom_node_attr = F.pad(atom_node_attr, (0, atom_update.shape[-1] - rec_node_attr.shape[-1]))
                atom_node_attr = atom_node_attr + atom_update + al_update + ar_update
                atom_node_attr_scalars = atom_node_attr[:, : self.ns]
                rec_node_attr = F.pad(rec_node_attr, (0, rec_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_update + ra_update + rl_update
                rec_node_attr_scalars = rec_node_attr[:, : self.ns]

        # confidence and affinity prediction
        if self.confidence_mode:
            scalar_lig_attr = (
                torch.cat([lig_node_attr[:, : self.ns], lig_node_attr[:, -self.ns :]], dim=1)
                if self.num_conv_layers >= 3
                else lig_node_attr[:, : self.ns]
            )
            confidence = self.confidence_predictor(
                scatter_mean(scalar_lig_attr, data["ligand"].batch, dim=0, dim_size=data.num_graphs)
            ).squeeze(dim=-1)
            return confidence

        # compute translational and rotational score vectors
        (
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
        ) = self.build_center_conv_graph(data)
        lig_c_ids = center_edge_index[0]
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_node_attr = torch.hstack((center_edge_attr, lig_node_attr_scalars[lig_c_ids]))
        global_pred = self.final_conv(
            lig_node_attr,
            center_edge_sh,
            center_edge_node_attr,
            (center_edge_index, (lig_node_attr.shape[0], data.num_graphs)),
        )
        if self.final_batch_norm is not None:
            global_pred = self.final_batch_norm(global_pred)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t["tr"])

        # adjust the magnitude of the score vectors
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
            lig_bond_edge_index,
            tor_edge_attr,
            tor_edge_sh,
        ) = self.build_bond_conv_graph(data)
        lig_b_ids, bond_b_ids = lig_bond_edge_index
        tor_bond_vec = data["ligand"].pos[tor_bonds[1]] - data["ligand"].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization="component")
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[bond_b_ids])

        tor_bond_attr_scalars_edge = tor_bond_attr[bond_b_ids, : self.ns]
        tor_edge_node_attr = torch.hstack(
            (tor_edge_attr, lig_node_attr_scalars[lig_b_ids], tor_bond_attr_scalars_edge)
        )
        tor_pred = self.tor_bond_conv(
            lig_node_attr,
            tor_edge_sh,
            tor_edge_node_attr,
            (lig_bond_edge_index, (lig_node_attr.shape[0], data["ligand"].edge_mask.sum())),
        )
        if self.tor_bond_batch_norm is not None:
            tor_pred = self.tor_bond_batch_norm(tor_pred)
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data["ligand"].batch][data["ligand", "ligand"].edge_index[0]][data["ligand"].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(
                torch.tensor(self.torus_score_norm(edge_sigma.cpu().numpy())).float().to(data["ligand"].x.device)
            )
        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data):
        # build the graph between ligand atoms
        data["ligand"].node_sigma_emb = self.timestep_emb_func(data["ligand"].node_t["tr"])

        radius_edges = radius_graph(data["ligand"].pos, self.lig_max_radius, data["ligand"].batch)
        edge_index = torch.cat([data["ligand", "ligand"].edge_index, radius_edges], 1).long()
        src, dst = edge_index.long()
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

        edge_sigma_emb = data["ligand"].node_sigma_emb[src]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data["ligand"].x, data["ligand"].node_sigma_emb], 1)

        edge_vec = data["ligand"].pos[dst] - data["ligand"].pos[src]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        return node_attr, edge_index.flip(dims=(0,)), edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # build the graph between receptor residues
        data["receptor"].node_sigma_emb = self.timestep_emb_func(data["receptor"].node_t["tr"])
        node_attr = torch.cat([data["receptor"].x, data["receptor"].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data["receptor", "receptor"].edge_index.long()
        src, dst = edge_index
        edge_vec = data["receptor"].pos[dst] - data["receptor"].pos[src]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["receptor"].node_sigma_emb[edge_index[0]]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        return node_attr, edge_index.flip(dims=(0,)), edge_attr, edge_sh

    def build_atom_conv_graph(self, data):
        # build the graph between receptor atoms
        data["atom"].node_sigma_emb = self.timestep_emb_func(data["atom"].node_t["tr"])
        node_attr = torch.cat([data["atom"].x, data["atom"].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data["atom", "atom"].edge_index.long()
        src, dst = edge_index
        edge_vec = data["atom"].pos[dst] - data["atom"].pos[src]

        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["atom"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        return node_attr, edge_index.flip(dims=(0,)), edge_attr, edge_sh

    def build_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        # build the cross edges between ligan atoms, receptor residues and receptor atoms

        # LIGAND to RECEPTOR
        if torch.is_tensor(lr_cross_distance_cutoff):
            # different cutoff for every graph
            lr_edge_index = radius(
                data["receptor"].pos / lr_cross_distance_cutoff[data["receptor"].batch],
                data["ligand"].pos / lr_cross_distance_cutoff[data["ligand"].batch],
                1,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )
        else:
            lr_edge_index = radius(
                data["receptor"].pos,
                data["ligand"].pos,
                lr_cross_distance_cutoff,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )

        lig_edge_id, rec_edge_id = lr_edge_index.long()

        lr_edge_vec = data["receptor"].pos[rec_edge_id] - data["ligand"].pos[lig_edge_id]
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        lr_edge_sigma_emb = data["ligand"].node_sigma_emb[lig_edge_id]
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], 1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization="component")

        # LIGAND to ATOM
        la_edge_index = radius(
            data["atom"].pos,
            data["ligand"].pos,
            self.lig_max_radius,
            data["atom"].batch,
            data["ligand"].batch,
            max_num_neighbors=10000,
        ).long()

        lig_edge_id, atom_edge_id = la_edge_index

        la_edge_vec = data["atom"].pos[atom_edge_id] - data["ligand"].pos[lig_edge_id]
        la_edge_length_emb = self.cross_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data["ligand"].node_sigma_emb[lig_edge_id]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization="component")

        # ATOM to RECEPTOR
        ar_edge_index = data["atom", "receptor"].edge_index.long()
        ar_edge_vec = data["receptor"].pos[ar_edge_index[1]] - data["atom"].pos[ar_edge_index[0]]
        ar_edge_length_emb = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sigma_emb = data["atom"].node_sigma_emb[ar_edge_index[0]]
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], 1)
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization="component")

        return (
            lr_edge_index,
            lr_edge_attr,
            lr_edge_sh,
            la_edge_index,
            la_edge_attr,
            la_edge_sh,
            ar_edge_index,
            ar_edge_attr,
            ar_edge_sh,
        )

    def build_center_conv_graph(self, data):
        # build the filter for the convolution of the center with the ligand atoms
        # for translational and rotational score
        edge_index = torch.cat(
            [
                data["ligand"].batch.unsqueeze(0),
                torch.arange(len(data["ligand"].batch)).to(data["ligand"].x.device).unsqueeze(0),
            ],
            dim=0,
        ).long()

        center_edge_id, lig_edge_id = edge_index

        center_pos = torch.zeros((data.num_graphs, 3)).to(data["ligand"].x.device)

        center_pos.index_add_(0, index=data["ligand"].batch, source=data["ligand"].pos)
        center_pos = center_pos / torch.bincount(data["ligand"].batch).unsqueeze(1)

        edge_vec = data["ligand"].pos[lig_edge_id] - center_pos[center_edge_id]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["ligand"].node_sigma_emb[lig_edge_id]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        return edge_index.flip(dims=(0,)), edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # build graph for the pseudotorque layer
        bonds = data["ligand", "ligand"].edge_index[:, data["ligand"].edge_mask].long()
        bond_pos = (data["ligand"].pos[bonds[0]] + data["ligand"].pos[bonds[1]]) / 2
        bond_batch = data["ligand"].batch[bonds[0]]
        edge_index = radius(
            data["ligand"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=data["ligand"].batch,
            batch_y=bond_batch,
        ).long()

        edge_vec = data["ligand"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        return bonds, edge_index.flip(dims=(0,)), edge_attr, edge_sh
