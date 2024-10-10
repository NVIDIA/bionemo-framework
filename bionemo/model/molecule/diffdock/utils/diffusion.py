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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bionemo.model.molecule.diffdock.utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from bionemo.model.molecule.diffdock.utils.torsion import modify_conformer_torsion_angles


def t_to_sigma(t_tr, t_rot, t_tor, cfg):
    tr_sigma = cfg.diffusion.tr_sigma_min ** (1 - t_tr) * cfg.diffusion.tr_sigma_max**t_tr
    rot_sigma = cfg.diffusion.rot_sigma_min ** (1 - t_rot) * cfg.diffusion.rot_sigma_max**t_rot
    tor_sigma = cfg.diffusion.tor_sigma_min ** (1 - t_tor) * cfg.diffusion.tor_sigma_max**t_tor
    return tr_sigma, rot_sigma, tor_sigma


def modify_conformer(data, tr_update, rot_update, torsion_updates):
    lig_center = torch.mean(data["ligand"].pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (data["ligand"].pos - lig_center) @ rot_mat.T + tr_update + lig_center

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(
            rigid_new_pos,
            data["ligand", "ligand"].edge_index.T[data["ligand"].edge_mask],
            (
                data["ligand"].mask_rotate
                if isinstance(data["ligand"].mask_rotate, np.ndarray)
                else data["ligand"].mask_rotate[0]
            ),
            torsion_updates,
        ).to(rigid_new_pos.device)
        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        data["ligand"].pos = aligned_flexible_pos
    else:
        data["ligand"].pos = rigid_new_pos
    return data


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == "sinusoidal":

        def emb_func(x):
            return sinusoidal_embedding(embedding_scale * x, embedding_dim)

    elif embedding_type == "fourier":
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplementedError
    return emb_func


class timestep_embedding(nn.Module):
    def __init__(self, embedding_type, embedding_dim, embedding_scale=10000):
        super(timestep_embedding, self).__init__()
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.embedding_scale = embedding_scale
        self.emb_func = get_timestep_embedding(embedding_type, embedding_dim, embedding_scale)

    def forward(self, *args, **kwargs):
        return self.emb_func(*args, **kwargs)

    def __getstate__(self):
        return {
            "embedding_type": self.embedding_type,
            "embedding_dim": self.embedding_dim,
            "embedding_scale": self.embedding_scale,
        }

    def __setstate__(self, d):
        super(timestep_embedding, self).__init__()
        self.embedding_type = d["embedding_type"]
        self.embedding_dim = d["embedding_dim"]
        self.embedding_scale = d["embedding_scale"]
        self.emb_func = get_timestep_embedding(**d)


def get_t_schedule(denoising_inference_steps):
    return np.linspace(1, 0, denoising_inference_steps + 1)[:-1]


def set_time(complex_graphs, t_tr, t_rot, t_tor, batchsize, all_atoms, device):
    complex_graphs["ligand"].node_t = {
        "tr": t_tr * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
        "rot": t_rot * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
        "tor": t_tor * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
    }
    complex_graphs["receptor"].node_t = {
        "tr": t_tr * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
        "rot": t_rot * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
        "tor": t_tor * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
    }
    complex_graphs.complex_t = {
        "tr": t_tr * torch.ones(batchsize).to(device),
        "rot": t_rot * torch.ones(batchsize).to(device),
        "tor": t_tor * torch.ones(batchsize).to(device),
    }
    if all_atoms:
        complex_graphs["atom"].node_t = {
            "tr": t_tr * torch.ones(complex_graphs["atom"].num_nodes).to(device),
            "rot": t_rot * torch.ones(complex_graphs["atom"].num_nodes).to(device),
            "tor": t_tor * torch.ones(complex_graphs["atom"].num_nodes).to(device),
        }
