# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import random
from copy import deepcopy
from typing import Callable, Generator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data.hetero_data import HeteroData

from bionemo.contrib.model.molecule.diffdock.utils import so3, torus
from bionemo.contrib.model.molecule.diffdock.utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from bionemo.contrib.model.molecule.diffdock.utils.torsion import modify_conformer_torsion_angles


def t_to_sigma(tr_sigma_min, tr_sigma_max, rot_sigma_min,
               rot_sigma_max, tor_sigma_min, tor_sigma_max,
               t_tr, t_rot, t_tor):
    tr_sigma = tr_sigma_min ** (1 - t_tr) * tr_sigma_max**t_tr
    rot_sigma = rot_sigma_min ** (1 - t_rot) * rot_sigma_max**t_rot
    tor_sigma = tor_sigma_min ** (1 - t_tor) * tor_sigma_max**t_tor
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


class GenerateNoise:
    """Apply forward diffusion on the ligand

    Args:
        t_to_sigma (Callable): Callable to embed time
        no_torsion (bool): if not to perturb ligand torsion degrees
        all_atom (bool): all atom or coarse grained/residue for protein
        copy_ref_pos (bool): whether or not make a copy of the input ligand position
    """

    def __init__(self, t_to_sigma: Callable[[float, float, float], Tuple[float,
                                                                         float,
                                                                         float]],
                 no_torsion: bool, all_atom: bool, copy_ref_pos: bool = False):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom
        self._copy_ref_pos = copy_ref_pos

    def __call__(self, source : Generator[HeteroData, None, None]) \
            -> Generator[HeteroData, None, None]:
        for (data,) in source:
            if self._copy_ref_pos:
                data["ligand"].aligned_pos = deepcopy(data["ligand"].pos)
            t = np.random.uniform()
            t_tr, t_rot, t_tor = t, t, t
            yield self.apply_noise(data, t_tr, t_rot, t_tor)

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
