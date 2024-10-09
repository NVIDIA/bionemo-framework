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


"""Implementation of gaussian noise based DSMBind model (https://www.biorxiv.org/content/10.1101/2023.12.10.570461v1) for binding energy prediction."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from chemprop.features import mol2graph
from omegaconf.dictconfig import DictConfig
from rdkit.Chem import Mol

from bionemo.model.molecule.dsmbind.fann import FANN
from bionemo.model.molecule.dsmbind.mpn import MPNEncoder


class DSMBind(nn.Module):
    def __init__(self, cfg_model: DictConfig):
        """
        Initialization of the DSMBind model.

        Args:
            cfg_model (DictConfig): model configurations.
        """
        super(DSMBind, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.threshold = cfg_model.threshold
        self.max_residue_atoms = cfg_model.max_residue_atoms
        self.mpn = MPNEncoder(cfg_model)
        self.encoder = FANN(cfg_model)
        self.binder_output = nn.Sequential(
            nn.Linear(cfg_model.hidden_size, cfg_model.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg_model.hidden_size, cfg_model.hidden_size),
        )
        self.target_output = nn.Sequential(
            nn.Linear(cfg_model.hidden_size, cfg_model.hidden_size),
            nn.SiLU(),
        )

    def forward(
        self,
        binder: Tuple[torch.Tensor, List[Mol], torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Define the forward of the energy-based model, and compute the denoising score matching loss.

        Args:
            binder (Tuple[torch.Tensor, List[Mol], torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second list is a list of RDKit molecules. The third tensor is a mask for indicating ligand atoms. Refer to the data/dsmbind/dataset.py to see how they are built and batched.
            target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the target atom coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all target atoms in the pocket. Refer to the data/dsmbind/dataset.py to see how they are built and batched.

        Returns:
            torch.Tensor: the denoising score matching loss value.
        """
        true_X, mol_batch, bind_A = binder
        tgt_X, tgt_S, tgt_A = target
        bind_S = self.mpn(mol2graph(mol_batch))

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        eps = np.random.uniform(0.1, 1.0, size=B)
        eps = torch.tensor(eps, dtype=torch.float, device=true_X.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        hat_t = torch.randn_like(true_X).to(true_X.device) * eps
        bind_X = true_X + hat_t * atom_mask
        bind_X = bind_X.requires_grad_()

        # Get the contact map
        mask_2D = (bind_A > 0).float().view(B, N * self.max_residue_atoms, 1) * (tgt_A > 0).float().view(
            B, 1, M * self.max_residue_atoms
        )
        dist = (
            bind_X.view(B, N * self.max_residue_atoms, 1, 3) - tgt_X.view(B, 1, M * self.max_residue_atoms, 3)
        ).norm(dim=-1)  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        mask_2D = mask_2D * (dist < self.threshold).float()

        # Compute the energy
        h = self.encoder(
            (bind_X, bind_S, bind_A),
            (tgt_X, tgt_S, tgt_A),
        )  # [B,N+M,self.max_residue_atoms,H]
        bind_h = self.binder_output(h[:, :N]).view(B, N * self.max_residue_atoms, -1)
        tgt_h = self.target_output(h[:, N:]).view(B, M * self.max_residue_atoms, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1, 2))  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        energy = (energy * mask_2D).sum(dim=(1, 2))  # [B]

        # Compute the DSM loss
        f_bind = torch.autograd.grad(energy.sum(), bind_X, create_graph=True, retain_graph=True)[0]
        loss = (hat_t / eps - f_bind * eps) * atom_mask
        return loss.sum() / atom_mask.sum()

    def predict(
        self,
        binder: Tuple[torch.Tensor, List[Mol], torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        The forward pass used for inference.

        Args:
            binder (Tuple[torch.Tensor, List[Mol], torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second list is a list of RDKit molecules. The third tensor is a mask for indicating ligand atoms. Refer to the data/dsmbind/dataset.py to see how they are built and batched.
            target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the target atom coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all target atoms in the pocket. Refer to the data/dsmbind/dataset.py to see how they are built and batched.

        Returns:
            torch.Tensor: the predicted energy.
        """
        bind_X, mol_batch, bind_A = binder
        tgt_X, tgt_S, tgt_A = target
        bind_S = self.mpn(mol2graph(mol_batch))

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A[:, :, 1].clamp(max=1).float()
        tgt_A[:, :, 1].clamp(max=1).float()
        bind_A = bind_A * (bind_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()

        mask_2D = (bind_A > 0).float().view(B, N * self.max_residue_atoms, 1) * (tgt_A > 0).float().view(
            B, 1, M * self.max_residue_atoms
        )
        dist = (
            bind_X.view(B, N * self.max_residue_atoms, 1, 3) - tgt_X.view(B, 1, M * self.max_residue_atoms, 3)
        ).norm(dim=-1)  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        mask_2D = mask_2D * (dist < self.threshold).float()

        h = self.encoder(
            (bind_X, bind_S, bind_A),
            (tgt_X, tgt_S, tgt_A),
        )  # [B,N+M,self.max_residue_atoms,H]

        bind_h = self.binder_output(h[:, :N]).view(B, N * self.max_residue_atoms, -1)
        tgt_h = self.target_output(h[:, N:]).view(B, M * self.max_residue_atoms, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1, 2))  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        return (energy * mask_2D).sum(dim=(1, 2))  # [B]
