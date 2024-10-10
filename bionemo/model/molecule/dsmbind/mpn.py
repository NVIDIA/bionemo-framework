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


"""A message passing neural network, which serves as the molecular graph embedding layer in DSMBind. This MPNEncoder is adapted from https://chemprop.readthedocs.io/en/v1.7.1/_modules/chemprop/models/mpn.html#MPNEncoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim
from chemprop.nn_utils import index_select_ND
from omegaconf.dictconfig import DictConfig


class MPNEncoder(nn.Module):
    def __init__(self, cfg_model: DictConfig):
        """
        Initialization of the message passing neural network.

        Args:
            cfg_model (DictConfig): model configurations.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()
        self.hidden_size = cfg_model.hidden_size
        self.depth = cfg_model.mpn_depth

        self.dropout_layer = nn.Dropout(cfg_model.dropout)
        self.act_func = nn.ReLU()

        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=False)

        w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=False)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, mol_graph: BatchMolGraph) -> torch.Tensor:
        """
        Forward pass of the molecular graph encoder layer.

        Args:
            mol_graph (BatchMolGraph): graph structure and featurization of a batch of molecules. Refer to https://chemprop.readthedocs.io/en/v1.7.1/features.html#chemprop.features.featurization.BatchMolGraph for details.
            get_components() will return the following components of the BatchMolGraph:
            f_atoms (torch.Tensor): atom features.
            f_bonds (torch.Tensor): bond features.
            a2b (torch.Tensor): mapping from atom index to incoming bond indices.
            b2a (torch.Tensor): mapping from bond index to the index of the atom the bond is coming from.
            b2revb (torch.Tensor): mapping from bond index to the index of the reverse bond.
            a_scope (list[Tuple]): list of tuples indicating (start_atom_index, num_atoms) for each molecule.
            b_scope (list[Tuple]): list of tuples indicating (start_bond_index, num_bonds) for each molecule.

        Returns:
            torch.Tensor: the output tensor with shape (batch_size, num_atoms, hidden_size).
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        # Get the device from an existing parameter
        device = next(self.parameters()).device

        f_atoms, f_bonds, a2b, b2a, b2revb = (
            f_atoms.to(device),
            f_bonds.to(device),
            a2b.to(device),
            b2a.to(device),
            b2revb.to(device),
        )

        # Input
        input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        mol_vecs = []
        L = max([a_size for (a_start, a_size) in a_scope])
        for i, (a_start, a_size) in enumerate(a_scope):
            h = atom_hiddens.narrow(0, a_start, a_size)
            h = F.pad(h, (0, 0, 0, L - a_size))
            mol_vecs.append(h)

        return torch.stack(mol_vecs, dim=0)
