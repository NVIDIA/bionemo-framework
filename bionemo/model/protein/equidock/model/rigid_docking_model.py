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

import dgl
import torch
from nemo.core import NeuralModule
from nemo.utils import logging

from bionemo.model.protein.equidock.model.block import IEGMN


class Rigid_Body_Docking_Net(NeuralModule):
    def __init__(self, args):
        """
        Implementation of Independent SE(3)-Equivariant Models for
        End-to-End Rigid Protein Docking
        """

        super(Rigid_Body_Docking_Net, self).__init__()

        self.debug = args["debug"]

        self.iegmn_original = IEGMN(args, n_lays=args["iegmn_n_lays"], fine_tune=False)
        if args["fine_tune"]:
            self.iegmn_fine_tune = IEGMN(args, n_lays=2, fine_tune=True)
            self.list_iegmns = [("original", self.iegmn_original), ("finetune", self.iegmn_fine_tune)]
        else:
            self.list_iegmns = [("finetune", self.iegmn_original)]  # just original

    # self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.0)
            else:
                torch.nn.init.zeros_(p)

    # FORWARD for Rigid_Body_Docking_Net
    def forward(self, batch_hetero_graph):
        last_outputs = None
        all_ligand_coors_deform_list = []

        device = batch_hetero_graph.device

        for stage, iegmn in self.list_iegmns:
            outputs = iegmn(batch_hetero_graph)
            assert len(outputs) == 4

            if stage == "finetune":
                last_outputs = outputs

            list_hetero_graph = dgl.unbatch(batch_hetero_graph)
            if stage == "original":
                new_list_hetero_graph = []

            for the_idx, hetero_graph in enumerate(list_hetero_graph):
                orig_coors_ligand = hetero_graph.nodes["ligand"].data["new_x"]
                # orig_coors_receptor = hetero_graph.nodes['receptor'].data['x']

                T_align = outputs[0][the_idx]
                b_align = outputs[1][the_idx]
                assert b_align.shape[0] == 1 and b_align.shape[1] == 3

                inner_coors_ligand = (T_align @ orig_coors_ligand.t()).t() + b_align  # (n,3)

                if stage == "original":
                    hetero_graph.nodes["ligand"].data["new_x"] = inner_coors_ligand
                    new_list_hetero_graph.append(hetero_graph)

                if self.debug:
                    logging.info("\n\n* DEBUG MODE *")
                    logging.info("T_align", T_align)
                    logging.info("T_align @ T_align.t() - eye(3)", T_align @ T_align.t() - torch.eye(3).to(device))
                    logging.info("b_align", b_align)
                    logging.info(
                        "\n ---> inner_coors_ligand mean - true ligand mean ",
                        inner_coors_ligand.mean(dim=0) - hetero_graph.nodes["ligand"].data["x"].mean(dim=0),
                        "\n",
                    )

                if stage == "finetune":
                    all_ligand_coors_deform_list.append(inner_coors_ligand)

            if stage == "original":
                batch_hetero_graph = dgl.batch(new_list_hetero_graph)

        all_keypts_ligand_list = last_outputs[2]
        all_keypts_receptor_list = last_outputs[3]
        all_rotation_list = last_outputs[0]
        all_translation_list = last_outputs[1]
        # TODO: write a test to make sure invariances are perserved

        return (
            all_ligand_coors_deform_list,
            all_keypts_ligand_list,
            all_keypts_receptor_list,
            all_rotation_list,
            all_translation_list,
        )
