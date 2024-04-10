# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import warnings

import dgl
import torch
import torch.multiprocessing


warnings.filterwarnings("ignore", category=FutureWarning)


def hetero_graph_from_sg_l_r_pair(ligand_graph, receptor_graph):
    ll = [('ligand', 'll', 'ligand'), (ligand_graph.edges()[0], ligand_graph.edges()[1])]
    rr = [('receptor', 'rr', 'receptor'), (receptor_graph.edges()[0], receptor_graph.edges()[1])]
    rl = [('receptor', 'cross', 'ligand'), (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    lr = [('ligand', 'cross', 'receptor'), (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    num_nodes = {'ligand': ligand_graph.num_nodes(), 'receptor': receptor_graph.num_nodes()}
    hetero_graph = dgl.heterograph({ll[0]: ll[1], rr[0]: rr[1], rl[0]: rl[1], lr[0]: lr[1]}, num_nodes_dict=num_nodes)

    hetero_graph.nodes['ligand'].data['res_feat'] = ligand_graph.ndata['res_feat']
    hetero_graph.nodes['ligand'].data['x'] = ligand_graph.ndata['x']
    hetero_graph.nodes['ligand'].data['new_x'] = ligand_graph.ndata['new_x']
    hetero_graph.nodes['ligand'].data['mu_r_norm'] = ligand_graph.ndata['mu_r_norm']

    hetero_graph.edges['ll'].data['he'] = ligand_graph.edata['he']

    hetero_graph.nodes['receptor'].data['res_feat'] = receptor_graph.ndata['res_feat']
    hetero_graph.nodes['receptor'].data['x'] = receptor_graph.ndata['x']
    hetero_graph.nodes['receptor'].data['mu_r_norm'] = receptor_graph.ndata['mu_r_norm']

    hetero_graph.edges['rr'].data['he'] = receptor_graph.edata['he']
    return hetero_graph


def batchify_and_create_hetero_graphs(data):
    (
        ligand_graph_list,
        receptor_graph_list,
        bound_ligand_repres_nodes_loc_array_list,
        bound_receptor_repres_nodes_loc_array_list,
        pocket_coors_ligand_list,
        pocket_coors_receptor_list,
    ) = map(list, zip(*data))

    hetero_graph_list = []
    for i, ligand_graph in enumerate(ligand_graph_list):
        receptor_graph = receptor_graph_list[i]
        hetero_graph = hetero_graph_from_sg_l_r_pair(ligand_graph, receptor_graph)
        hetero_graph_list.append(hetero_graph)

    batch_hetero_graph = dgl.batch(hetero_graph_list)
    return (
        batch_hetero_graph,
        bound_ligand_repres_nodes_loc_array_list,
        bound_receptor_repres_nodes_loc_array_list,
        pocket_coors_ligand_list,
        pocket_coors_receptor_list,
    )


def batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph):
    hetero_graph_list = []
    hetero_graph = hetero_graph_from_sg_l_r_pair(ligand_graph, receptor_graph)
    hetero_graph_list.append(hetero_graph)
    batch_hetero_graph = dgl.batch(hetero_graph_list)
    return batch_hetero_graph
