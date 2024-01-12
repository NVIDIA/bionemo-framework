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

"""
Entry point to EquiDock.

modify parameters from conf/*.cfg
"""
import os
import pathlib
import tempfile
import time

import numpy as np
import torch
from biopandas.pdb import PandasPdb
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import bionemo
from bionemo.data.equidock.protein_utils import (
    extract_to_dir,
)
from bionemo.model.protein.equidock.infer import EquiDockInference
from bionemo.model.protein.equidock.utils.train_utils import batchify_and_create_hetero_graphs_inference


os.environ['DGLBACKEND'] = 'pytorch'
torch.set_float32_matmul_precision("high")
BIONEMO_ROOT = pathlib.Path(bionemo.__file__).parent.parent.as_posix()


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    logging.info(f"\nTrain dataset name {cfg.data.data_name}")

    seed_everything(cfg.seed)

    #######################################
    # set data IO dirs
    #######################################
    output_dir = cfg.exp_manager.exp_dir
    #######################################
    # load model and chpt from cfg
    # model = EquiDockInference(cfg)
    #######################################
    logging.info("\n\n************** Loading EquiDockInference ***********")
    model = EquiDockInference(cfg=cfg)
    model.eval()
    data_dir = cfg.data.data_dir

    with tempfile.TemporaryDirectory() as temp_dir:
        # random transformed zip pdb directory
        extract_to_dir(os.path.join(data_dir, 'ligands.zip'), temp_dir)
        extract_to_dir(os.path.join(data_dir, 'receptors.zip'), temp_dir)

        pdb_files = [
            f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f)) and f.endswith('.pdb')
        ]
        pdb_files.sort()
        time_list = []

        for file in pdb_files:
            start = time.time()
            if not file.endswith('_l_b.pdb'):
                continue

            ll = len('_l_b.pdb')
            ligand_filename = os.path.join(temp_dir, f'{file[:-ll]}_l_b.pdb')
            receptor_filename = os.path.join(temp_dir, f'{file[:-ll]}_r_b.pdb')
            out_filename = f'{file[:-ll]}_l_b_COMPLEX.pdb'
            logging.info(f"Processing {out_filename} ...")

            # Create ligand and receptor graphs and arrays
            (
                ligand_graph,
                receptor_graph,
                bound_ligand_repres_nodes_loc_clean_array,
                _,
            ) = model.model.create_ligand_receptor_graphs_arrays(ligand_filename, receptor_filename, cfg.data)

            # Create a batch of a single DGL graph
            batch_hetero_graph = batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph)

            batch_hetero_graph = batch_hetero_graph.to(model.device)
            (
                model_ligand_coors_deform_list,
                _,
                _,
                all_rotation_list,
                all_translation_list,
            ) = model(batch_hetero_graph)

            # Get rotation and translation
            rotation = all_rotation_list[0].detach().cpu().numpy()
            translation = all_translation_list[0].detach().cpu().numpy()

            new_residues = (rotation @ bound_ligand_repres_nodes_loc_clean_array.T).T + translation
            assert np.linalg.norm(new_residues - model_ligand_coors_deform_list[0].detach().cpu().numpy()) < 1e-1

            # Get initial ligand position
            ppdb_ligand = PandasPdb().read_pdb(ligand_filename)
            unbound_ligand_all_atoms_pre_pos = (
                ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
            )
            unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T + translation

            ppdb_ligand.df['ATOM'][
                ['x_coord', 'y_coord', 'z_coord']
            ] = unbound_ligand_new_pos  # unbound_ligand_new_pos
            unbound_ligand_save_filename = os.path.join(output_dir, out_filename)
            ppdb_ligand.to_pdb(path=unbound_ligand_save_filename, records=['ATOM'], gz=False)

            time_list.append((time.time() - start))

        time_array = np.array(time_list)
        logging.info(f"Mean runtime: {np.mean(time_array)}, std runtime: {np.std(time_array)}")
        logging.info(f"\nResults are written into {output_dir}")


if __name__ == "__main__":
    main()
