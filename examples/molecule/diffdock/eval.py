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


"""
Entry point to DiffDock: evaluating generated ligand poses.

modify parameters from conf/*.yaml
"""

import os

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

from bionemo.data.diffdock.inference import compute_performance_metrics, compute_rmsd
from bionemo.model.molecule.diffdock.utils.utils import read_strings_from_txt


@hydra_runner(config_path="conf", config_name="eval")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    logging.info("Reading paths and names.")

    if cfg.get("protein_ligand_csv") is not None:
        df = pd.read_csv(cfg.protein_ligand_csv)
        names = df["complex_name"].tolist()
        ligand_description_list = df["ligand_description"].tolist()
        protein_path_list = df["protein_path"].tolist()
    else:
        names = read_strings_from_txt(cfg.data.split_test)
        ligand_description_list = [None for _ in names]
        protein_path_list = [None for _ in names]
    names_no_rec_overlap = read_strings_from_txt(cfg.names_no_rec_overlap)
    os.makedirs(cfg.results_path, exist_ok=True)
    results_path_containments = os.listdir(cfg.results_path)

    successful_names_list = []
    rmsds_list = []
    centroid_distances_list = []
    min_cross_distances_list = []
    min_self_distances_list = []
    without_rec_overlap_list = []

    for i, name in enumerate(tqdm(names)):
        # read the reference and predicted ligand molecules and compute RMSDs.
        try:
            rmsd, ligand_pos, orig_ligand_pos = compute_rmsd(
                cfg, name, results_path_containments, ref_ligand=ligand_description_list[i]
            )
        except Exception as e:
            logging.warning(f"{name} failed because of error:\n{e}")
            continue

        if rmsd is None:
            continue

        rmsds_list.append(rmsd)
        centroid_distances_list.append(
            np.linalg.norm(ligand_pos.mean(axis=1) - orig_ligand_pos[None, :].mean(axis=1), axis=1)
        )

        if protein_path_list[i] is not None:
            rec_path = protein_path_list[i]
        else:
            rec_path = os.path.join(cfg.data.data_dir, name, f"{name}_protein_processed.pdb")
            if not os.path.exists(rec_path):
                rec_path = os.path.join(cfg.data.data_dir, name, f"{name}_protein_obabel_reduce.pdb")
        rec = PandasPdb().read_pdb(rec_path)
        rec_df = rec.df["ATOM"]
        receptor_pos = rec_df[["x_coord", "y_coord", "z_coord"]].to_numpy().squeeze().astype(np.float32)
        receptor_pos = np.tile(receptor_pos, (cfg.num_predictions, 1, 1))

        cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
        self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
        self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
        min_cross_distances_list.append(np.min(cross_distances, axis=(1, 2)))
        min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))
        successful_names_list.append(name)
        without_rec_overlap_list.append(1 if name in names_no_rec_overlap else 0)

    # Compute the performance metrics for top 1, top 5, and top 10 predictions
    # Save results to cfg.results_path
    performance_metrics = compute_performance_metrics(
        cfg=cfg,
        rmsds_list=rmsds_list,
        min_cross_distances_list=min_cross_distances_list,
        min_self_distances_list=min_self_distances_list,
        centroid_distances_list=centroid_distances_list,
        successful_names_list=successful_names_list,
        without_rec_overlap_list=without_rec_overlap_list,
        save_results=True,
    )

    logging.info("Performance Metrics:")
    for k in performance_metrics:
        logging.info(f"{k}: {performance_metrics[k]}")
    logging.info(f"Results are in {cfg.results_path}")


if __name__ == "__main__":
    main()
