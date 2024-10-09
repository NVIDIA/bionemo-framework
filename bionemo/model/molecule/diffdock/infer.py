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


import copy
import os
import time
from functools import partial
from typing import List, Tuple

import numpy as np
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning import LightningModule
from rdkit.Chem import RemoveHs
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from bionemo.data.diffdock.process_mols import write_mol_with_coords
from bionemo.model.molecule.diffdock.utils.diffusion import get_t_schedule
from bionemo.model.molecule.diffdock.utils.diffusion import t_to_sigma as t_to_sigma_compl
from bionemo.model.molecule.diffdock.utils.sampling import randomize_position, sampling
from bionemo.model.molecule.diffdock.utils.visualise import PDBFile
from bionemo.model.utils import (
    load_model_for_inference,
)


class DiffDockModelInference(LightningModule):
    """
    Base class for inference.
    """

    def __init__(self, cfg, interactive: bool = False):
        super().__init__()
        self.cfg = cfg
        self.model = load_model_for_inference(cfg, strict=False, interactive=interactive)
        # move self to same device as loaded model
        self.to(self.model.device)
        self._trainer = self.model.trainer

    def forward(self, batch):
        """Forward pass of the model"""
        return self.model.model.net(batch)


def do_inference_sampling(
    cfg: DictConfig,
    model: LightningModule,
    confidence_model: LightningModule,
    complex_name_list: List,
    test_loader: DataLoader,
    test_dataset: Dataset,
    confidence_test_dataset: Dataset,
) -> Tuple[int, int]:
    """Perform the inference, which as following steps
        1. randomize the initial ligand positions.
        2. Doing reverse diffusion sampling with the score model and get confidence scores for the generated ligand poses.
        3. Write out results.

    Args:
        cfg (DictConfig): Inference conf
        model (LightningModule): Score model loaded in the diffdock inference model
        confidence_model (LightningModule): Confidence model loaded in the diffdock inference model
        complex_name_list (List): List of complex names for inference
        test_loader (DataLoader): Dataloader for inference
        test_dataset (Dataset): Dataset for score model inference
        confidence_test_dataset (Dataset): Dataset for confidence model inference

    Returns:
        Tuple[int, int]: Number of Failed cases, number of skipped cases
    """
    model.eval()
    if confidence_model is not None:
        confidence_model.eval()

    for name in complex_name_list:
        write_dir = f"{cfg.out_dir}/{name}"
        os.makedirs(write_dir, exist_ok=True)

    tr_schedule = get_t_schedule(denoising_inference_steps=cfg.denoising_inference_steps)
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    logging.info(f"common t schedule {tr_schedule}")
    score_model_cfg = cfg.score_infer
    if cfg.confidence_infer is not None:
        confidence_model_cfg = cfg.confidence_infer

    t_to_sigma = partial(t_to_sigma_compl, cfg=score_model_cfg.model)

    failures, skipped, confidences_list, names_list, run_times, min_self_distances_list = 0, 0, [], [], [], []
    N = cfg.samples_per_complex
    logging.info(f"Size of test dataset: {len(test_dataset)}")

    with tqdm(total=len(test_loader), desc="sampling") as pbar:
        for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
            if not orig_complex_graph.success[0]:
                skipped += 1
                logging.info(
                    f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} "
                    f"for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}."
                    " We are skipping this complex."
                )
                pbar.update()
                continue
            try:
                if confidence_test_dataset is not None:
                    confidence_complex_graph = confidence_test_dataset[idx]
                    if not confidence_complex_graph.success:
                        skipped += 1
                        logging.info(
                            f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex."
                        )
                        pbar.update()
                        continue
                    confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
                else:
                    confidence_data_list = None
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                randomize_position(
                    data_list,
                    score_model_cfg.model.diffusion.no_torsion,
                    cfg.no_random,
                    score_model_cfg.model.diffusion.tr_sigma_max,
                )
                lig = orig_complex_graph.mol[0]

                # initialize visualisation
                pdb = None
                if cfg.save_visualisation:
                    visualization_list = []
                    for graph in data_list:
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        pdb.add(
                            (orig_complex_graph["ligand"].pos + orig_complex_graph.original_center).detach().cpu(),
                            1,
                            0,
                        )
                        pdb.add((graph["ligand"].pos + graph.original_center).detach().cpu(), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None

                start_time = time.time()

                # run reverse diffusion
                with autocast(enabled=cfg.amp):
                    data_list, confidence = sampling(
                        data_list=data_list,
                        model=model.model.net,
                        denoising_inference_steps=(
                            cfg.denoising_actual_steps
                            if cfg.denoising_actual_steps is not None
                            else score_model_cfg.model.denoising_inference_steps
                        ),
                        tr_schedule=tr_schedule,
                        rot_schedule=rot_schedule,
                        tor_schedule=tor_schedule,
                        device=model.device,
                        t_to_sigma=t_to_sigma,
                        model_cfg=score_model_cfg.model,
                        no_random=cfg.no_random,
                        ode=cfg.ode,
                        visualization_list=visualization_list,
                        confidence_model=confidence_model.model.net,
                        confidence_data_list=confidence_data_list,
                        confidence_model_cfg=confidence_model_cfg.model,
                        batch_size=cfg.micro_batch_size,
                        no_final_step_noise=cfg.no_final_step_noise,
                    )
                ligand_pos = np.asarray(
                    [
                        complex_graph["ligand"].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                        for complex_graph in data_list
                    ]
                )
                run_times.append(time.time() - start_time)

                # reorder predictions based on confidence output
                if confidence is not None and isinstance(confidence_model_cfg.rmsd_classification_cutoff, ListConfig):
                    confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = confidence.cpu().numpy()
                    re_order = np.argsort(confidence)[::-1]
                    confidence = confidence[re_order]
                    confidences_list.append(confidence)
                    ligand_pos = ligand_pos[re_order]

                # save predictions
                write_dir = f"{cfg.out_dir}/{complex_name_list[idx]}"
                for rank, pos in enumerate(ligand_pos):
                    if np.abs(pos.mean(axis=0)).max() > 1000.0:
                        pos = pos - pos.mean(axis=0) + 800.0 * np.sign(pos.mean(axis=0))
                    mol_pred = copy.deepcopy(lig)
                    if score_model_cfg.data.remove_hs:
                        mol_pred = RemoveHs(mol_pred)
                    if rank == 0:
                        write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f"rank{rank+1}.sdf"))
                    if confidence is not None:
                        path = os.path.join(write_dir, f"rank{rank+1}_confidence{confidence[rank]:.2f}.sdf")
                    else:
                        path = os.path.join(write_dir, f"rank{rank+1}.sdf")
                    write_mol_with_coords(mol_pred, pos, path)

                self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
                self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
                min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))

                # save visualisation frames
                if cfg.save_visualisation:
                    if confidence is not None:
                        for rank, batch_idx in enumerate(re_order):
                            visualization_list[batch_idx].write(
                                os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb")
                            )
                    else:
                        for rank in range(len(ligand_pos)):
                            visualization_list[rank].write(os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb"))

                names_list.append(orig_complex_graph.name[0])
            except Exception as e:
                logging.error(f'Failed on {orig_complex_graph["name"]} with the error {e}')
                failures += 1
            pbar.update()

    min_self_distances = np.array(min_self_distances_list)
    confidences = np.array(confidences_list)
    names = np.array(names_list)
    run_times = np.array(run_times)
    np.save(f"{cfg.out_dir}/min_self_distances.npy", min_self_distances)
    np.save(f"{cfg.out_dir}/confidences.npy", confidences)
    np.save(f"{cfg.out_dir}/run_times.npy", run_times)
    np.save(f"{cfg.out_dir}/complex_names.npy", names)

    return failures, skipped
