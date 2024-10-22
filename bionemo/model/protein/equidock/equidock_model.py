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

import random
from functools import partial
from typing import List, Tuple

import dgl
import numpy as np
import torch
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.aggregation import CatMetric

from bionemo.data.equidock.data_manager import DataManager
from bionemo.data.equidock.protein_utils import (
    get_residues,
    preprocess_unbound_bound,
    protein_to_graph_unbound_bound,
)
from bionemo.model.protein.equidock.loss_metrics.eval import metrics_statistics, rmsd_compute
from bionemo.model.protein.equidock.loss_metrics.intersection_loss import compute_body_intersection_loss
from bionemo.model.protein.equidock.loss_metrics.ot_utils import compute_ot_emd, compute_sq_dist_mat
from bionemo.model.protein.equidock.model import Rigid_Body_Docking_Net
from bionemo.model.protein.equidock.utils.train_utils import batchify_and_create_hetero_graphs


torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NeMoExtension(ModelPT):
    def __init__(
        self, cfg: OmegaConf, net: torch.nn.Module, trainer: Trainer = None, data_manager: DataManager = None
    ):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        self.cfg = model_utils.maybe_update_config_version(cfg)

        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices
        else:
            self.world_size = 1

        logging.info("Fetching train/test/validation splits of protein-protein heterographs")
        if data_manager is not None:
            self._train_ds = data_manager.train_ds
            self._validation_ds = data_manager.validation_ds
            self._test_ds = data_manager.test_ds

        self.metrics_train = [CatMetric() for _ in range(3)]
        self.metrics_val = [CatMetric() for _ in range(3)]
        self.metrics_test = [CatMetric() for _ in range(3)]
        super().__init__(cfg, trainer)

        self.net = net
        self.pocket_ot_loss_weight = self.cfg.pocket_ot_loss_weight
        self.intersection_loss_weight = self.cfg.intersection_loss_weight

    def log_stats(self):
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, batch_size=self.cfg.micro_batch_size)

    def equidock_build_dataloader(self, data_config, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=data_config.micro_batch_size,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=False,
            shuffle=data_config.shuffle,
            drop_last=data_config.drop_last,
            collate_fn=partial(batchify_and_create_hetero_graphs),
        )

    def setup_training_data(self, train_data_config: OmegaConf):
        logging.info(f"Length of train dataset: {len(self._train_ds)}")
        self._train_dl = self.equidock_build_dataloader(train_data_config, self._train_ds)

    def setup_validation_data(self, val_data_config: OmegaConf):
        logging.info(f"Length of validation dataset: {len(self._validation_ds)}")
        self._validation_dl = self.equidock_build_dataloader(val_data_config, self._validation_ds)

    def setup_test_data(self, test_data_config: OmegaConf):
        logging.info(f"Length of test dataset: {len(self._test_ds)}")
        self._test_dl = self.equidock_build_dataloader(test_data_config, self._test_ds)

    def training_step(self, train_batch, batch_idx):
        (
            batch_hetero_graph,
            bound_ligand_repres_nodes_loc_array_list,
            bound_receptor_repres_nodes_loc_array_list,
            pocket_coors_ligand_list,
            pocket_coors_receptor_list,
        ) = train_batch
        (
            model_ligand_coors_deform_list,
            model_keypts_ligand_list,
            model_keypts_receptor_list,
            _,
            _,
        ) = self.net.forward(batch_hetero_graph)

        loss, batch_ot_loss, batch_intersection_loss = self.loss_function(
            model_ligand_coors_deform_list,
            model_keypts_ligand_list,
            model_keypts_receptor_list,
            pocket_coors_ligand_list,
            bound_ligand_repres_nodes_loc_array_list,
            pocket_coors_receptor_list,
            bound_receptor_repres_nodes_loc_array_list,
        )

        train_log = {
            "train_L": loss.cpu().detach(),
            "train_ot_L": batch_ot_loss.cpu().detach(),
            "train_intersect_L": batch_intersection_loss.cpu().detach(),
        }

        self.log_dict(
            train_log,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            batch_size=self.cfg.micro_batch_size,
            sync_dist=True,
        )

        rmsd_log = metrics_statistics(self.metrics_train, "train")

        self.log_dict(
            rmsd_log,
            prog_bar=False,
            logger=True,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
            batch_size=self.cfg.micro_batch_size,
        )

        self.log_stats()

        return loss

    def on_test_epoch_end(self):
        rmsd_log = metrics_statistics(self.metrics_test, "test")

        logging.info("\n")
        logging.info(f"Testing rmsd computed for  {rmsd_log['test_shape']} / {len(self._test_ds)} points!")
        logging.info(
            f"Testing dataset rmsd mean (complex/receptor/ligand):   {rmsd_log['test_complex_rmsd_mean']:.3f}/{rmsd_log['test_receptor_rmsd_mean']:.3f}/{rmsd_log['test_ligand_rmsd_mean']:.3f}"
        )
        logging.info(
            f"Testing dataset rmsd median (complex/receptor/ligand): {rmsd_log['test_complex_rmsd_median']:.3f}/{rmsd_log['test_receptor_rmsd_median']:.3f}/{rmsd_log['test_ligand_rmsd_median']:.3f}"
        )
        logging.info(
            f"Testing dataset rmsd std (complex/receptor/ligand):    {rmsd_log['test_complex_rmsd_std']:.3f}/{rmsd_log['test_receptor_rmsd_std']:.3f}/{rmsd_log['test_ligand_rmsd_std']:.3f}"
        )

        self.log_dict(
            rmsd_log,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.micro_batch_size,
        )

    def on_validation_epoch_end(self):
        rmsd_log = metrics_statistics(self.metrics_val, "val")

        logging.info("\n")
        logging.info(f"Validation rmsd computed for  {rmsd_log['val_shape']} / {len(self._validation_ds)} points!")
        logging.info(
            f"Validation dataset rmsd mean (complex/receptor/ligand):   {rmsd_log['val_complex_rmsd_mean']:.3f}/{rmsd_log['val_receptor_rmsd_mean']:.3f}/{rmsd_log['val_ligand_rmsd_mean']:.3f}"
        )
        logging.info(
            f"Validation dataset rmsd median (complex/receptor/ligand): {rmsd_log['val_complex_rmsd_median']:.3f}/{rmsd_log['val_receptor_rmsd_median']:.3f}/{rmsd_log['val_ligand_rmsd_median']:.3f}"
        )
        logging.info(
            f"Validation dataset rmsd std (complex/receptor/ligand):    {rmsd_log['val_complex_rmsd_std']:.3f}/{rmsd_log['val_receptor_rmsd_std']:.3f}/{rmsd_log['val_ligand_rmsd_std']:.3f}"
        )

        if self.trainer.early_stopping_callback is not None and self.current_epoch != 0:
            logging.info(
                f"Early stopping (wait/patience), (best_score) : {self.trainer.early_stopping_callback.wait_count}/{self.trainer.early_stopping_callback.patience}, {self.trainer.early_stopping_callback.best_score}"
            )
            early_stopping_stats = {"best_score": self.trainer.early_stopping_callback.best_score}
            self.log_dict(
                early_stopping_stats,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.cfg.micro_batch_size,
            )

        self.log_dict(
            rmsd_log,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.micro_batch_size,
        )

    def validation_step(self, val_batch, batch_idx):
        (
            batch_hetero_graph,
            bound_ligand_repres_nodes_loc_array_list,
            bound_receptor_repres_nodes_loc_array_list,
            pocket_coors_ligand_list,
            pocket_coors_receptor_list,
        ) = val_batch
        (
            model_ligand_coors_deform_list,
            model_keypts_ligand_list,
            model_keypts_receptor_list,
            _,
            _,
        ) = self.net(batch_hetero_graph)

        loss, batch_ot_loss, batch_intersection_loss = self.loss_function(
            model_ligand_coors_deform_list,
            model_keypts_ligand_list,
            model_keypts_receptor_list,
            pocket_coors_ligand_list,
            bound_ligand_repres_nodes_loc_array_list,
            pocket_coors_receptor_list,
            bound_receptor_repres_nodes_loc_array_list,
            dataset_type="validation",
        )

        val_log = {
            "val_L": loss.cpu().detach(),
            "val_ot_L": batch_ot_loss.cpu().detach(),
            "val_intersect_L": batch_intersection_loss.cpu().detach(),
        }

        self.log_dict(
            val_log,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            batch_size=self.cfg.micro_batch_size,
            sync_dist=True,
        )

        return loss

    def test_step(self, test_batch, batch_idx):
        (
            batch_hetero_graph,
            bound_ligand_repres_nodes_loc_array_list,
            bound_receptor_repres_nodes_loc_array_list,
            pocket_coors_ligand_list,
            pocket_coors_receptor_list,
        ) = test_batch
        (
            model_ligand_coors_deform_list,
            model_keypts_ligand_list,
            model_keypts_receptor_list,
            _,
            _,
        ) = self.net.forward(batch_hetero_graph)

        loss, batch_ot_loss, batch_intersection_loss = self.loss_function(
            model_ligand_coors_deform_list,
            model_keypts_ligand_list,
            model_keypts_receptor_list,
            pocket_coors_ligand_list,
            bound_ligand_repres_nodes_loc_array_list,
            pocket_coors_receptor_list,
            bound_receptor_repres_nodes_loc_array_list,
            dataset_type="testing",
        )

        test_loss = {
            "test_L": loss.cpu().detach(),
            "test_ot_L": batch_ot_loss.cpu().detach(),
            "test_intersect_L": batch_intersection_loss.cpu().detach(),
        }
        self.log_dict(
            test_loss,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            batch_size=self.cfg.micro_batch_size,
            sync_dist=True,
        )

        return test_loss

    def loss_function(
        self,
        model_ligand_coors_deform_list: List[Tensor],
        model_keypts_ligand_list: List[Tensor],
        model_keypts_receptor_list: List[Tensor],
        pocket_coors_ligand_list: List[Tensor],
        bound_ligand_repres_nodes_loc_array_list: List[Tensor],
        pocket_coors_receptor_list: List[Tensor],
        bound_receptor_repres_nodes_loc_array_list: List[Tensor],
        dataset_type: str = "train",
    ):
        pocket_ot_loss_weight = self.pocket_ot_loss_weight
        intersection_loss_weight = self.intersection_loss_weight

        intersection_surface_ct = self.cfg.intersection_surface_ct
        intersection_sigma = self.cfg.intersection_sigma

        device = model_ligand_coors_deform_list[0].device

        # Compute MSE loss for each protein individually, then average over the minibatch.
        batch_ligand_coors_loss = torch.zeros([]).to(device)
        batch_receptor_coors_loss = torch.zeros([]).to(device)  # This is not used!
        batch_ot_loss = torch.zeros([]).to(device)
        batch_intersection_loss = torch.zeros([]).to(device)

        loss_fn_coors = torch.nn.MSELoss(reduction="mean")

        for i in range(len(model_ligand_coors_deform_list)):
            # Compute average MSE loss (which is 3 times smaller than average squared RMSD)
            batch_ligand_coors_loss = batch_ligand_coors_loss + loss_fn_coors(
                model_ligand_coors_deform_list[i], bound_ligand_repres_nodes_loc_array_list[i].to(device)
            )

            # Compute the OT loss for the binding pocket:
            ligand_pocket_coors = pocket_coors_ligand_list[i].to(device)  # (N, 3), N = num pocket nodes
            receptor_pocket_coors = pocket_coors_receptor_list[i].to(device)  # (N, 3), N = num pocket nodes

            # (K, 3), K = num keypoints
            ligand_keypts_coors = model_keypts_ligand_list[i]
            # (K, 3), K = num keypoints
            receptor_keypts_coors = model_keypts_receptor_list[i]

            # (N, K) cost matrix
            cost_mat_ligand = compute_sq_dist_mat(ligand_pocket_coors, ligand_keypts_coors)
            cost_mat_receptor = compute_sq_dist_mat(receptor_pocket_coors, receptor_keypts_coors)

            ot_dist, _ = compute_ot_emd(cost_mat_ligand + cost_mat_receptor, device)
            batch_ot_loss = batch_ot_loss + ot_dist

            batch_intersection_loss = batch_intersection_loss + compute_body_intersection_loss(
                model_ligand_coors_deform_list[i],
                bound_receptor_repres_nodes_loc_array_list[i].to(device),
                intersection_sigma,
                intersection_surface_ct,
            )

            if i < 2 or dataset_type.startswith("val") or dataset_type.startswith("test") or (random.random() < 0.1):
                complex_rmsd, ligand_rmsd, receptor_rmsd = rmsd_compute(
                    model_ligand_coors_deform_list[i],
                    bound_receptor_repres_nodes_loc_array_list[i],
                    bound_ligand_repres_nodes_loc_array_list[i],
                    bound_receptor_repres_nodes_loc_array_list[i],
                )

                if dataset_type.startswith("val"):
                    for cnt, cur_rmsd in enumerate([complex_rmsd, ligand_rmsd, receptor_rmsd]):
                        self.metrics_val[cnt].update(cur_rmsd)
                elif dataset_type.startswith("test"):
                    for cnt, cur_rmsd in enumerate([complex_rmsd, ligand_rmsd, receptor_rmsd]):
                        self.metrics_test[cnt].update(cur_rmsd)
                elif dataset_type.startswith("train"):
                    for cnt, cur_rmsd in enumerate([complex_rmsd, ligand_rmsd, receptor_rmsd]):
                        self.metrics_train[cnt].update(cur_rmsd)
                else:
                    raise ValueError(f"Unknown dataset type {dataset_type}!")

        batch_ligand_coors_loss = batch_ligand_coors_loss / float(len(model_ligand_coors_deform_list))
        batch_receptor_coors_loss = batch_receptor_coors_loss / float(len(model_ligand_coors_deform_list))
        batch_ot_loss = batch_ot_loss / float(len(model_ligand_coors_deform_list))
        batch_intersection_loss = batch_intersection_loss / float(len(model_ligand_coors_deform_list))

        loss_coors = batch_ligand_coors_loss + batch_receptor_coors_loss

        loss = loss_coors + pocket_ot_loss_weight * batch_ot_loss + intersection_loss_weight * batch_intersection_loss

        return loss, batch_ot_loss, batch_intersection_loss

    def create_ligand_receptor_graphs_arrays(
        self,
        ligand_filename: str,
        receptor_filename: str,
        data_cfg: OmegaConf,
    ) -> Tuple[dgl.graph, dgl.graph, np.ndarray, np.ndarray]:
        """Creates ligand and receptor graphs and arrays (coordinates) from ligand and receptor files"""

        # Preprocess ligand and receptor
        (
            unbound_predic_ligand,
            unbound_predic_receptor,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
        ) = preprocess_unbound_bound(
            get_residues(ligand_filename),
            get_residues(receptor_filename),
            graph_nodes=self.cfg.graph_nodes,
            pos_cutoff=data_cfg.pocket_cutoff,
            inference=True,
        )

        # Make graphs
        ligand_graph, receptor_graph = protein_to_graph_unbound_bound(
            unbound_predic_ligand,
            unbound_predic_receptor,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
            graph_nodes=self.cfg.graph_nodes,
            cutoff=data_cfg.graph_cutoff,
            max_neighbor=data_cfg.graph_max_neighbor,
            one_hot=False,
            residue_loc_is_alphaC=self.cfg.graph_residue_loc_is_alphaC,
        )

        if self.cfg.input_edge_feats_dim < 0:
            self.cfg.input_edge_feats_dim = ligand_graph.edata["he"].shape[1]

        ligand_graph.ndata["new_x"] = ligand_graph.ndata["x"]

        assert (
            np.linalg.norm(bound_ligand_repres_nodes_loc_clean_array - ligand_graph.ndata["x"].detach().cpu().numpy())
            < 1e-1
        )

        return (
            ligand_graph,
            receptor_graph,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
        )

    @staticmethod
    def empty_safe_mean(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    @classmethod
    def list_available_models(cls) -> PretrainedModelInfo:
        return None


class EquiDock(NeMoExtension):
    def __init__(
        self, cfg: OmegaConf, trainer: Trainer = None, data_manager: DataManager = None, net: torch.nn.Module = None
    ):
        if "model" in cfg:
            cfg_ = cfg.model
        else:
            cfg_ = cfg

        super().__init__(
            cfg=cfg_,
            trainer=trainer,
            data_manager=data_manager,
            net=Rigid_Body_Docking_Net(args=cfg_) if net is None else net,
        )

    def reload_nemo_model(self, cfg: OmegaConf, trainer: Trainer, data_manager: DataManager):
        if data_manager is None:
            raise ValueError(f"Error: data_manager {data_manager} and tariner {Trainer} cannot be None")

        self._train_ds = data_manager.train_ds
        self._test_ds = data_manager.test_ds
        self._validation_ds = data_manager.validation_ds
        self.setup_training_data(cfg.model.train_ds)
        self.setup_validation_data(cfg.model.validation_ds)
        self.setup_test_data(cfg.model.test_ds)
