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
Entry point to DSMBind training.

Typical usage example:
    cd $BIONEMO_HOME
    python examples/molecule/dsmbind/pretrain.py

Notes:
    See examples/molecule/dsmbind/conf/pretrain.yaml for hyperparameters and settings for this script.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bionemo.data.dsmbind.dataset import DSMBindDataset
from bionemo.model.molecule.dsmbind.dsmbind_model import DSMBind


def train_one_epoch(model, dataloader, optimizer, clip_norm, device):
    """
    This is the training logic for one epoch.
    """
    model.train()
    running_loss = 0.0
    for batched_binder, batched_target in dataloader:
        # batched_binder[1] is a complicated rdkit Mol, which will be moved to device inside the model
        batched_binder = (
            batched_binder[0].to(device),
            batched_binder[1],
            batched_binder[2].to(device),
        )
        batched_target = (
            batched_target[0].to(device),
            batched_target[1].to(device),
            batched_target[2].to(device),
        )
        optimizer.zero_grad()
        loss = model(batched_binder, batched_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)  # Clip the gradient norm
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)  # The average loss per batch
    return epoch_loss


def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir):
    """
    This is the function to save the checkpoint.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    logging.info(f"Checkpoint saved at {checkpoint_path}")


@hydra_runner(config_path="conf", config_name="pretrain")
def main(cfg) -> None:
    """
    This is the main function conducting data loading and model training for DSMBind.
    """
    logging.info("\n\n************** Experiment Configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Set up
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    devices = [torch.device(f"cuda:{i}") for i in range(cfg.train.num_gpus)]
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    # Data loading
    logging.info("************** Loading Dataset ***********")
    train_dataset = DSMBindDataset(
        processed_data_path=cfg.data.processed_training_data_path,
        aa_size=cfg.model.aa_size,
        max_residue_atoms=cfg.model.max_residue_atoms,
    )
    train_loader = DataLoader(
        train_dataset, cfg.train.batch_size, collate_fn=train_dataset.pl_collate_fn, shuffle=True
    )

    # Setup model and potentially use DataParallel
    logging.info("************** Setup Model ***********")
    model = DSMBind(cfg.model)
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(devices))))
    model.to(devices[0])  # Move model to the first device
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"# Model parameters: {total_params}")

    # Setup training
    logging.info("************** Setup Training ***********")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, cfg.train.anneal_rate)

    # Training
    logging.info("************** Starting Training ***********")
    for epoch in range(cfg.train.max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.train.clip_norm, devices[0])
        scheduler.step()
        logging.info(f"Epoch {epoch+1}/{cfg.train.max_epochs}, Loss: {train_loss:.4f}")
        if (epoch + 1) % cfg.train.ckpt_interval == 0:
            save_checkpoint(epoch + 1, model, optimizer, train_loss, cfg.train.ckpt_dir)
    logging.info("************** Done ***********")

    # Skip validation now since we don't have the licence for the validation data


if __name__ == "__main__":
    main()
