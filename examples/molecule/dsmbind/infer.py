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
Entry point to DSMBind inference.

Typical usage example:
    cd $BIONEMO_HOME
    python examples/molecule/dsmbind/infer.py

Notes:
    See conf/infer.yaml for hyperparameters and settings for this script.
"""

import os

import torch
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bionemo.data.dsmbind.dataset import DSMBindDataset
from bionemo.data.dsmbind.preprocess import preprocess
from bionemo.model.molecule.dsmbind.dsmbind_model import DSMBind


def inference(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
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
            pred = model.predict(batched_binder, batched_target)
            predictions.append(pred.item())
    return predictions


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    """
    This is the main function conducting data loading and model training for DSMBind.
    """
    logging.info("\n\n************** Experiment Configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Set up
    devices = [torch.device(f"cuda:{i}") for i in range(cfg.inference.num_gpus)]

    # Load model
    logging.info("************** Load Trained Model ***********")
    checkpoint_path = cfg.inference.ckpt_path
    model = DSMBind(cfg.model)
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(devices))))
    model.to(devices[0])  # Move model to the first device
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"# Model parameters: {total_params}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if cfg.data.processed:
        logging.info("************** Loading Inference Dataset ***********")
        test_dataset = DSMBindDataset(
            processed_data_path=cfg.data.processed_inference_data_path,
            aa_size=cfg.model.aa_size,
            max_residue_atoms=cfg.model.max_residue_atoms,
        )
    else:
        logging.info("************** Processing Inference Dataset ***********")
        preprocess(raw_data_dir=cfg.data.raw_inference_data_dir)
        logging.info("************** Loading Inference Dataset ***********")
        test_dataset = DSMBindDataset(
            processed_data_path=os.path.join(cfg.data.raw_inference_data_dir, "processed.pkl"),
            aa_size=cfg.model.aa_size,
            max_residue_atoms=cfg.model.max_residue_atoms,
        )

    test_loader = DataLoader(test_dataset, 1, collate_fn=test_dataset.pl_collate_fn, shuffle=False)
    logging.info("************** Starting Inference ***********")
    predictions = inference(model, test_loader, devices[0])
    logging.info(f"Predictions: {predictions}")  # Only the rank of the predictions matters


if __name__ == "__main__":
    main()