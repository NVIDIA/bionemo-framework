# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from bionemo.data.dsmbind.dataset import DSMBindDataset
from bionemo.data.dsmbind.preprocess import preprocess
from bionemo.model.molecule.dsmbind.dsmbind_model import DSMBind
from bionemo.utils.hydra import load_model_config

from .inference_shared_test_code import get_config_dir


def dsmbind_test_raw_data_dir(bionemo_home) -> str:
    path = bionemo_home / "examples" / "tests" / "test_data" / "molecule" / "dsmbind" / "inference"
    return str(path)


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


@pytest.mark.needs_checkpoint
def test_dsmbind_inference(bionemo_home: Path):
    config_path = get_config_dir(bionemo_home, "dsmbind")
    cfg = load_model_config(config_name="infer", config_path=config_path)

    # Set up
    device = torch.device("cpu")

    # Load model
    checkpoint_path = cfg.inference.ckpt_path
    model = DSMBind(cfg.model)
    model.to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Process and load dataset
    raw_data_dir = dsmbind_test_raw_data_dir(bionemo_home)
    preprocess(raw_data_dir=raw_data_dir)
    inference_dataset = DSMBindDataset(
        processed_data_path=os.path.join(raw_data_dir, "processed.pkl"),
        aa_size=cfg.model.aa_size,
        max_residue_atoms=cfg.model.max_residue_atoms,
    )
    os.remove(os.path.join(raw_data_dir, "processed.pkl"))
    test_loader = DataLoader(inference_dataset, 1, collate_fn=inference_dataset.pl_collate_fn, shuffle=False)

    predictions = inference(model, test_loader, device)

    assert np.allclose(np.sort(np.array(predictions)), np.array([-2277249.5, -2054088.0]), atol=1.0e-3)
