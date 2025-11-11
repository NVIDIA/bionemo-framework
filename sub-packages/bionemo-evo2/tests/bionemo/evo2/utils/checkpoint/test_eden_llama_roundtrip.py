# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from pathlib import Path

import pytest
import torch
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from nemo.collections.llm.gpt.model.llama import HFLlamaExporter

from bionemo.evo2.models.llama import HFEdenLlamaImporter
from bionemo.llm.lightning import batch_collator
from bionemo.testing.subprocess_utils import run_command_in_subprocess


REPO_PATH = Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent


@pytest.fixture(scope="module")
def checkpoint_eden_path() -> Path:
    """
    mkdir -p $REPO_PATH/tmp_checkpoints
    scp -r jstjohn@computelab-sc-01:/home/jstjohn/scratch/checkpoints/eden_llama_og2_step_182313 $REPO_PATH/tmp_checkpoints/
    """
    return REPO_PATH / "tmp_checkpoints" / "eden_llama_og2_step_182313"


@pytest.fixture(scope="module")
def metagenome_fasta_path() -> Path:
    """
    mkdir -p $REPO_PATH/tmp_data
    scp -r jstjohn@computelab-sc-01:/home/jstjohn/scratch/experiments/evo2_activations/ckpt_lm_loss_evals/lm_loss_work/evo2_metagenomics_test_only_sl8192_sd42.fasta $REPO_PATH/tmp_data/
    """
    return REPO_PATH / "tmp_data" / "evo2_metagenomics_test_only_sl8192_sd42.fasta"


def predict_metagenome(
    model_checkpoint_path: Path, metagenome_fasta_path: Path, output_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    port = find_free_network_port()
    cmd = f"""NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 --master-port={port} --no-python  \
        predict_evo2 \
            --eden-tokenizer \
            --devices=2 \
            --model-size 7B \
            --tensor-parallel-size=2 \
            --fasta {metagenome_fasta_path} \
            --ckpt-dir {model_checkpoint_path} \
            --output-log-prob-seqs \
            --log-prob-collapse-option per_token \
            --output-dir {output_path}"""
    run_command_in_subprocess(cmd, str(REPO_PATH))
    with open(output_path / "seq_idx_map.json", "r") as jsonf:
        fasta_to_index = json.load(jsonf)
    preds_list = [torch.load(f) for f in output_path.glob("*.pt")]
    all_pt_data = batch_collator([item for item in preds_list if item is not None])
    return all_pt_data, fasta_to_index  # type: ignore


@pytest.mark.slow
def test_eden_llama_roundtrip(tmp_path, checkpoint_eden_path: Path, metagenome_fasta_path: Path):
    """Test that converting NeMo -> HF -> NeMo produces the same model."""
    if not checkpoint_eden_path.exists() or not metagenome_fasta_path.exists():
        pytest.skip("Skipping test, first download the checkpoint and the metagenome fasta.")

    exporter = HFLlamaExporter(checkpoint_eden_path)
    hf_path = tmp_path / "hf_checkpoint"
    exporter.apply(hf_path)
    importer = HFEdenLlamaImporter(hf_path)
    importer.apply(tmp_path / "nemo_checkpoint")
    original_predictions, original_fasta_to_index = predict_metagenome(
        checkpoint_eden_path, metagenome_fasta_path, tmp_path / "original_predictions"
    )
    new_predictions, new_fasta_to_index = predict_metagenome(
        tmp_path / "nemo_checkpoint", metagenome_fasta_path, tmp_path / "new_predictions"
    )
    assert original_fasta_to_index == new_fasta_to_index, "Fasta to index mapping is not the same, need better logic."
    for key in ["seq_idx", "log_probs_seqs", "loss_mask"]:
        torch.testing.assert_close(
            original_predictions[key],
            new_predictions[key],
            msg=lambda diff: f"Results for {key} are not the same:\n{diff}",
        )
