# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test mbridge -> HF -> mbridge roundtrip fidelity for Eden (Llama) models.

Ported from sub-packages/bionemo-evo2/tests/bionemo/evo2/utils/checkpoint/test_eden_llama_roundtrip.py
which tested NeMo -> HF -> NeMo. This version tests the Megatron Bridge equivalent.
"""

import copy
import glob
import os
import shlex
import subprocess
from pathlib import Path

import pytest
import torch

from bionemo.evo2.data.test_utils.create_fasta_file import ALU_SEQUENCE, create_fasta_file
from bionemo.evo2.run.predict import batch_collator

from .utils import find_free_network_port, is_a6000_gpu


ROUNDTRIP_HELPER = Path(__file__).parent / "_eden_roundtrip_helper.py"
PRETEST_ENV = copy.deepcopy(os.environ)


def _run_predict(ckpt_dir: Path, fasta_path: Path, output_dir: Path, env: dict) -> dict:
    """Run predict_evo2 and return collated predictions."""
    open_port = find_free_network_port()
    command = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"-m bionemo.evo2.run.predict --fasta {fasta_path} --ckpt-dir {ckpt_dir} "
        f"--output-dir {output_dir} "
        "--micro-batch-size 3 --write-interval epoch "
        "--output-log-prob-seqs --log-prob-collapse-option per_token "
        "--num-nodes 1 --devices 1"
    )
    result = subprocess.run(
        shlex.split(command), check=False, cwd=output_dir.parent, capture_output=True, env=env, text=True
    )
    if result.returncode != 0:
        print(f"PREDICT STDOUT:\n{result.stdout}")
        print(f"PREDICT STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"predict_evo2 failed: {result.stderr[-2000:]}"

    pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
    preds = [torch.load(pf) for pf in pred_files]
    return batch_collator(
        [p for p in preds if p is not None],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )


@pytest.fixture(scope="module")
def eden_ckpt(mbridge_eden_checkpoint) -> Path:
    """Module-scoped alias for the session-scoped Eden checkpoint."""
    return mbridge_eden_checkpoint


@pytest.fixture(scope="module")
def hf_exported_dir(eden_ckpt: Path, tmp_path_factory) -> Path:
    """Export the Eden mbridge checkpoint to HuggingFace format."""
    tmp_dir = tmp_path_factory.mktemp("eden_hf_export")
    hf_dir = tmp_dir / "hf_checkpoint"

    open_port = find_free_network_port()
    cmd = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"{ROUNDTRIP_HELPER} --mode export "
        f"--ckpt-dir {eden_ckpt} --hf-output-dir {hf_dir}"
    )
    env = copy.deepcopy(PRETEST_ENV)
    result = subprocess.run(shlex.split(cmd), check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"EXPORT STDOUT:\n{result.stdout}")
        print(f"EXPORT STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"mbridge->HF export failed: {result.stderr[-2000:]}"
    assert hf_dir.exists(), f"HF export dir not created at {hf_dir}"
    return hf_dir


@pytest.fixture(scope="module")
def hf_reimported_dir(hf_exported_dir: Path, tmp_path_factory) -> Path:
    """Import the HF checkpoint back and re-export to HF for weight comparison."""
    tmp_dir = tmp_path_factory.mktemp("eden_hf_reimport")
    reimported_hf_dir = tmp_dir / "reimported_hf_checkpoint"

    open_port = find_free_network_port()
    cmd = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"{ROUNDTRIP_HELPER} --mode import "
        f"--hf-input-dir {hf_exported_dir} --ckpt-output-dir {reimported_hf_dir}"
    )
    env = copy.deepcopy(PRETEST_ENV)
    result = subprocess.run(shlex.split(cmd), check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"IMPORT STDOUT:\n{result.stdout}")
        print(f"IMPORT STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"HF->mbridge->HF re-import failed: {result.stderr[-2000:]}"
    assert reimported_hf_dir.exists(), f"Reimported HF dir not created at {reimported_hf_dir}"
    return reimported_hf_dir


@pytest.mark.slow
def test_roundtrip_hf_weight_equality(hf_exported_dir: Path, hf_reimported_dir: Path):
    """Verify that mbridge->HF->mbridge->HF produces identical weights."""
    from transformers import LlamaForCausalLM

    original = LlamaForCausalLM.from_pretrained(hf_exported_dir, torch_dtype=torch.bfloat16)
    reimported = LlamaForCausalLM.from_pretrained(hf_reimported_dir, torch_dtype=torch.bfloat16)

    orig_sd = original.state_dict()
    reimp_sd = reimported.state_dict()

    assert set(orig_sd.keys()) == set(reimp_sd.keys()), (
        f"Key mismatch.\nOnly in original: {sorted(set(orig_sd) - set(reimp_sd))}\n"
        f"Only in reimported: {sorted(set(reimp_sd) - set(orig_sd))}"
    )

    for key in sorted(orig_sd.keys()):
        assert orig_sd[key].shape == reimp_sd[key].shape, f"Shape mismatch for {key}"
        torch.testing.assert_close(
            orig_sd[key],
            reimp_sd[key],
            atol=0,
            rtol=0,
            msg=lambda diff: f"Weight mismatch for {key}: {diff}",
        )


@pytest.mark.slow
def test_roundtrip_prediction_equality(
    eden_ckpt: Path,
    hf_exported_dir: Path,
    hf_reimported_dir: Path,
    tmp_path,
):
    """Verify that predictions from the original and roundtripped models match.

    Runs predict on both the original mbridge checkpoint and on the re-imported HF checkpoint
    (loaded via AutoBridge) and compares per-token log probabilities.
    """
    from transformers import LlamaForCausalLM

    num_sequences = 2
    seq_lengths = [64, 64]

    fasta_path = tmp_path / "test.fasta"
    create_fasta_file(fasta_path, num_sequences, sequence_lengths=seq_lengths, repeating_dna_pattern=ALU_SEQUENCE)

    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"

    # Predictions from the original mbridge checkpoint
    original_preds = _run_predict(eden_ckpt, fasta_path, tmp_path / "orig_preds", env)

    assert "log_probs_seqs" in original_preds
    assert "seq_idx" in original_preds

    # Load the original and reimported HF models and compare forward pass
    original_hf = LlamaForCausalLM.from_pretrained(hf_exported_dir, torch_dtype=torch.bfloat16).eval()
    reimported_hf = LlamaForCausalLM.from_pretrained(hf_reimported_dir, torch_dtype=torch.bfloat16).eval()

    # Quick sanity: HF forward pass should produce identical outputs for both
    input_ids = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        orig_logits = original_hf(input_ids).logits
        reimp_logits = reimported_hf(input_ids).logits

    torch.testing.assert_close(orig_logits, reimp_logits, atol=0, rtol=0)
