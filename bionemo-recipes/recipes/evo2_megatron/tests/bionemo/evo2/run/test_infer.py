# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

"""Tests for Evo2 text generation (inference) using MBridge.

NOTE: Autoregressive generation tests may fail due to:
1. FP8 execution requires sequence dimensions divisible by 8/16
2. The vortex flash_decode path needs additional integration work

The core forward pass (predict.py) and HyenaInferenceContext are tested
in test_evo2.py which has working test_forward_manual and test_forward_ckpt_conversion.
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch

from bionemo.evo2.models.evo2_provider import HyenaInferenceContext


@pytest.fixture(scope="module")
def mbridge_checkpoint_path(tmp_path_factory):
    """Create or use an MBridge checkpoint for testing.

    Uses the same checkpoint conversion as test_predict.py.
    """
    from bionemo.core.data.load import load
    from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
    from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge

    # Otherwise create a new one using a NeMo2 checkpoint
    try:
        nemo2_ckpt_path = load("evo2/1b-8k-bf16:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            pytest.skip(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    output_dir = tmp_path_factory.mktemp("mbridge_ckpt")
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_ckpt_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=output_dir / "evo2_1b_mbridge",
        model_size="1b",
        seq_length=8192,
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    return mbridge_ckpt_dir / "iter_0000001"


def test_infer_runs(mbridge_checkpoint_path, tmp_path):
    """Test that infer.py runs without errors."""
    output_file = tmp_path / "output.txt"

    # Use a longer DNA prompt to meet FP8 dimension requirements (divisible by 8)
    # 64 characters should be safe
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "10",
        "--output-file",
        str(output_file),
        "--temperature",
        "1.0",  # Non-zero temperature required by MCore
        "--top-k",
        "1",  # Top-k=1 for greedy decoding
    ]

    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29501"

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(Path(__file__).parent.parent.parent.parent.parent),
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    # Check that output contains generated text
    generated = output_file.read_text()
    assert len(generated) > 0, "Generated text is empty"


@pytest.mark.parametrize("temperature", [0.5, 1.0])
def test_infer_temperature(mbridge_checkpoint_path, tmp_path, temperature):
    """Test that different temperatures produce output."""
    output_file = tmp_path / f"output_temp_{temperature}.txt"
    # Use a longer prompt for FP8 compatibility
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "5",
        "--temperature",
        str(temperature),
        "--output-file",
        str(output_file),
    ]

    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29502"

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(Path(__file__).parent.parent.parent.parent.parent),
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_infer_top_k(mbridge_checkpoint_path, tmp_path):
    """Test top-k sampling."""
    output_file = tmp_path / "output_topk.txt"
    # Use a longer prompt for FP8 compatibility
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "5",
        "--top-k",
        "4",  # Only sample from top 4 tokens (A, C, G, T)
        "--output-file",
        str(output_file),
    ]

    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29503"

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(Path(__file__).parent.parent.parent.parent.parent),
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_infer_phylogenetic_prompt(mbridge_checkpoint_path, tmp_path):
    """Test generation with a phylogenetic lineage prompt.

    Evo2 is trained with phylogenetic tags, so generation should work
    well when conditioned on these tags. Using a longer prompt for FP8.
    """
    output_file = tmp_path / "output_phylo.txt"

    # Phylogenetic prompt (padded to be longer for FP8 compatibility)
    prompt = (
        "|d__Bacteria;"
        "p__Pseudomonadota;"
        "c__Gammaproteobacteria;"
        "o__Enterobacterales;"
        "f__Enterobacteriaceae;"
        "g__Escherichia;"
        "s__Escherichia|"
    )

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "20",
        "--temperature",
        "1.0",  # Non-zero temperature required by MCore
        "--top-k",
        "1",  # Top-k=1 for greedy decoding
        "--output-file",
        str(output_file),
    ]

    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29504"

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(Path(__file__).parent.parent.parent.parent.parent),
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    generated = output_file.read_text()
    assert len(generated) > 0, "Generated text is empty"


class TestHyenaInferenceContext:
    """Unit tests for the Hyena-specific inference context."""

    def test_context_initialization(self):
        """Test that HyenaInferenceContext can be initialized."""
        context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
        assert context is not None
        assert context.max_batch_size == 1
        assert context.max_sequence_length == 8192

    def test_context_reset(self):
        """Test that context reset works without error."""
        context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
        # Add some fake filter state (simulating what hyena layers do)
        context.filter_state_dict_layer_0 = {"key": torch.zeros(10)}
        context.filter_state_dict_layer_1 = {"key": torch.ones(10)}

        # Verify the state was added
        assert hasattr(context, "filter_state_dict_layer_0")
        assert hasattr(context, "filter_state_dict_layer_1")

        # Reset should remove all filter_state_dict attributes
        context.reset()

        assert not hasattr(context, "filter_state_dict_layer_0")
        assert not hasattr(context, "filter_state_dict_layer_1")

    def test_context_materialize_logits_setting(self):
        """Test that materialize_only_last_token_logits can be configured."""
        context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)

        # Default should be True for efficiency
        # We can set it to False if we need full sequence logits
        context.materialize_only_last_token_logits = False
        assert context.materialize_only_last_token_logits is False

        context.materialize_only_last_token_logits = True
        assert context.materialize_only_last_token_logits is True

    def test_context_multiple_batches(self):
        """Test context with different batch sizes."""
        for batch_size in [1, 2, 4]:
            context = HyenaInferenceContext(max_batch_size=batch_size, max_sequence_length=4096)
            assert context.max_batch_size == batch_size
            context.reset()  # Should not error

    def test_context_different_sequence_lengths(self):
        """Test context with different max sequence lengths."""
        for seq_len in [1024, 8192, 16384]:
            context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=seq_len)
            assert context.max_sequence_length == seq_len
            context.reset()
