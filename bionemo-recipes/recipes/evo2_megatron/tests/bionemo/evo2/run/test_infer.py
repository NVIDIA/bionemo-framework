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

import copy
import os
import subprocess

import pytest
import torch

from bionemo.evo2.models.evo2_provider import HyenaInferenceContext

from ..utils import find_free_network_port


# Capture environment at import time (consistent with test_predict.py)
PRETEST_ENV = copy.deepcopy(os.environ)

# Note: mbridge_checkpoint_path fixture is provided by conftest.py at session scope


def test_infer_runs(mbridge_checkpoint_path, tmp_path):
    """Test that infer.py runs without errors."""
    output_file = tmp_path / "output.txt"

    # Use a longer DNA prompt to meet FP8 dimension requirements (divisible by 8)
    # 64 characters should be safe
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
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

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
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
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
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

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_infer_top_k(mbridge_checkpoint_path, tmp_path):
    """Test top-k sampling."""
    output_file = tmp_path / "output_topk.txt"
    # Use a longer prompt for FP8 compatibility
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
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

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
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
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
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

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    generated = output_file.read_text()
    assert len(generated) > 0, "Generated text is empty"


# DNA prompts for reproducibility tests (from test_prompt.py)
PROMPT_1 = "GAATAGGAACAGCTCCGGTCTACAGCTCCCAGCGTGAGCGACGCAGAAGACGGTGATTTCTGCATTTCCATCTGAGGTACCGGGTTCATCTCACTAGGGAGTGCCAGACAGTGGGCGCAGGCCAGTGTGTGTGCGCACCGTGCGCGAGCCGAAGCAGGG"
PROMPT_2 = "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGGTATGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTCGCAGTATCTGTCTTTGATTCCTGCCTCATCCTATTATTT"


def run_infer_subprocess(
    mbridge_checkpoint_path,
    prompt: str,
    output_file,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    top_k: int = 1,
    seed: int = 42,
):
    """Helper function to run inference as a subprocess.

    Args:
        mbridge_checkpoint_path: Path to the MBridge checkpoint
        prompt: Input prompt for the model
        output_file: Path to write output
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (1 for greedy)
        seed: Random seed for reproducibility

    Returns:
        The generated text from the output file
    """
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-file",
        str(output_file),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    return output_file.read_text()


def test_identical_prompts_should_be_identical(mbridge_checkpoint_path, tmp_path):
    """Test that identical prompts produce identical sequences.

    With greedy decoding (top_k=1) and the same seed, identical prompts
    should produce identical outputs.
    """
    output_file_1 = tmp_path / "output_prompt1_run1.txt"
    output_file_2 = tmp_path / "output_prompt1_run2.txt"

    # Run inference twice with the same prompt
    generated_1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_1,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding for determinism
        seed=42,
    )

    generated_2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_2,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding for determinism
        seed=42,
    )

    assert len(generated_1) > 0, "First generation produced empty output"
    assert len(generated_2) > 0, "Second generation produced empty output"
    assert generated_1 == generated_2, (
        f"Identical prompts with same seed and greedy decoding produced different outputs:\n"
        f"Run 1: {generated_1}\n"
        f"Run 2: {generated_2}"
    )


def test_different_prompts_produce_different_outputs(mbridge_checkpoint_path, tmp_path):
    """Test that different prompts produce different sequences.

    Different input prompts should produce different outputs, demonstrating
    that the model is actually responding to the prompt content.
    """
    output_file_1 = tmp_path / "output_prompt1.txt"
    output_file_2 = tmp_path / "output_prompt2.txt"

    # Run inference with two different prompts
    generated_1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_1,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
    )

    generated_2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_2,
        output_file=output_file_2,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
    )

    assert len(generated_1) > 0, "First generation produced empty output"
    assert len(generated_2) > 0, "Second generation produced empty output"

    # The outputs should be different since the prompts are different
    # We check that the generated portions (after the prompt) are not identical
    assert generated_1 != generated_2, (
        f"Different prompts produced identical outputs:\n"
        f"Prompt 1 output: {generated_1}\n"
        f"Prompt 2 output: {generated_2}"
    )


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
