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
import csv
import os
import subprocess
from pathlib import Path

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


def mid_point_split(*, seq, num_tokens: int | None = None, fraction: float = 0.5):
    """Split a sequence at a midpoint for prompt/target evaluation."""
    mid_point = int(fraction * len(seq))
    prompt = seq[:mid_point]
    if num_tokens is not None:
        target = seq[mid_point : mid_point + num_tokens]
    else:
        target = seq[mid_point:]
    return prompt, target


def calculate_sequence_identity(seq1: str, seq2: str) -> float | None:
    """Calculate sequence identity between two sequences through direct comparison."""
    if not seq1 or not seq2:
        return None
    min_length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_length], seq2[:min_length]))
    return (matches / min_length) * 100


def _recipe_root() -> Path:
    """Return the recipe root directory (evo2_megatron/)."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def _infer_script_path() -> Path:
    """Return the path to the source infer.py script.

    Uses the source version directly (rather than the installed module via ``-m``)
    so that local fixes to infer.py are picked up without reinstalling the package.
    """
    return _recipe_root() / "src" / "bionemo" / "evo2" / "run" / "infer.py"


def run_infer_subprocess_parallel(
    mbridge_checkpoint_path,
    prompt_file: Path,
    output_file: Path,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int = 1,
    seed: int = 42,
    tensor_parallel_size: int = 1,
    context_parallel_size: int = 1,
):
    """Helper to run inference as a subprocess with model parallelism.

    Runs the source infer.py script directly (not the installed module) so that
    local fixes are picked up without reinstalling the package.

    Args:
        mbridge_checkpoint_path: Path to the MBridge checkpoint.
        prompt_file: Path to a text file containing the prompt.
        output_file: Path to write output.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter (1 for greedy).
        seed: Random seed for reproducibility.
        tensor_parallel_size: Tensor parallelism degree.
        context_parallel_size: Context parallelism degree.

    Returns:
        The generated text from the output file.
    """
    nproc_per_node = tensor_parallel_size * context_parallel_size
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(nproc_per_node),
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        str(_infer_script_path()),
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt-file",
        str(prompt_file),
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
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--context-parallel-size",
        str(context_parallel_size),
    ]

    env = copy.deepcopy(PRETEST_ENV)
    # Prepend the source src/ directory to PYTHONPATH so that local model code
    # (hyena_mixer.py, hyena_utils.py, etc.) is used instead of the installed package.
    src_dir = str(_recipe_root() / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,  # 15 minutes for parallel configs
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


@pytest.fixture
def dna_sequences():
    """Load DNA sequences from prompts.csv test data."""
    prompts_csv = Path(__file__).resolve().parent.parent / "data" / "prompts.csv"
    with prompts_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row["Sequence"] for row in reader]


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tp, cp",
    [
        # The 1b model only supports TP=1 through infer.py due to divisibility constraints
        # (15 attention heads and 128-width HyenaMixer). TP>1 requires the 7b model.
        pytest.param(1, 1, id="tp=1,cp=1"),
        pytest.param(
            1,
            2,
            id="tp=1,cp=2",
            marks=pytest.mark.xfail(reason="CP>1 is known broken for inference", strict=False),
        ),
    ],
)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
def test_parallel_inference_accuracy(mbridge_checkpoint_path, tmp_path, dna_sequences, tp, cp):
    """Test that parallel inference produces accurate generation results.

    Loads real DNA sequences, splits them in half, generates 500 tokens from the first half,
    and compares the generated tokens against the known second half using sequence identity.
    This mirrors the pattern in test_batch_generate_mbridge in test_evo2.py but exercises
    the subprocess-based infer.py CLI with parallelism.
    """
    num_gpus_required = tp * cp
    if torch.cuda.device_count() < num_gpus_required:
        pytest.skip(f"Not enough GPUs: need {num_gpus_required}, have {torch.cuda.device_count()}")

    num_tokens = 500
    # Expected sequence identity percentages for the 1b-8k-bf16 checkpoint (from test_evo2.py)
    expected_matchpercents = [96.8, 29.7, 76.6, 71.6]

    match_percents = []
    for i, seq in enumerate(dna_sequences):
        prompt, target = mid_point_split(seq=seq, num_tokens=num_tokens, fraction=0.5)

        prompt_file = tmp_path / f"prompt_seq{i}.txt"
        output_file = tmp_path / f"output_seq{i}.txt"
        prompt_file.write_text(prompt)

        generated_text = run_infer_subprocess_parallel(
            mbridge_checkpoint_path,
            prompt_file=prompt_file,
            output_file=output_file,
            max_new_tokens=num_tokens,
            temperature=1.0,
            top_k=1,  # Greedy decoding
            seed=42,
            tensor_parallel_size=tp,
            context_parallel_size=cp,
        )

        identity = calculate_sequence_identity(target, generated_text)
        match_percents.append(identity)

    matchperc_print = [f"{mp:.2f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.2f}%" for ep in expected_matchpercents]

    assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
    )


@pytest.fixture(scope="module")
def mbridge_checkpoint_7b_1m_path(tmp_path_factory) -> Path:
    """Create or load a MBridge checkpoint for 7b-1m model testing."""
    from bionemo.core.data.load import load
    from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
    from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge

    try:
        nemo2_checkpoint_path = load("evo2/7b-1m:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            pytest.skip(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    tmp_dir = tmp_path_factory.mktemp("mbridge_ckpt_7b")
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_checkpoint_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=tmp_dir / "mbridge_checkpoint",
        model_size="7b_arc_longcontext",
        seq_length=8192,
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tp, cp",
    [
        # The 7b model has 32 attention heads, supporting TP=1, 2, 4, 8
        pytest.param(1, 1, id="tp=1,cp=1"),
        pytest.param(2, 1, id="tp=2,cp=1"),
        pytest.param(4, 1, id="tp=4,cp=1"),
        pytest.param(8, 1, id="tp=8,cp=1"),
        pytest.param(
            1,
            2,
            id="tp=1,cp=2",
            marks=pytest.mark.xfail(reason="CP>1 is known broken for inference", strict=False),
        ),
    ],
)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
def test_parallel_inference_accuracy_7b(mbridge_checkpoint_7b_1m_path, tmp_path, dna_sequences, tp, cp):
    """Test that parallel inference with the 7b model produces accurate generation results.

    Uses the 7b-1m checkpoint which supports TP>1 (32 attention heads), enabling
    proper tensor parallel accuracy testing that the 1b model cannot support.
    """
    num_gpus_required = tp * cp
    if torch.cuda.device_count() < num_gpus_required:
        pytest.skip(f"Not enough GPUs: need {num_gpus_required}, have {torch.cuda.device_count()}")

    num_tokens = 500
    # Expected sequence identity percentages for the 7b model (from test_evo2.py)
    expected_matchpercents = [97.60, 89.63, 80.03, 84.57]

    match_percents = []
    for i, seq in enumerate(dna_sequences):
        prompt, target = mid_point_split(seq=seq, num_tokens=num_tokens, fraction=0.5)

        prompt_file = tmp_path / f"prompt_seq{i}.txt"
        output_file = tmp_path / f"output_seq{i}.txt"
        prompt_file.write_text(prompt)

        generated_text = run_infer_subprocess_parallel(
            mbridge_checkpoint_7b_1m_path,
            prompt_file=prompt_file,
            output_file=output_file,
            max_new_tokens=num_tokens,
            temperature=1.0,
            top_k=1,  # Greedy decoding
            seed=42,
            tensor_parallel_size=tp,
            context_parallel_size=cp,
        )

        identity = calculate_sequence_identity(target, generated_text)
        match_percents.append(identity)

    matchperc_print = [f"{mp:.2f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.2f}%" for ep in expected_matchpercents]

    assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
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
