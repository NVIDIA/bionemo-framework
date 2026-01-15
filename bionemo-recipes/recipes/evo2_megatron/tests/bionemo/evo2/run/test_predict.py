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

"""Tests for Evo2 prediction (inference) workflow using Megatron Bridge."""

import copy
import glob
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import torch

from bionemo.evo2.data.test_utils.create_fasta_file import ALU_SEQUENCE, create_fasta_file
from bionemo.evo2.run.predict import batch_collator

from ..utils import check_fp8_support, find_free_network_port, is_a6000_gpu


# Do this at collection time before we run any tests.
PRETEST_ENV = copy.deepcopy(os.environ)


@pytest.fixture(scope="module")
def mbridge_checkpoint_1b_8k_bf16_path(tmp_path_factory) -> Path:
    """Create or load a MBridge checkpoint for testing.

    This fixture creates an MBridge checkpoint from a NeMo2 checkpoint if one doesn't exist,
    or can be adapted to download/load an existing MBridge checkpoint.
    """
    from bionemo.core.data.load import load
    from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
    from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge

    try:
        nemo2_checkpoint_path = load("evo2/1b-8k-bf16:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            pytest.skip(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    # Create a temporary directory for the MBridge checkpoint
    tmp_dir = tmp_path_factory.mktemp("mbridge_ckpt")
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_checkpoint_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=tmp_dir / "mbridge_checkpoint",
        model_size="1b",
        seq_length=8192,
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    # Return the checkpoint iteration directory
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.mark.parametrize(
    "ddp,pp,wi",
    [
        pytest.param(1, 1, "epoch", id="ddp=1,pp=1,wi=epoch"),
        pytest.param(2, 1, "epoch", id="ddp=2,pp=1,wi=epoch"),
        pytest.param(2, 1, "batch", id="ddp=2,pp=1,wi=batch"),
        pytest.param(
            1,
            2,
            "epoch",
            id="ddp=1,pp=2,wi=epoch",
            marks=pytest.mark.skip("Pipeline parallelism test currently hangs."),
        ),
    ],
)
@pytest.mark.slow
def test_predict_evo2_runs(
    tmp_path,
    ddp: int,
    pp: int,
    wi: str,
    mbridge_checkpoint_1b_8k_bf16_path: Path,
    num_sequences: int = 5,
    target_sequence_lengths: list[int] | None = None,
):
    """Test that the predict_evo2 command runs successfully with MBridge checkpoints.

    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.

    Since it's the full output this does not support CP, so we only test with TP=1. We also want coverage of the
        case where the sequence lengths are different and not necessarily divisible by CP.
    """
    if target_sequence_lengths is None:
        target_sequence_lengths = [3149, 3140, 1024, 3148, 3147]

    world_size = ddp * pp
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is greater than the number of GPUs {torch.cuda.device_count()}")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )

    # Create a local copy of the environment
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
        env["NCCL_P2P_DISABLE"] = "1"

    # Build the command string
    output_dir = tmp_path / "test_output"
    open_port = find_free_network_port()
    command = (
        f"torchrun --nproc_per_node {world_size} --nnodes 1 --master_port {open_port} "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_1b_8k_bf16_path} "
        f"--output-dir {output_dir} "
        f"--micro-batch-size 3 --write-interval {wi} "
        f"--pipeline-model-parallel-size {pp} --num-nodes 1 --devices {world_size}"
    )

    # Run the command in a subshell
    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )

    # For debugging purposes, print the output if the test fails
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)

    # Assert that the command completed successfully
    assert result.returncode == 0, f"predict_evo2 command failed with code {result.returncode}"

    # Assert that the output directory was created and contains predictions
    # With DDP, each DP rank produces its own file with dp_rank in the filename
    # File naming convention:
    #   Batch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}__batch_{batch_idx}.pt
    #   Epoch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt
    if wi == "batch":
        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*__batch_*.pt")))
        # With batch write interval, we expect multiple files (batches * dp_ranks)
        assert len(pred_files) >= ddp, f"Expected at least {ddp} prediction files, got {len(pred_files)}"
    else:
        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
        # With epoch write interval, we expect one file per DP rank
        assert len(pred_files) == ddp, f"Expected {ddp} prediction files (one per DP rank), got {len(pred_files)}"

    # Check sequence index map exists
    seq_idx_map_path = output_dir / "seq_idx_map.json"
    assert seq_idx_map_path.exists(), f"seq_idx_map.json not found at {seq_idx_map_path}"

    with open(seq_idx_map_path) as f:
        seq_idx_map = json.load(f)

    # Load and collate predictions
    # Note: predict.py outputs are all batch-first (batch_dim=0), seq-second (seq_dim=1)
    preds = [torch.load(pf) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    assert isinstance(preds, dict)
    assert "token_logits" in preds
    assert "pad_mask" in preds
    assert "seq_idx" in preds

    assert len(preds["token_logits"]) == len(preds["pad_mask"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences

    for original_idx, pad_mask, token_logits in zip(preds["seq_idx"], preds["pad_mask"], preds["token_logits"]):
        # seq_idx is not sorted necessarily, so use the saved "seq_idx" to determine the original order
        expected_len = target_sequence_lengths[original_idx]
        assert pad_mask.sum() == expected_len
        # Vocab size should be 512 for the nucleotide tokenizer
        assert token_logits.shape[-1] == 512


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

    # Create a temporary directory for the MBridge checkpoint
    tmp_dir = tmp_path_factory.mktemp("mbridge_ckpt_7b")
    # Note: run_nemo2_to_mbridge uses full model config from model_size
    # For testing we use the full 7b model but with shorter sequences
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_checkpoint_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=tmp_dir / "mbridge_checkpoint",
        model_size="7b_arc_longcontext",
        seq_length=8192,  # Use shorter seq length for tests
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.fixture(scope="module")
def baseline_predictions_7b_1m_results(
    mbridge_checkpoint_7b_1m_path: Path,
    tmp_path_factory,
    num_sequences: int = 5,
    target_sequence_lengths: list[int] | None = None,
) -> dict[int, float]:
    """Generate baseline predictions for 7b-1m model comparison."""
    if target_sequence_lengths is None:
        target_sequence_lengths = [2048, 2048, 2048, 2048, 2048]

    tmp_path = tmp_path_factory.mktemp("baseline_preds")
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path,
        num_sequences,
        sequence_lengths=target_sequence_lengths,
        repeating_dna_pattern=ALU_SEQUENCE,
    )
    output_dir = tmp_path / "test_output"
    open_port = find_free_network_port()
    command = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 "
        f"--output-dir {output_dir} "
        f"--num-nodes 1 --write-interval epoch "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )

    env = copy.deepcopy(PRETEST_ENV)
    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, f"predict_evo2 command failed: {result.stderr}"

    # Use the updated glob pattern matching the new naming convention
    # Epoch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt
    pred_files = glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt"))
    preds = [torch.load(pf) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    return dict(zip([i.item() for i in preds["seq_idx"]], [p.item() for p in preds["log_probs_seqs"]]))


@pytest.mark.parametrize(
    "ddp,cp,pp,tp,fp8,wi",
    [
        pytest.param(1, 1, 1, 1, False, "epoch", id="ddp=1,cp=1,pp=1,tp=1,fp8=False,wi=epoch"),
        pytest.param(2, 1, 1, 1, False, "epoch", id="ddp=2,cp=1,pp=1,tp=1,fp8=False,wi=epoch"),
        pytest.param(
            2, 1, 1, 1, False, "batch", id="ddp=2,cp=1,pp=1,tp=1,fp8=False,wi=batch"
        ),  # simulate a large prediction run with dp parallelism
        pytest.param(1, 2, 1, 1, False, "epoch", id="ddp=1,cp=2,pp=1,tp=1,fp8=False,wi=epoch"),
        pytest.param(1, 2, 1, 1, False, "batch", id="ddp=1,cp=2,pp=1,tp=1,fp8=False,wi=batch"),
        pytest.param(
            1,
            1,
            2,
            1,
            False,
            "epoch",
            id="ddp=1,cp=1,pp=2,tp=1,fp8=False,wi=epoch",
            marks=pytest.mark.skip("Pipeline parallelism test currently hangs."),
        ),
        pytest.param(
            1, 1, 1, 2, True, "epoch", id="ddp=1,cp=1,pp=1,tp=2,fp8=True,wi=epoch"
        ),  # Cover case where FP8 was not supported with TP=2
        pytest.param(1, 1, 1, 2, False, "epoch", id="ddp=1,cp=1,pp=1,tp=2,fp8=False,wi=epoch"),
    ],
)
@pytest.mark.slow
def test_predict_evo2_equivalent_with_log_probs(
    tmp_path,
    ddp: int,
    cp: int,
    pp: int,
    tp: int,
    fp8: bool,
    wi: str,
    mbridge_checkpoint_7b_1m_path: Path,
    baseline_predictions_7b_1m_results: dict[int, float],
    num_sequences: int = 5,
    target_sequence_lengths: list[int] | None = None,
):
    """Test that predict_evo2 produces equivalent log probabilities with different parallelism settings.

    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.

    For this test, we want coverage of CP, so we make sure sequence lengths are all the same and divisible by CP.

    The other thing this test does is check that the log probabilities are equivalent to the baseline predictions
     without model parallelism.
    """
    if target_sequence_lengths is None:
        target_sequence_lengths = [2048, 2048, 2048, 2048, 2048]

    world_size = ddp * cp * pp * tp
    mp_size = cp * pp * tp
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is greater than the number of GPUs {torch.cuda.device_count()}")
    is_fp8_supported, _, _ = check_fp8_support(torch.cuda.current_device())
    if not is_fp8_supported and fp8:
        pytest.skip("FP8 is not supported on this GPU.")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )

    # Create a local copy of the environment
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
        env["NCCL_P2P_DISABLE"] = "1"

    fp8_option = "--mixed-precision-recipe bf16_with_fp8_current_scaling_mixed" if fp8 else ""
    output_dir = tmp_path / "test_output"
    open_port = find_free_network_port()
    command = (
        f"torchrun --nproc_per_node {world_size} --nnodes 1 --master_port {open_port} "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 --write-interval {wi} "
        f"--output-dir {output_dir} --tensor-parallel-size {tp} {fp8_option} "
        f"--pipeline-model-parallel-size {pp} --context-parallel-size {cp} --num-nodes 1 --devices {world_size} "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )

    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )

    # For debugging purposes, print the output if the test fails
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)

    # Assert that the command completed successfully
    assert result.returncode == 0, f"predict_evo2 command failed with code {result.returncode}"

    # Assert that the output directory was created
    # With DDP, each DP rank produces its own file with dp_rank in the filename
    # File naming convention:
    #   Batch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}__batch_{batch_idx}.pt
    #   Epoch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt
    if wi == "batch":
        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*__batch_*.pt")))
        # With batch write interval, we expect multiple files (batches * dp_ranks)
        assert len(pred_files) >= ddp, f"Expected at least {ddp} prediction files, got {len(pred_files)}"
    else:
        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
        # With epoch write interval, we expect one file per DP rank
        assert len(pred_files) == ddp, f"Expected {ddp} prediction files (one per DP rank), got {len(pred_files)}"

    with open(output_dir / "seq_idx_map.json") as f:
        seq_idx_map = json.load(f)

    # Load and collate predictions from all DP ranks
    preds = [torch.load(pf) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    assert isinstance(preds, dict)
    assert "log_probs_seqs" in preds
    assert "seq_idx" in preds
    assert len(preds["log_probs_seqs"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences

    for original_idx, log_probs in zip(preds["seq_idx"], preds["log_probs_seqs"]):
        if mp_size > 1 and not fp8:
            # FIXME changing batch size so it doesn't match also required dropping rel=1e-6 to rel=1e-3.
            #  This should be investigated. TP=2 on some GPUs needs even more tolerance.
            rel = 2e-3
        elif fp8:
            # NOTE: This is hand-tuned on a b300 to pass for now as of 9/10/2025.
            rel = 1e-2
        else:
            rel = 1e-6
        assert log_probs.item() == pytest.approx(baseline_predictions_7b_1m_results[original_idx.item()], rel=rel)


# Note: The PEFT/LoRA test is commented out as it requires training infrastructure and LoRA support
# which may need additional updates for the Megatron Bridge API
# @pytest.mark.timeout(512)
# @pytest.mark.slow
# def test_different_results_with_without_peft(tmp_path):
#     """Test that predictions differ when using PEFT/LoRA adapters."""
#     pass


@pytest.mark.parametrize(
    "embedding_layer,expected_num_layers",
    [
        pytest.param(-1, 25, id="embedding_layer=-1_expects_25_layers"),
        pytest.param(-2, 24, id="embedding_layer=-2_expects_24_layers"),
        pytest.param(0, 1, id="embedding_layer=0_expects_1_layer"),
        pytest.param(5, 6, id="embedding_layer=5_expects_6_layers"),
    ],
)
@pytest.mark.slow
def test_predict_evo2_embedding_extraction(
    tmp_path,
    embedding_layer: int,
    expected_num_layers: int,
    mbridge_checkpoint_1b_8k_bf16_path: Path,
    num_sequences: int = 3,
    target_sequence_lengths: list[int] | None = None,
):
    """Test that embedding extraction produces outputs with expected shapes and keys.

    This test verifies:
    1. The model is initialized with the correct number of layers (logged and verified)
    2. Output contains 'hidden_embeddings' key instead of 'token_logits'
    3. Embeddings have expected shape [B, S, H] where H is hidden dimension
    4. Other expected keys (pad_mask, seq_idx, tokens) are present

    The 1b model has 25 layers, so:
    - embedding_layer=-1 -> 25 layers (last layer)
    - embedding_layer=-2 -> 24 layers (second-to-last)
    - embedding_layer=0 -> 1 layer (first layer only)
    - embedding_layer=5 -> 6 layers (layers 0-5)
    """
    original_num_layers = 25  # 1b model has 25 layers

    if target_sequence_lengths is None:
        target_sequence_lengths = [1024, 1024, 1024]

    world_size = 1
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is greater than the number of GPUs {torch.cuda.device_count()}")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )

    # Create a local copy of the environment
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"

    output_dir = tmp_path / "test_output"
    open_port = find_free_network_port()
    command = (
        f"torchrun --nproc_per_node {world_size} --nnodes 1 --master_port {open_port} "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_1b_8k_bf16_path} "
        f"--output-dir {output_dir} "
        f"--micro-batch-size 2 --write-interval epoch "
        f"--embedding-layer {embedding_layer}"
    )

    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )

    # For debugging purposes, print the output if the test fails
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)

    # Assert that the command completed successfully
    assert result.returncode == 0, f"predict_evo2 command failed with code {result.returncode}"

    # Combine stdout and stderr for log checking
    combined_output = result.stdout + result.stderr

    # Verify logging about model layers is present and extract the layer count
    assert "Model initialized with" in combined_output, "Expected logging about model layer count"
    assert "Embedding extraction" in combined_output, "Expected logging about embedding extraction mode"

    # Parse and verify the actual number of layers from the log
    # Look for pattern: "Model initialized with N layers"
    layer_match = re.search(r"Model initialized with (\d+) layers", combined_output)
    assert layer_match is not None, "Could not parse 'Model initialized with N layers' from output"
    actual_num_layers = int(layer_match.group(1))
    assert actual_num_layers == expected_num_layers, (
        f"Expected model to have {expected_num_layers} layers for embedding_layer={embedding_layer}, "
        f"but got {actual_num_layers} layers"
    )

    # Verify the embedding extraction log shows correct layer info
    # Look for pattern: "using N of M layers"
    extraction_match = re.search(r"using (\d+) of (\d+) layers", combined_output)
    assert extraction_match is not None, "Could not parse 'using N of M layers' from output"
    layers_used = int(extraction_match.group(1))
    layers_original = int(extraction_match.group(2))
    assert layers_used == expected_num_layers, (
        f"Expected 'using {expected_num_layers}' layers, but log shows 'using {layers_used}'"
    )
    assert layers_original == original_num_layers, (
        f"Expected original model to have {original_num_layers} layers, but log shows {layers_original}"
    )

    # Load predictions
    pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
    assert len(pred_files) == 1, f"Expected 1 prediction file, got {len(pred_files)}"

    preds = torch.load(pred_files[0])
    assert isinstance(preds, dict)

    # Verify expected keys for embedding extraction
    assert "hidden_embeddings" in preds, "Expected 'hidden_embeddings' key in embedding extraction mode"
    assert "token_logits" not in preds, "Should not have 'token_logits' in embedding extraction mode"
    assert "pad_mask" in preds, "Expected 'pad_mask' key"
    assert "seq_idx" in preds, "Expected 'seq_idx' key"
    assert "tokens" in preds, "Expected 'tokens' key"

    # Verify shapes
    hidden_embeddings = preds["hidden_embeddings"]
    pad_mask = preds["pad_mask"]
    tokens = preds["tokens"]

    # hidden_embeddings should be [B, S, H] where H is hidden dimension (1920 for 1b model)
    assert len(hidden_embeddings.shape) == 3, f"Expected 3D tensor, got shape {hidden_embeddings.shape}"
    batch_size, seq_len, hidden_dim = hidden_embeddings.shape

    assert batch_size == num_sequences, f"Expected batch size {num_sequences}, got {batch_size}"
    # Sequence length should match padded length
    max_seq_len = max(target_sequence_lengths)
    assert seq_len == max_seq_len, f"Expected seq_len {max_seq_len}, got {seq_len}"
    # Hidden dim should be 1920 for 1b model
    assert hidden_dim == 1920, f"Expected hidden_dim 1920 for 1b model, got {hidden_dim}"

    # Verify pad_mask and tokens have matching shapes
    assert pad_mask.shape == (batch_size, seq_len), f"pad_mask shape mismatch: {pad_mask.shape}"
    assert tokens.shape == (batch_size, seq_len), f"tokens shape mismatch: {tokens.shape}"

    # Verify seq_idx has correct count
    assert len(preds["seq_idx"]) == num_sequences, f"Expected {num_sequences} seq_idx entries"

    # Check sequence index map exists
    seq_idx_map_path = output_dir / "seq_idx_map.json"
    assert seq_idx_map_path.exists(), f"seq_idx_map.json not found at {seq_idx_map_path}"

    with open(seq_idx_map_path) as f:
        seq_idx_map = json.load(f)
    assert len(seq_idx_map) == num_sequences


@pytest.mark.slow
def test_predict_evo2_embedding_layer_validation(
    tmp_path,
    mbridge_checkpoint_1b_8k_bf16_path: Path,
):
    """Test that invalid embedding layer values are rejected with appropriate errors."""
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(fasta_file_path, 1, sequence_lengths=[512], repeating_dna_pattern=ALU_SEQUENCE)

    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"

    output_dir = tmp_path / "test_output"
    open_port = find_free_network_port()

    # Test with an invalid embedding layer (too large positive index)
    # The 1b model has 25 layers, so layer 100 should be invalid
    command = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_1b_8k_bf16_path} "
        f"--output-dir {output_dir} --embedding-layer 100"
    )

    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )

    # Should fail with an error about invalid embedding layer
    assert result.returncode != 0, "Expected command to fail with invalid embedding layer"
    assert "Invalid embedding_layer" in result.stderr or "Invalid embedding_layer" in result.stdout, (
        "Expected error message about invalid embedding layer"
    )


@pytest.mark.slow
def test_predict_evo2_embedding_with_log_probs_rejected(
    tmp_path,
    mbridge_checkpoint_1b_8k_bf16_path: Path,
):
    """Test that using both --embedding-layer and --output-log-prob-seqs is rejected."""
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(fasta_file_path, 1, sequence_lengths=[512], repeating_dna_pattern=ALU_SEQUENCE)

    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"

    output_dir = tmp_path / "test_output"
    open_port = find_free_network_port()

    # Test combining embedding extraction with log prob output (should fail)
    command = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_1b_8k_bf16_path} "
        f"--output-dir {output_dir} --embedding-layer -1 --output-log-prob-seqs"
    )

    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )

    # Should fail with an error about incompatible options
    assert result.returncode != 0, "Expected command to fail with incompatible options"
    assert "Cannot use --output-log-prob-seqs with --embedding-layer" in result.stderr or (
        "Cannot use --output-log-prob-seqs with --embedding-layer" in result.stdout
    ), "Expected error message about incompatible options"
