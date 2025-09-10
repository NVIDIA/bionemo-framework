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

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.core.data.load import load
from bionemo.llm.lightning import batch_collator
from bionemo.testing.data.fasta import ALU_SEQUENCE, create_fasta_file
from bionemo.testing.torch import check_fp8_support


def is_a6000_gpu() -> bool:
    # Check if any of the visible GPUs is an A6000
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        if "A6000" in device_name:
            return True
    return False


@pytest.fixture(scope="module")
def checkpoint_1b_8k_bf16_path() -> Path:
    try:
        checkpoint_path = load("evo2/1b-8k-bf16:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    return checkpoint_path


@pytest.fixture(scope="module")
def checkpoint_7b_1m_path() -> Path:
    try:
        checkpoint_path = load("evo2/7b-1m:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    return checkpoint_path


@pytest.mark.parametrize(
    "ddp,pp,tp,wi",
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
def test_predict_evo2_runs(
    tmp_path,
    ddp: int,
    pp: int,
    tp: int,
    wi: str,
    checkpoint_1b_8k_bf16_path: Path,
    num_sequences: int = 5,
    target_sequence_lengths: list[int] = [3149, 3140, 1024, 3148, 3147],
):
    """
    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.

    Since it's the full output this does not support CP, so we only test with TP=1. We also want coverage of the
        case where the sequence lengths are different and not necessarily divisible by CP.
    """
    world_size = ddp * pp * tp
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is less than the number of GPUs {torch.cuda.device_count()}")
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )
    # Create a mock data directory.
    # a local copy of the environment
    env = dict(**os.environ)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
        env["NCCL_P2P_DISABLE"] = "1"

    # Build the command string.
    # Note: The command assumes that `train_evo2` is in your PATH.
    output_dir = tmp_path / "test_output"
    command = (
        f"torchrun --nproc_per_node {world_size} --nnodes 1 --no-python "
        f"predict_evo2 --fasta {fasta_file_path} --ckpt-dir {checkpoint_1b_8k_bf16_path} "
        f"--output-dir {output_dir} --model-size 1b --tensor-parallel-size {tp} "
        f"--micro-batch-size 3 --write-interval {wi} "
        f"--pipeline-model-parallel-size {pp} --num-nodes 1 --devices {world_size}"
    )

    # Run the command in a subshell, using the temporary directory as the current working directory.
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    result = subprocess.run(
        command,
        shell=True,  # Use the shell to interpret wildcards (e.g. SDH*)
        cwd=tmp_path,  # Run in the temporary directory
        capture_output=True,  # Capture stdout and stderr for debugging
        env=env,  # Pass in the env where we override the master port.
        text=True,  # Decode output as text
    )

    # For debugging purposes, print the output if the test fails.
    if result.returncode != 0:
        sys.stderr.write("STDOUT:\n" + result.stdout + "\n")
        sys.stderr.write("STDERR:\n" + result.stderr + "\n")

    # Assert that the command completed successfully.
    assert result.returncode == 0, "train_evo2 command failed."

    # Assert that the output directory was created.
    pred_files = glob.glob(os.path.join(output_dir, "predictions__rank_*.pt"))
    if wi == "batch":
        assert len(pred_files) == 2, f"Expected 2 prediction file (for this test), got {len(pred_files)}"
    else:
        assert len(pred_files) == ddp, f"Expected {ddp} prediction file (for this test), got {len(pred_files)}"
    with open(output_dir / "seq_idx_map.json", "r") as f:
        seq_idx_map = json.load(
            f
        )  # This gives us the mapping from the sequence names to the indices in the predictions.
    preds = [torch.load(pf) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
        batch_dim_key_defaults={"token_logits": 0},
        seq_dim_key_defaults={"token_logits": 1},
    )
    assert isinstance(preds, dict)
    assert "token_logits" in preds
    assert "pad_mask" in preds
    assert "seq_idx" in preds

    assert len(preds["token_logits"]) == len(preds["pad_mask"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences
    for original_idx, pad_mask, token_logits in zip(preds["seq_idx"], preds["pad_mask"], preds["token_logits"]):
        # seq_idx is not sorted necessarily, so use the saved "seq_idx" to determine the original order.
        expected_len = target_sequence_lengths[original_idx]
        assert pad_mask.sum() == expected_len
        assert token_logits.shape == (max(target_sequence_lengths), 512)


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
    ids=lambda x: f"ddp={x[0]},cp={x[1]},pp={x[2]},tp={x[3]},fp8={x[4]},wi={x[5]}",
)
def test_predict_evo2_runs_with_log_probs(
    tmp_path,
    ddp: int,
    cp: int,
    pp: int,
    tp: int,
    fp8: bool,
    wi: str,
    checkpoint_7b_1m_path: Path,
    num_sequences: int = 5,
    target_sequence_lengths: list[int] = [2048, 2048, 2048, 2048, 2048],
):
    """
    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.

    For this test, we want coverage of CP, so we make sure sequence lengths are all the same and divisible by CP.
    """

    world_size = ddp * cp * pp * tp
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is less than the number of GPUs {torch.cuda.device_count()}")
    is_fp8_supported, _, _ = check_fp8_support(torch.cuda.current_device())
    if not is_fp8_supported and fp8:
        pytest.skip("FP8 is not supported on this GPU.")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )
    # Create a mock data directory.
    # a local copy of the environment
    env = dict(**os.environ)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
        env["NCCL_P2P_DISABLE"] = "1"

    fp8_option = "--fp8" if fp8 else ""
    # Build the command string.
    # Note: The command assumes that `train_evo2` is in your PATH.
    output_dir = tmp_path / "test_output"
    command = (
        f"torchrun --nproc_per_node {world_size} --nnodes 1 --no-python "
        f"predict_evo2 --fasta {fasta_file_path} --ckpt-dir {checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 --write-interval {wi} "
        f"--num-layers 4 --hybrid-override-pattern SDH* "  # subset of layers for testing
        f"--output-dir {output_dir} --model-size 7b_arc_longcontext --tensor-parallel-size {tp} {fp8_option} "
        f"--pipeline-model-parallel-size {pp} --context-parallel-size {cp} --num-nodes 1 --devices {world_size} "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )

    # Run the command in a subshell, using the temporary directory as the current working directory.
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    result = subprocess.run(
        command,
        shell=True,  # Use the shell to interpret wildcards (e.g. SDH*)
        cwd=tmp_path,  # Run in the temporary directory
        capture_output=True,  # Capture stdout and stderr for debugging
        env=env,  # Pass in the env where we override the master port.
        text=True,  # Decode output as text
    )

    # For debugging purposes, print the output if the test fails.
    if result.returncode != 0:
        sys.stderr.write("STDOUT:\n" + result.stdout + "\n")
        sys.stderr.write("STDERR:\n" + result.stderr + "\n")

    # Assert that the command completed successfully.
    assert result.returncode == 0, "train_evo2 command failed."

    # Assert that the output directory was created.
    pred_files = glob.glob(os.path.join(output_dir, "predictions__rank_*.pt"))
    if wi == "batch":
        assert len(pred_files) == 2, f"Expected 2 prediction file (for this test), got {len(pred_files)}"
    else:
        assert len(pred_files) == ddp, f"Expected {ddp} prediction file (for this test), got {len(pred_files)}"
    with open(output_dir / "seq_idx_map.json", "r") as f:
        seq_idx_map = json.load(
            f
        )  # This gives us the mapping from the sequence names to the indices in the predictions.
    preds = [torch.load(pf) for pf in pred_files]
    preds = [torch.load(pf) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
    )
    assert isinstance(preds, dict)
    assert "log_probs_seqs" in preds
    assert "seq_idx" in preds
    assert len(preds["log_probs_seqs"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences
    # TODO consider some kind of numerical test on the log probabilities returned. For now though there is no
    #  correct answer, and the model is just a subset so it is not even a real model we would expect a good result
    #  from. Checking that output is made without error will still capture API drift.
