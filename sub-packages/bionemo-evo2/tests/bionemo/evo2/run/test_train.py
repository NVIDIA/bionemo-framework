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
import argparse
import io
import os
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Tuple

import pytest
import torch
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from nemo import lightning as nl

from bionemo.evo2.run.train import parse_args, train
from bionemo.testing.megatron_parallel_state_utils import (
    distributed_model_parallel_state,
)


def run_train_with_std_redirect(args: argparse.Namespace) -> Tuple[str, nl.Trainer]:
    """
    Run a function with output capture.
    """
    with distributed_model_parallel_state():
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            trainer: nl.Trainer = train(args)

    train_stdout = stdout_buf.getvalue()
    train_stderr = stderr_buf.getvalue()
    print("Captured STDOUT:\n", train_stdout)
    print("Captured STDERR:\n", train_stderr)
    return train_stdout, trainer


def extract_global_steps_from_log(log_string):
    pattern = r"\| global_step: (\d+) \|"
    matches = re.findall(pattern, log_string)
    return [int(step) for step in matches]


def small_training_cmd(path, max_steps, val_check, additional_args=None):
    cmd = (
        f"train_evo2 --mock-data --result-dir {path} "
        "--model-size 1b_nv --num-layers 4 --hybrid-override-pattern SDH* --limit-val-batches 1 "
        "--no-activation-checkpointing --add-bias-output --create-tensorboard-logger --create-tflops-callback "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} --limit-val-batche 1 "
        f"--seq-length 64 --hidden-dropout 0.1 --attention-dropout 0.1 {additional_args}"
    )
    return cmd


@pytest.mark.timeout(256)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_runs(tmp_path, num_steps=5):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    open_port = find_free_network_port()
    # a local copy of the environment
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(open_port)

    # Build the command string.
    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_cmd(tmp_path, max_steps=num_steps, val_check=num_steps)

    # Run the command in a subshell, using the temporary directory as the current working directory.
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
    assert "reduced_train_loss:" in result.stdout
    assert result.returncode == 0, "train_evo2 command failed."


@pytest.mark.timeout(256)  # Optional: fail if the test takes too long.
@pytest.mark.slow
@pytest.mark.skip(
    reason="This test fails due to error when the checkpoints are saved. "
    "Issue: https://github.com/NVIDIA/bionemo-framework/issues/760"
)
def test_train_evo2_stops(tmp_path):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    open_port = find_free_network_port()
    # a local copy of the environment
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(open_port)
    max_steps = 500000
    early_stop_steps = 2
    val_check = 1
    additional_args = f"--early-stop-on-step {early_stop_steps}"
    # Expected location of logs and checkpoints
    log_dir = tmp_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"
    tensorboard_dir = log_dir / "dev"

    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_cmd(tmp_path, max_steps=max_steps, val_check=val_check, additional_args=additional_args)
    command_parts_no_program = shlex.split(command)[1:]
    args = parse_args(args=command_parts_no_program)
    train_stdout, trainer = run_train_with_std_redirect(args)

    assert f"Training epoch 0, iteration 0/{max_steps - 1}" in train_stdout
    # Extract and validate global steps
    global_steps = extract_global_steps_from_log(train_stdout)
    assert global_steps[0] == 0
    assert global_steps[-1] == (early_stop_steps - 1)
    assert len(global_steps) == max_steps

    expected_checkpoint_suffix = f"{early_stop_steps}.0-last"
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."
    )

    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir}"

    assert "reduced_train_loss" in trainer.logged_metrics  # validation logging on by default
    assert "tflops_per_sec_per_gpu" in trainer.logged_metrics  # ensuring that tflops logger can be added
    assert "train_step_timing in s" in trainer.logged_metrics


@pytest.mark.timeout(256)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_stop_at_max_steps_and_continue(tmp_path):
    # Setup
    open_port = find_free_network_port()
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(open_port)

    max_steps_first_run = 4
    max_steps_second_run = 6
    val_check_interval = 2
    # Expected location of logs and checkpoints
    log_dir = tmp_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"

    ######### FIRST TRAINING max_steps=max_steps_first_run #########
    # Parse args
    command_first_run = small_training_cmd(tmp_path, max_steps_first_run, val_check_interval)
    args_first_run = parse_args(shlex.split(command_first_run)[1:])
    train_stdout_first_run, _ = run_train_with_std_redirect(args_first_run)

    assert f"Training epoch 0, iteration 0/{max_steps_first_run - 1}" in train_stdout_first_run
    # Extract and validate global steps
    global_steps_first_run = extract_global_steps_from_log(train_stdout_first_run)
    assert global_steps_first_run[0] == 0
    assert global_steps_first_run[-1] == (max_steps_first_run - 1)
    assert len(global_steps_first_run) == max_steps_first_run

    expected_checkpoint_first_run_suffix = f"{max_steps_first_run}.0-last"
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_first_run_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_first_run_suffix}' found in {checkpoints_dir}."
    )

    ######### SECOND TRAINING continues previous training since max_steps=max_steps_second_run #########
    # Parse args
    command_second_run = small_training_cmd(tmp_path, max_steps_second_run, val_check_interval)
    args_second_run = parse_args(shlex.split(command_second_run)[1:])
    train_stdout_second_run, _ = run_train_with_std_redirect(args_second_run)

    # Assertions
    assert f"Training epoch 0, iteration 0/{max_steps_second_run - 1}" not in train_stdout_second_run
    assert f"Training epoch 0, iteration {max_steps_first_run}/{max_steps_second_run - 1}" in train_stdout_second_run
    # Extract and validate global steps
    global_steps_second_run = extract_global_steps_from_log(train_stdout_second_run)
    assert global_steps_second_run[0] == max_steps_first_run
    assert global_steps_second_run[-1] == max_steps_second_run - 1
    assert len(global_steps_second_run) == (max_steps_second_run - max_steps_first_run)
    expected_checkpoint_second_run_suffix = f"{max_steps_second_run}.0-last"

    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_second_run_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_second_run_suffix}' found in {checkpoints_dir}."
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_size",
    ["1b_nv"],
)
@pytest.mark.skip(reason="This test requires a gpu larger than the 24Gb L4s available on GitHub Actions.")
def test_train_single_gpu(tmp_path, model_size: str):
    """
    This test runs them single gpu evo2 training command with sample data in a temporary directory.
    """
    num_steps = 5
    open_port = find_free_network_port()
    # a local copy of the environment
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(open_port)

    additional_args = [
        "--result-dir",
        str(tmp_path),
        "--model",
        model_size,
        "--num-layers",
        str(4),
        "--val-check-interval",
        str(1),
        "--limit-val-batches",
        str(1),
        "--hybrid-override-pattern",
        "SDH*",
        "--no-activation-checkpointing",
        "--add-bias-output",
        "--max-steps",
        str(num_steps),
        "--warmup-steps",
        str(1),
        "--seq-length",
        str(128),
        "--wandb-offline",
        "--wandb-anonymous",
        "--mock-data",
    ]
    args = parse_args(args=additional_args)
    with distributed_model_parallel_state():
        train(args=args)


@pytest.mark.slow
@pytest.mark.distributed
@pytest.mark.parametrize("model_size", ["1b_nv"])
@pytest.mark.skip(
    reason="This tests requires to be run on a multi-gpu machine with torchrun --nproc_per_node=N_GPU -m pytest TEST_NAME"
)
def test_train_multi_gpu(tmp_path, model_size: str):
    """
    This test runs multi gpu distributed (tensor_model_parallel_size>1) evo2 training with sample data in a temporary directory.
    """
    num_steps = 5
    world_size = torch.cuda.device_count()
    print(f"Number of GPUs available: {world_size}")
    if world_size < 2:
        pytest.fail("This test requires at least 2 GPUs.")

    additional_args = [
        "--result-dir",
        str(tmp_path),
        "--model",
        model_size,
        "--add-bias-output",
        "--max-steps",
        str(num_steps),
        "--warmup-steps",
        str(1),
        "--wandb-offline",
        "--wandb-anonymous",
        "--devices",
        str(world_size),
        "--tensor-parallel-size",
        str(world_size),
    ]

    with distributed_model_parallel_state(devices=world_size, tensor_model_parallel_size=world_size):
        args = parse_args(args=additional_args)
        train(args=args)
