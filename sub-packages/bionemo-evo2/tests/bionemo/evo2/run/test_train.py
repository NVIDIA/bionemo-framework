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
import shlex
from contextlib import redirect_stderr, redirect_stdout
from typing import Tuple

import pytest
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from nemo import lightning as nl
from transformer_engine.pytorch.fp8 import check_fp8_support

from bionemo.evo2.run.train import parse_args, train
from bionemo.testing.lightning import extract_global_steps_from_log
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
from bionemo.testing.subprocess_utils import run_command_in_subprocess


fp8_available, reason_for_no_fp8 = check_fp8_support()


def run_train_with_std_redirect(args: argparse.Namespace) -> Tuple[str, nl.Trainer]:
    """
    Run a function with output capture.
    """
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        with distributed_model_parallel_state():
            trainer: nl.Trainer = train(args)

    train_stdout = stdout_buf.getvalue()
    train_stderr = stderr_buf.getvalue()
    print("Captured STDOUT:\n", train_stdout)
    print("Captured STDERR:\n", train_stderr)
    return train_stdout, trainer


def small_training_cmd(path, max_steps, val_check, devices: int = 1, additional_args: str = ""):
    cmd = (
        f"train_evo2 --mock-data --result-dir {path} --devices {devices} "
        "--model-size 1b_nv --num-layers 4 --hybrid-override-pattern SDH* --limit-val-batches 1 "
        "--no-activation-checkpointing --add-bias-output --create-tensorboard-logger --create-tflops-callback "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} --limit-val-batches 1 "
        f"--seq-length 8 --hidden-dropout 0.1 --attention-dropout 0.1 {additional_args}"
    )
    return cmd


@pytest.mark.timeout(256)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_runs(tmp_path):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    num_steps = 2
    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_cmd(tmp_path, max_steps=num_steps, val_check=num_steps)
    run_command_in_subprocess(command=command, path=str(tmp_path))

    log_dir = tmp_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"
    tensorboard_dir = log_dir / "dev"

    # Check if logs dir exists
    assert log_dir.exists(), "Logs folder should exist."
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    expected_checkpoint_suffix = f"{num_steps}.0-last"
    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."
    )

    # Check if directory with tensorboard logs exists
    assert tensorboard_dir.exists(), "TensorBoard logs folder does not exist."
    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir}"


@pytest.mark.timeout(256)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_stops(tmp_path):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    max_steps = 500000
    early_stop_steps = 4
    val_check = 2
    additional_args = f"--early-stop-on-step {early_stop_steps}"
    # Expected location of logs and checkpoints
    log_dir = tmp_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"

    assert not log_dir.exists(), "Logs folder shouldn't exist yet."

    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_cmd(tmp_path, max_steps=max_steps, val_check=val_check, additional_args=additional_args)
    command_parts_no_program = shlex.split(command)[1:]
    args = parse_args(args=command_parts_no_program)
    train_stdout, trainer = run_train_with_std_redirect(args)

    assert f"Training epoch 0, iteration 0/{early_stop_steps - 1}" in train_stdout
    # Extract and validate global steps
    global_steps = extract_global_steps_from_log(train_stdout)
    assert global_steps[0] == 0
    assert global_steps[-1] == (early_stop_steps - 1)
    assert trainer.global_step == early_stop_steps
    assert len(global_steps) == early_stop_steps

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

    assert "reduced_train_loss" in trainer.logged_metrics  # validation logging on by default
    assert "TFLOPS_per_GPU" in trainer.logged_metrics  # ensuring that tflops logger can be added
    assert "train_step_timing in s" in trainer.logged_metrics


@pytest.mark.slow
@pytest.mark.parametrize("model_size", ["7b_nv", "7b_arc_longcontext"])
def test_train_single_gpu(tmp_path, model_size: str):
    """
    This test runs them single gpu evo2 training command with sample data in a temporary directory.
    """
    num_steps = 7
    open_port = find_free_network_port()
    # a local copy of the environment
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(open_port)
    # Part 1: Make sure training runs for only --early-stop-on-step steps
    additional_args1 = [
        "--result-dir",
        str(tmp_path),
        "--model-size",
        model_size,
        "--num-layers",
        str(4),
        "--hybrid-override-pattern",
        "SDH*",
        "--no-activation-checkpointing",
        "--use-precision-aware-optimizer",
        "--add-bias-output",
        "--bf16-main-grads",
        "--val-check-interval",
        str(5),
        "--max-steps",
        str(num_steps),
        "--early-stop-on-step",
        str(num_steps - 2),
        "--warmup-steps",
        str(1),
        "--seq-length",
        str(128),
        "--wandb-offline",
        "--wandb-anonymous",
        "--mock-data",
    ]
    args1 = parse_args(args=additional_args1)
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf), distributed_model_parallel_state():
        train(args=args1)
    train_stdout = stdout_buf.getvalue()
    train_lines = train_stdout.split("\n")
    iteration_lines = [line for line in train_lines if "Training epoch" in line]
    assert len(iteration_lines) == 5
    iteration_line_1 = iteration_lines[0]
    # No strong opinion on how the total should be computed in the case of early stopping. We allow either for now
    #  unless there is an issue, such as with the LR scheduler...
    # TODO: Add a test somewhere that covers that early stopping callback has no impact on the LR scheduler
    assert "iteration 0/4" in iteration_line_1 or "iteration 0/6" in iteration_line_1
    iteration_line_final = iteration_lines[-1]
    assert "iteration 4/4" in iteration_line_final or "iteration 4/6" in iteration_line_final

    # Part 2: Make sure training picks up where it left off
    additional_args2 = [
        "--result-dir",
        str(tmp_path),
        "--model-size",
        model_size,
        "--num-layers",
        str(4),
        "--hybrid-override-pattern",
        "SDH*",
        "--no-activation-checkpointing",
        "--use-precision-aware-optimizer",
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
    args2 = parse_args(args=additional_args2)
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf), distributed_model_parallel_state():
        train(args=args2)
    train_stdout = stdout_buf.getvalue()
    train_lines = train_stdout.split("\n")
    iteration_lines = [line for line in train_lines if "Training epoch" in line]
    assert len(iteration_lines) == 2
    iteration_line_1 = iteration_lines[0]
    assert "iteration 5/6" in iteration_line_1
    iteration_line_2 = iteration_lines[1]
    assert "iteration 6/6" in iteration_line_2


@pytest.mark.parametrize(
    "additional_args",
    [
        pytest.param("", id="no_fp8"),
        pytest.param(
            "--fp8",
            marks=[
                pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8),
                pytest.mark.xfail(reason="FP8 test currently broken - TODO: fix"),
            ],
            id="fp8",
        ),
    ],
)
@pytest.mark.timeout(256)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_stop_at_max_steps_and_continue(tmp_path, additional_args):
    max_steps_first_run = 4
    max_steps_second_run = 6
    val_check_interval = 2
    # Expected location of logs and checkpoints
    log_dir = tmp_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"

    command_first_run = small_training_cmd(
        tmp_path, max_steps_first_run, val_check_interval, additional_args=additional_args
    )

    # The first training command to finish at max_steps_first_run
    stdout_first_run = run_command_in_subprocess(command=command_first_run, path=str(tmp_path))

    assert f"Training epoch 0, iteration 0/{max_steps_first_run - 1}" in stdout_first_run
    # Extract and validate global steps
    global_steps_first_run = extract_global_steps_from_log(stdout_first_run)

    assert global_steps_first_run[0] == 0
    assert global_steps_first_run[-1] == max_steps_first_run - 1
    assert len(global_steps_first_run) == max_steps_first_run

    expected_checkpoint_first_run_suffix = f"{max_steps_first_run}.0-last"
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."
    # Check if any ckpt subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_first_run_suffix in p.name)
    ]
    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_first_run_suffix}' found in {checkpoints_dir}."
    )

    # The second training command to continue from max_steps_first_run and finish at max_steps_second_run
    command_second_run = small_training_cmd(
        tmp_path, max_steps_second_run, val_check_interval, additional_args=additional_args
    )
    stdout_second_run = run_command_in_subprocess(command=command_second_run, path=str(tmp_path))
    global_steps_second_run = extract_global_steps_from_log(stdout_second_run)

    assert global_steps_second_run[0] == max_steps_first_run
    assert global_steps_second_run[-1] == max_steps_second_run - 1
    assert len(global_steps_second_run) == max_steps_second_run - max_steps_first_run

    expected_checkpoint_second_run_suffix = f"{max_steps_second_run}.0-last"
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_second_run_suffix in p.name)
    ]
    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_second_run_suffix}' found in {checkpoints_dir}."
    )
