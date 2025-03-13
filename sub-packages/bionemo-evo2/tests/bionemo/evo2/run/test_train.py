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

import io
import os
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

import pytest
import torch
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.evo2.run.train import parse_args, train
from bionemo.testing.megatron_parallel_state_utils import (
    distributed_model_parallel_state,
)


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
    command = (
        f"train_evo2 --mock-data --result-dir {tmp_path} "
        "--model-size 1b_nv --num-layers 4 --hybrid-override-pattern SDH* "
        "--no-activation-checkpointing --add-bias-output "
        f"--max-steps {num_steps} --warmup-steps 1 "
        "--seq-length 128 --hidden-dropout 0.1 --attention-dropout 0.1 "
    )

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
def test_train_evo2_stops(tmp_path, num_steps=500000, early_stop_steps=3):
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
    command = (
        f"train_evo2 --mock-data --result-dir {tmp_path} "
        "--model-size 1b_nv --num-layers 4 --hybrid-override-pattern SDH* "
        "--no-activation-checkpointing --add-bias-output "
        f"--max-steps {num_steps} --early-stop-on-step {early_stop_steps} --warmup-steps 1 "
        "--seq-length 128 --hidden-dropout 0.1 --attention-dropout 0.1 "
    )
    command_parts_no_program = shlex.split(command)[1:]
    args = parse_args(args=command_parts_no_program)
    with distributed_model_parallel_state():
        # Capture stdout/stderr during train function execution
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            train(args=args)
        # Get the captured output
        train_stdout = stdout_buffer.getvalue()
        train_stderr = stderr_buffer.getvalue()
        # Print the captured output for debugging
        print("TRAIN FUNCTION STDOUT:")
        print(train_stdout)
        print("TRAIN FUNCTION STDERR:")
        print(train_stderr)

    # Assert that the command completed successfully.
    assert "reduced_train_loss:" in train_stdout
    pattern = r"\| global_step: (\d+) \|"

    def extract_global_steps(log_string):
        matches = re.findall(pattern, log_string)
        return [int(step) for step in matches]

    global_step_ints = extract_global_steps(train_stdout)
    assert global_step_ints[-1] == early_stop_steps - 1
    assert len(global_step_ints) == early_stop_steps


@pytest.mark.slow
@pytest.mark.parametrize("model_size", ["1b_nv"])
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
