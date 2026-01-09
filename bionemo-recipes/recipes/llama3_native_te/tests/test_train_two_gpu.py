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

"""Multi-GPU training tests for Llama3.

These tests validate that the training scripts work correctly with multiple GPUs.
They require at least 2 GPUs to run and will be skipped on single-GPU machines.

Tests:
- DDP training on 2 GPUs
- FSDP2 training on 2 GPUs

Note: These tests don't check convergence, they just verify the training
      scripts run successfully without errors on multiple GPUs.
"""

import subprocess

import pytest
import torch


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

# TODO(@jomitchell): Delete once https://nvbugspro.nvidia.com/bug/5458694 is fixed.
requires_datacenter_hardware = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not any(
        gpu_name in torch.cuda.get_device_name(0).upper() for gpu_name in ["H100", "H200", "B100", "B200", "B300"]
    ),
    reason="Test requires datacenter hardware (H100, H200, B100, B200, B300)",
)


def run_train_cmd(cmd, recipe_path):
    """Run a training command and check for errors.

    Args:
        cmd: List of command arguments to run
        recipe_path: Path to the recipe directory (working directory for command)

    Raises:
        pytest.fail: If command returns non-zero exit code
    """
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,  # 4 minutes timeout
        cwd=str(recipe_path),
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command:\n{' '.join(cmd)}\nfailed with exit code {result.returncode}")


@requires_multi_gpu
def test_multi_gpu_train_ddp(tmp_path, recipe_path):
    """Test DDP training on 2 GPUs.

    This test validates:
    - DDP launches successfully with 2 processes
    - Both GPUs are utilized
    - Training completes without errors
    - Gradient synchronization works across GPUs

    The test runs only 4 training steps for speed.
    """
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",  # 2 processes = 2 GPUs
            "--standalone",  # Single node mode
            "train_ddp.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",  # Just 4 steps for speed
        ],
        recipe_path,
    )


@requires_multi_gpu
def test_multi_gpu_train_fsdp2(tmp_path, recipe_path):
    """Test FSDP2 training on 2 GPUs.

    This test validates:
    - FSDP2 launches successfully with 2 processes
    - Model sharding works across 2 GPUs
    - Training completes without errors
    - Parameter gathering/scattering works correctly

    The test runs only 4 training steps for speed.
    """
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",  # 2 processes = 2 GPUs
            "--standalone",  # Single node mode
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",  # Just 4 steps for speed
        ],
        recipe_path,
    )


@requires_multi_gpu
def test_multi_gpu_train_ddp_with_checkpointing(tmp_path, recipe_path):
    """Test DDP training on 2 GPUs with checkpoint saving.

    This test validates:
    - DDP can save checkpoints with multiple processes
    - Checkpoint files are created correctly
    - No race conditions in checkpoint saving
    """
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",
            "--standalone",
            "train_ddp.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=10",
            f"checkpoint.ckpt_dir={tmp_path}",
            "checkpoint.save_every_n_steps=5",
            "dataset.use_stateful_dataloader=true",  # Enable for checkpoint testing
        ],
        recipe_path,
    )

    # Verify checkpoint was created
    ckpt_dir = tmp_path / "train_ddp"
    assert ckpt_dir.exists(), f"Checkpoint directory not created: {ckpt_dir}"
    assert (ckpt_dir / "step_5").exists(), "Checkpoint at step 5 not found"


@requires_multi_gpu
def test_multi_gpu_train_fsdp2_with_checkpointing(tmp_path, recipe_path):
    """Test FSDP2 training on 2 GPUs with checkpoint saving.

    This test validates:
    - FSDP2 can save checkpoints with multiple processes
    - Sharded checkpoints are created correctly
    - No race conditions in checkpoint saving
    """
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",
            "--standalone",
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=10",
            f"checkpoint.ckpt_dir={tmp_path}",
            "checkpoint.save_every_n_steps=5",
            "dataset.use_stateful_dataloader=true",  # Enable for checkpoint testing
        ],
        recipe_path,
    )

    # Verify checkpoint was created
    ckpt_dir = tmp_path / "train_fsdp2"
    assert ckpt_dir.exists(), f"Checkpoint directory not created: {ckpt_dir}"
    assert (ckpt_dir / "step_5").exists(), "Checkpoint at step 5 not found"


@requires_multi_gpu
@requires_datacenter_hardware
@pytest.mark.xfail(reason="BIO-5: CP is still WIP")
def test_multi_gpu_train_te_fsdp2_cp(tmp_path, recipe_path):
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node=2",
            "--standalone",
            "train_fsdp2_cp.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
            "+cp_size=2",
        ],
        recipe_path,
    )
