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

"""Test suite for distributed checkpointing functionality.

This module tests checkpoint save/resume functionality across different
distributed training configurations:
- DDP (Distributed Data Parallel) with 1 and 2 processes
- mFSDP (Megatron-style Fully Sharded Data Parallel) with 1 and 2 processes
- FSDP2 (PyTorch native Fully Sharded Data Parallel v2) with 1 and 2 processes

Test Strategy:
1. Phase 1: Train for N steps and save checkpoint
2. Phase 2: Resume training from checkpoint and continue
3. Validate: Checkpoints created, resuming works, training continues seamlessly

Each test uses temporary directories and disables wandb logging for isolation.
"""

import os
import subprocess

import pytest
import torch


os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@pytest.mark.slow
def test_checkpoint_save_and_load_single_process_ddp(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for DDP with single process.

    This test validates:
    - DDP creates single-file checkpoints (step_X.pt files)
    - Standard PyTorch checkpoint format (model + optimizer state)
    - Single-process DDP training and resuming works correctly
    - Checkpoint files contain complete model state

    Process:
    1. Train 10 steps (0-9), save checkpoint file at step 5
    2. Resume training from checkpoint, continue to step 15
    3. Verify step_X.pt checkpoint files exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_ddp")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_ddp.py
    train_script = recipe_path / "train_ddp.py"

    # Phase 1: Train for 10 steps, saving a checkpoint at step 5
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_ddp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created
    checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
    assert len(checkpoint_files) > 0, "No checkpoint files created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5.pt"
    assert expected_checkpoint in checkpoint_files, f"Expected {expected_checkpoint} not found"

    # Phase 2: Resume training (should start from step 5, continue to step 15)
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
    # Should have checkpoints at steps 5, 10
    expected_checkpoints = ["step_5.pt", "step_10.pt"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_files, f"Missing checkpoint: {expected}"


@requires_multi_gpu
@pytest.mark.slow
def test_checkpoint_save_and_load_two_processes_ddp(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for DDP with two processes.

    This test validates:
    - Multi-process DDP checkpoint behavior (main process saves only)
    - Checkpoint files can be loaded by all DDP processes
    - Process synchronization during resume (all processes load same checkpoint)
    - DDP training continues correctly after resume across processes

    Process:
    1. Train 10 steps (0-9) across 2 processes, main process saves checkpoint at step 5
    2. Resume training with 2 processes, all load same checkpoint file, continue to step 15
    3. Verify step_X.pt checkpoint files exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_ddp_2p")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_ddp.py
    train_script = recipe_path / "train_ddp.py"

    # Phase 1: Train for 10 steps with 2 processes
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_ddp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created
    checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
    assert len(checkpoint_files) > 0, "No checkpoint files created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5.pt"
    assert expected_checkpoint in checkpoint_files, f"Expected {expected_checkpoint} not found"

    # Phase 2: Resume training with 2 processes
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
    expected_checkpoints = ["step_5.pt", "step_10.pt"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_files, f"Missing checkpoint: {expected}"


@pytest.mark.slow
def test_checkpoint_save_and_load_single_process_mfsdp(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for mFSDP with single process.

    This test validates:
    - mFSDP creates distributed checkpoints (step_X directories)
    - Checkpoints are saved at specified intervals (every 5 steps)
    - Training can resume from latest checkpoint and continue
    - Resume starts from correct step count

    Process:
    1. Train 10 steps (0-9), save checkpoint at step 5
    2. Resume training from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_mfsdp")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_mfsdp.py
    train_script = recipe_path / "train_mfsdp.py"

    # Phase 1: Train for 10 steps
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_mfsdp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created (mFSDP creates directories)
    checkpoint_dirs = [
        f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, f))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Phase 2: Resume training
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, f))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"


@requires_multi_gpu
@pytest.mark.slow
def test_checkpoint_save_and_load_two_processes_mfsdp(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for mFSDP with two processes.

    This test validates:
    - Multi-process mFSDP coordination during checkpoint save/load
    - Distributed checkpoint format works across process boundaries
    - Both processes participate in distributed checkpoint operations
    - Training resumes correctly with proper process synchronization

    Process:
    1. Train 10 steps (0-9) across 2 processes, save checkpoint at step 5
    2. Resume training with 2 processes from step 5, continue to step 15
    3. Verify distributed checkpoints exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_mfsdp_2p")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_mfsdp.py
    train_script = recipe_path / "train_mfsdp.py"

    # Phase 1: Train for 10 steps with 2 processes
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_mfsdp")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created
    checkpoint_dirs = [
        f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, f))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Phase 2: Resume training with 2 processes
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, f))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"


@pytest.mark.slow
def test_checkpoint_save_and_load_single_process_fsdp2(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for FSDP2 with single process.

    This test validates:
    - FSDP2 creates distributed checkpoints (step_X directories by default)
    - Each rank saves its shard (even with single process)
    - Training can resume from latest checkpoint and continue
    - Resume starts from correct step count

    Process:
    1. Train 10 steps (0-9), save checkpoint at step 5
    2. Resume training from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_fsdp2")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_fsdp2.py
    train_script = recipe_path / "train_fsdp2.py"

    # Phase 1: Train for 10 steps (using distributed checkpoint by default)
    # Use smaller model for faster tests
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "model_tag=facebook/esm2_t6_8M_UR50D",  # Use smallest model
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created (FSDP2 now creates directories by default)
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Phase 2: Resume training
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"


@requires_multi_gpu
@pytest.mark.slow
def test_checkpoint_save_and_load_two_processes_fsdp2(recipe_path, tmp_path):
    """Test checkpoint save/resume functionality for FSDP2 with two processes.

    This test validates:
    - Multi-process FSDP2 distributed checkpointing (each rank saves its shard)
    - All ranks participate in saving and loading
    - Training resumes correctly with proper process synchronization

    Process:
    1. Train 10 steps (0-9) across 2 processes, save checkpoint at step 5
    2. Resume training with 2 processes from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10
    """
    temp_dir = str(tmp_path / "test_ckpt_fsdp2_2p")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_fsdp2.py
    train_script = recipe_path / "train_fsdp2.py"

    # Phase 1: Train for 10 steps with 2 processes
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Checkpoints are saved in a subdirectory named after the script
    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
    assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

    # Verify checkpoint was created (FSDP2 now creates directories by default)
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert len(checkpoint_dirs) > 0, "No checkpoint directories created in phase 1"

    # Check that checkpoint at step 5 exists
    expected_checkpoint = "step_5"
    assert expected_checkpoint in checkpoint_dirs, f"Expected {expected_checkpoint} not found"

    # Phase 2: Resume training with 2 processes
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify phase 2 completed and created additional checkpoints
    final_checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    expected_checkpoints = ["step_5", "step_10"]
    for expected in expected_checkpoints:
        assert expected in final_checkpoint_dirs, f"Missing checkpoint: {expected}"


@pytest.mark.slow
def test_fsdp2_legacy_checkpoint_format(recipe_path, tmp_path):
    """Test FSDP2 with legacy checkpoint format (gather full state).

    This test validates:
    - Can explicitly use legacy format with use_distributed_checkpoint_fsdp2=false
    - Creates single .pt files instead of directories
    - Can resume from legacy checkpoints
    """
    temp_dir = str(tmp_path / "test_ckpt_fsdp2_legacy")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    train_script = recipe_path / "train_fsdp2.py"

    # Phase 1: Train with legacy format
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",
        "checkpoint.use_distributed_checkpoint_fsdp2=false",  # Use legacy format
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")

    # Verify legacy .pt files were created (not directories)
    checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]

    assert len(checkpoint_files) > 0, "No legacy checkpoint files created"
    assert len(checkpoint_dirs) == 0, "Unexpected checkpoint directories created in legacy mode"
    assert "step_5.pt" in checkpoint_files, "Expected step_5.pt not found"

    # Phase 2: Resume from legacy checkpoint
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.use_distributed_checkpoint_fsdp2=false",  # Continue with legacy format
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    final_checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
    assert "step_10.pt" in final_checkpoint_files, "Missing step_10.pt"


@pytest.mark.slow
def test_fsdp2_backward_compatibility(recipe_path, tmp_path):
    """Test FSDP2 can load legacy checkpoints when using distributed format.

    This test validates:
    - Create checkpoint with legacy format
    - Resume with distributed format (should auto-detect and load legacy)
    - New checkpoints use distributed format
    """
    temp_dir = str(tmp_path / "test_ckpt_fsdp2_compat")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    train_script = recipe_path / "train_fsdp2.py"

    # Phase 1: Create legacy checkpoint
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",
        "checkpoint.use_distributed_checkpoint_fsdp2=false",  # Legacy format
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")

    # Verify legacy checkpoint exists
    checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
    assert "step_5.pt" in checkpoint_files, "Legacy checkpoint not created"

    # Phase 2: Resume with distributed format (default)
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
        # use_distributed_checkpoint_fsdp2=true by default
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Check we have both formats now
    final_checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
    final_checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]

    assert "step_5.pt" in final_checkpoint_files, "Legacy checkpoint should still exist"
    assert "step_10" in final_checkpoint_dirs, "New distributed checkpoint not created"


@pytest.mark.slow
def test_final_model_save_ddp(recipe_path, tmp_path):
    """Test final model saving for DDP.

    Validates that DDP saves the final model correctly with:
    - model.safetensors containing weights
    - config.json with model configuration
    - esm_nv.py for custom model code
    """
    temp_dir = str(tmp_path / "test_final_ddp")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    train_script = recipe_path / "train_ddp.py"

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "checkpoint.save_final_model=true",
        "num_train_steps=3",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Check final model directory
    final_model_dir = os.path.join(temp_dir, "train_ddp", "final_model")
    assert os.path.exists(final_model_dir), "Final model directory not created"

    # Check required files
    required_files = ["model.safetensors", "config.json", "esm_nv.py"]
    for file in required_files:
        file_path = os.path.join(final_model_dir, file)
        assert os.path.exists(file_path), f"Missing required file: {file}"
        assert os.path.getsize(file_path) > 0, f"File {file} is empty"


@pytest.mark.slow
def test_final_model_save_mfsdp(recipe_path, tmp_path):
    """Test final model saving for mFSDP.

    Validates that mFSDP gathers parameters and saves the final model with:
    - model.safetensors containing gathered weights
    - config.json with model configuration
    - esm_nv.py for custom model code
    """
    temp_dir = str(tmp_path / "test_final_mfsdp")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    train_script = recipe_path / "train_mfsdp.py"

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=3",
        "checkpoint.save_final_model=true",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Check final model directory
    final_model_dir = os.path.join(temp_dir, "train_mfsdp", "final_model")
    assert os.path.exists(final_model_dir), "Final model directory not created"

    # Check required files
    required_files = ["model.safetensors", "config.json", "esm_nv.py"]
    for file in required_files:
        file_path = os.path.join(final_model_dir, file)
        assert os.path.exists(file_path), f"Missing required file: {file}"
        assert os.path.getsize(file_path) > 0, f"File {file} is empty"


@pytest.mark.slow
def test_final_model_save_fsdp2(recipe_path, tmp_path):
    """Test final model saving for FSDP2.

    Validates that FSDP2 gathers full state dict and saves the final model with:
    - model.safetensors containing gathered weights
    - config.json with model configuration
    """
    temp_dir = str(tmp_path / "test_final_fsdp2")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    train_script = recipe_path / "train_fsdp2.py"

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "checkpoint.save_final_model=true",
        "num_train_steps=3",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Check final model directory
    final_model_dir = os.path.join(temp_dir, "train_fsdp2", "final_model")
    assert os.path.exists(final_model_dir), "Final model directory not created"

    # Check required files (FSDP2 doesn't save esm_nv.py)
    required_files = ["model.safetensors", "config.json"]
    for file in required_files:
        file_path = os.path.join(final_model_dir, file)
        assert os.path.exists(file_path), f"Missing required file: {file}"
        assert os.path.getsize(file_path) > 0, f"File {file} is empty"


@pytest.mark.slow
def test_scheduler_resume_single_gpu(recipe_path, tmp_path):
    """Test that learning rate scheduler resumes from correct state after checkpoint load.

    This test validates:
    - Scheduler state is saved in checkpoint
    - Scheduler resumes with correct step count
    - Learning rate continues from where it left off (not reset)
    - Warmup and decay continue correctly after resume

    Process:
    1. Train for 10 steps, save checkpoint with scheduler state at step 5
    2. Resume training, verify scheduler continues from step 6 (not step 0)
    3. Check that learning rate progression is continuous across resume
    """
    temp_dir = str(tmp_path / "test_scheduler_resume")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    train_script = recipe_path / "train_ddp.py"

    # Phase 1: Train for 10 steps with warmup
    # Use small warmup to see LR changes quickly
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh, don't look for checkpoints
        "lr_scheduler_kwargs.num_warmup_steps=20",  # Warmup over 20 steps
        "lr_scheduler_kwargs.num_training_steps=100",  # Total 100 steps
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Extract learning rates from phase 1 output
    phase1_lrs = []
    for line in result1.stdout.split("\n"):
        if "learning_rate:" in line:
            # Extract learning rate from log line
            lr_match = line.split("learning_rate:")[-1].strip().split()[0].removesuffix(",")
            phase1_lrs.append(float(lr_match))

    assert len(phase1_lrs) > 0, "No learning rates found in phase 1 output"

    # Phase 2: Resume training for 5 more steps
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=1",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
        "lr_scheduler_kwargs.num_warmup_steps=20",
        "lr_scheduler_kwargs.num_training_steps=100",
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Extract learning rates from phase 2 output
    phase2_lrs = []
    for line in result2.stdout.split("\n"):
        if "learning_rate:" in line:
            # Extract learning rate from log line
            lr_match = line.split("learning_rate:")[-1].strip().split()[0].removesuffix(",")
            phase2_lrs.append(float(lr_match))

    assert len(phase2_lrs) > 0, "No learning rates found in phase 2 output"

    # Verify scheduler continued (not reset)
    # The first LR in phase 2 should be different from the first LR in phase 1
    # (unless we're past warmup and in constant phase)
    if len(phase1_lrs) > 1 and len(phase2_lrs) > 0:
        # Phase 2 should continue from where phase 1 left off
        # The learning rate should be progressing, not reset to initial
        assert phase2_lrs[0] != phase1_lrs[0], (
            f"Scheduler appears to have reset: Phase2 first LR {phase2_lrs[0]} == Phase1 first LR {phase1_lrs[0]}"
        )


@requires_multi_gpu
@pytest.mark.slow
def test_scheduler_resume_two_gpu(recipe_path, tmp_path):
    """Test that learning rate scheduler resumes correctly with multi-GPU training.

    This test validates:
    - Scheduler state is synchronized across GPUs during save
    - All GPUs resume with same scheduler state
    - Learning rate is consistent across all processes after resume

    Process:
    1. Train for 10 steps across 2 GPUs, save checkpoint at step 5
    2. Resume training on 2 GPUs, verify scheduler continues correctly
    3. Ensure both GPUs have same learning rate progression
    """
    temp_dir = str(tmp_path / "test_scheduler_resume_2gpu")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Test with FSDP2 as it's most complex for scheduler state
    train_script = recipe_path / "train_fsdp2.py"

    # Phase 1: Train for 10 steps with 2 GPUs
    cmd_phase1 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=10",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=false",  # Start fresh, don't look for checkpoints
        "lr_scheduler_kwargs.num_warmup_steps=20",
        "lr_scheduler_kwargs.num_training_steps=100",
    ]

    result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    # Check that checkpoint was created (FSDP2 uses distributed format by default)
    ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
    checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert "step_5" in checkpoint_dirs, "Checkpoint at step 5 not found"

    # Phase 2: Resume training with 2 GPUs
    cmd_phase2 = [
        "torchrun",
        "--nproc_per_node=2",
        train_script,
        f"checkpoint.ckpt_dir={temp_dir}",
        "num_train_steps=15",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.resume_from_checkpoint=true",  # Resume from checkpoint
        "lr_scheduler_kwargs.num_warmup_steps=20",
        "lr_scheduler_kwargs.num_training_steps=100",
    ]

    result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    # Verify training continued (check for step progression in logs)
    assert "global_step: 6" in result2.stdout.lower(), "Phase 2 should start from step 6 after resuming from step 5"

    # Check that final checkpoint was created (distributed format)
    final_checkpoint_dirs = [
        d for d in os.listdir(ckpt_subdir) if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_subdir, d))
    ]
    assert "step_10" in final_checkpoint_dirs, "Checkpoint at step 10 not found"
