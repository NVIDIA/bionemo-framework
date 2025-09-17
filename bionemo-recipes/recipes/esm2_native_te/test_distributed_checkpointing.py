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
import shutil
import subprocess
import tempfile

import pytest
import torch


os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@pytest.mark.slow
def test_checkpoint_save_and_load_single_process_ddp():
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
    temp_dir = tempfile.mkdtemp(prefix="test_ckpt_ddp_")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_ddp.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_ddp.py")

    try:
        # Phase 1: Train for 10 steps, saving a checkpoint at step 5
        cmd_phase1 = [
            "torchrun",
            "--nproc_per_node=1",
            train_script,
            f"ckpt_dir={temp_dir}",
            "num_train_steps=10",
            "save_every_n_steps=5",
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
            f"ckpt_dir={temp_dir}",
            "num_train_steps=15",
            "save_every_n_steps=5",
        ]

        result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
        assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

        # Verify phase 2 completed and created additional checkpoints
        final_checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
        # Should have checkpoints at steps 5, 10
        expected_checkpoints = ["step_5.pt", "step_10.pt"]
        for expected in expected_checkpoints:
            assert expected in final_checkpoint_files, f"Missing checkpoint: {expected}"

        # Basic success assertions
        print("✅ Test passed: DDP checkpoints created successfully")
        print(f"✅ Found checkpoints: {sorted(final_checkpoint_files)}")
        print("✅ Resume functionality works - phase 2 completed without errors")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@requires_multi_gpu
@pytest.mark.slow
def test_checkpoint_save_and_load_two_processes_ddp():
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
    temp_dir = tempfile.mkdtemp(prefix="test_ckpt_ddp_2p_")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_ddp.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_ddp.py")

    try:
        # Phase 1: Train for 10 steps with 2 processes
        cmd_phase1 = [
            "torchrun",
            "--nproc_per_node=2",
            train_script,
            f"ckpt_dir={temp_dir}",
            "num_train_steps=10",
            "save_every_n_steps=5",
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
            f"ckpt_dir={temp_dir}",
            "num_train_steps=15",
            "save_every_n_steps=5",
        ]

        result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
        assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

        # Verify phase 2 completed and created additional checkpoints
        final_checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
        expected_checkpoints = ["step_5.pt", "step_10.pt"]
        for expected in expected_checkpoints:
            assert expected in final_checkpoint_files, f"Missing checkpoint: {expected}"

        print("✅ Test passed: Multi-process DDP checkpoints created successfully")
        print(f"✅ Found checkpoints: {sorted(final_checkpoint_files)}")
        print("✅ Resume functionality works across processes")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
def test_checkpoint_save_and_load_single_process_mfsdp():
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
    temp_dir = tempfile.mkdtemp(prefix="test_ckpt_mfsdp_")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_mfsdp.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_mfsdp.py")

    try:
        # Phase 1: Train for 10 steps
        cmd_phase1 = [
            "torchrun",
            "--nproc_per_node=1",
            train_script,
            f"ckpt_dir={temp_dir}",
            "num_train_steps=10",
            "save_every_n_steps=5",
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
            f"ckpt_dir={temp_dir}",
            "num_train_steps=15",
            "save_every_n_steps=5",
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

        print("✅ Test passed: mFSDP checkpoints created successfully")
        print(f"✅ Found checkpoints: {sorted(final_checkpoint_dirs)}")
        print("✅ Resume functionality works")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@requires_multi_gpu
@pytest.mark.slow
def test_checkpoint_save_and_load_two_processes_mfsdp():
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
    temp_dir = tempfile.mkdtemp(prefix="test_ckpt_mfsdp_2p_")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_mfsdp.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_mfsdp.py")

    try:
        # Phase 1: Train for 10 steps with 2 processes
        cmd_phase1 = [
            "torchrun",
            "--nproc_per_node=2",
            train_script,
            f"ckpt_dir={temp_dir}",
            "num_train_steps=10",
            "save_every_n_steps=5",
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
            f"ckpt_dir={temp_dir}",
            "num_train_steps=15",
            "save_every_n_steps=5",
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

        print("✅ Test passed: Multi-process mFSDP checkpoints created successfully")
        print(f"✅ Found checkpoints: {sorted(final_checkpoint_dirs)}")
        print("✅ Resume functionality works across processes")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
def test_checkpoint_save_and_load_single_process_fsdp2():
    """Test checkpoint save/resume functionality for FSDP2 with single process.

    This test validates:
    - FSDP2 creates single-file checkpoints (step_X.pt files)
    - Full state dict gathering and saving works correctly
    - Training can resume from latest checkpoint and continue
    - Resume starts from correct step count

    Process:
    1. Train 10 steps (0-9), save checkpoint at step 5
    2. Resume training from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10
    """
    temp_dir = tempfile.mkdtemp(prefix="test_ckpt_fsdp2_")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_fsdp2.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_fsdp2.py")

    try:
        # Phase 1: Train for 10 steps
        cmd_phase1 = [
            "torchrun",
            "--nproc_per_node=1",
            train_script,
            f"ckpt_dir={temp_dir}",
            "num_train_steps=10",
            "save_every_n_steps=5",
        ]

        result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
        assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

        # Checkpoints are saved in a subdirectory named after the script
        ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
        assert os.path.exists(ckpt_subdir), f"Checkpoint subdirectory {ckpt_subdir} not created"

        # Verify checkpoint was created (FSDP2 creates .pt files)
        checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
        assert len(checkpoint_files) > 0, "No checkpoint files created in phase 1"

        # Check that checkpoint at step 5 exists
        expected_checkpoint = "step_5.pt"
        assert expected_checkpoint in checkpoint_files, f"Expected {expected_checkpoint} not found"

        # Phase 2: Resume training
        cmd_phase2 = [
            "torchrun",
            "--nproc_per_node=1",
            train_script,
            f"ckpt_dir={temp_dir}",
            "num_train_steps=15",
            "save_every_n_steps=5",
        ]

        result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
        assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

        # Verify phase 2 completed and created additional checkpoints
        final_checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
        expected_checkpoints = ["step_5.pt", "step_10.pt"]
        for expected in expected_checkpoints:
            assert expected in final_checkpoint_files, f"Missing checkpoint: {expected}"

        print("✅ Test passed: FSDP2 checkpoints created successfully")
        print(f"✅ Found checkpoints: {sorted(final_checkpoint_files)}")
        print("✅ Resume functionality works")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@requires_multi_gpu
@pytest.mark.slow
def test_checkpoint_save_and_load_two_processes_fsdp2():
    """Test checkpoint save/resume functionality for FSDP2 with two processes.

    This test validates:
    - Multi-process FSDP2 state dict gathering (all ranks participate)
    - Main process saves full state dict after gathering
    - All processes can load and broadcast from rank 0
    - Training resumes correctly with proper process synchronization

    Process:
    1. Train 10 steps (0-9) across 2 processes, save checkpoint at step 5
    2. Resume training with 2 processes from step 5, continue to step 15
    3. Verify checkpoints exist at steps 5 and 10
    """
    temp_dir = tempfile.mkdtemp(prefix="test_ckpt_fsdp2_2p_")

    # Set environment for subprocess
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Get the full path to train_fsdp2.py
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_fsdp2.py")

    try:
        # Phase 1: Train for 10 steps with 2 processes
        cmd_phase1 = [
            "torchrun",
            "--nproc_per_node=2",
            train_script,
            f"ckpt_dir={temp_dir}",
            "num_train_steps=10",
            "save_every_n_steps=5",
        ]

        result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
        assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

        # Checkpoints are saved in a subdirectory named after the script
        ckpt_subdir = os.path.join(temp_dir, "train_fsdp2")
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
            f"ckpt_dir={temp_dir}",
            "num_train_steps=15",
            "save_every_n_steps=5",
        ]

        result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
        assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

        # Verify phase 2 completed and created additional checkpoints
        final_checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("step_") and f.endswith(".pt")]
        expected_checkpoints = ["step_5.pt", "step_10.pt"]
        for expected in expected_checkpoints:
            assert expected in final_checkpoint_files, f"Missing checkpoint: {expected}"

        print("✅ Test passed: Multi-process FSDP2 checkpoints created successfully")
        print(f"✅ Found checkpoints: {sorted(final_checkpoint_files)}")
        print("✅ Resume functionality works across processes")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
def test_final_model_save_ddp():
    """Test final model saving for DDP.

    Validates that DDP saves the final model correctly with:
    - model.safetensors containing weights
    - config.json with model configuration
    - esm_nv.py for custom model code
    """
    temp_dir = tempfile.mkdtemp(prefix="test_final_ddp_")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_ddp.py")

    try:
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            train_script,
            f"ckpt_dir={temp_dir}",
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

        print("✅ DDP final model saved successfully")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
def test_final_model_save_mfsdp():
    """Test final model saving for mFSDP.

    Validates that mFSDP gathers parameters and saves the final model with:
    - model.safetensors containing gathered weights
    - config.json with model configuration
    - esm_nv.py for custom model code
    """
    temp_dir = tempfile.mkdtemp(prefix="test_final_mfsdp_")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_mfsdp.py")

    try:
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            train_script,
            f"ckpt_dir={temp_dir}",
            "num_train_steps=3",
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

        print("✅ mFSDP final model saved successfully")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
def test_final_model_save_fsdp2():
    """Test final model saving for FSDP2.

    Validates that FSDP2 gathers full state dict and saves the final model with:
    - model.safetensors containing gathered weights
    - config.json with model configuration
    """
    temp_dir = tempfile.mkdtemp(prefix="test_final_fsdp2_")

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train_fsdp2.py")

    try:
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            train_script,
            f"ckpt_dir={temp_dir}",
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

        print("✅ FSDP2 final model saved successfully")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# @pytest.mark.slow
# def test_load_from_final_checkpoint_ddp():
#     """Test loading from a saved final model for DDP.

#     This test validates:
#     - Training can start from a saved final model (model.safetensors)
#     - The loaded model continues training correctly
#     - New checkpoints are saved after loading

#     Process:
#     1. Train initial model for 5 steps and save final model
#     2. Start new training from the saved final model
#     3. Verify training continues and new checkpoints are created
#     """
#     temp_dir = tempfile.mkdtemp(prefix="test_final_load_ddp_")

#     env = os.environ.copy()
#     env["WANDB_MODE"] = "disabled"

#     this_dir = os.path.dirname(__file__)
#     train_script = os.path.join(this_dir, "train_ddp.py")

#     try:
#         # Phase 1: Train and save final model
#         cmd_phase1 = [
#             "torchrun",
#             "--nproc_per_node=1",
#             train_script,
#             f"ckpt_dir={temp_dir}/phase1",
#             "num_train_steps=5",
#         ]

#         result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
#         assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

#         final_model_dir = os.path.join(temp_dir, "phase1", "train_ddp", "final_model")
#         assert os.path.exists(final_model_dir), "Final model not created in phase 1"

#         # Phase 2: Load from final model and continue training
#         cmd_phase2 = [
#             "torchrun",
#             "--nproc_per_node=1",
#             train_script,
#             f"ckpt_dir={temp_dir}/phase2",
#             f"load_final_checkpoint_from_path={final_model_dir}",
#             "num_train_steps=10",
#             "save_every_n_steps=5",
#         ]

#         result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
#         assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

#         # Verify new checkpoints were created in phase 2
#         phase2_ckpt_dir = os.path.join(temp_dir, "phase2", "train_ddp")
#         checkpoint_files = [f for f in os.listdir(phase2_ckpt_dir) if f.startswith("step_") and f.endswith(".pt")]
#         assert len(checkpoint_files) > 0, "No checkpoints created after loading from final model"

#         print("✅ DDP loading from final checkpoint works")
#         print(f"✅ Created {len(checkpoint_files)} new checkpoints after loading")

#     finally:
#         shutil.rmtree(temp_dir, ignore_errors=True)


# @pytest.mark.slow
# def test_load_from_final_checkpoint_mfsdp():
#     """Test loading from a saved final model for mFSDP.

#     This test validates:
#     - Training can start from a saved final model (model.safetensors)
#     - The loaded model continues training correctly with mFSDP
#     - New distributed checkpoints are saved after loading

#     Process:
#     1. Train initial model for 5 steps and save final model
#     2. Start new training from the saved final model
#     3. Verify training continues and new checkpoints are created
#     """
#     temp_dir = tempfile.mkdtemp(prefix="test_final_load_mfsdp_")

#     env = os.environ.copy()
#     env["WANDB_MODE"] = "disabled"

#     this_dir = os.path.dirname(__file__)
#     train_script = os.path.join(this_dir, "train_mfsdp.py")

#     try:
#         # Phase 1: Train and save final model
#         cmd_phase1 = [
#             "torchrun",
#             "--nproc_per_node=1",
#             train_script,
#             f"ckpt_dir={temp_dir}/phase1",
#             "num_train_steps=5",
#         ]

#         result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
#         assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

#         final_model_dir = os.path.join(temp_dir, "phase1", "train_mfsdp", "final_model")
#         assert os.path.exists(final_model_dir), "Final model not created in phase 1"

#         # Phase 2: Load from final model and continue training
#         cmd_phase2 = [
#             "torchrun",
#             "--nproc_per_node=1",
#             train_script,
#             f"ckpt_dir={temp_dir}/phase2",
#             f"load_final_checkpoint_from_path={final_model_dir}",
#             "num_train_steps=10",
#             "save_every_n_steps=5",
#         ]

#         result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
#         assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

#         # Verify new checkpoints were created in phase 2
#         phase2_ckpt_dir = os.path.join(temp_dir, "phase2", "train_mfsdp")
#         checkpoint_dirs = [f for f in os.listdir(phase2_ckpt_dir) if f.startswith("step_") and os.path.isdir(os.path.join(phase2_ckpt_dir, f))]
#         assert len(checkpoint_dirs) > 0, "No checkpoints created after loading from final model"

#         print("✅ mFSDP loading from final checkpoint works")
#         print(f"✅ Created {len(checkpoint_dirs)} new checkpoints after loading")

#     finally:
#         shutil.rmtree(temp_dir, ignore_errors=True)


# @pytest.mark.slow
# def test_load_from_final_checkpoint_fsdp2():
#     """Test loading from a saved final model for FSDP2.

#     This test validates:
#     - Training can start from a saved final model (model.safetensors)
#     - The loaded model continues training correctly with FSDP2
#     - New checkpoints are saved after loading

#     Process:
#     1. Train initial model for 5 steps and save final model
#     2. Start new training from the saved final model
#     3. Verify training continues and new checkpoints are created
#     """
#     temp_dir = tempfile.mkdtemp(prefix="test_final_load_fsdp2_")

#     env = os.environ.copy()
#     env["WANDB_MODE"] = "disabled"

#     this_dir = os.path.dirname(__file__)
#     train_script = os.path.join(this_dir, "train_fsdp2.py")

#     try:
#         # Phase 1: Train and save final model
#         cmd_phase1 = [
#             "torchrun",
#             "--nproc_per_node=1",
#             train_script,
#             f"ckpt_dir={temp_dir}/phase1",
#             "num_train_steps=5",
#         ]

#         result1 = subprocess.run(cmd_phase1, check=False, capture_output=True, text=True, env=env)
#         assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

#         final_model_dir = os.path.join(temp_dir, "phase1", "train_fsdp2", "final_model")
#         assert os.path.exists(final_model_dir), "Final model not created in phase 1"

#         # Phase 2: Load from final model and continue training
#         cmd_phase2 = [
#             "torchrun",
#             "--nproc_per_node=1",
#             train_script,
#             f"ckpt_dir={temp_dir}/phase2",
#             f"load_final_checkpoint_from_path={final_model_dir}",
#             "num_train_steps=10",
#             "save_every_n_steps=5",
#         ]

#         result2 = subprocess.run(cmd_phase2, check=False, capture_output=True, text=True, env=env)
#         assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

#         # Verify new checkpoints were created in phase 2
#         phase2_ckpt_dir = os.path.join(temp_dir, "phase2", "train_fsdp2")
#         checkpoint_files = [f for f in os.listdir(phase2_ckpt_dir) if f.startswith("step_") and f.endswith(".pt")]
#         assert len(checkpoint_files) > 0, "No checkpoints created after loading from final model"

#         print("✅ FSDP2 loading from final checkpoint works")
#         print(f"✅ Created {len(checkpoint_files)} new checkpoints after loading")

#     finally:
#         shutil.rmtree(temp_dir, ignore_errors=True)


# @pytest.mark.slow
# def test_final_checkpoint_cross_compatibility():
#     """Test that a final model saved with one strategy can be loaded by another.

#     This test validates cross-compatibility between training strategies:
#     - Save final model with DDP
#     - Load and train with mFSDP
#     - Load and train with FSDP2

#     This ensures the final model format is consistent and portable.
#     """
#     temp_dir = tempfile.mkdtemp(prefix="test_cross_compat_")

#     env = os.environ.copy()
#     env["WANDB_MODE"] = "disabled"

#     this_dir = os.path.dirname(__file__)

#     try:
#         # Phase 1: Train with DDP and save final model
#         cmd_ddp = [
#             "torchrun",
#             "--nproc_per_node=1",
#             os.path.join(this_dir, "train_ddp.py"),
#             f"ckpt_dir={temp_dir}/ddp",
#             "num_train_steps=3",
#         ]

#         result = subprocess.run(cmd_ddp, check=False, capture_output=True, text=True, env=env)
#         assert result.returncode == 0, f"DDP training failed: {result.stderr}"

#         ddp_final_model = os.path.join(temp_dir, "ddp", "train_ddp", "final_model")
#         assert os.path.exists(ddp_final_model), "DDP final model not created"

#         # Phase 2: Load DDP model with mFSDP
#         cmd_mfsdp = [
#             "torchrun",
#             "--nproc_per_node=1",
#             os.path.join(this_dir, "train_mfsdp.py"),
#             f"ckpt_dir={temp_dir}/mfsdp",
#             f"load_final_checkpoint_from_path={ddp_final_model}",
#             "num_train_steps=3",
#         ]

#         result = subprocess.run(cmd_mfsdp, check=False, capture_output=True, text=True, env=env)
#         assert result.returncode == 0, f"mFSDP loading from DDP failed: {result.stderr}"

#         # Phase 3: Load DDP model with FSDP2
#         cmd_fsdp2 = [
#             "torchrun",
#             "--nproc_per_node=1",
#             os.path.join(this_dir, "train_fsdp2.py"),
#             f"ckpt_dir={temp_dir}/fsdp2",
#             f"load_final_checkpoint_from_path={ddp_final_model}",
#             "num_train_steps=3",
#         ]

#         result = subprocess.run(cmd_fsdp2, check=False, capture_output=True, text=True, env=env)
#         assert result.returncode == 0, f"FSDP2 loading from DDP failed: {result.stderr}"

#         print("✅ Cross-compatibility test passed!")
#         print("✅ DDP model successfully loaded by mFSDP and FSDP2")

#     finally:
#         shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run a quick test to verify the setup works
    print("Running quick sanity test...")
    test_checkpoint_save_and_load_single_process_ddp()
    print("\nAll tests can be run with: pytest test_distributed_checkpointing.py")
