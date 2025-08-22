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

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# Set TRITON_LIBCUDA_PATH before check_fp8_support import that triggers Triton initialization
os.environ["TRITON_LIBCUDA_PATH"] = "/usr/local/cuda/lib64"

import pytest
import torch
from hydra import compose, initialize_config_dir
from transformer_engine.pytorch.fp8 import check_fp8_support

from train import main


_fp8_available, _fp8_reason = check_fp8_support()


@pytest.fixture(scope="session")
def session_temp_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("my-session-tempdir")
    return temp_dir


@pytest.mark.skipif(not _fp8_available, reason=f"FP8 is not supported on this GPU: {_fp8_reason}")
def test_train_sanity_config(monkeypatch, session_temp_dir: Path):
    """Test that train.py runs successfully with sanity config and creates expected outputs."""

    # Get the recipe directory
    recipe_dir = Path(__file__).parent

    # Set required environment variables for distributed training
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp8")
    monkeypatch.setenv("ACCELERATE_FP8_BACKEND", "TE")

    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(config_name="L0_sanity", overrides=[f"trainer.output_dir={session_temp_dir}"])

    main(sanity_config)

    output_dir = session_temp_dir

    # Check that the output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check for checkpoint directories
    checkpoint_dirs = [d for d in output_dir.iterdir() if d.is_dir() and re.match(r"checkpoint-\d+", d.name)]
    assert len(checkpoint_dirs) == 2, (
        f"Expected 2 checkpoint directories, found {len(checkpoint_dirs)}: {[d.name for d in checkpoint_dirs]}"
    )

    # Check for the final model checkpoint
    final_checkpoint = output_dir / "checkpoint-last"
    assert final_checkpoint.exists(), f"Final checkpoint directory {final_checkpoint} does not exist"

    # Verify the final checkpoint contains model files
    model_files = list(final_checkpoint.glob("*.safetensors"))
    config_files = list(final_checkpoint.glob("*.json"))

    assert len(model_files) > 0, f"No model files found in {final_checkpoint}"
    assert len(config_files) > 0, f"No config files found in {final_checkpoint}"

    # Check that training metrics were saved
    train_metrics_file = output_dir / "train_results.json"
    assert train_metrics_file.exists(), f"Training metrics file {train_metrics_file} does not exist"

    print(
        f"Successfully completed training test with fp8 precision. Found {len(checkpoint_dirs)} checkpoint directories and final model checkpoint."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not _fp8_available, reason=f"FP8 is not supported on this GPU: {_fp8_reason}")
def test_train_sanity_resume_from_checkpoint(monkeypatch, session_temp_dir: Path):
    """Test that train.py runs successfully with sanity config and resumes from checkpoint."""

    # Get the recipe directory
    recipe_dir = Path(__file__).parent

    # Remove the checkpoint-10 and checkpoint-last directories
    checkpoint_4 = session_temp_dir / "checkpoint-4"
    checkpoint_last = session_temp_dir / "checkpoint-last"
    if checkpoint_4.exists():
        shutil.rmtree(checkpoint_4)
    if checkpoint_last.exists():
        shutil.rmtree(checkpoint_last)

    assert (session_temp_dir / "checkpoint-2").exists(), (
        f"Checkpoint-2 directory {session_temp_dir / 'checkpoint-2'} does not exist."
    )

    # Set required environment variables for distributed training
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp8")
    monkeypatch.setenv("ACCELERATE_FP8_BACKEND", "TE")

    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(config_name="L0_sanity", overrides=[f"trainer.output_dir={session_temp_dir}"])

    main(sanity_config)

    output_dir = session_temp_dir

    # Check that the output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check for checkpoint directories
    checkpoint_dirs = [d for d in output_dir.iterdir() if d.is_dir() and re.match(r"checkpoint-\d+", d.name)]
    assert len(checkpoint_dirs) == 2, (
        f"Expected 2 checkpoint directories, found {len(checkpoint_dirs)}: {[d.name for d in checkpoint_dirs]}"
    )

    # Check for the final model checkpoint
    final_checkpoint = output_dir / "checkpoint-last"
    assert final_checkpoint.exists(), f"Final checkpoint directory {final_checkpoint} does not exist"

    # Verify the final checkpoint contains model files
    model_files = list(final_checkpoint.glob("*.safetensors"))
    config_files = list(final_checkpoint.glob("*.json"))

    assert len(model_files) > 0, f"No model files found in {final_checkpoint}"
    assert len(config_files) > 0, f"No config files found in {final_checkpoint}"

    # Check that training metrics were saved
    train_metrics_file = output_dir / "train_results.json"
    assert train_metrics_file.exists(), f"Training metrics file {train_metrics_file} does not exist"

    print(
        f"Successfully completed training test with fp8 precision. Found {len(checkpoint_dirs)} checkpoint directories and final model checkpoint."
    )


@pytest.mark.parametrize("accelerate_config", ["deepspeed_config.yaml", "fp8_config.yaml"])
def test_accelerate_launch(accelerate_config, tmp_path):
    """Test that accelerate launch runs successfully."""
    # Skip FP8 config test if FP8 is not supported
    if accelerate_config == "fp8_config.yaml" and not check_fp8_support()[0]:
        pytest.skip("FP8 not supported on this hardware")

    # Find the recipe directory and train.py
    recipe_dir = Path(__file__).parent
    train_py = recipe_dir / "train.py"
    accelerate_config_path = recipe_dir / "accelerate_config" / accelerate_config

    assert train_py.exists(), f"train.py not found at {train_py}"
    assert accelerate_config_path.exists(), f"{accelerate_config} not found at {accelerate_config_path}"

    # Run 'accelerate launch train.py' as a subprocess
    env = os.environ.copy()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(accelerate_config_path),
            str(train_py),
            "--config-name",
            "L0_sanity.yaml",
            f"trainer.output_dir={tmp_path}",
        ],
        cwd=recipe_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        timeout=240,
        env=env,
    )
