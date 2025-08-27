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

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@pytest.fixture(scope="session")
def session_temp_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("my-session-tempdir")
    return temp_dir


def test_train_can_resume_from_checkpoint(monkeypatch, session_temp_dir: Path):
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

    ## Remove last two checkpoints and re-train

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

    # Re-train
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


@pytest.mark.parametrize(
    "accelerate_config,model_tag",
    [
        ("default.yaml", "nvidia/esm2_t6_8M_UR50D"),
        # ("fsdp1_te.yaml", "nvidia/esm2_t6_8M_UR50D"),
        ("fsdp2_te.yaml", "nvidia/esm2_t6_8M_UR50D"),
        ("default.yaml", "facebook/esm2_t6_8M_UR50D"),
        # ("fsdp1_hf.yaml", "facebook/esm2_t6_8M_UR50D"),
        ("fsdp2_hf.yaml", "facebook/esm2_t6_8M_UR50D"),
    ],
    # FSDP1 seems to be failing for single-node / NO_SHARD until
    # https://github.com/pytorch/pytorch/pull/154369 is brought in.
)
def test_accelerate_launch(accelerate_config, model_tag, tmp_path):
    """Test that accelerate launch runs successfully."""

    # Find the recipe directory and train.py
    recipe_dir = Path(__file__).parent
    train_py = recipe_dir / "train.py"
    accelerate_config_path = recipe_dir / "accelerate_config" / accelerate_config

    assert train_py.exists(), f"train.py not found at {train_py}"
    assert accelerate_config_path.exists(), f"deepspeed_config.yaml not found at {accelerate_config_path}"

    # Run 'accelerate launch train.py' as a subprocess
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(accelerate_config_path),
        "--num_processes",
        "1",
        str(train_py),
        "--config-name",
        "L0_sanity.yaml",
        f"model_tag={model_tag}",
        f"trainer.output_dir={tmp_path}",
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command:\n{' '.join(cmd)}\nfailed with exit code {result.returncode}")


@requires_multi_gpu
@pytest.mark.parametrize(
    "accelerate_config,model_tag",
    [
        ("default.yaml", "nvidia/esm2_t6_8M_UR50D"),
        # Currently failing for some reason with device types not matching, oddly a local modeling_esm_te import seems
        # to fix it.
        # ("fsdp1_te.yaml", "nvidia/esm2_t6_8M_UR50D"),
        ("fsdp2_te.yaml", "nvidia/esm2_t6_8M_UR50D"),
        ("default.yaml", "facebook/esm2_t6_8M_UR50D"),
        ("fsdp1_hf.yaml", "facebook/esm2_t6_8M_UR50D"),
        ("fsdp2_hf.yaml", "facebook/esm2_t6_8M_UR50D"),
    ],
)
def test_accelerate_launch_multi_gpu(accelerate_config, model_tag, tmp_path):
    """Test that accelerate launch runs successfully."""

    # Find the recipe directory and train.py
    recipe_dir = Path(__file__).parent
    train_py = recipe_dir / "train.py"
    accelerate_config_path = recipe_dir / "accelerate_config" / accelerate_config

    assert train_py.exists(), f"train.py not found at {train_py}"
    assert accelerate_config_path.exists(), f"deepspeed_config.yaml not found at {accelerate_config_path}"

    # Run 'accelerate launch train.py' as a subprocess
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(accelerate_config_path),
        "--num_processes",
        "2",
        str(train_py),
        "--config-name",
        "L0_sanity.yaml",
        f"model_tag={model_tag}",
        f"trainer.output_dir={tmp_path}",
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command:\n{' '.join(cmd)}\nfailed with exit code {result.returncode}")
