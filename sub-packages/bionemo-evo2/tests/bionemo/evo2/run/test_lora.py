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
import pytest

from bionemo.testing.subprocess_utils import run_command_in_subprocess

from .common import small_training_cmd, small_training_finetune_cmd


@pytest.mark.timeout(512)  # Optional: fail if the test takes too long.
@pytest.mark.slow
@pytest.mark.parametrize("with_peft", [True, False])
def test_train_evo2_finetune_runs_lora(tmp_path, with_peft: bool):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    num_steps = 2
    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_cmd(tmp_path / "pretrain", max_steps=num_steps, val_check=num_steps)
    stdout_pretrain: str = run_command_in_subprocess(command=command, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" not in stdout_pretrain

    log_dir = tmp_path / "pretrain" / "evo2"
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
    assert len(matching_subfolders) == 1, "Only one checkpoint subfolder should be found."
    if with_peft:
        result_dir = tmp_path / "lora_finetune"
        additional_args = "--lora-finetune"
    else:
        result_dir = tmp_path / "finetune"
        additional_args = ""

    command_finetune = small_training_finetune_cmd(
        result_dir,
        max_steps=num_steps,
        val_check=num_steps,
        prev_ckpt=matching_subfolders[0],
        additional_args=additional_args,
    )
    stdout_finetune: str = run_command_in_subprocess(command=command_finetune, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" in stdout_finetune

    log_dir_ft = result_dir / "evo2"
    checkpoints_dir_ft = log_dir_ft / "checkpoints"
    tensorboard_dir_ft = log_dir_ft / "dev"

    # Check if logs dir exists
    assert log_dir_ft.exists(), "Logs folder should exist."
    # Check if checkpoints dir exists
    assert checkpoints_dir_ft.exists(), "Checkpoints folder does not exist."

    expected_checkpoint_suffix = f"{num_steps}.0-last"
    # Check if any subfolder ends with the expected suffix
    matching_subfolders_ft = [
        p for p in checkpoints_dir_ft.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders_ft, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir_ft}."
    )

    # Check if directory with tensorboard logs exists
    assert tensorboard_dir_ft.exists(), "TensorBoard logs folder does not exist."
    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir_ft.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir_ft}"

    assert len(matching_subfolders_ft) == 1, "Only one checkpoint subfolder should be found."

    if with_peft:
        result_dir = tmp_path / "finetune_resume"

        log_dir_ft = result_dir / "evo2"
        checkpoints_dir_ft = log_dir_ft / "checkpoints"
        tensorboard_dir_ft = log_dir_ft / "dev"

        # Check if logs dir exists
        assert log_dir_ft.exists(), "Logs folder should exist."
        # Check if checkpoints dir exists
        assert checkpoints_dir_ft.exists(), "Checkpoints folder does not exist."

        # Resume from LoRA checkpoint
        command_resume_finetune = small_training_finetune_cmd(
            result_dir,
            max_steps=num_steps,
            val_check=num_steps,
            prev_ckpt=matching_subfolders[0],
            additional_args=f"--lora-finetune --lora-ckpt-path {tmp_path / 'something' }"
        )
        stdout_finetune: str = run_command_in_subprocess(command=command_resume_finetune, path=str(tmp_path))
