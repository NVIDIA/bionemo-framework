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

import torch
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.core.data.load import load
from bionemo.noodles.nvfaidx import NvFaidx
from bionemo.testing.data.fasta import ALU_SEQUENCE, create_fasta_file


def test_train_evo2_runs(
    tmp_path, num_sequences: int = 5, target_sequence_lengths: list[int] = [3149, 3140, 1024, 3149, 3149]
):
    """
    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )
    # Create a mock data directory.
    open_port = find_free_network_port()
    # a local copy of the environment
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(open_port)
    checkpoint_path = load("evo2/1b-8k:1.0", source="pbss")
    # Build the command string.
    # Note: The command assumes that `train_evo2` is in your PATH.
    output_dir = tmp_path / "test_output"
    command = (
        f"predict_evo2 --fasta {fasta_file_path} --ckpt-dir {checkpoint_path} "
        f"--output-dir {output_dir} --model-size 1b --tensor-parallel-size 1 "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1"
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
    assert result.returncode == 0, "train_evo2 command failed."

    # Assert that the output directory was created.
    pred_files = glob.glob(os.path.join(output_dir, "predictions__rank_*.pt"))
    assert len(pred_files) == 1, "Expected 1 prediction file (for this test), got {}".format(len(pred_files))
    with open(output_dir / "seq_idx_map.json", "r") as f:
        seq_idx_map = json.load(
            f
        )  # This gives us the mapping from the sequence names to the indices in the predictions.
    preds = torch.load(pred_files[0])
    assert isinstance(preds, dict)
    assert "token_logits" in preds
    assert "pad_mask" in preds
    assert "seq_idx" in preds
    assert len(preds["token_logits"]) == len(preds["pad_mask"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences
    fasta = NvFaidx(fasta_file_path)
    for i, seq_name in enumerate(sorted(fasta.keys())):
        expected_len = target_sequence_lengths[i]
        idx = seq_idx_map[seq_name]  # look up the out of order prediction index for this sequence.
        assert preds["pad_mask"][idx].sum() == expected_len
        assert preds["token_logits"][idx].shape == (max(target_sequence_lengths), 512)
