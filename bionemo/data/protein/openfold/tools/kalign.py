# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Python wrapper for Kalign."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from nemo.utils import logging


class Kalign:
    """Python wrapper of the Kalign binary."""

    def __init__(self, binary_path: str, verbose: bool = False) -> None:
        """Initializes the Python Kalign wrapper.

        Args:
            binary_path: The path to the Kalign binary.
            verbose: Whether to print relevant information.

        """
        self.binary_path = binary_path
        self.verbose = verbose

    def align(self, sequences: List[str]) -> str:
        """Aligns the sequences and returns the alignment in A3M string.

        Args:
            sequences: A list of query sequence strings. The sequences have to be at
                least 6 residues long (Kalign requires this). Note that the order in
                which you give the sequences might alter the output slightly as
                different alignment tree might get constructed.

        Returns:
            A string with the alignment in a3m format.

        Raises:
            RuntimeError: If Kalign fails.
            ValueError: If any of the sequences is less than 6 residues long.

        """
        if self.verbose:
            logging.debug(f"Kalign input: {sequences}")
        assert len(sequences) >= 2

        for s in sequences:
            if len(s) < 6:
                raise ValueError(
                    "Kalign requires all sequences to be at least 6 residues long."
                    f" Sequence={repr(s)} has {len(s)} residues."
                )

        tmp_dirpath = Path(tempfile.mkdtemp(dir="/tmp"))

        input_fasta_path = tmp_dirpath / "input.fasta"
        output_a3m_path = tmp_dirpath / "output.a3m"

        with open(input_fasta_path, "w") as f:
            f.write(_to_a3m(sequences))

        kalign_cmd = [
            self.binary_path,
            "-i",
            str(input_fasta_path),
            "-o",
            str(output_a3m_path),
            "-format",
            "fasta",
        ]

        if self.verbose:
            logging.debug(f"Kalign command: {kalign_cmd}")

        try:
            process = subprocess.Popen(
                args=kalign_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Kalign is not installed or is not discoverable. You can install tested 3.3.5 version "
                " via install_third_party.sh located in examples/protein/openfold/scripts"
            ) from e

        try:
            retcode = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            logging.warning("Kalign timeout expired!")
            raise RuntimeError("Kalign timeout expired!\n")

        stdout, stderr = process.communicate()

        if self.verbose:
            logging.debug(f"Kalign stdout:\n{stdout.decode('utf-8')}\n" f"Kalign stderr:\n{stderr.decode('utf-8')}\n")

        if retcode != 0:
            shutil.rmtree(tmp_dirpath, ignore_errors=True)
            raise RuntimeError(
                "Kalign error!\n"
                f"Kalign stdout:\n{stdout.decode('utf-8')}\n"
                f"Kalign stderr:\n{stderr.decode('utf-8')}\n"
            )

        with open(output_a3m_path) as f:
            a3m = f.read()

        if self.verbose:
            logging.debug(f"Kalign output: {a3m}")

        shutil.rmtree(tmp_dirpath, ignore_errors=True)

        return a3m


def _to_a3m(sequences: List[str]) -> str:
    """Converts sequences to an a3m file."""
    names = [f"sequence {i}" for i in range(1, len(sequences) + 1)]
    a3m = []
    for sequence, name in zip(sequences, names):
        a3m.append(">" + name + "\n")
        a3m.append(sequence + "\n")
    return "".join(a3m)
