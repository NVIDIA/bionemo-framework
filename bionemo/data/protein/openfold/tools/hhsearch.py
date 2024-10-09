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

"""Library to run HHsearch from Python."""

import glob
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List


class HHSearch:
    """Python wrapper of the HHsearch binary."""

    def __init__(
        self,
        databases: List[str],
        binary_path: str = "hhsearch",
        n_cpu: int = 2,
        maxseq: int = 1_000_000,
    ):
        """Initializes the Python HHsearch wrapper.

        Args:
          binary_path: The path to the HHsearch executable.
          databases: A list of HHsearch database paths. This should be the
            common prefix for the database files (i.e. up to but not including
            _hhm.ffindex etc.)
          n_cpu: The number of CPUs to use
          maxseq: The maximum number of rows in an input alignment. Note that this
            parameter is only supported in HHBlits version 3.1 and higher.

        Raises:
          FileNotFoundError: If hh-search binary not found within the path.
          RunTimeError: If hh-search failed.
        """
        self.binary_path = binary_path
        self.databases = databases
        self.n_cpu = n_cpu
        self.maxseq = maxseq

        for database_path in self.databases:
            if not glob.glob(database_path + "pdb70_*"):
                raise ValueError(f"Could not find HHsearch database {repr(database_path)}")

    def query(self, a3m: str) -> str:
        """Queries the database using HHsearch using a given a3m."""
        tmp_dirpath = Path(tempfile.mkdtemp(dir="/tmp"))
        input_path = str(tmp_dirpath / "query.a3m")
        hhr_path = str(tmp_dirpath / "output.hhr")

        with open(input_path, "w") as f:
            f.write(a3m)

        db_cmd = []
        for db_path in self.databases:
            db_cmd.append("-d")
            db_cmd.append(db_path)
        cmd = [
            self.binary_path,
            "-i",
            input_path,
            "-o",
            hhr_path,
            "-maxseq",
            str(self.maxseq),
            "-cpu",
            str(self.n_cpu),
        ] + db_cmd
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()
            retcode = process.wait()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "hh-search is required for finding templates. Please install HH-suite v3.3.0 available in "
                "https://github.com/soedinglab/hh-suite/archive/refs/tags/v3.3.0.tar.gz. If you have already "
                "installed it, please make sure it is discoverable by shell (usually through PATH variable)."
            ) from e

        if retcode:
            # Stderr is truncated to prevent proto size errors in Beam.
            raise RuntimeError(
                "HHSearch failed:\nstdout:\n%s\n\nstderr:\n%s\n"
                % (stdout.decode("utf-8"), stderr[:100_000].decode("utf-8"))
            )

        with open(hhr_path) as f:
            hhr = f.read()

        shutil.rmtree(tmp_dirpath, ignore_errors=True)

        return hhr
