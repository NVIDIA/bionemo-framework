# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Exhaustive small-reference protein search shared by online and final scoring."""

import shutil
import subprocess
import time
import uuid
from pathlib import Path


def _write_exhaustive_prefilter(query_index: Path, target_index: Path, output: Path) -> None:
    """Give every query the complete target list as one NUL-terminated MMseqs entry."""
    # MMseqs f71d0a6 blastp.sh/fake_pref symlinks a plain target.index here,
    # but Alignment.cpp reads until NUL. An exactly page-aligned index can then
    # overrun its mapping. Copy the bytes privately; never append to target.index.
    with output.open("xb") as stream, target_index.open("rb") as targets:
        shutil.copyfileobj(targets, stream)
        stream.write(b"\0")
        size = stream.tell()
    with Path(f"{output}.index").open("x") as index, query_index.open() as queries:
        index.writelines(f"{query.split()[0]}\t0\t{size}\n" for query in queries)
    with Path(f"{output}.dbtype").open("xb") as dbtype:
        dbtype.write(b"\x07\0\0\0")  # MMseqs prefilter-result database.


def run_reference_protein_search(
    *,
    reference_fasta: Path,
    candidate_fasta: Path,
    output_tsv: Path,
    temporary_dir: Path,
    threads: int,
    env: dict[str, str],
    timeout: float,
) -> None:
    """Align the small reference panel against the current scoring batch's called ORFs.

    MMseqs E-values depend on the target pool's total residues. Changing the
    batch size or ORF content can alter weak-hit credit and which hits pass E<=1;
    dropping significance from the reward alone would not remove this cutoff.

    These are the protein easy-search --prefilter-mode 2 stages, with a safe
    private candidate stream in place of its unterminated fake_pref symlink.
    Keep native alignment/scoring defaults, including full sequence identities
    (mode 3), and the entire target pool. Native failures propagate, without retry.
    """
    work = temporary_dir / uuid.uuid4().hex
    work.mkdir(parents=True)
    query, target, pref, result = (work / name for name in ("query", "target", "pref", "result"))
    common = ["--threads", str(max(1, int(threads))), "-v", "0"]
    deadline = time.monotonic() + timeout

    def run(*args: str | Path) -> None:
        command = ["mmseqs", *map(str, args), *common]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        subprocess.run(command, check=True, env=env, timeout=remaining)

    run("createdb", reference_fasta, query)
    run("createdb", candidate_fasta, target, "--write-lookup", "0", "--gpu", "0")
    _write_exhaustive_prefilter(Path(f"{query}.index"), Path(f"{target}.index"), pref)
    run(
        "align",
        query,
        target,
        pref,
        result,
        "--alignment-mode",
        "3",
        "-e",
        "1",
        "--min-seq-id",
        "0",
        "-c",
        "0",
        "--max-accept",
        "2147483647",
        "--max-rejected",
        "2147483647",
    )
    # Only completed searches may be reused as evidence by other objectives.
    # convertalis can leave an empty/partial file on failure or timeout.
    completed_tsv = work / "hits.tsv"
    run(
        "convertalis",
        query,
        target,
        result,
        completed_tsv,
        "--format-output",
        "query,target,evalue,pident,alnlen,qlen,tlen,qcov,tcov",
    )
    completed_tsv.replace(output_tsv)
