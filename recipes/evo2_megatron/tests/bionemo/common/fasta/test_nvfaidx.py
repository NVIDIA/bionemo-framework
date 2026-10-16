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

"""Tests for the pure-Python NvFaidx stand-in."""

from pathlib import Path

import pytest

from bionemo.common.fasta.nvfaidx import NvFaidx


DUPLICATE_FASTA = ">chr1 first copy\nAAAA\n>chr2\nCCCC\n>chr1 second copy\nGGGG\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_reads_a_plain_fasta(tmp_path: Path) -> None:
    fasta = _write(tmp_path, "ok.fa", ">chr1\nAAAACC\n>chr2\nGG\n")

    index = NvFaidx(fasta)

    assert len(index) == 2
    assert index["chr1"][:] == "AAAACC"
    assert index["chr2"][:] == "GG"
    assert index.records["chr1"].length == 6


def test_duplicate_seqid_is_rejected(tmp_path: Path) -> None:
    fasta = _write(tmp_path, "dup.fa", DUPLICATE_FASTA)

    with pytest.raises(ValueError, match="Non-unique sequence-id"):
        NvFaidx(fasta)


def test_duplicate_seqid_keeps_the_records_apart_when_allowed(tmp_path: Path) -> None:
    fasta = _write(tmp_path, "dup.fa", DUPLICATE_FASTA)

    index = NvFaidx(fasta, allow_duplicate_seqids=True)

    # Three records of four bases each, not two with one of them doubled.
    assert len(index) == 3
    assert [record.name for record in index.records.values()] == ["chr1", "chr2", "chr1"]
    assert [record.length for record in index.records.values()] == [4, 4, 4]


def test_faidx_agrees_with_the_in_memory_index(tmp_path: Path) -> None:
    fasta = _write(tmp_path, "dup.fa", DUPLICATE_FASTA)

    faidx = Path(NvFaidx.create_faidx(fasta, force=True))
    index = NvFaidx(fasta, allow_duplicate_seqids=True)

    assert len(faidx.read_text().strip().splitlines()) == len(index.records)


def test_sequence_data_before_a_header_is_rejected(tmp_path: Path) -> None:
    fasta = _write(tmp_path, "bad.fa", "AAAA\n>chr1\nCCCC\n")

    with pytest.raises(ValueError, match="before a FASTA header"):
        NvFaidx(fasta)
