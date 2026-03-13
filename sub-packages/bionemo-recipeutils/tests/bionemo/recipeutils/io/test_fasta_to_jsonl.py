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

"""Tests for bionemo.recipeutils.io.fasta_to_jsonl."""

import json

from bionemo.recipeutils.io.fasta_to_jsonl import fasta_to_jsonl


def test_single_sequence(tmp_path):
    """Convert a single-record FASTA to JSONL."""
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">seq1\nATCGATCG\nGGCC\n")
    out = tmp_path / "output.jsonl"

    count = fasta_to_jsonl(fasta, out)

    assert count == 1
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["id"] == "seq1"
    assert records[0]["prompt"] == "ATCGATCGGGCC"


def test_multiple_sequences(tmp_path):
    """Multiple FASTA records produce one JSONL line each."""
    fasta = tmp_path / "multi.fasta"
    fasta.write_text(">alpha description text\nAAAA\n>beta\nTTTT\nCCCC\n>gamma\nGG\n")
    out = tmp_path / "multi.jsonl"

    count = fasta_to_jsonl(fasta, out)

    assert count == 3
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records[0] == {"id": "alpha", "prompt": "AAAA"}
    assert records[1] == {"id": "beta", "prompt": "TTTTCCCC"}
    assert records[2] == {"id": "gamma", "prompt": "GG"}


def test_uppercase_conversion(tmp_path):
    """The uppercase flag converts sequences to upper case."""
    fasta = tmp_path / "lower.fasta"
    fasta.write_text(">s1\natcg\n")
    out = tmp_path / "upper.jsonl"

    count = fasta_to_jsonl(fasta, out, uppercase=True)

    assert count == 1
    record = json.loads(out.read_text().strip())
    assert record["prompt"] == "ATCG"


def test_empty_fasta(tmp_path):
    """An empty FASTA produces an empty JSONL file."""
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    out = tmp_path / "empty.jsonl"

    count = fasta_to_jsonl(fasta, out)

    assert count == 0
    assert out.read_text() == ""


def test_blank_lines_are_skipped(tmp_path):
    """Blank lines between sequences are ignored."""
    fasta = tmp_path / "blanks.fasta"
    fasta.write_text(">s1\nAA\n\n\n>s2\nTT\n\n")
    out = tmp_path / "blanks.jsonl"

    count = fasta_to_jsonl(fasta, out)

    assert count == 2
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records[0]["prompt"] == "AA"
    assert records[1]["prompt"] == "TT"
