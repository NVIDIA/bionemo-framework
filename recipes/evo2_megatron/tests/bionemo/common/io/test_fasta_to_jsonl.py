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

import json
import os
from pathlib import Path

import pytest

from bionemo.common.io.fasta_to_jsonl import fasta_to_jsonl, main


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
def test_reject_source_as_output(tmp_path, monkeypatch, alias):
    source = tmp_path / "input.fasta"
    original = ">sample description\nacgt\nTG\n"
    source.write_text(original)
    output = source
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path("input.fasta")
    elif alias == "symlink":
        output = tmp_path / "link.fasta"
        output.symlink_to(source)
    elif alias == "hardlink":
        output = tmp_path / "hardlink.fasta"
        os.link(source, output)
    with pytest.raises(ValueError, match="different files"):
        fasta_to_jsonl(source, output)
    assert source.read_text() == original


@pytest.mark.parametrize("uppercase", [False, True])
def test_distinct_output(tmp_path, uppercase):
    source = tmp_path / "input.fasta"
    source.write_text(">sample description\nacgt\nTG\n>second\nNn\n")
    output = tmp_path / "output.jsonl"
    output.write_text("old output\n")
    assert fasta_to_jsonl(source, output, uppercase=uppercase) == 2
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": "sample", "prompt": "ACGTTG" if uppercase else "acgtTG"},
        {"id": "second", "prompt": "NN" if uppercase else "Nn"},
    ]
    assert source.read_text().startswith(">sample")


def test_cli_rejects_source(tmp_path, monkeypatch):
    source = tmp_path / "input.fasta"
    source.write_text(">id\nACGT\n")
    monkeypatch.setattr("sys.argv", ["bionemo_fasta_to_jsonl", str(source), str(source)])
    with pytest.raises(ValueError, match="different files"):
        main()
    assert source.read_text() == ">id\nACGT\n"
