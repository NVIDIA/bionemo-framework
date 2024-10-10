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


import os
from tempfile import NamedTemporaryFile

from bionemo.utils.fasta import FastaUtil


def test_constructor_right_length_entries():
    with NamedTemporaryFile("w") as fd:
        fd.write(">seq1\n")
        fd.write("A" * 100 + "\n" + "C" * 100 + "\n")
        fd.write(">seq2\n")
        fd.write("A" * 100 + "\n" + "C" * 100 + "\n")
        fd.write(">seq3\n")
        fd.write("A" * 100 + "\n" + "C" * 100 + "\n")
        fd.flush()
        fasta = FastaUtil.from_filename(fd.name)
        assert len(fasta.seq_lookup.keys()) == 3
        for sequences in fasta.seq_lookup.values():
            assert len(sequences) == 2
            assert all(len(seq) == 100 for seq in sequences)


def test_split_ns():
    with NamedTemporaryFile("w") as (fd):
        fd.write(">seq1\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        )
        fd.write("N")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )

        fd.write(">seq2\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        )
        fd.write("NN")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )

        fd.write(">seq3\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        )
        fd.write("NCN")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )
        fd.flush()

        fasta = FastaUtil.from_filename(fd.name)
        # Now we have a simple method for doing this
        splitted = fasta._split_ns(fasta.seq_lookup[">seq1"])
        # Both should be length 100

        assert isinstance(splitted[0], str)
        assert len(splitted[0]) == 100 and len(splitted[1]) == 100
        assert all(x == "A" for x in splitted[0])
        assert all(x == "C" for x in splitted[1])

        splitted = fasta._split_ns(fasta.seq_lookup[">seq2"])
        # Both should be length 100
        assert isinstance(splitted[0], str)
        assert len(splitted[0]) == 100 and len(splitted[1]) == 100
        assert all(x == "A" for x in splitted[0])
        assert all(x == "C" for x in splitted[1])
        splitted = fasta._split_ns(fasta.seq_lookup[">seq3"])

        # Three entries
        assert isinstance(splitted[0], str)
        assert len(splitted[0]) == 100 and len(splitted[1]) == 1 and len(splitted[2]) == 100
        assert all(x == "A" for x in splitted[0])
        assert all(x == "C" for x in splitted[1])
        assert all(x == "C" for x in splitted[2])


def test_split_on_ns():
    with NamedTemporaryFile("w") as (fd):
        fd.write(">seq1\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        )
        fd.write("N")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )

        fd.write(">seq2\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        )
        fd.write("NN")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )

        fd.write(">seq3\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        )
        fd.write("NCN")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )
        fd.flush()
        fasta = FastaUtil.from_filename(fd.name)

    # TODO: Might be more appropriate to get this working with a NamedTemporaryFile
    result = fasta.split_on_ns().write("test.fa")

    # Assumptions:
    #   there are 7 regions because of 'N'
    #   they have the same total length as before, 601.
    fasta = FastaUtil.from_filename(result)
    assert len(fasta.seq_lookup.keys()) == 7

    lens = []
    for seqs in fasta.seq_lookup.values():
        lens.extend([len(seq) for seq in seqs])

    assert all(lambda x: isinstance(x, str) for x in seqs)
    assert sum(lens) == 601  # 100 + 100, 100 + 100, 100 + 1 + 100 = 601

    os.remove("test.fa")


def test_split_on_ns_nogap():
    with NamedTemporaryFile("w") as (fd):
        fd.write(">seq1\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        )
        fd.write("N\n")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )

        fd.write(">seq2\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        )
        fd.write("NN\n")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )

        fd.write(">seq3\n")
        fd.write(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        )
        fd.write("NCN\n")
        fd.write(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        )

        fd.write(">seq4\n")
        fd.write("AAAAANAAAAA\n")
        fd.write("CCCCCNCCCCC\n")
        fd.write("CCCCCCCCCCN\n")
        fd.flush()
        fasta = FastaUtil.from_filename(fd.name)

    result = fasta.split_on_ns().write("test.fa")
    fasta = FastaUtil.from_filename(result)

    lens = []
    for seqs in fasta.seq_lookup.values():
        lens.extend([len(seq) for seq in seqs])

    # Matches the joining behavior, split on Ns, and joined on newlines.
    assert lens == [100, 100, 100, 100, 100, 1, 100, 5, 10, 15]
    assert sum(lens) == 631
    assert len(fasta.seq_lookup.keys()) == 10
    os.remove("test.fa")
