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
import pathlib
import random
import tempfile
import time
from collections import defaultdict

import pytest

from bionemo.noodles import back_transcribe_sequence, complement_sequence, reverse_sequence, transcribe_sequence
from bionemo.noodles.nvfaidx import NvFaidx


@pytest.fixture
def sample_fasta():
    return str(pathlib.Path(__file__).parent.parent.parent / "bionemo/noodles/data/sample.fasta")


def test_reverse_sequence():
    assert reverse_sequence("ACGTACGTACGT") == "TGCATGCATGCA"


def test_reverse_sequence_equivalence(sample_fasta):
    idx = NvFaidx(sample_fasta)

    # compare to results generated from biopython:
    assert reverse_sequence(idx["chr1"].sequence()) == "GTCAGTCAGTCA"
    assert complement_sequence(idx["chr1"].sequence()) == "TGACTGACTGAC"
    assert transcribe_sequence(idx["chr1"].sequence()) == "ACUGACUGACUG"
    assert back_transcribe_sequence(idx["chr1"].sequence()) == "ACTGACTGACTG"


@pytest.mark.skip("Requires Biopython")
def test_benchmark_vs_biopython():
    """Must install biopython to actually run this. Timings below.
    reverse 0.0005855560302734375 1.9788742065429688e-05 29.59036144578313
    transcribe 0.0012478828430175781 4.1961669921875e-05 29.738636363636363
    back_transcribe 9.417533874511719e-05 8.821487426757812e-06 10.675675675675675
    complement 0.0005459785461425781 8.416175842285156e-05 6.4872521246458925
    """
    from Bio import SeqIO

    test_fasta = create_test_fasta(num_seqs=100, seq_length=10000)
    fasta_biop = SeqIO.parse(test_fasta, "fasta")
    # Time transcribe
    results = defaultdict(lambda: 0.0)
    for record in fasta_biop:
        start = time.time()
        record.seq[::-1]
        end = time.time()
        results["reverse"] += end - start

        start = time.time()
        record.seq.transcribe()
        end = time.time()
        results["transcribe"] += end - start

        start = time.time()
        record.seq.back_transcribe()
        end = time.time()
        results["back_transcribe"] += end - start

        start = time.time()
        record.seq.complement()
        end = time.time()
        results["complement"] += end - start

    biop_results = results

    idx = NvFaidx(test_fasta)
    results = defaultdict(lambda: 0.0)
    for seq in idx.values():
        start = time.time()
        reverse_sequence(seq)
        end = time.time()
        results["reverse"] = end - start

        start = time.time()
        transcribe_sequence(seq)
        end = time.time()
        results["transcribe"] = end - start

        start = time.time()
        back_transcribe_sequence(seq)
        end = time.time()
        results["back_transcribe"] = end - start

        start = time.time()
        complement_sequence(seq)
        end = time.time()
        results["complement"] = end - start

    noodles_results = results
    print("func", "biopython-time", "noodles-time", "noodles-speedup")
    for key in results:
        biop, noodles = biop_results[key], noodles_results[key]
        print(key, biop, noodles, biop / noodles)
        assert biop / noodles > 1
    assert False  # So they print out


def test_complement_sequence():
    assert complement_sequence("ACGTACGTACGT") == "TGCATGCATGCA"
    assert complement_sequence(complement_sequence("ACGTACGTACGT")) == "ACGTACGTACGT"


def test_transcribe_sequence():
    assert transcribe_sequence("ACGTACGTACGT") == "ACGUACGUACGU"
    assert back_transcribe_sequence(transcribe_sequence("ACGTACGTACGT")) == "ACGTACGTACGT"


def test_back_transcribe_sequence():
    assert back_transcribe_sequence("ACGUACGUACGU") == "ACGTACGTACGT"
    assert transcribe_sequence(back_transcribe_sequence("ACGUACGUACGU")) == "ACGUACGUACGU"


def create_test_fasta(num_seqs=2, seq_length=1000):
    """
    Creates a FASTA file with random sequences.

    Args:
        num_seqs (int): Number of sequences to include in the FASTA file.
        seq_length (int): Length of each sequence.

    Returns:
        str: File path to the generated FASTA file.
    """
    temp_dir = tempfile.mkdtemp()
    fasta_path = os.path.join(temp_dir, "test.fasta")

    with open(fasta_path, "w") as fasta_file:
        for i in range(1, num_seqs + 1):
            # Write the header
            fasta_file.write(f">contig{i}\n")

            # Generate a random sequence of the specified length
            sequence = "".join(random.choices("ACGT", k=seq_length))

            # Split the sequence into lines of 60 characters for FASTA formatting
            for j in range(0, len(sequence), 80):
                fasta_file.write(sequence[j : j + 80] + "\n")

    return fasta_path
