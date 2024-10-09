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


import pyfastx
import pytest

from bionemo.utils.gff import (
    GFFEntry,
    _build_donor_acceptors_midpoints,
    _get_donors_acceptors,
    _get_exons_by_parent,
    _get_intron_midpoints,
    parse_gff3,
)


def test_midpoint():
    # 3, 22
    odds = [
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=1,
            end=2,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=5,
            end=20,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=25,
            end=30,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
    ]

    result = _get_intron_midpoints(odds)
    assert result == [3, 22]

    # 3, 22
    evens = [
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=1,
            end=2,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=4,
            end=20,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=24,
            end=30,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
    ]
    result = _get_intron_midpoints(evens)
    assert result == [3, 22]


def test_midpoint_empty():
    result = _get_intron_midpoints([])
    assert result == []


def test_midpoint_singleton():
    exon = [
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=1,
            end=2,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
    ]
    result = _get_intron_midpoints(exon)
    assert result == []


def test_donor_acceptor_empty():
    exons = []
    expected = ([], [])
    result = _get_donors_acceptors(exons)
    assert expected == result


def test_donor_acceptor_singleton():
    exons = [
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=1,
            end=2,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
    ]
    expected = ([], [])
    result = _get_donors_acceptors(exons)
    assert expected == result


def test_donor_acceptor():
    # Unit test for sampling the correct positions of donors and acceptors.

    # donors: 2, 12
    # acceptors: 10, 20
    # >>>[1   2]d---a[10   12]d---a[20   22]>>>
    forward_exons = [
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=1,
            end=2,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=10,
            end=12,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=20,
            end=22,
            score=".",
            strand="+",
            phase=".",
            attributes="",
        ),
    ]

    # donors: 10, 20
    # acceptors: 2, 12
    # <<<[1   2]a---d[10   12]a---d[20   22]<<<
    reverse_exons = [
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=1,
            end=2,
            score=".",
            strand="-",
            phase=".",
            attributes="",
        ),
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=10,
            end=12,
            score=".",
            strand="-",
            phase=".",
            attributes="",
        ),
        GFFEntry(
            seqid="seqid",
            source="source",
            type="exon",
            start=20,
            end=22,
            score=".",
            strand="-",
            phase=".",
            attributes="",
        ),
    ]

    expected = ([2, 12], [10, 20])
    result = _get_donors_acceptors(forward_exons)
    assert expected == result

    expected = ([10, 20], [2, 12])
    result = _get_donors_acceptors(reverse_exons)
    assert expected == result


# NOTE functional test.
@pytest.mark.skip
def test_end_to_end():
    from collections import Counter

    # entries = parse_gff3("/workspace/nemo_chem/test.10000.16.gff3")

    entries = parse_gff3("/workspace/nemo_chem/Homo_sapiens.GRCh38.99.chromosome.16.gff3")
    parent_lookup = _get_exons_by_parent(entries)
    donor_acceptors_midpoints = _build_donor_acceptors_midpoints(parent_lookup)
    print(list(parent_lookup.keys())[0])
    print(list(donor_acceptors_midpoints.keys())[0])

    fasta = pyfastx.Fasta("/workspace/nemo_chem/Homo_sapiens.GRCh38.dna.chromosome.16.fa")
    d_counter = Counter()
    a_counter = Counter()
    for _, (donors, acceptors, _) in donor_acceptors_midpoints.items():
        # We expect AT/TC(GA), note that the acceptor is reverse complimented.
        # TODO: are there other cases there is some revcomp nonsense going on?
        d_seqs = map(str, (fasta["16"][x : x + 2] for x in donors))
        d_counter.update(d_seqs)

        a_seqs = map(str, (fasta["16"][x : x + 2] for x in acceptors))
        a_counter.update(a_seqs)

    # We expect 'GT'
    ((first_key, first_count), (_, second_count)) = d_counter.most_common(2)
    assert first_key == "GT"
    assert second_count / first_count < 0.35

    # We expect revcomp of 'AG' = 'CT'
    ((first_key, first_count), (_, second_count)) = a_counter.most_common(2)
    assert first_key == "CT"
    assert second_count / first_count < 0.35
