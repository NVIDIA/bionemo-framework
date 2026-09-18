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

"""Focused tests for smooth, ORF-gated reference evidence."""

from itertools import combinations
from pathlib import Path
from random import Random

import pandas as pd
import pytest
import yaml
from Bio import SeqIO

from bionemo.evo2_phage_gen import protein_evidence


SMOOTH_SYNTENY_MATCH = {
    "identity_full_credit": 0.90,
    "reference_coverage_full_credit": 0.95,
    "candidate_coverage_full_credit": 0.95,
}
SMOOTH_TROPISM_MATCH = {
    **SMOOTH_SYNTENY_MATCH,
    "identity_full_credit": 0.95,
    "reference_coverage_full_credit": 0.99,
    "candidate_coverage_full_credit": 0.99,
}


@pytest.mark.parametrize(
    ("identity", "evalue", "ref_cov", "cand_cov", "expected"),
    [
        (47.5, 10**-2.5, 0.475, 0.475, 0.5),  # All four normalized factors are one half.
        (5.00085, 10**-0.00005, 0.0000095, 0.0000095, 0.00001),  # No cutoff or bonus near zero.
        (90.0, 0.0, 0.95, 0.95, 1.0),
        (90.0, 1e-5, 0.95, 0.95, 1.0),
        (5.0, 1e-20, 0.95, 0.95, 0.0),
        (0.0, 1e-20, 0.95, 0.95, 0.0),
        (90.0, 1.0, 0.95, 0.95, 0.0),
        (90.0, 2.0, 0.95, 0.95, 0.0),
        (90.0, 1e-20, 0.0, 0.95, 0.0),
        (90.0, 1e-20, 0.95, 0.0, 0.0),
    ],
)
def test_smooth_match_four_factors(identity, evalue, ref_cov, cand_cov, expected):
    """Equal evidence has the same score; any zero factor must veto the match."""
    observed = protein_evidence.smooth_protein_match_integrity(
        identity,
        evalue,
        100,
        100,
        100,
        reference_coverage=ref_cov,
        candidate_coverage=cand_cov,
        **SMOOTH_SYNTENY_MATCH,
    )
    assert observed == pytest.approx(expected)


def test_smooth_match_rejects_infinite_endpoint():
    """An infinite E-value endpoint must not turn NaN interpolation into full credit."""
    with pytest.raises(ValueError, match="smooth protein-match configuration"):
        protein_evidence.smooth_protein_match_integrity(
            30.0,
            0.1,
            100,
            100,
            100,
            reference_coverage=0.2,
            candidate_coverage=0.2,
            significance_zero_evalue=float("inf"),
            **SMOOTH_SYNTENY_MATCH,
        )


def test_native_coverage_excludes_gap_columns():
    """Alignment columns include gaps; they must not manufacture intact-gene credit."""
    hits = pd.DataFrame(
        {
            "protein_database_mmseqs_percent_identity": [100.0, 100.0],
            "protein_database_mmseqs_alignment_length": [100, 100],
            "protein_database_mmseqs_query_length": [100, 100],
            "protein_database_mmseqs_target_length": [100, 100],
            "protein_database_mmseqs_query_coverage": [0.70, 1.0],
            "protein_database_mmseqs_target_coverage": [1.0, 0.60],
        }
    )
    measured, available = protein_evidence.add_protein_alignment_evidence(hits, "protein_database")
    assert available
    assert measured["protein_database_alignment_integrity"].tolist() == [0.70, 0.60]
    legacy = hits.drop(columns=["protein_database_mmseqs_query_coverage"])
    missing, available = protein_evidence.add_protein_alignment_evidence(legacy, "protein_database")
    assert not available
    assert missing["protein_database_alignment_integrity"].tolist() == [0.0, 0.0]


def test_smooth_match_uses_native_coverage():
    """A gapped partial match stays partial even when alnlen equals both protein lengths."""
    observed = protein_evidence.smooth_protein_match_integrity(
        100.0,
        1e-20,
        100,
        100,
        100,
        reference_coverage=0.70,
        candidate_coverage=1.0,
        identity_full_credit=1.0,
        reference_coverage_full_credit=1.0,
        candidate_coverage_full_credit=1.0,
    )
    assert observed == pytest.approx(0.9146912192)


def test_aai_best_evalue_per_orf():
    hits = pd.DataFrame(
        {
            "id_prompt": ["sample_ORF.1", "sample_ORF.1", "sample_ORF.2", "other_ORF.1"],
            "protein_database_mmseqs_target": ["family_a|1", "family_b|2", "family_a|3", "family_a|1"],
            "protein_database_mmseqs_e_value": [1e-30, 1e-10, 1e-20, 1e-40],
            "protein_database_mmseqs_percent_identity": [80.0, 99.0, 100.0, 97.0],
        }
    )
    result = protein_evidence.summarize_best_hit_aai(hits).set_index("id_prompt")
    # Minimum E-value, not maximum identity, and two ORFs in the same family count twice.
    assert result.loc["sample", "average_protein_percent_identity"] == 90.0
    assert result.loc["sample", "average_protein_identity_gene_count"] == 2
    assert result.loc["other", "average_protein_percent_identity"] == 97.0
    assert protein_evidence.summarize_best_hit_aai(hits.iloc[:0]).empty


def _required_hits(*rows):
    """Build compact required-gene MMseqs evidence rows for assignment tests."""
    return pd.DataFrame(
        [
            {
                "id_prompt": candidate,
                "annot": annotation,
                "protein_database_mmseqs_target": target,
                "protein_database_mmseqs_percent_identity": 100.0,
                "protein_database_mmseqs_alignment_length": int(integrity * 400),
                "protein_database_mmseqs_query_length": 400,
                "protein_database_mmseqs_target_length": 400,
                "protein_database_mmseqs_query_coverage": integrity,
                "protein_database_mmseqs_target_coverage": integrity,
            }
            for candidate, annotation, target, integrity in rows
        ]
    )


def _required_sequences(*genome_ids):
    return pd.DataFrame(
        {
            "id_prompt": list(genome_ids),
            "genome_id": [f"genome_{index}" for index, _ in enumerate(genome_ids, start=1)],
        }
    )


def test_pseudocircular_filter_removes_a_prefix_tail_repeated_by_a_cross_origin_orf(tmp_path):
    """A native-prefix tail must not be counted again when a longer circular ORF contains it."""
    source = tmp_path / "source.fasta"
    source.write_text(">genome\nATGTAAATGAAA\n")
    nucleotide_orfs = tmp_path / "orfs.fasta"
    protein_orfs = tmp_path / "proteins.fasta"
    nucleotide_orfs.write_text(
        ">genome_ORF.1 [0-6](+) type:complete length:6\nATGAAA\n"
        ">genome_ORF.2 [6-18](+) type:complete length:12\nATGAAAATGAAA\n"
        ">genome_ORF.3 [6-12](+) type:complete length:6\nATGAAA\n"
    )
    protein_orfs.write_text(
        ">genome_ORF.1 [0-6](+) type:complete length:6\nMK\n"
        ">genome_ORF.2 [6-18](+) type:complete length:12\nQQMK\n"
        ">genome_ORF.3 [6-12](+) type:complete length:6\nQQ\n"
    )

    protein_evidence.remove_pseudocircular_extension_orfs(source, nucleotide_orfs, protein_orfs)

    assert [record.id for record in SeqIO.parse(protein_orfs, "fasta")] == ["genome_ORF.2", "genome_ORF.3"]


def test_reference_gff_proteins_and_candidate_orf_order_are_loaded_for_smooth_search(tmp_path):
    """The search inputs must preserve reference loci and coordinate-order called ORFs."""
    reference_gff = tmp_path / "reference.gff"
    reference_gff.write_text(
        "##gff-version 3\n"
        "ref\ttest\tCDS\t1\t9\t.\t+\t0\tID=A\n"
        "ref\ttest\tCDS\t10\t18\t.\t+\t0\tID=G\n"
        "##FASTA\n"
        ">ref\n"
        "ATGAAATAAATGCCCTAA\n"
    )
    reference_fasta = tmp_path / "reference-proteins.fasta"
    candidate_orfs = tmp_path / "candidate-orfs.fasta"
    candidate_orfs.write_text(
        ">umi1_ORF.5 [300-450](+) type:complete length:150\nATG\n"
        ">umi1_ORF.2 [10-100](+) type:complete length:90\nATG\n"
    )

    reference_order = protein_evidence.write_reference_protein_fasta(reference_gff, reference_fasta)
    candidate_sequences, candidate_orders = protein_evidence.load_candidate_orf_context(candidate_orfs)

    assert reference_order == ("A", "G")
    assert [(record.id, str(record.seq)) for record in SeqIO.parse(reference_fasta, "fasta")] == [
        ("A", "MK"),
        ("G", "MP"),
    ]
    assert candidate_orders == {"umi1": ("umi1_ORF.2", "umi1_ORF.5")}
    assert set(candidate_sequences) == {"umi1_ORF.2", "umi1_ORF.5"}


def test_smooth_reference_summary_reuses_orf_hits_for_synteny_tropism_and_gene_a_origin():
    """One permissive ORF search must drive all three graded reference objectives."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    a_orf = "G" * 6 + motif + "G" * 30
    hits = pd.DataFrame(
        [
            {
                "query": "A",
                "target": "umi1_ORF.1",
                "evalue": 1e-20,
                "pident": 90.0,
                "alnlen": 95,
                "qlen": 100,
                "tlen": 100,
                "qcov": 0.95,
                "tcov": 0.95,
            },
            {
                "query": "G",
                "target": "umi1_ORF.2",
                "evalue": 1e-20,
                "pident": 95.0,
                "alnlen": 99,
                "qlen": 100,
                "tlen": 100,
                "qcov": 0.99,
                "tcov": 0.99,
            },
        ]
    )

    observed = protein_evidence.summarize_smooth_reference_evidence(
        hits,
        genome_sequences={"umi1": a_orf, "umi2": "G" * len(a_orf)},
        candidate_orf_sequences={"umi1_ORF.1": a_orf, "umi1_ORF.2": "ATG" * 34},
        candidate_orders={"umi1": ("umi1_ORF.1", "umi1_ORF.2"), "umi2": ()},
        reference_order=("A", "G"),
        core_gene_match_parameters=SMOOTH_SYNTENY_MATCH,
        tropism_match_parameters=SMOOTH_TROPISM_MATCH,
        core_gene_order_weight=0.75,
        core_gene_duplicate_penalty_weight=0.75,
        gene_a_reference_locus="A",
        tropism_reference_locus="G",
        gene_a_origin_motif=motif,
        gene_a_origin_offset_nt=6,
        gene_a_origin_offset_tolerance_nt=6,
    ).set_index("id_prompt")

    assert observed.loc["umi1", "reward_external_core_gene_ordered_conservation"] == 1.0
    assert observed.loc["umi1", "reward_external_tropism"] == 1.0
    assert observed.loc["umi1", "reward_gene_a_origin"] == 1.0
    assert observed.loc["umi1", "smooth_reference_matched_loci"] == 2
    assert observed.loc["umi2", "reward_external_core_gene_ordered_conservation"] == 0.0
    assert observed.loc["umi2", "reward_external_tropism"] == 0.0
    assert observed.loc["umi2", "reward_gene_a_origin"] == 0.0


@pytest.mark.parametrize(
    ("functions", "full_credit"),
    [
        (("A", "G", "J"), True),
        (("J", "A", "G"), True),  # Circular origin changes preserve order.
        (("A", "J", "G"), False),
        (("A", "G", "J", "J"), False),
        (("A", "G"), False),
        (("A", "G", "unrelated"), False),
        (("G", "A|J", "A|J"), True),  # A tied assignment must not hide valid circular order.
    ],
)
def test_function_synteny_preserves_order_and_copy_checks(functions, full_credit):
    """Supported replacements fill slots without relaxing order, copies, or other objectives."""
    families = {"A": ["phrog:713"], "G": ["phrog:1483"], "J": ["phrog:2354", "phrog:3780"]}
    targets = {"A": "phrog_713", "G": "phrog_1483", "J": "phrog_3780", "unrelated": "phrog_11693"}
    candidates = tuple(f"umi1_ORF.{i}" for i in range(len(functions)))
    hits = _required_hits(
        *[
            (candidate, "", targets[alternative], 1.0)
            for candidate, function in zip(candidates, functions, strict=True)
            for alternative in function.split("|")
        ]
    )
    matches, available = protein_evidence.score_function_matches(hits, families)
    assert available
    reference_functions = {"ref_A": "A", "ref_G": "G", "ref_J": "J"}
    reference_order = ("ref_A", "optional_K", "ref_G", "ref_J")
    observed = protein_evidence.summarize_smooth_reference_evidence(
        pd.DataFrame(columns=["query", "target"]),
        genome_sequences={"umi1": "ATG" * 400},
        candidate_orf_sequences=dict.fromkeys(candidates, "ATG" * 100),
        candidate_orders={"umi1": candidates},
        reference_order=reference_order,
        core_gene_match_parameters=SMOOTH_SYNTENY_MATCH,
        tropism_match_parameters=SMOOTH_TROPISM_MATCH,
        core_gene_order_weight=0.75,
        core_gene_duplicate_penalty_weight=0.75,
        gene_a_reference_locus="ref_A",
        tropism_reference_locus="ref_G",
        gene_a_origin_motif="CAACTTGATATTAATAACACTATAGACCAC",
        gene_a_origin_offset_nt=6,
        gene_a_origin_offset_tolerance_nt=6,
        function_matches=matches,
        reference_functions=reference_functions,
    ).iloc[0]
    assert bool(observed.reward_external_core_gene_ordered_conservation == 1.0) is full_credit
    assert observed.reward_external_tropism == 0.0
    assert observed.reward_gene_a_origin == 0.0
    hard = protein_evidence.summarize_core_gene_ordered_conservation(
        matches,
        pd.DataFrame({"id_prompt": ["umi1"], "genome_id": ["genome_1"]}),
        candidate_orders={"umi1": candidates},
        reference_order=reference_order,
        reference_functions=reference_functions,
    )
    assert protein_evidence.core_gene_ordered_conservation_pass_mask(hard).tolist() == [full_credit]


def test_smooth_reference_summary_rejects_invalid_match_settings_without_hits():
    """A no-hit batch must not turn an invalid reward configuration into measured zero."""
    with pytest.raises(ValueError, match="smooth protein-match configuration"):
        protein_evidence.summarize_smooth_reference_evidence(
            pd.DataFrame(columns=["query", "target", "evalue", "pident", "alnlen", "qlen", "tlen"]),
            genome_sequences={"umi1": "A" * 100},
            candidate_orf_sequences={},
            candidate_orders={"umi1": ()},
            reference_order=("A",),
            core_gene_match_parameters={**SMOOTH_SYNTENY_MATCH, "identity_full_credit": 0.0},
            tropism_match_parameters=SMOOTH_TROPISM_MATCH,
            core_gene_order_weight=0.75,
            core_gene_duplicate_penalty_weight=0.75,
            gene_a_reference_locus="A",
            tropism_reference_locus="G",
            gene_a_origin_motif="CAACTTGATATTAATAACACTATAGACCAC",
            gene_a_origin_offset_nt=6,
            gene_a_origin_offset_tolerance_nt=6,
        )


def test_smooth_match_grades_weak_and_partial_evidence():
    """Weak evidence receives less credit than a significant, well-covered partial match."""
    kwargs = {
        "identity_full_credit": 0.85,
        "reference_coverage_full_credit": 0.95,
        "candidate_coverage_full_credit": 0.95,
    }

    decoy = protein_evidence.smooth_protein_match_integrity(
        percent_identity=21.8,
        e_value=0.46,
        alignment_length=147,
        reference_length=522,
        candidate_length=427,
        reference_coverage=0.282,
        candidate_coverage=0.344,
        **kwargs,
    )
    partial = protein_evidence.smooth_protein_match_integrity(
        percent_identity=35.0,
        e_value=1e-8,
        alignment_length=70,
        reference_length=100,
        candidate_length=100,
        reference_coverage=0.70,
        candidate_coverage=0.70,
        **kwargs,
    )
    complete = protein_evidence.smooth_protein_match_integrity(
        percent_identity=85.0,
        e_value=1e-20,
        alignment_length=95,
        reference_length=100,
        candidate_length=100,
        reference_coverage=0.95,
        candidate_coverage=0.95,
        **kwargs,
    )

    assert 0.0 < decoy < partial < 1.0
    assert complete == 1.0


def test_smooth_match_penalizes_both_truncations_and_fusions():
    """Dropping either reciprocal-coverage side must lower an otherwise identical match."""
    kwargs = {
        "percent_identity": 85.0,
        "e_value": 1e-20,
        "identity_full_credit": 0.85,
        "reference_coverage_full_credit": 0.95,
        "candidate_coverage_full_credit": 0.95,
    }
    complete = protein_evidence.smooth_protein_match_integrity(
        alignment_length=95,
        reference_length=100,
        candidate_length=100,
        reference_coverage=0.95,
        candidate_coverage=0.95,
        **kwargs,
    )
    truncation = protein_evidence.smooth_protein_match_integrity(
        alignment_length=70,
        reference_length=100,
        candidate_length=70,
        reference_coverage=0.70,
        candidate_coverage=1.0,
        **kwargs,
    )
    fusion = protein_evidence.smooth_protein_match_integrity(
        alignment_length=95,
        reference_length=100,
        candidate_length=140,
        reference_coverage=0.95,
        candidate_coverage=0.679,
        **kwargs,
    )

    assert complete == 1.0
    assert 0.0 < truncation < complete
    assert 0.0 < fusion < complete


@pytest.mark.parametrize(
    ("reference", "candidates", "expected"),
    [
        pytest.param("", ({"A": 1.0},), 0.0, id="empty-reference"),
        pytest.param("ABC", (), 0.0, id="empty-candidate"),
        pytest.param("ABC", ({}, {}), 0.0, id="no-matches"),
        pytest.param("ABC", ({"A": 0.2}, {"B": 0.5}, {"C": 0.8}), 1.5, id="weighted-in-order"),
        pytest.param("ABCD", ({"A": 1.0}, {"C": 1.0}, {"B": 1.0}, {"D": 1.0}), 3.0, id="swap"),
        pytest.param("ABC", ({"C": 1.0}, {"B": 1.0}, {"A": 1.0}), 1.0, id="reversed-linear-order"),
        pytest.param("ABC", ({"C": 1.0}, {"A": 1.0}, {"B": 1.0}), 2.0, id="linear-cut-is-fixed"),
        pytest.param("ABCD", ({"A": 1.0}, {}, {"D": 0.5}), 1.5, id="skip-both-kinds-of-gap"),
        pytest.param("AB", ({"A": 1.0, "B": 1.0},), 1.0, id="cannot-reuse-candidate"),
        pytest.param("A", ({"A": 0.4}, {"A": 0.9}), 0.9, id="cannot-reuse-reference"),
        # B(.9)+C(.2) beats the longer A(.2)+B(.2)+C(.2) alignment.
        pytest.param("ABC", ({"B": 0.9}, {"A": 0.2}, {"B": 0.2}, {"C": 0.2}), 1.1, id="weight-over-count"),
        # A(.8)+B(.8)+C(.8) beats greedily taking the two .9 edges.
        pytest.param("ABC", ({"A": 0.8, "B": 0.9}, {"B": 0.8, "C": 0.9}, {"C": 0.8}), 2.4, id="global-choice"),
    ],
)
def test_linear_ordered_gold(reference, candidates, expected):
    """Hand-solved weighted alignments; each dictionary describes one candidate ORF."""
    candidate_order = tuple(f"orf{i}" for i in range(len(candidates)))
    edges = {(gene, candidate_order[i]): credit for i, hits in enumerate(candidates) for gene, credit in hits.items()}
    assert protein_evidence._linear_ordered_integrity(edges, tuple(reference), candidate_order) == pytest.approx(
        expected
    )


def test_linear_ordered_exhaustive():
    """Check the DP against all equally sized ordered subsequence pairs, without a recurrence."""
    random = Random(20260916)
    for n_reference in range(5):
        for n_candidate in range(5):
            references = tuple(f"r{i}" for i in range(n_reference))
            candidates = tuple(f"c{i}" for i in range(n_candidate))
            for _ in range(8):
                edges = {(r, c): random.choice((0.0, 0.25, 0.5, 1.0)) for r in references for c in candidates}
                # Enumerate every order-preserving one-to-one pairing. Small matrices
                # make this independent, deliberately slow definition practical.
                expected = max(
                    sum(edges[(r, c)] for r, c in zip(rs, cs, strict=True))
                    for length in range(min(n_reference, n_candidate) + 1)
                    for rs in combinations(references, length)
                    for cs in combinations(candidates, length)
                )
                observed = protein_evidence._linear_ordered_integrity(edges, references, candidates)
                assert observed == pytest.approx(expected), (references, candidates, edges)


@pytest.mark.parametrize(
    ("reference", "candidates", "expected"),
    [
        # Expected tuple: content sum, circular ordered sum, excess mass, reward.
        # With four reference slots the reward is (content + 3*ordered - 3*excess)/16.
        pytest.param("ABCD", (), (0, 0, 0, 0), id="empty-candidate"),
        pytest.param("ABCD", ({}, {}), (0, 0, 0, 0), id="no-homologs"),
        pytest.param("ABCD", ({"A": 1}, {"B": 1}, {"C": 1}, {"D": 1}), (4, 4, 0, 1), id="complete"),
        pytest.param("ABCD", ({"A": 1}, {"C": 1}, {"B": 1}, {"D": 1}), (4, 3, 0, 0.8125), id="one-swap"),
        pytest.param("ABCD", ({"D": 1}, {"C": 1}, {"B": 1}, {"A": 1}), (4, 2, 0, 0.625), id="reversed-order"),
        pytest.param("ABCD", ({"A": 1}, {"B": 1}, {"D": 1}), (3, 3, 0, 0.75), id="missing-one"),
        pytest.param("ABCD", ({"A": 1}, {"D": 1}), (2, 2, 0, 0.5), id="missing-two"),
        pytest.param("ABCD", ({"A": 1},), (1, 1, 0, 0.25), id="single-gene-foothold"),
        pytest.param("ABCD", ({"A": 0.2}, {"B": 0.2}, {"C": 0.2}, {"D": 0.2}), (0.8, 0.8, 0, 0.2), id="partial"),
        pytest.param(
            "ABCD", ({"A": 0.2}, {"C": 0.2}, {"B": 0.2}, {"D": 0.2}), (0.8, 0.6, 0, 0.1625), id="partial-swap"
        ),
        pytest.param("ABCD", ({"A": 1}, {}, {"B": 1}, {"C": 1}, {"D": 1}), (4, 4, 0, 1), id="unrelated-extra-gene"),
        # Repairing A-C-B-D with another B recovers one unit of order and pays
        # one unit of excess: it must tie the one-swap score, not improve on it.
        pytest.param(
            "ABCD", ({"A": 1}, {"B": 1}, {"C": 1}, {"B": 1}, {"D": 1}), (4, 4, 1, 0.8125), id="duplicate-repair"
        ),
        pytest.param(
            "ABCD", ({"A": 1}, {"B": 1}, {"B": 0.5}, {"C": 1}, {"D": 1}), (4, 4, 0.5, 0.90625), id="partial-copy"
        ),
        pytest.param("ABCD", ({"A": 1}, {"A": 1}, {"A": 1}), (1, 1, 2, 0), id="excess-clips-to-zero"),
        pytest.param("AB", ({"A": 1, "B": 1},), (1, 1, 0, 0.5), id="one-orf-two-possible-functions"),
    ],
)
def test_smooth_synteny_gold(reference, candidates, expected):
    """Exact score ladder, repeated at every circular cut of both genomes.

    These are mathematical gold cases for admitted match credits, not claims that
    a particular score predicts biological viability. Expected values are hand
    calculations, never snapshots of the implementation under test.
    """
    candidate_order = tuple(f"orf{i}" for i in range(len(candidates)))
    edges = {(gene, candidate_order[i]): credit for i, hits in enumerate(candidates) for gene, credit in hits.items()}
    for reference_cut in range(len(reference)):
        reference_order = tuple(reference[reference_cut:] + reference[:reference_cut])
        for candidate_cut in range(max(1, len(candidate_order))):
            result = protein_evidence.score_core_gene_ordered_conservation(
                edges,
                reference_order=reference_order,
                candidate_order=candidate_order[candidate_cut:] + candidate_order[:candidate_cut],
                order_weight=0.75,
                duplicate_penalty_weight=0.75,
            )
            assert (
                result.content_integrity_sum,
                result.ordered_integrity_sum,
                result.duplicate_integrity_sum,
                result.reward,
            ) == pytest.approx(expected)
            assert (result.content_score, result.ordered_score, result.duplicate_score) == pytest.approx(
                tuple(total / len(reference) for total in expected[:3])
            )
            assert len({r for r, _ in result.assignment}) == len(result.assignment)
            assert len({c for _, c in result.assignment}) == len(result.assignment)
            assert sum(edges[pair] for pair in result.assignment) == pytest.approx(expected[0])


def test_smooth_synteny_scales_beyond_twenty_reference_loci():
    """Genome-scale synteny matching must not retain the bitmask implementation's locus cap."""
    reference_order = tuple(f"reference_{index}" for index in range(34))
    candidate_order = tuple(f"candidate_{index}" for index in range(34))
    edges = {
        (reference, candidate): 1.0 for reference, candidate in zip(reference_order, candidate_order, strict=True)
    }

    result = protein_evidence.score_core_gene_ordered_conservation(
        edges,
        reference_order=reference_order,
        candidate_order=candidate_order,
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )

    assert result.reward == 1.0
    assert result.content_integrity_sum == 34.0
    assert result.assignment == tuple(zip(reference_order, candidate_order, strict=True))


def test_gene_a_origin_requires_the_functional_site_inside_the_assigned_a_orf():
    """An exact site outside its A-ORF context must not earn origin credit."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    expected_offset = 345
    exact_a = "G" * expected_offset + motif + "G" * 200
    misplaced_a = "G" * 100 + motif + "G" * (len(exact_a) - 130)

    exact = protein_evidence.score_gene_a_origin(
        candidate_a_orf_nt=exact_a,
        candidate_genome_nt=exact_a,
        a_match_integrity=1.0,
        motif=motif,
        expected_offset_nt=expected_offset,
        offset_tolerance_nt=30,
    )
    misplaced = protein_evidence.score_gene_a_origin(
        candidate_a_orf_nt=misplaced_a,
        candidate_genome_nt=misplaced_a,
        a_match_integrity=1.0,
        motif=motif,
        expected_offset_nt=expected_offset,
        offset_tolerance_nt=30,
    )

    assert exact.reward == 1.0
    assert exact.exact_functional_site is True
    assert misplaced.reward == 0.0


def test_origin_partial_match_shaping():
    """Partial sites provide graded credit while baseline matches and absent A evidence do not."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    offset = 345

    def score(site: str, a_integrity: float = 1.0):
        candidate = "G" * offset + site + "G" * 200
        return protein_evidence.score_gene_a_origin(
            candidate_a_orf_nt=candidate,
            candidate_genome_nt=candidate,
            a_match_integrity=a_integrity,
            motif=motif,
            expected_offset_nt=offset,
            offset_tolerance_nt=1,  # Isolate one in-frame site when checking its match fractions.
        )

    critical = score(motif[:3] + "A" + motif[4:])
    binding = score(motif[:14] + "C" + motif[15:])
    tail = score(motif[:28] + "TT")

    assert 0.0 < critical.reward < binding.reward < 1.0
    assert tail.reward == 1.0
    assert tail.exact_functional_site is True

    # Seven recognition matches and thirteen binding matches were discarded by
    # the old 8/10 and 14/18 cutoffs. Improving either now gives incremental credit.
    recognition_seven = score("GGG" + motif[3:])
    recognition_eight = score("GG" + motif[2:])
    binding_thirteen = score(motif[:10] + "GGGGG" + motif[15:])
    binding_fourteen = score(motif[:10] + "GGGG" + motif[14:])
    assert 0.0 < recognition_seven.reward < recognition_eight.reward < 1.0
    assert 0.0 < binding_thirteen.reward < binding_fourteen.reward < 1.0

    # One of four nicking bases matches: exactly the uniform-DNA baseline.
    assert score(motif[:3] + "GGGG" + motif[7:]).reward == 0.0
    assert score("G" * 28).reward == 0.0
    assert score(motif, a_integrity=0.0).reward == 0.0
    assert score(motif, a_integrity=0.25).reward == pytest.approx(0.7071067812)


def test_gene_a_origin_penalizes_duplicate_strong_sites():
    """Adding a second strong origin must not improve a candidate's reward."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    offset = 345
    candidate_a = "G" * offset + motif + "G" * 200
    near_exact_functional_site = motif[:14] + "C" + motif[15:]
    duplicated_genome = candidate_a + "G" * 50 + near_exact_functional_site

    result = protein_evidence.score_gene_a_origin(
        candidate_a_orf_nt=candidate_a,
        candidate_genome_nt=duplicated_genome,
        a_match_integrity=1.0,
        motif=motif,
        expected_offset_nt=offset,
        offset_tolerance_nt=30,
    )

    assert result.strong_site_count == 2
    assert result.reward == pytest.approx(0.8408964153)


@pytest.mark.parametrize("offset", [315, 318, 345, 372, 375])
def test_origin_context_window_has_no_preferred_start(offset):
    """An intact origin in the accepted frame/window is neutral to A-start variation."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    candidate = "G" * offset + motif
    result = protein_evidence.score_gene_a_origin(
        candidate_a_orf_nt=candidate,
        candidate_genome_nt=candidate,
        a_match_integrity=1.0,
        motif=motif + "TT",  # Bases 29-30 are not part of the functional site.
        expected_offset_nt=345,
        offset_tolerance_nt=30,
    )
    assert result.reward == 1.0
    assert result.position_score == 1.0
    assert result.exact_functional_site


@pytest.mark.parametrize("offset", [312, 316, 344, 376, 378])
def test_origin_context_limits_full_credit(offset):
    """An exact site outside the accepted frame/window cannot give full credit."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    candidate = "G" * offset + motif + "G" * 100
    result = protein_evidence.score_gene_a_origin(
        candidate_a_orf_nt=candidate,
        candidate_genome_nt=candidate,
        a_match_integrity=1.0,
        motif=motif,
        expected_offset_nt=345,
        offset_tolerance_nt=30,
    )
    # Imperfect matches at eligible offsets may still earn shaped partial credit.
    assert result.reward < 1.0
    assert not result.exact_functional_site


def test_required_family_selector_matches_supported_phrog_target_forms_with_missing_annotations():
    """An explicit family matches native target IDs even without an annotation label."""
    hits = _required_hits(
        ("numeric_ORF.1", pd.NA, 1713, 1.0),
        ("string_ORF.1", float("nan"), "1713", 1.0),
        ("prefixed_ORF.1", None, "phrog_1713", 1.0),
        ("unrelated_ORF.1", pd.NA, "phrog_9999", 1.0),
    )

    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("numeric", "string", "prefixed", "unrelated"),
        {"K": ["phrog:1713"]},
    )

    assert observed["required_genes_integrity_sum"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert observed["required_genes_full_length_count"].tolist() == [1, 1, 1, 0]


@pytest.mark.parametrize(
    "required",
    [
        {"A": ["nan"]},
        {"A": ["head morphogenesis"]},
        {"A": ["phrog:1473"], "B": ["phrog:1473"]},
        {"J": []},
        {"J": "phrog:2354"},
    ],
)
def test_required_families_reject_ambiguous_profiles(required):
    with pytest.raises(ValueError, match="required families"):
        protein_evidence.summarize_required_gene_evidence(pd.DataFrame(), _required_sequences("sample"), required)


def test_required_families_ignore_annotation_text():
    """Curated families still match without labels; an unrelated labeled hit cannot substitute."""
    hits = _required_hits(
        ("valid_ORF.1", "wrong label", "phrog_1473", 1.0),
        ("unrelated_ORF.1", "head morphogenesis", "phrog_9999", 1.0),
    ).drop(columns="annot")
    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("valid", "unrelated"),
        {"B": ["phrog:1473"]},
    )
    assert observed["required_genes_integrity_sum"].tolist() == [1.0, 0.0]
    assert observed["required_genes_alignment_evidence_available"].tolist() == [True, True]


def test_phix_profile_requires_its_specific_families():
    """An unrelated endolysin cannot replace E merely by sharing its annotation label."""
    families = [713, 1473, 1465, 1386, 1472, 514, 1483, 1471, 2354]
    labels = [
        "DNA replication initiation",
        "head morphogenesis",
        "terminase",
        "head morphogenesis",
        "endolysin",
        "major head protein",
        "major spike protein",
        "pilot protein for DNA ejection",
        "DNA condensation",
    ]
    hits = _required_hits(
        *[
            (
                f"{genome}_ORF.{i}",
                label,
                f"phrog_{9999 if genome == 'wrong' and family == 1472 else 3780 if genome == 'alternate' and family == 2354 else family}",
                1.0,
            )
            for genome in ("valid", "wrong", "alternate")
            for i, (family, label) in enumerate(zip(families, labels, strict=True))
        ]
    )
    hits = pd.concat([hits, _required_hits(("wrong_ORF.extra", "DNA condensation", "phrog_3780", 1.0))])
    config_path = Path(__file__).parents[3] / "configs/arc_genome_design_filtering_local.yaml"
    config = yaml.safe_load(config_path.read_text())
    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("valid", "wrong", "alternate"),
        config["required_gene_families"],
        family_coverage_thresholds=config["required_gene_family_coverage"],
    )
    assert observed["required_genes_integrity_sum"].tolist() == [9.0, 8.0, 9.0]
    assert observed["required_genes_full_length_count"].tolist() == [9, 8, 9]


def test_required_families_assign_distinct_orfs_and_ignore_extra_copies():
    """One ambiguous ORF cannot fill two families; extra copies cannot inflate completeness."""
    hits = _required_hits(
        ("sample_ORF.1", "same label", "phrog_1", 1.0),
        ("sample_ORF.1", "same label", "phrog_2", 1.0),
        ("sample_ORF.2", "same label", "phrog_1", 0.375),
        ("sample_ORF.3", "same label", "phrog_1", 0.30),
        ("sample_ORF.4", "other", "phrog_3", 1.0),
    )
    scores = []
    for frame in (hits, hits.iloc[::-1]):
        result = protein_evidence.summarize_required_gene_evidence(
            frame,
            _required_sequences("sample"),
            {"A": ["phrog:1"], "B": ["phrog:2"]},
        )
        scores.append(result.loc[0, "required_genes_integrity_sum"])
        assert result.loc[0, "required_genes_matched_count"] == 2
        assert result.loc[0, "required_genes_full_length_count"] == 1
    assert scores == [1.5, 1.5]


@pytest.mark.parametrize(
    "family,aligned,query,target,qcov,tcov",
    [
        (1465, 33, 45, 68, 0.733, 0.471),  # Evo-phi75: 33 columns, only 32 target residues.
        (1465, 33, 47, 68, 0.702, 0.471),  # Evo-phi100 has the same target-side gap.
        (1472, 74, 127, 76, 0.583, 0.974),
        (1473, 80, 82, 117, 0.976, 0.684),
    ],
)
def test_calibrated_family_coverage_accepts_viable_variants(family, aligned, query, target, qcov, tcov):
    """Observed C/B truncations and E extension are intact evidence in their calibrated envelopes."""
    hits = _required_hits(("sample_ORF.1", "function", f"phrog_{family}", 1.0))
    hits["protein_database_mmseqs_alignment_length"] = aligned
    hits["protein_database_mmseqs_query_length"] = query
    hits["protein_database_mmseqs_target_length"] = target
    hits["protein_database_mmseqs_query_coverage"] = qcov
    hits["protein_database_mmseqs_target_coverage"] = tcov
    config_path = Path(__file__).parents[3] / "configs" / "arc_genome_design_filtering_local.yaml"
    profile = yaml.safe_load(config_path.read_text())["required_gene_family_coverage"]
    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("sample"),
        {"gene": [f"phrog:{family}"]},
        family_coverage_thresholds=profile,
    )
    assert observed.loc[0, "required_genes_full_length_count"] == 1
    assert observed.loc[0, "required_genes_integrity_sum"] == 1.0

    # A smaller alignment than the observed viable bound is partial, not intact.
    hits["protein_database_mmseqs_alignment_length"] = aligned - 1
    hits["protein_database_mmseqs_query_coverage"] = qcov - 1 / query
    hits["protein_database_mmseqs_target_coverage"] = tcov - 1 / target
    damaged = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("sample"),
        {"gene": [f"phrog:{family}"]},
        family_coverage_thresholds=profile,
    )
    assert damaged.loc[0, "required_genes_full_length_count"] == 0
    assert 0.0 < damaged.loc[0, "required_genes_integrity_sum"] < 1.0


def test_calibrated_coverage_does_not_relax_other_families():
    """A C-specific allowance cannot make an arbitrary truncated head protein pass."""
    hits = _required_hits(("sample_ORF.1", "major head protein", "phrog_514", 0.5))
    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("sample"),
        {"F": ["phrog:514"]},
        family_coverage_thresholds={"phrog:1465": (0.70, 0.47)},
    )
    assert observed.loc[0, "required_genes_full_length_count"] == 0
    assert observed.loc[0, "required_genes_integrity_sum"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("thresholds", [(0, 0.5), (1.1, 0.5), (float("nan"), 0.5), (0.5,)])
def test_bad_family_coverage_is_rejected_even_without_hits(thresholds):
    with pytest.raises(ValueError, match="coverage"):
        protein_evidence.summarize_required_gene_evidence(
            pd.DataFrame(),
            _required_sequences("empty"),
            {"C": ["phrog:1465"]},
            family_coverage_thresholds={"phrog:1465": thresholds},
        )


def test_synteny_deficit_allowance_keeps_order_copy_and_measurement_checks():
    metrics = pd.DataFrame(
        {
            "reference_num_genes": [10] * 7,
            "num_syntenic_genes": [10, 9, 8, 9, 9, 9, 0],
            "duplicate_reference_gene_count": [0, 0, 0, 1, 0, 0, 0],
            "reference_order_violation_count": [0, 0, 0, 0, 1, 0, 0],
            "missing_synteny_output": [False, False, False, False, False, True, False],
        }
    )
    assert protein_evidence.core_gene_ordered_conservation_pass_mask(metrics).tolist() == [True] + [False] * 6
    assert protein_evidence.core_gene_ordered_conservation_pass_mask(metrics, 1).tolist() == [True, True] + [False] * 5


@pytest.mark.parametrize("allowance", [-1, 1.5, True])
def test_synteny_deficit_allowance_requires_nonnegative_integer(allowance):
    with pytest.raises(ValueError, match="missing reference"):
        protein_evidence.core_gene_ordered_conservation_pass_mask(pd.DataFrame(), allowance)
