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

import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from Bio import SeqIO

from bionemo.evo2_phage_gen import protein_evidence


SMOOTH_SYNTENY_MATCH = {
    "identity_full_credit": 0.90,
    "reference_coverage_full_credit": 0.95,
    "candidate_coverage_full_credit": 0.95,
    "gamma": 1.5,
    "raw_integrity_min": 0.001,
    "min_credit": 0.01,
}
SMOOTH_TROPISM_MATCH = {
    **SMOOTH_SYNTENY_MATCH,
    "identity_full_credit": 0.95,
    "reference_coverage_full_credit": 0.99,
    "candidate_coverage_full_credit": 0.99,
}


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
        gamma=1.0,
        raw_integrity_min=0.0,
        min_credit=0.0,
    )
    assert observed == pytest.approx(0.70)


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
        synteny_match_parameters=SMOOTH_SYNTENY_MATCH,
        tropism_match_parameters=SMOOTH_TROPISM_MATCH,
        synteny_order_weight=0.75,
        synteny_duplicate_penalty_weight=0.75,
        gene_a_reference_locus="A",
        tropism_reference_locus="G",
        gene_a_origin_motif=motif,
        gene_a_origin_offset_nt=6,
        gene_a_origin_offset_tolerance_nt=6,
    ).set_index("id_prompt")

    assert observed.loc["umi1", "reward_external_synteny"] == 1.0
    assert observed.loc["umi1", "reward_external_tropism"] == 1.0
    assert observed.loc["umi1", "reward_gene_a_origin"] == 1.0
    assert observed.loc["umi1", "smooth_reference_matched_loci"] == 2
    assert observed.loc["umi2", "reward_external_synteny"] == 0.0
    assert observed.loc["umi2", "reward_external_tropism"] == 0.0
    assert observed.loc["umi2", "reward_gene_a_origin"] == 0.0


def test_smooth_reference_summary_rejects_invalid_match_settings_without_hits():
    """A no-hit batch must not turn an invalid reward configuration into measured zero."""
    with pytest.raises(ValueError, match="smooth protein-match configuration"):
        protein_evidence.summarize_smooth_reference_evidence(
            pd.DataFrame(columns=["query", "target", "evalue", "pident", "alnlen", "qlen", "tlen"]),
            genome_sequences={"umi1": "A" * 100},
            candidate_orf_sequences={},
            candidate_orders={"umi1": ()},
            reference_order=("A",),
            synteny_match_parameters={**SMOOTH_SYNTENY_MATCH, "identity_full_credit": 0.0},
            tropism_match_parameters=SMOOTH_TROPISM_MATCH,
            synteny_order_weight=0.75,
            synteny_duplicate_penalty_weight=0.75,
            gene_a_reference_locus="A",
            tropism_reference_locus="G",
            gene_a_origin_motif="CAACTTGATATTAATAACACTATAGACCAC",
            gene_a_origin_offset_nt=6,
            gene_a_origin_offset_tolerance_nt=6,
        )


def test_smooth_match_rejects_decoy_scale_evidence_and_grades_real_partial_matches():
    """A shuffled-scale edge must stay zero while a credible fragment starts above zero."""
    kwargs = {
        "identity_full_credit": 0.85,
        "reference_coverage_full_credit": 0.95,
        "candidate_coverage_full_credit": 0.95,
        "gamma": 1.5,
        "raw_integrity_min": 0.001,
        "min_credit": 0.01,
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

    assert decoy == 0.0
    assert 0.01 < partial < 1.0
    assert complete == 1.0


def test_smooth_match_penalizes_both_truncations_and_fusions():
    """Dropping either reciprocal-coverage side must lower an otherwise identical match."""
    kwargs = {
        "percent_identity": 85.0,
        "e_value": 1e-20,
        "identity_full_credit": 0.85,
        "reference_coverage_full_credit": 0.95,
        "candidate_coverage_full_credit": 0.95,
        "gamma": 1.5,
        "raw_integrity_min": 0.001,
        "min_credit": 0.01,
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


def test_ordered_partial_matches_outscore_the_same_scrambled_matches():
    """Synteny must add signal beyond reference content for identical edge weights."""
    reference_order = ("A", "B", "C", "D")
    ordered = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 0.2, ("B", "b"): 0.2, ("C", "c"): 0.2, ("D", "d"): 0.2},
        reference_order=reference_order,
        candidate_order=("a", "b", "c", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )
    scrambled = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 0.2, ("B", "b"): 0.2, ("C", "c"): 0.2, ("D", "d"): 0.2},
        reference_order=reference_order,
        candidate_order=("a", "c", "b", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )

    assert ordered.content_score == pytest.approx(scrambled.content_score)
    assert ordered.reward == pytest.approx(0.2)
    assert 0.0 < scrambled.reward < ordered.reward


def test_smooth_architecture_does_not_reward_deletion_or_order_repair_by_duplication():
    """A candidate cannot raise synteny by deleting evidence or adding a second homolog."""
    reference_order = ("A", "B", "C", "D")
    swapped = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 1.0, ("B", "b"): 1.0, ("C", "c"): 1.0, ("D", "d"): 1.0},
        reference_order=reference_order,
        candidate_order=("a", "c", "b", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )
    deleted = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 1.0, ("B", "b"): 1.0, ("D", "d"): 1.0},
        reference_order=reference_order,
        candidate_order=("a", "b", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )
    duplicated = protein_evidence.score_smooth_reference_architecture(
        {
            ("A", "a"): 1.0,
            ("B", "b"): 1.0,
            ("B", "b_ordered"): 1.0,
            ("C", "c"): 1.0,
            ("D", "d"): 1.0,
        },
        reference_order=reference_order,
        candidate_order=("a", "b_ordered", "c", "b", "d"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )

    assert deleted.reward < swapped.reward
    assert duplicated.reward == pytest.approx(swapped.reward)
    assert duplicated.duplicate_score == pytest.approx(0.25)


def test_smooth_architecture_is_rotation_invariant_and_one_to_one():
    """Circular rotation is neutral and one ORF cannot satisfy two reference loci."""
    rotated = protein_evidence.score_smooth_reference_architecture(
        {("A", "a"): 1.0, ("B", "b"): 1.0, ("C", "c"): 1.0, ("D", "d"): 1.0},
        reference_order=("A", "B", "C", "D"),
        candidate_order=("d", "a", "b", "c"),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )
    ambiguous = protein_evidence.score_smooth_reference_architecture(
        {("A", "shared"): 1.0, ("B", "shared"): 1.0},
        reference_order=("A", "B"),
        candidate_order=("shared",),
        order_weight=0.75,
        duplicate_penalty_weight=0.75,
    )

    assert rotated.reward == 1.0
    assert ambiguous.content_integrity_sum == 1.0


def test_smooth_architecture_scales_beyond_twenty_reference_loci():
    """Genome-scale architecture matching must not retain the bitmask implementation's locus cap."""
    reference_order = tuple(f"reference_{index}" for index in range(34))
    candidate_order = tuple(f"candidate_{index}" for index in range(34))
    edges = {
        (reference, candidate): 1.0 for reference, candidate in zip(reference_order, candidate_order, strict=True)
    }

    result = protein_evidence.score_smooth_reference_architecture(
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


def test_gene_a_origin_weights_the_nicking_core_and_ignores_nonfunctional_tail_bases():
    """Known nicking-core mutations matter more than bases 29-30, which are dispensable in vitro."""
    motif = "CAACTTGATATTAATAACACTATAGACCAC"
    offset = 345

    def score(site: str):
        candidate = "G" * offset + site + "G" * 200
        return protein_evidence.score_gene_a_origin(
            candidate_a_orf_nt=candidate,
            candidate_genome_nt=candidate,
            a_match_integrity=1.0,
            motif=motif,
            expected_offset_nt=offset,
            offset_tolerance_nt=30,
        )

    critical = score(motif[:3] + "A" + motif[4:])
    binding = score(motif[:14] + "C" + motif[15:])
    tail = score(motif[:28] + "TT")

    assert 0.0 < critical.reward < binding.reward < 1.0
    assert tail.reward == 1.0
    assert tail.exact_functional_site is True


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
    assert result.reward == pytest.approx(0.5)


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
def test_origin_context_rejects_wrong_frame_or_outside_window(offset):
    """Removing the within-window preference must retain frame and context checks."""
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
    assert result.reward == 0.0


def test_required_family_selector_matches_supported_phrog_target_forms_with_missing_annotations():
    """The known unknown-function slot must select PHROG1713, not missing annotations generally."""
    hits = _required_hits(
        ("numeric_ORF.1", pd.NA, 1713, 1.0),
        ("string_ORF.1", float("nan"), "1713", 1.0),
        ("prefixed_ORF.1", None, "phrog_1713", 1.0),
        ("unrelated_ORF.1", pd.NA, "phrog_9999", 1.0),
    )

    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("numeric", "string", "prefixed", "unrelated"),
        ("phrog:1713",),
    )

    assert observed["required_genes_integrity_sum"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert observed["required_genes_full_length_count"].tolist() == [1, 1, 1, 0]


def test_required_nan_compatibility_alias_selects_only_phrog1713():
    """Legacy `nan` configs must mean PHROG1713 without accepting unrelated missing annotations."""
    hits = _required_hits(
        ("family_ORF.1", pd.NA, "phrog_1713", 1.0),
        ("unrelated_ORF.1", float("nan"), "phrog_9999", 1.0),
    )

    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("family", "unrelated"),
        ("nan",),
    )

    assert observed["required_genes_integrity_sum"].tolist() == [1.0, 0.0]
    assert observed["required_genes_full_length_count"].tolist() == [1, 0]


def test_required_product_labels_keep_exact_annotation_matching():
    """Ordinary required products must continue to match their annotation labels."""
    hits = _required_hits(
        ("labeled_ORF.1", "terminase", "phrog_1", 1.0),
        ("missing_ORF.1", pd.NA, "phrog_2", 1.0),
    )

    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("labeled", "missing"),
        ("terminase",),
    )

    assert observed["required_genes_integrity_sum"].tolist() == [1.0, 0.0]


def test_required_integrity_assignment_is_invariant_to_orf_names_and_row_order():
    """Renumbering the same 0.675 and 1.0 family hits must not change the shaped score."""
    first = _required_hits(
        ("first_ORF.1", "DNA condensation", "phrog_2354", 0.675),
        ("first_ORF.2", "DNA condensation", "phrog_2354", 1.0),
    )
    rotated = _required_hits(
        ("rotated_ORF.2", "DNA condensation", "phrog_2354", 0.675),
        ("rotated_ORF.1", "DNA condensation", "phrog_2354", 1.0),
    ).iloc[::-1]

    first_score = protein_evidence.summarize_required_gene_evidence(
        first,
        _required_sequences("first"),
        ("DNA condensation",),
    )
    rotated_score = protein_evidence.summarize_required_gene_evidence(
        rotated,
        _required_sequences("rotated"),
        ("DNA condensation",),
    )

    assert first_score.loc[0, "required_genes_integrity_sum"] == 1.0
    assert rotated_score.loc[0, "required_genes_integrity_sum"] == 1.0


def test_required_integrity_capped_assignment_uses_best_single_edge():
    """A one-copy quota must optimize one edge rather than truncate a full matching."""
    hits = _required_hits(
        ("sample_ORF.1", "head", "family_1", 1.0),
        ("sample_ORF.1", "head", "family_2", 0.2),
        ("sample_ORF.2", "head", "family_1", 0.9),
        ("sample_ORF.2", "head", "family_2", 0.8),
    )

    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("sample"),
        ("head",),
    )

    assert observed.loc[0, "required_genes_integrity_sum"] == 1.0


def test_required_integrity_repeated_quota_uses_maximum_weight_one_to_one_assignment():
    """Repeated products must maximize evidence while keeping candidates and families unique."""
    hits = _required_hits(
        ("sample_ORF.1", "head", "family_1", 1.0),
        ("sample_ORF.1", "head", "family_2", 0.2),
        ("sample_ORF.2", "head", "family_1", 0.9),
        ("sample_ORF.2", "head", "family_2", 0.8),
    )

    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("sample"),
        ("head", "head"),
    )

    assert observed.loc[0, "required_genes_integrity_sum"] == pytest.approx(1.8)
    assert observed.loc[0, "required_genes_matched_count"] == 2


def test_required_integrity_does_not_reuse_one_target_family_for_a_repeated_product():
    """Extra ORFs hitting one family cannot satisfy a repeated-product quota twice."""
    hits = _required_hits(
        ("sample_ORF.1", "head", "family_1", 1.0),
        ("sample_ORF.2", "head", "family_1", 0.9),
    )

    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("sample"),
        ("head", "head"),
    )

    assert observed.loc[0, "required_genes_integrity_sum"] == 1.0
    assert observed.loc[0, "required_genes_matched_count"] == 1


def test_required_integrity_does_not_reuse_ambiguous_candidate_across_products():
    """One ORF with hits under two annotations may fill only one required-product slot."""
    hits = _required_hits(
        ("sample_ORF.1", "product A", "family_A", 1.0),
        ("sample_ORF.1", "product B", "family_B", 1.0),
        ("sample_ORF.2", "product A", "family_A", 0.5),
    )

    observed = protein_evidence.summarize_required_gene_evidence(
        hits,
        _required_sequences("sample"),
        ("product A", "product B"),
    )

    assert observed.loc[0, "required_genes_integrity_sum"] == 1.5
    assert observed.loc[0, "required_genes_matched_count"] == 2
    assert observed.loc[0, "required_genes_full_length_count"] == 1


def test_required_integrity_never_decreases_when_weaker_evidence_is_added():
    """Adding a weaker hit to an already satisfied slot must leave its best score intact."""
    strong = _required_hits(("sample_ORF.9", "head", "family_1", 1.0))
    with_weaker = pd.concat(
        [_required_hits(("sample_ORF.1", "head", "family_1", 0.675)), strong],
        ignore_index=True,
    )

    baseline = protein_evidence.summarize_required_gene_evidence(
        strong,
        _required_sequences("sample"),
        ("head",),
    )
    augmented = protein_evidence.summarize_required_gene_evidence(
        with_weaker,
        _required_sequences("sample"),
        ("head",),
    )

    assert augmented.loc[0, "required_genes_integrity_sum"] == baseline.loc[0, "required_genes_integrity_sum"]


def test_required_assignment_bounds_highs_threads_and_only_suppresses_its_forwarding_warning(monkeypatch):
    """The tiny MILP must be single-threaded without hiding unrelated solver warnings."""
    observed_options = None

    def fake_milp(**kwargs):
        nonlocal observed_options
        observed_options = kwargs.get("options")
        warnings.warn(
            "Unrecognized options detected: {'threads'}. These will be passed to HiGHS verbatim.",
            RuntimeWarning,
        )
        warnings.warn("independent solver warning", RuntimeWarning)
        return SimpleNamespace(success=True, x=[1.0], message="optimal")

    monkeypatch.setattr(protein_evidence, "milp", fake_milp)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assigned = protein_evidence._maximum_weight_required_assignment(
            {("candidate", "target", "product"): 1.0},
            {"product": 1},
        )

    assert assigned == (1.0,)
    assert observed_options == {"threads": 1}
    assert [(warning.category, str(warning.message)) for warning in caught] == [
        (RuntimeWarning, "independent solver warning")
    ]


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
        ("function",),
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
        ("function",),
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
        ("major head protein",),
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
            ("function",),
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
    assert protein_evidence.reference_synteny_pass_mask(metrics).tolist() == [True] + [False] * 6
    assert protein_evidence.reference_synteny_pass_mask(metrics, 1).tolist() == [True, True] + [False] * 5


@pytest.mark.parametrize("allowance", [-1, 1.5, True])
def test_synteny_deficit_allowance_requires_nonnegative_integer(allowance):
    with pytest.raises(ValueError, match="missing reference"):
        protein_evidence.reference_synteny_pass_mask(pd.DataFrame(), allowance)
