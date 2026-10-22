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

"""Regression tests preventing silent replacement of ARC gates with RL objectives."""

from unittest.mock import patch

import pandas as pd
import pytest

from bionemo.evo2_phage_gen.arc_final_filters import (
    ADD_ONE_GENE_CONTRACT,
    CIRCULAR_AAI_CONTRACT,
    CONTRACT,
    arc_synteny_pass_mask,
    classify_syntenic_diversity,
    load_arc_functions,
    stage_arc_reference,
    validate_core_safety,
)


def test_arc_count_pairs_are_independent_of_core():
    rows = [(a, b) for a in range(8, 14) for b in range(8, 14)]
    frame = pd.DataFrame(rows, columns=["num_syntenic_genes", "total_num_genes"])
    actual = arc_synteny_pass_mask(frame).tolist()
    assert actual == [10 <= a <= 12 and 10 <= b <= 12 and (a, b) != (11, 11) for a, b in rows]
    assert not arc_synteny_pass_mask(pd.DataFrame({"num_syntenic_genes": [9], "total_num_genes": [11]})).iloc[0]


@pytest.mark.parametrize("value", [None, float("nan"), 10.5])
def test_missing_or_fractional_synteny_cannot_pass(value):
    with pytest.raises(ValueError):
        arc_synteny_pass_mask(pd.DataFrame({"num_syntenic_genes": [value], "total_num_genes": [10]}))


def test_original_reference_preserves_eleven_features(tmp_path):
    from bionemo.evo2_phage_gen.arc_final_filters import RECIPE_ROOT

    if not (RECIPE_ROOT / "data/external/arc_evo2/phage_gen/data/NC_001422.1_pseudocircular.gff").exists():
        pytest.skip("pinned ARC assets unavailable")
    staged = stage_arc_reference(tmp_path / "original-reference.gff")
    assert sum("\tCDS\t" in line for line in staged.read_text().splitlines()) == 11
    assert "NC_001422.1_ORF.9" in staged.read_text()


def test_pinned_source_rejects_a_modified_scorer(tmp_path):
    source = tmp_path / "pipeline.py"
    source.write_text("def substituted_core_gate(): return True\n")
    with pytest.raises(ValueError, match="pinned"):
        load_arc_functions(source)


def evidence():
    config = {
        "final_arc_contract": CONTRACT,
        "core_gene_reference_functions": dict(zip("ABCDEFGHJ", "ABCDEFGHJ")),
        "sequence_safety_manifest": "safety.json",
        "evo_gen_seqs_fasta_file_save_location": "input.fasta",
    }
    frame = pd.DataFrame(
        {
            "id_prompt": ["x"],
            "sequence": ["ACGT"],
            "num_syntenic_genes": [9],
            "reference_num_genes": [9],
            "duplicate_reference_gene_count": [0],
            "reference_order_violation_count": [0],
            "missing_synteny_output": [False],
            "average_protein_percent_identity": [94.0],
            "average_protein_identity_gene_count": [10],
        }
    )
    return config, frame


def test_core_evidence_and_safety_are_both_required():
    config, frame = evidence()
    with patch("bionemo.evo2_phage_gen.rollout_evidence.load_safety_pass_sequences", return_value={"ACGT"}):
        validate_core_safety(frame, config)
        frame.loc[0, "duplicate_reference_gene_count"] = 1
        with pytest.raises(ValueError, match="nine-core"):
            validate_core_safety(frame, config)
    config, frame = evidence()
    with patch("bionemo.evo2_phage_gen.rollout_evidence.load_safety_pass_sequences", return_value=set()):
        with pytest.raises(ValueError, match="safety PASS"):
            validate_core_safety(frame, config)
    config["online_measurement_mode"] = True
    with pytest.raises(ValueError, match="offline"):
        validate_core_safety(frame, config)


def test_missing_core_measurements_are_not_boolean_passes():
    config, frame = evidence()
    with pytest.raises(ValueError, match="Missing"):
        validate_core_safety(frame.drop(columns="missing_synteny_output"), config)


def test_original_synteny_formula_matches_pinned_function(tmp_path):
    from bionemo.evo2_phage_gen.arc_final_filters import RECIPE_ROOT

    source = RECIPE_ROOT / "data/external/arc_evo2/phage_gen/pipelines/genome_design_filtering_pipeline.py"
    if not source.exists():
        pytest.skip("pinned ARC assets unavailable")
    api = load_arc_functions(source)
    frame = pd.DataFrame(
        [(f"g{a}_{b}", a, b) for a in range(8, 14) for b in range(8, 14)],
        columns=["genome_id", "num_syntenic_genes", "total_num_genes"],
    )
    source_csv = tmp_path / "input.csv"
    result_csv = tmp_path / "result.csv"
    frame.to_csv(source_csv, index=False)
    api["valid_syntenic_gene_count"](
        str(source_csv), str(result_csv), [10, 12], [10, 12], [[11, 11]], *[str(tmp_path / "unused")] * 4
    )
    assert list(pd.read_csv(result_csv).genome_id) == list(frame.loc[arc_synteny_pass_mask(frame), "genome_id"])


def test_current_aai_cannot_be_dropped_by_original_arc_branch():
    config, frame = evidence()
    frame.loc[0, "average_protein_percent_identity"] = 96
    with pytest.raises(ValueError, match="physical-ORF AAI"):
        validate_core_safety(frame, config)


@pytest.mark.parametrize("contract", [CONTRACT, CIRCULAR_AAI_CONTRACT, ADD_ONE_GENE_CONTRACT])
def test_both_contracts_require_the_same_physical_aai_core_and_safety(contract):
    config, frame = evidence()
    config["final_arc_contract"] = contract
    with patch("bionemo.evo2_phage_gen.rollout_evidence.load_safety_pass_sequences", return_value={"ACGT"}):
        validate_core_safety(frame, config)
        frame.loc[0, "average_protein_percent_identity"] = 95.1
        with pytest.raises(ValueError, match="physical-ORF AAI"):
            validate_core_safety(frame, config)


def test_unrecognized_contract_cannot_silently_choose_an_aai_definition():
    config, frame = evidence()
    config["final_arc_contract"] = "unversioned-aai"
    with pytest.raises(ValueError, match="offline"):
        validate_core_safety(frame, config)


def diversity_evidence(counts=(11, 13)):
    synteny = pd.DataFrame({"id_prompt": ["g"], "num_syntenic_genes": [counts[0]], "total_num_genes": [counts[1]]})
    metrics = pd.DataFrame(
        {
            "id_prompt": ["g"],
            "accessory_k_credit": [1.0],
            "accessory_x_credit": [1.0],
            "accessory_distinct_type_mass": [2.0],
            "accessory_duplicate_mass": [0.0],
            "accessory_gene_diversification_measurement_available": [1.0],
        }
    )
    assignments = pd.DataFrame(
        [
            {
                "id_prompt": "g_ORF.1",
                "genome_id": "g",
                "family": "phrog:1713",
                "accessory_type": "K",
                "credit": 1.0,
                "eligible": True,
            },
            {
                "id_prompt": "g_ORF.2",
                "genome_id": "g",
                "family": "phrog:1670",
                "accessory_type": "phrog:1670",
                "credit": 1.0,
                "eligible": True,
            },
        ]
    )
    return synteny, metrics, assignments


def classify(*args):
    return classify_syntenic_diversity(*args, core_families={"phrog:1"}, k_families={"phrog:1713"})


def test_structured_addition_has_separate_category():
    result = classify(*diversity_evidence()).iloc[0]
    assert not result.syntenic_diversity_arc_count_pass
    assert result.syntenic_diversity_add_one_gene_pass
    assert result.syntenic_diversity_pass
    assert result.syntenic_diversity_category == "add_one_gene"
    assert result.syntenic_diversity_x_family == "phrog:1670"
    assert result.syntenic_diversity_x_orf_id == "g_ORF.2"


@pytest.mark.parametrize(
    "column,value",
    [
        ("accessory_x_credit", 0.99),
        ("accessory_k_credit", 0.99),
        ("accessory_k_credit", 0.0),
        ("accessory_duplicate_mass", 0.01),
        ("accessory_distinct_type_mass", 2.01),
        ("accessory_distinct_type_mass", 1.0),
        ("accessory_gene_diversification_measurement_available", 0.0),
        ("accessory_x_credit", float("nan")),
        ("accessory_k_credit", float("inf")),
    ],
)
def test_count_pair_never_bypasses_structured_evidence(column, value):
    counts, metrics, assignments = diversity_evidence()
    metrics.loc[0, column] = value
    result = classify(counts, metrics, assignments).iloc[0]
    assert not result.syntenic_diversity_pass
    assert result.syntenic_diversity_category == "fail"


@pytest.mark.parametrize(
    "change", ["missing_x", "reserved_x", "core_family", "k_family", "partial_x", "extra_copy", "extra_type"]
)
def test_orf_evidence_required_even_with_perfect_scores(change):
    counts, metrics, assignments = diversity_evidence()
    if change == "missing_x":
        assignments = assignments.iloc[:1]
    elif change == "reserved_x":
        assignments.loc[1, "eligible"] = False
    elif change in {"core_family", "k_family"}:
        family = "phrog:1" if change == "core_family" else "phrog:1713"
        assignments.loc[1, ["family", "accessory_type"]] = family
    elif change == "partial_x":
        assignments.loc[1, "credit"] = 0.99
    else:
        extra = assignments.iloc[[1]].copy()
        extra["id_prompt"] = "g_ORF.3"
        if change == "extra_type":
            extra[["family", "accessory_type"]] = "phrog:999"
        assignments = pd.concat([assignments, extra], ignore_index=True)
    assert not classify(counts, metrics, assignments).syntenic_diversity_pass.iloc[0]


def test_original_route_and_overlap_remain_explicit():
    counts, metrics, assignments = diversity_evidence((10, 10))
    result = classify(counts, metrics, assignments).iloc[0]
    assert result.syntenic_diversity_category == "both"
    metrics.loc[0, "accessory_x_credit"] = 0
    result = classify(counts, metrics, assignments).iloc[0]
    assert result.syntenic_diversity_category == "original_arc_count"
    assert result.syntenic_diversity_pass
    assert not result.syntenic_diversity_add_one_gene_pass


def test_missing_or_misaligned_native_evidence_fails_closed():
    counts, metrics, assignments = diversity_evidence()
    metrics.loc[0, "id_prompt"] = "other"
    with pytest.raises(ValueError, match="match"):
        classify(counts, metrics, assignments)
    counts, metrics, assignments = diversity_evidence()
    assignments["eligible"] = assignments.eligible.astype(object)
    assignments.loc[1, "eligible"] = "True"
    with pytest.raises(ValueError, match="booleans"):
        classify(counts, metrics, assignments)
