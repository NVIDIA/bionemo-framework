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

import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from bionemo.evo2_phage_gen import accessory_genes, arc_final_filters
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


@pytest.fixture
def final_replay(tmp_path, monkeypatch):
    """Replay real upstream decisions with synthetic external-search/ORF outputs.

    Every candidate advertises perfect online reward. Literal external evidence
    includes one failure for each common ARC gate, distinct original/added/both
    synteny routes, and an original-inventory AAI failure. No acceptance mask,
    aggregation, GFF annotation, or accessory measurement is mocked.
    """
    upstream_fixture = Path(__file__).parent / "fixtures/arc_pipeline_upstream.py"
    with monkeypatch.context() as fixture_source:
        # The committed excerpt has separately verified per-function upstream
        # provenance. Use the real loader's coordinate correction on that excerpt.
        fixture_source.setattr(
            arc_final_filters, "UPSTREAM_SHA256", hashlib.sha256(upstream_fixture.read_bytes()).hexdigest()
        )
        api = arc_final_filters.load_arc_functions(upstream_fixture)
    # (original ORF count, cross-reference cluster pairs, original AAI, native X)
    panel = {
        "original": (10, 10, 90, False),
        "added": (13, 11, 90, True),
        "both": (10, 10, 90, True),
        "count_only": (13, 11, 90, False),
        "near_copy": (10, 10, 98, True),
        "bad_chars": (10, 10, 90, True),
        "short": (10, 10, 90, True),
        "high_gc": (10, 10, 90, True),
        "homopolymer": (10, 10, 90, True),
        "six_hits": (6, 6, 90, True),
        "low_tropism": (10, 10, 90, True),
    }
    sequences = {}
    for index, name in enumerate(panel):
        suffix = "".join("ACGT"[(index >> shift) & 3] for shift in (6, 4, 2, 0))
        sequences[name] = "ACGT" * 1249 + suffix
    sequences["bad_chars"] = "N" + sequences["bad_chars"][1:]
    sequences["short"] = sequences["short"][:3999]
    sequences["high_gc"] = "GCGT" * 1249 + "ACGT"
    sequences["homopolymer"] = "A" * 11 + sequences["homopolymer"][11:]
    fasta = tmp_path / "input.fasta"
    fasta.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in sequences.items()))
    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"records": [{"record_id": name, "state": "PASS"} for name in panel]}))
    qualified = pd.DataFrame(
        {
            "id_prompt": list(panel),
            "sequence": list(sequences.values()),
            "num_syntenic_genes": 9,
            "reference_num_genes": 9,
            "duplicate_reference_gene_count": 0,
            "reference_order_violation_count": 0,
            "missing_synteny_output": False,
            "average_protein_percent_identity": 94.0,
            "average_protein_identity_gene_count": 9,
            "reward": 1.0,
            "all_objectives_max_score": 1.0,
        }
    )
    natural_db = tmp_path / "natural-db"
    natural_db.write_bytes(b"natural-member database v1")
    tropism_db = tmp_path / "tropism-db"
    tropism_db.write_bytes(b"tropism database v1")
    annotation = tmp_path / "annotations.tsv"
    annotation.write_text("phrog\tannot\tcategory\n1\ttest protein\ttest category\n")
    reference = tmp_path / "physical-reference.gff"
    reference.write_text(
        "##gff-version 3\n"
        + "".join(
            f"ref\ttest\tCDS\t{i * 90 + 1}\t{(i + 1) * 90}\t.\t+\t0\tID=ref{gene}\n"
            for i, gene in enumerate("ABCDEFGHJ")
        )
        + "##FASTA\n>ref\n"
        + "ATG" * 270
        + "\n"
    )
    physical_proteins = tmp_path / "physical.fasta"
    physical_proteins.write_text("".join(f">{name}_ORF.{i}\n{'M' * 30}\n" for name in panel for i in (1, 2)))
    family_dir = tmp_path / "families"
    family_dir.mkdir()
    family_rows = []
    for name, (_, _, _, has_x) in panel.items():
        for i, family in ((1, 1713), (2, 1670)) if has_x else ((1, 1713),):
            family_rows.append(
                {
                    "id_prompt": f"{name}_ORF.{i}",
                    "protein_database_mmseqs_target": f"phrog_{family}",
                    "protein_database_mmseqs_e_value": 1e-20,
                    "protein_database_mmseqs_percent_identity": 90,
                    "protein_database_mmseqs_alignment_length": 30,
                    "protein_database_mmseqs_query_length": 30,
                    "protein_database_mmseqs_target_length": 30,
                    "protein_database_mmseqs_query_coverage": 1.0,
                    "protein_database_mmseqs_target_coverage": 1.0,
                }
            )
    pd.DataFrame(family_rows).to_csv(family_dir / "mmseqs2_all_hits.csv", index=False)
    config = {
        "final_arc_contract": ADD_ONE_GENE_CONTRACT,
        "core_gene_reference_functions": {f"ref{gene}": gene for gene in "ABCDEFGHJ"},
        "required_gene_families": {gene: [f"phrog:{i}"] for i, gene in enumerate("ABCDEFGHJ", 1)},
        "accessory_gene_k_families": ["phrog:1713"],
        "sequence_safety_manifest": str(safety),
        "evo_gen_seqs_fasta_file_save_location": str(fasta),
        "results_save_dir": str(tmp_path),
        "mmseqs_db_aai_database": str(natural_db),
        "mmseqs_db_tropism_protein": str(tropism_db),
        "protein_annotation_file": str(annotation),
        "reference_genome_gff_file_save_location": str(reference),
        "orfipy_proteins_file_save_location": physical_proteins.name,
        "mmseqs_protein_database_results_dir_save_location": family_dir.name,
        "final_arc_workers": 1,
        "mmseqs_threads": 1,
    }

    def orf_output(input_fasta, threads, starts, stops, strand, minimum, maximum, out, nuc, temp, proteins):
        nucleotide_records, protein_records = [], []
        for record in SeqIO.parse(input_fasta, "fasta"):
            for i in range(panel[record.id][0]):
                identifier = f"{record.id}_ORF.{i + 1}"
                description = f"[{i * 90}-{(i + 1) * 90}](+)"
                nucleotide_records.append(SeqRecord(Seq("ATG" * 30), id=identifier, description=description))
                protein_records.append(SeqRecord(Seq("M" * 30), id=identifier, description=description))
        SeqIO.write(nucleotide_records, Path(out) / nuc, "fasta")
        SeqIO.write(protein_records, Path(out) / proteins, "fasta")

    def search_output(query, database, results, output_csv, prefix, **kwargs):
        rows = []
        for record in SeqIO.parse(query, "fasta"):
            name = record.id.rsplit("_", 1)[0]
            if prefix == "tropism_protein" and not record.id.endswith("_ORF.1"):
                continue
            identity = panel[name][2] if prefix == "protein_database" else 59 if name == "low_tropism" else 90
            rows.append(
                {
                    "id_prompt": record.id,
                    "sequence": str(record.seq),
                    f"{prefix}_mmseqs_target": "phrog_1|natural-member",
                    f"{prefix}_mmseqs_e_value": 1e-20,
                    f"{prefix}_mmseqs_percent_identity": identity,
                }
            )
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    def staged_reference(output):
        output.write_text(
            "".join(f"ref\ttest\tCDS\t{i * 90 + 1}\t{(i + 1) * 90}\t.\t+\t0\tID=ORF.{i + 1}\n" for i in range(11))
        )
        return output

    def cluster_output():
        inputs = Path(sys.argv[sys.argv.index("-gff") + 1])
        candidate = next(inputs.glob("genome_*.gff"))
        name = next(
            line.removeprefix("##description ")
            for line in candidate.read_text().splitlines()
            if line.startswith("##description ")
        )
        total, paired, _, _ = panel[name]
        rows = []
        for i in range(1, 12):
            ref_id = f"ref-ORF.{i}"
            rows.append(f"{ref_id}\t{ref_id}\n")
            if i <= paired:
                rows.append(f"{ref_id}\t{candidate.stem}-ORF.{i}\n")
        for i in range(paired + 1, total + 1):
            identifier = f"{candidate.stem}-ORF.{i}"
            rows.append(f"{identifier}\t{identifier}\n")
        table = Path(sys.argv[sys.argv.index("-o") + 1]) / "mmseqs/mmseqs_clustering.tsv"
        table.parent.mkdir(parents=True)
        table.write_text("".join(rows))

    def reference_search_output(**kwargs):
        # A successfully completed, empty direct-reference search is measured evidence.
        kwargs["output_tsv"].write_text("")

    api.update(run_orfipy=orf_output, run_mmseqs_search_proteins=search_output)
    monkeypatch.setattr(arc_final_filters, "load_arc_functions", lambda source: api)
    monkeypatch.setattr(arc_final_filters, "stage_arc_reference", staged_reference)
    monkeypatch.setattr(
        arc_final_filters.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="test-mmseqs")
    )
    monkeypatch.setattr(arc_final_filters.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(
        arc_final_filters.concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    cluster_module = SimpleNamespace(main=cluster_output)
    monkeypatch.setitem(sys.modules, "bionemo.evo2_phage_gen.lovis4u_metrics", cluster_module)
    # Another collected test may already have imported the real submodule, so
    # `from ... import lovis4u_metrics` can resolve the cached package attribute.
    monkeypatch.setattr(sys.modules["bionemo.evo2_phage_gen"], "lovis4u_metrics", cluster_module, raising=False)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setenv("LOVIS4U_MMSEQS_THREADS", "1")
    monkeypatch.setenv("LOVIS4U_MMSEQS_BINARY", "mmseqs")
    monkeypatch.setattr(accessory_genes, "run_reference_protein_search", reference_search_output)
    return SimpleNamespace(qualified=qualified, config=config, output=tmp_path / "replay", api=api)


@pytest.mark.parametrize(
    "contract,accepted,gate_counts,route_counts",
    [
        (CONTRACT, ["original", "both"], [10, 9, 8, 7, 6, 5, 4, 2], [2, 0, 0, 0, 2]),
        (CIRCULAR_AAI_CONTRACT, ["original", "both", "near_copy"], [10, 9, 8, 7, 6, 5, 5, 3], [3, 0, 0, 0, 3]),
        (
            ADD_ONE_GENE_CONTRACT,
            ["original", "added", "both", "near_copy"],
            [10, 9, 8, 7, 6, 5, 5, 4],
            [3, 3, 2, 1, 4],
        ),
    ],
)
def test_final_replay_keeps_independent_gates(final_replay, contract, accepted, gate_counts, route_counts):
    run = final_replay
    run.config["final_arc_contract"] = contract
    result, summary = arc_final_filters.apply_arc_final_filters(run.qualified, run.config, output_dir=run.output)
    assert result.id_prompt.tolist() == accepted
    assert result.reward.eq(1).all()
    assert result.num_syntenic_genes.eq(9).all()  # Original core evidence survives the independent ARC annotation.
    assert result.average_protein_percent_identity.eq(94).all()
    assert summary["state"] == "succeeded"
    assert summary["passed_observations"] == len(accepted)
    assert [entry["remaining"] for entry in summary["gate_order"]] == gate_counts
    assert summary["syntenic_diversity_counts"] == dict(
        zip(["original_arc_count", "add_one_gene", "both", "add_one_gene_only", "union"], route_counts, strict=True)
    )
    assert summary["original_inventory_aai_passers"] == 4
    assert json.loads((run.output / "summary.json").read_text()) == summary
    assert pd.read_csv(run.output / "accepted-metrics.csv").id_prompt.tolist() == accepted
    assert [record.id for record in SeqIO.parse(run.output / "accepted-observations.fasta", "fasta")] == accepted
    if contract == ADD_ONE_GENE_CONTRACT:
        decisions = pd.read_csv(run.output / "syntenic-diversity-decisions.csv").set_index("id_prompt")
        assert decisions.syntenic_diversity_category.to_dict() == {
            "original": "original_arc_count",
            "added": "add_one_gene",
            "both": "both",
            "count_only": "fail",
            "near_copy": "both",
        }
        assert result.set_index("id_prompt").loc["near_copy", "arc_average_protein_percent_identity"] == 98


@pytest.mark.parametrize("changed", ["tropism", "tropism-sidecar", "natural", "annotation", "physical", "input"])
def test_replay_cache_rejects_changed_evidence(final_replay, changed):
    run = final_replay
    first, summary = arc_final_filters.apply_arc_final_filters(run.qualified, run.config, output_dir=run.output)
    fingerprint = (run.output / "input-fingerprint.txt").read_text()
    repeated, repeated_summary = arc_final_filters.apply_arc_final_filters(
        run.qualified, run.config, output_dir=run.output
    )
    pd.testing.assert_frame_equal(first, repeated)
    assert repeated_summary == summary
    if changed == "input":
        run.qualified.loc[0, "reward"] = 0.5
    else:
        path = {
            "tropism": Path(run.config["mmseqs_db_tropism_protein"]),
            "tropism-sidecar": Path(run.config["mmseqs_db_tropism_protein"] + ".index"),
            "natural": Path(run.config["mmseqs_db_aai_database"]),
            "annotation": Path(run.config["protein_annotation_file"]),
            "physical": Path(run.config["results_save_dir"]) / run.config["orfipy_proteins_file_save_location"],
        }[changed]
        path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\nchanged evidence\n")
    with pytest.raises(ValueError, match="different input/configuration"):
        arc_final_filters.apply_arc_final_filters(run.qualified, run.config, output_dir=run.output)
    assert (run.output / "input-fingerprint.txt").read_text() == fingerprint
    assert json.loads((run.output / "summary.json").read_text()) == summary


def test_missing_tropism_database_fails_closed(final_replay):
    run = final_replay
    Path(run.config["mmseqs_db_tropism_protein"]).unlink()
    with pytest.raises(ValueError, match=r"tropism.*database"):
        arc_final_filters.apply_arc_final_filters(run.qualified, run.config, output_dir=run.output)


@pytest.mark.parametrize("missing", ["core", "physical_aai", "safety", "accessory"])
def test_perfect_reward_cannot_replace_evidence(final_replay, missing):
    run = final_replay
    if missing == "core":
        run.qualified = run.qualified.drop(columns="missing_synteny_output")
    elif missing == "physical_aai":
        run.qualified.loc[0, "average_protein_identity_gene_count"] = 0
    elif missing == "safety":
        path = Path(run.config["sequence_safety_manifest"])
        manifest = json.loads(path.read_text())
        manifest["records"][0]["state"] = "INDETERMINATE"
        path.write_text(json.dumps(manifest))
    else:
        path = Path(run.config["results_save_dir"]) / "families/mmseqs2_all_hits.csv"
        pd.read_csv(path).drop(columns="protein_database_mmseqs_query_coverage").to_csv(path, index=False)
    with pytest.raises(ValueError):
        arc_final_filters.apply_arc_final_filters(run.qualified, run.config, output_dir=run.output)
    assert not (run.output / "accepted-observations.fasta").exists()


def test_empty_gate_reports_zero_remaining(final_replay):
    run = final_replay
    qualified = run.qualified.loc[run.qualified.id_prompt.eq("bad_chars")]
    accepted, summary = arc_final_filters.apply_arc_final_filters(qualified, run.config, output_dir=run.output)
    assert accepted.empty
    assert summary["gate_order"] == [
        {"gate": gate, "remaining": 0}
        for gate in (
            "valid_nt_chars",
            "valid_genome_len",
            "valid_gc_content",
            "valid_nt_homopolymer_len",
            "arc_seven_protein_hits",
            "arc_tropism",
            "physical_aai_reconfirmed",
            "syntenic_diversity",
        )
    ]
    assert summary["passed_observations"] == 0
    assert summary["state"] == "succeeded"
    assert (run.output / "accepted-observations.fasta").read_text() == ""
    assert pd.read_csv(run.output / "accepted-metrics.csv").empty


@pytest.mark.parametrize("outcome", ["no_hits", "failed"])
def test_external_failure_is_not_a_successful_zero(final_replay, outcome):
    run = final_replay

    def search_output(query, database, results, output_csv, prefix, **kwargs):
        if outcome == "failed":
            raise RuntimeError("synthetic search execution failure")
        pd.DataFrame(columns=["id_prompt", f"{prefix}_mmseqs_target"]).to_csv(output_csv, index=False)

    run.api["run_mmseqs_search_proteins"] = search_output
    if outcome == "failed":
        with pytest.raises(RuntimeError, match="search execution failure"):
            arc_final_filters.apply_arc_final_filters(run.qualified, run.config, output_dir=run.output)
        assert json.loads((run.output / "summary.json").read_text())["state"] == "running"
        assert not (run.output / "accepted-observations.fasta").exists()
    else:
        accepted, summary = arc_final_filters.apply_arc_final_filters(run.qualified, run.config, output_dir=run.output)
        assert accepted.empty
        assert summary["state"] == "succeeded"
        assert [entry["remaining"] for entry in summary["gate_order"]] == [10, 9, 8, 7, 0, 0, 0, 0]
