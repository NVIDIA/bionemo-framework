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

"""Behavioral controls for accessory diversification, independent of core completeness."""

from itertools import pairwise

import pandas as pd
import pytest

from bionemo.evo2_phage_gen import accessory_genes


def _hit(orf, family, credit=1.0, evalue=1e-20):
    return {
        "id_prompt": f"g_ORF.{orf}",
        "protein_database_mmseqs_target": family,
        "protein_database_mmseqs_e_value": evalue,
        "protein_database_mmseqs_percent_identity": 80.0,
        "protein_database_mmseqs_alignment_length": 75,
        "protein_database_mmseqs_query_length": 100,
        "protein_database_mmseqs_target_length": 100,
        "protein_database_mmseqs_query_coverage": 0.75 * credit,
        "protein_database_mmseqs_target_coverage": 0.75 * credit,
    }


def _measure(hits, *, reserved=(), proteins=None):
    table = pd.DataFrame(hits, columns=_hit(1, 1713))
    return accessory_genes.summarize_accessory_gene_evidence(
        table,
        pd.DataFrame({"id_prompt": ["g"]}),
        reserved_core_orfs=set(reserved),
        core_families={"phrog:713"},
        k_families={"phrog:1713"},
        candidate_proteins=proteins or {f"g_ORF.{i}": "M" + "A" * 99 for i in range(1, 8)},
        reference_k_protein="M" + "A" * 99,
    )


def test_k_x_surface():
    score = accessory_genes.score_accessory_diversification
    assert score(0, 0) == 0.75
    assert score(1, 0) == 0.50
    assert score(0, 1) == score(1, 1) == 1
    assert score(0.5, 1) == 0.8
    assert score(0.5, 0.1) == pytest.approx(0.6425)
    assert score(0.1, 0.1) == score(0.9, 0.1) == pytest.approx(0.7485)
    for k in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1):
        assert score(k, 0) == pytest.approx(0.75 - 0.25 * k)
        values = [score(k, x) for x in (0, 1e-6, 0.1, 0.5, 1)]
        assert all(0 <= v <= 1 for v in values)
        assert all(a < b for a, b in pairwise(values))
        for x in (1e-6, 0.1, 1):
            assert score(k, x) == pytest.approx(score(1 - k, x))
            if 0 < k <= 0.5:
                assert score(k - 0.01, x) > score(k, x)
            if 0.5 <= k < 1:
                assert score(k + 0.01, x) > score(k, x)


def test_accessory_pool_excludes_core():
    # A secondary partial core hit reserves ORF 1; a direct-reference core hit
    # reserves ORF 2. A different, unapproved family on ORF 3 remains an extra.
    hits = [_hit(1, 900), _hit(1, 713, 0.2), _hit(2, 901), _hit(3, 902, 0.4)]
    metrics, assignments = _measure(hits, reserved={"g_ORF.2"})
    row = metrics.iloc[0]
    assert row.accessory_x_credit == pytest.approx(0.4)
    assert row.accessory_distinct_type_mass == pytest.approx(0.4)
    assert set(assignments.loc[assignments.eligible, "id_prompt"]) == {"g_ORF.3"}
    # A reserved K ORF must not turn into an apparent K deletion.
    metrics, _ = _measure([_hit(1, 1713)], reserved={"g_ORF.1"})
    assert metrics.iloc[0].accessory_k_credit == 1
    assert metrics.iloc[0].reward_external_accessory_gene_diversification == 0.5


@pytest.mark.parametrize(
    ("families", "expected"),
    [
        ([], 0.75),
        ([1713], 0.5),
        ([900], 1.0),
        ([1713, 900], 1.0),
        ([900, 901], 1.0),
        ([1713, 900, 901], 0.5),
        ([1713, 900, 900], 0.5),
        ([1713, 1713, 1713], 1 / 6),
        ([1713, 8511], 1.0),
    ],
)
def test_accessory_copy_budget(families, expected):
    hits = [_hit(i, family) for i, family in enumerate(families, 1)]
    metrics, _ = _measure(hits)
    row = metrics.iloc[0]
    assert row.reward_external_accessory_gene_diversification == pytest.approx(expected)


def test_accessory_evidence_and_k_swap():
    # One ORF's weaker alternate family cannot manufacture another accessory type.
    metrics, _ = _measure([_hit(1, 900), _hit(1, 901, evalue=1e-5)])
    assert metrics.iloc[0].accessory_distinct_type_mass == 1
    # A distant K-like family remains a distinct extra despite a weaker K hit.
    metrics, assignments = _measure([_hit(1, 1713), _hit(2, 8511), _hit(2, 1713, 0.2, evalue=1e-5)])
    assert metrics.iloc[0].reward_external_accessory_gene_diversification == 1
    assert metrics.iloc[0].accessory_duplicate_mass == 0
    assert set(assignments.accessory_type) == {"K", "phrog:8511"}
    # An intact supported K homolog can provide novelty without a non-K family.
    metrics, _ = _measure([_hit(1, 1713)], proteins={"g_ORF.1": "M" + "C" * 10 + "A" * 89})
    assert metrics.iloc[0].reward_external_accessory_gene_diversification == 1
    # A truncated identical K fragment is incomplete, not a divergent K swap.
    metrics, _ = _measure([_hit(1, 1713, 0.5)], proteins={"g_ORF.1": "M" + "A" * 49})
    assert metrics.iloc[0].accessory_x_credit == 0
    # Missing search columns cannot be interpreted as a supported K deletion.
    with pytest.raises(ValueError, match=r"coverage|evidence"):
        accessory_genes.summarize_accessory_gene_evidence(
            pd.DataFrame(),
            pd.DataFrame({"id_prompt": ["g"]}),
            reserved_core_orfs=set(),
            core_families={"phrog:713"},
            k_families={"phrog:1713"},
            candidate_proteins={"g_ORF.1": "MA"},
            reference_k_protein="MA",
        )


def test_accessory_artifact_measurement(tmp_path, monkeypatch):
    """Shared reference evidence reserves partial core ORFs; empty cohorts earn no loss credit."""
    dna = "ATG" + "GCT" * 99
    (tmp_path / "reference.gff").write_text(
        "##gff-version 3\n"
        "ref\t.\tCDS\t1\t300\t.\t+\t0\tID=refA\n"
        "ref\t.\tCDS\t301\t600\t.\t+\t0\tID=refK\n"
        f"##FASTA\n>ref\n{dna * 2}\n"
    )
    (tmp_path / "proteins.faa").write_text("".join(f">g_ORF.{i}\nM{'A' * 99}\n" for i in (1, 2, 3)))
    (tmp_path / "phrogs").mkdir()
    pd.DataFrame([_hit(1, 1713), _hit(2, 8511), _hit(3, 900)]).to_csv(
        tmp_path / "phrogs/mmseqs2_all_hits.csv", index=False
    )
    config = {
        "results_save_dir": str(tmp_path),
        "smooth_reference_genome_gff_file": str(tmp_path / "reference.gff"),
        "core_gene_reference_functions": {"refA": "A"},
        "required_gene_families": {"A": ["phrog:713"]},
        "accessory_gene_k_reference_locus": "refK",
        "accessory_gene_k_families": ["phrog:1713"],
        "orfipy_proteins_file_save_location": "proteins.faa",
        "mmseqs_protein_database_results_dir_save_location": "phrogs",
    }
    searches = []

    def search(**kwargs):
        searches.append(kwargs)
        assert kwargs["env"] == {"PATH": "/prepared/tools"}
        kwargs["output_tsv"].write_text("refA\tg_ORF.3\t1e-10\t80\t30\t100\t100\t0.3\t0.3\n")

    monkeypatch.setattr(accessory_genes, "run_reference_protein_search", search)
    cohort = pd.DataFrame({"id_prompt": ["g", "no_called_proteins"]})
    for _ in range(2):
        metrics = accessory_genes.measure_accessory_gene_artifacts(config, cohort, env={"PATH": "/prepared/tools"})
        assert metrics.reward_external_accessory_gene_diversification.tolist() == [1, 0]
        assert metrics.accessory_gene_diversification_measurement_available.tolist() == [True, False]
    assert len(searches) == 1
    assignments = pd.read_csv(tmp_path / "qc6_accessory_genes_assignments.csv")
    assert set(assignments.loc[assignments.eligible, "id_prompt"]) == {"g_ORF.1", "g_ORF.2"}
