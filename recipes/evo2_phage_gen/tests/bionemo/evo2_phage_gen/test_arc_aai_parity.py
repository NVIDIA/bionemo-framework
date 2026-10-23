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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Independent AAI aggregation oracles; not a claim of fresh search parity.

The workbook supplies sequences and published numbers, the archived search
supplies alignments, and verbatim upstream functions supply a second algorithm.
The fixture README explains the deliberately retained publication discrepancies.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from Bio.Seq import Seq

from bionemo.evo2_phage_gen.protein_evidence import summarize_best_hit_aai


FIXTURES = Path(__file__).with_name("fixtures")
AAI = "average_protein_percent_identity"
COUNT = "average_protein_identity_gene_count"
EVALUE = "protein_database_mmseqs_e_value"
PIDENT = "protein_database_mmseqs_percent_identity"


@pytest.fixture
def publication():
    return json.loads((FIXTURES / "arc_aai_publication.json").read_text())


def hit_frame(rows):
    return pd.DataFrame(rows).rename(columns={"query": "id_prompt", "evalue": EVALUE, "pident": PIDENT})


def test_workbook_sequence_binding(publication):
    """Every retained search peptide actually translates from the XLSX sequence."""
    sequences = {record["id"]: record["sequence"] for record in publication["records"]}
    for record in publication["records"]:
        assert hashlib.sha256(record["sequence"].encode()).hexdigest() == record["sequence_sha256"]
    orf_ids = set()
    for orf in publication["orfs"]:
        assert orf["strand"] == "+"
        sequence = sequences[orf["id"].rsplit("_", 1)[0]]
        assert 0 <= orf["start"] < orf["end"] <= 2 * len(sequence)
        dna = (sequence * 2)[orf["start"] : orf["end"]]
        assert dna[:3] == "ATG" and dna[-3:] in {"TAA", "TAG", "TGA"}
        assert str(Seq(dna).translate(to_stop=True)) == orf["protein"]
        assert orf["id"] not in orf_ids
        orf_ids.add(orf["id"])
    assert {hit["query"] for hit in publication["hits"]} == orf_ids


def test_aai_upstream_and_paper(publication, load_script, tmp_path):
    """Upstream hit selection + GFF arithmetic vs the current vectorized helper."""
    upstream = load_script(FIXTURES / "arc_pipeline_upstream.py", "arc_aai_oracle")
    proteins = tmp_path / "proteins.faa"
    proteins.write_text("".join(f">{row['id']}\n{row['protein']}\n" for row in publication["orfs"]))
    hits = [(row["query"], row["target"], row["evalue"], row["pident"]) for row in publication["hits"]]
    best = upstream.mmseqs_results_to_df(hits, proteins, tmp_path / "best.csv", "protein_database")
    gff = tmp_path / "gff"
    gff.mkdir()
    for record in publication["records"]:
        selected = best.loc[best.id_prompt.str.rsplit("_", n=1).str[0] == record["id"]]
        # GFF attributes come from upstream-selected rows, never the new helper.
        features = "".join(
            f"genome\toracle\tCDS\t1\t3\t.\t+\t0\tpercent_identity={identity}\n" for identity in selected[PIDENT]
        )
        (gff / f"{record['id']}.gff").write_text(f"##description {record['id']}\n{features}")
    pd.DataFrame({"id_prompt": [record["id"] for record in publication["records"]]}).to_csv(
        tmp_path / "input.csv", index=False
    )
    upstream.calculate_average_protein_percent_identity(gff, tmp_path / "input.csv", tmp_path / "upstream.csv")
    oracle = pd.read_csv(tmp_path / "upstream.csv").set_index("id_prompt")
    measured = summarize_best_hit_aai(hit_frame(publication["hits"])).set_index("id_prompt")
    assert set(measured.index) == set(oracle.index)
    for record in publication["records"]:
        row = measured.loc[record["id"]]
        tolerance = publication["absolute_tolerance_pp"]
        assert row[AAI] == pytest.approx(oracle.loc[record["id"], AAI], abs=tolerance, rel=0)
        assert row[AAI] == pytest.approx(record["archived_probe_aai"], abs=tolerance, rel=0)
        assert row[COUNT] == record["archived_probe_hit_count"]
        if record["publication_parity_expected"]:
            assert row[AAI] == pytest.approx(record["published_aai"], abs=tolerance, rel=0)
            assert row[COUNT] == record["published_hit_count"]
        else:
            # Known search/inventory discrepancies are evidence, not tolerance.
            assert abs(row[AAI] - record["published_aai"]) > tolerance
            assert (row[AAI] <= 95) != (record["published_aai"] <= 95)


def test_aai_selection_not_max_identity(publication):
    """The retained real hits distinguish the old drift from correct selection."""
    hits = hit_frame(publication["hits"])
    wrong = hits.sort_values(PIDENT, ascending=False).drop_duplicates("id_prompt")
    wrong["genome"] = wrong.id_prompt.str.rsplit("_", n=1).str[0]
    wrong_means = wrong.groupby("genome")[PIDENT].mean()
    measured = summarize_best_hit_aai(hits).set_index("id_prompt")
    assert (wrong_means - measured[AAI]).abs().max() > 0.01


@pytest.mark.parametrize("evalue", [0.0, 1e-20])
def test_aai_first_tie_and_orf_weight(evalue, load_script, tmp_path):
    upstream = load_script(FIXTURES / "arc_pipeline_upstream.py", "arc_aai_ties")
    # Two ORFs from the same family still contribute two observations. Unequal
    # lengths, coverages and target multiplicities must not change their weight.
    rows = [
        {"query": "g_ORF.1", "target": "same_family_a", "evalue": evalue, "pident": 70.0, "qcov": 0.1},
        {"query": "g_ORF.1", "target": "same_family_b", "evalue": evalue, "pident": 100.0, "qcov": 1.0},
        {"query": "g_ORF.2", "target": "same_family_a", "evalue": 1e-30, "pident": 90.0, "qcov": 0.9},
    ]
    proteins = tmp_path / "proteins.faa"
    proteins.write_text(">g_ORF.1\nMKK\n>g_ORF.2\nMKKKKKKKKKKKKKK\n>no_hit_ORF.1\nMK\n")
    for order, expected in [(rows, 80.0), ([rows[1], rows[0], rows[2]], 95.0)]:
        best = upstream.mmseqs_results_to_df(
            [(row["query"], row["target"], row["evalue"], row["pident"]) for row in order],
            proteins,
            tmp_path / "best.csv",
            "protein_database",
        )
        result = summarize_best_hit_aai(hit_frame(order)).set_index("id_prompt")
        assert result.index.tolist() == ["g"]  # No-hit is unavailable, not a novel AAI=0 genome.
        assert result.loc["g", COUNT] == 2
        assert result.loc["g", AAI] == expected == best[PIDENT].mean()


@pytest.mark.parametrize(
    "column,value",
    [
        (EVALUE, -1),
        (EVALUE, float("inf")),
        (EVALUE, float("nan")),
        (PIDENT, -1),
        (PIDENT, 101),
        (PIDENT, float("nan")),
    ],
)
def test_aai_invalid_hits_fail(column, value):
    hits = pd.DataFrame({"id_prompt": ["g_ORF.1"], EVALUE: [1e-20], PIDENT: [80.0]})
    hits.loc[0, column] = value
    with pytest.raises(ValueError):
        summarize_best_hit_aai(hits)
