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

"""Tests for ``bionemo.evo2_phage_gen.arc_pipeline``."""

import ast
import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from Bio import SeqIO
from Bio.Seq import Seq

import bionemo.evo2_phage_gen.arc_pipeline as arc_pipeline
from bionemo.evo2_phage_gen.arc_pipeline import (
    ARC_EVO2_GIT_URL,
    ARC_EVO2_REV,
    ARC_PIPELINE_FILES,
    DEFAULT_ARC_PIPELINE_SOURCE_DIR,
    DEFAULT_PHIX174_FASTA,
    _apply_online_measurement_patches,
    _assert_arc_source_revision,
    prepare_arc_pipeline_workdir,
)
from bionemo.evo2_phage_gen.external_qc import ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA
from bionemo.evo2_phage_gen.protein_evidence import core_gene_ordered_conservation_pass_mask


def _load_prepared_arc_pipeline(tmp_path: Path, module_name: str, monkeypatch):
    """Prepare Arc into tmp_path and import the patched pipeline module from there."""
    if not DEFAULT_ARC_PIPELINE_SOURCE_DIR.exists() or not DEFAULT_PHIX174_FASTA.exists():
        pytest.skip("Arc source assets are not available")
    workdir = tmp_path / "patched_arc"
    prepare_arc_pipeline_workdir(
        DEFAULT_ARC_PIPELINE_SOURCE_DIR,
        workdir,
        phix174_fasta=DEFAULT_PHIX174_FASTA,
    )
    pipeline_path = workdir / "genome_design_filtering_pipeline.py"
    monkeypatch.syspath_prepend(str(pipeline_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_synthetic_mmseqs_pipeline(tmp_path: Path, module_name: str):
    """Load the Arc protein-search fragment after applying the runtime evidence patch."""
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import os
import subprocess
import time

import pandas as pd
from Bio import SeqIO


def mmseqs_search_proteins(query_fasta: str, mmseqs_db: str, results_dir: str, threads: int=8, split: int=0, sensitivity: float=4.0) -> None:
    os.makedirs(results_dir, exist_ok=True)
    mmseqs_out = os.path.join(results_dir, "mmseqs_result.m8")
    log_file = os.path.join(results_dir, "mmseqs_search.log")
    cmd = f"mmseqs easy-search {query_fasta} {mmseqs_db} {mmseqs_out} {results_dir} --threads {threads} --split {split} -s {sensitivity} --remove-tmp-files 1 --format-output 'query,target,evalue,pident'"
    start_time = time.time()
    with open(log_file, "w") as log:
        subprocess.run(cmd, shell=True, check=True, stdout=log, stderr=log, text=True)
    end_time = time.time()
    print(f"MMseqs2 search completed in {end_time - start_time:.2f} seconds.")
    if not os.path.isfile(mmseqs_out):
        raise FileNotFoundError(f"Output file not found: {mmseqs_out}")
    return mmseqs_out


def parse_mmseqs_results(mmseqs_out):
    hits = []
    with open(mmseqs_out, "r") as f:
        for line in f:
            query, target, evalue, pident = line.strip().split('\\t')
            hits.append((query, target, float(evalue), float(pident)))
    return hits


def mmseqs_results_to_df(hits, query_fasta: str, output_csv: str, descriptive_prefix: str, only_top_hits: bool=True) -> pd.DataFrame:
    sequences = {record.id: record.seq for record in SeqIO.parse(query_fasta, "fasta")}
    data = []
    for query, target, evalue, pident in hits:
        if query in sequences:
            data.append([query, sequences[query], target, evalue, pident])
    df = pd.DataFrame(data, columns=["id_prompt", "sequence", f"{descriptive_prefix}_mmseqs_target", f"{descriptive_prefix}_mmseqs_e_value", f"{descriptive_prefix}_mmseqs_percent_identity"])
    if only_top_hits==True and not df.empty:
        df = df.loc[df.groupby("id_prompt")[f"{descriptive_prefix}_mmseqs_e_value"].idxmin()]
    df.to_csv(output_csv, index=False)
    return df


def run_mmseqs_search_proteins(query_fasta: str, mmseqs_db: str, results_dir: str, output_csv: str, descriptive_prefix: str, threads: int=8, split: int=0, sensitivity: float=4.0, only_top_hits: bool=True) -> pd.DataFrame:
    mmseqs_out = mmseqs_search_proteins(query_fasta, mmseqs_db, results_dir, threads, split, sensitivity)
    hits = parse_mmseqs_results(mmseqs_out)
    if not hits:
        df = pd.DataFrame(
            columns=[
                "id_prompt",
                "sequence",
                f"{descriptive_prefix}_mmseqs_target",
                f"{descriptive_prefix}_mmseqs_e_value",
                f"{descriptive_prefix}_mmseqs_percent_identity",
            ]
        )
        df.to_csv(output_csv, index=False)
        return df
    return mmseqs_results_to_df(hits, query_fasta, output_csv, descriptive_prefix, only_top_hits)
"""
    )
    arc_pipeline._apply_mmseqs_protein_evidence_patch(tmp_path)
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("online", "filter7", "empty_at"),
    [
        (False, False, None),
        (False, True, None),
        (True, False, None),
        (False, True, "required"),
        (False, True, "core_gene_ordered_conservation"),
        (False, True, "safety"),
    ],
)
def test_final_gate_order(tmp_path, online, filter7, empty_at):
    """Exercise emitted stage dataflow: biology and safety before novelty, with RL unfiltered."""
    if not DEFAULT_ARC_PIPELINE_SOURCE_DIR.exists() or not DEFAULT_PHIX174_FASTA.exists():
        pytest.skip("Arc source assets are not available")
    workdir = tmp_path / "patched"
    prepare_arc_pipeline_workdir(DEFAULT_ARC_PIPELINE_SOURCE_DIR, workdir, phix174_fasta=DEFAULT_PHIX174_FASTA)
    tree = ast.parse((workdir / "genome_design_filtering_pipeline.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    first = next(
        i
        for i, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "run_genetic_architecture_visualization_and_synteny_filtering"
            for t in node.targets
        )
    )
    config = yaml.safe_load((arc_pipeline.RECIPE_ROOT / "configs/arc_genome_design_filtering_local.yaml").read_text())
    config.update(
        results_save_dir=str(tmp_path),
        homology_filtering=True,
        diversification_filtering=not online,
        genetic_architecture_visualization_and_synteny_filtering=True,
        genetic_architecture_remove_filter=filter7,
    )
    sequences = ["AAAA", "AAAC", "AAAG", "AAAT", "AACA", "AACC", "AACG"]
    ids = ["keep", "high_aai", "missing_gene", "wrong_order", "unsafe", "unknown_safety", "filter7"]
    rows = pd.DataFrame({"id_prompt": ids, "sequence": sequences})
    original = tmp_path / "original.fasta"
    original.write_text("".join(f">original-{i}\n{seq}\n" for i, seq in enumerate(sequences)))
    manifest = tmp_path / "safety.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": f"original-{i}",
                        "state": "FAIL" if empty_at == "safety" or i == 4 else "INDETERMINATE" if i == 5 else "PASS",
                    }
                    for i in range(len(ids))
                ]
            }
        )
    )
    config.update(
        evo_gen_seqs_fasta_file_save_location=str(original),
        sequence_safety_manifest=str(manifest),
        sequence_safety_input_fasta=str(original),
    )
    rows.to_csv(tmp_path / config["homology_filter_seqs_csv_file_save_location"], index=False)
    counts = pd.DataFrame({"count_tropism_protein_sequence_identity_filter": [len(rows)]})
    counts.to_csv(tmp_path / config["homology_filter_counts_file_save_location"], index=False)
    hits = tmp_path / config["mmseqs_protein_database_results_dir_save_location"]
    hits.mkdir()
    (hits / "mmseqs2_all_hits.csv").write_text("id_prompt\n")
    calls = []

    def no_op(*args, **kwargs):
        pass

    def add_mapping(**kwargs):
        df = pd.read_csv(kwargs["input_csv"])
        df["genome_id"] = [f"genome-{i}" for i in range(len(df))]
        df.to_csv(kwargs["output_csv"], index=False)

    def required(**kwargs):
        df = kwargs["sequences_df"]
        assert "genome_id" in df  # Must not depend on AAI running first to load the ID map.
        calls.append(("required", df.id_prompt.tolist()))
        if not kwargs["filter_results"]:
            return df
        return df.iloc[:0] if empty_at == "required" else df[df.id_prompt != "missing_gene"]

    def synteny(**kwargs):
        df = pd.read_csv(kwargs["input_csv"])
        calls.append(("core_gene_ordered_conservation", df.id_prompt.tolist()))
        if kwargs["filter_results"]:
            df = df.iloc[:0] if empty_at == "core_gene_ordered_conservation" else df[df.id_prompt != "wrong_order"]
        df.to_csv(kwargs["output_csv"], index=False)

    def remove_architecture(df, *args):
        assert args[-2] == "remove"
        calls.append(("filter7", df.id_prompt.tolist()))
        return df[df.id_prompt != "filter7"]

    def aai(*args, **kwargs):
        df = pd.read_csv(args[2])
        calls.append(("aai", df.id_prompt.tolist()))
        assert kwargs["query_fasta"] == str(tmp_path / config["orfipy_proteins_file_save_location"])
        if kwargs["filter_results"]:
            df = df[df.id_prompt != "high_aai"]
        df.to_csv(args[3], index=False)

    def save_fasta(df, path):
        Path(path).write_text("".join(f">{row.id_prompt}\n{row.sequence}\n" for row in df.itertuples()))

    namespace = dict(
        config=config,
        pd=pd,
        os=os,
        shutil=shutil,
        online_measurement_mode=online,
        filtered_df=rows.copy(),
        seq_df=rows.copy(),
        seq_fasta=str(original),
        filter_counts=counts,
        annotate_protein_hits=no_op,
        batch_create_gff_files=no_op,
        batch_convert_gff_to_gbk=no_op,
        add_genome_id_mapping=add_mapping,
        valid_gene_annotations=required,
        run_lovis4u_in_conda_env=no_op,
        count_syntenic_genes_all=no_op,
        count_total_num_genes=no_op,
        valid_syntenic_gene_count=synteny,
        calculate_average_protein_percent_identity=no_op,
        valid_average_protein_percent_identity=aai,
        save_df_as_fasta=save_fasta,
        valid_genetic_architecture_score=remove_architecture,
        ga=SimpleNamespace(
            phix174_truth_matrix_blurred_sigma5=None,
            phix174_weight_vector=None,
            phix174_normalization_vector_blurred_sigma5=None,
        ),
    )
    exec(compile(ast.Module(body=main.body[first:], type_ignores=[]), str(workdir), "exec"), namespace)
    observed = dict(calls)
    assert observed["required"] == ids
    if online:
        assert observed["core_gene_ordered_conservation"] == ids
        assert observed["aai"] == ids
        expected = ids
    elif empty_at:
        assert "aai" not in observed
        assert "filter7" not in observed
        expected = []
    else:
        assert observed["core_gene_ordered_conservation"] == [i for i in ids if i != "missing_gene"]
        qualified = ["keep", "high_aai", "filter7"]
        if filter7:
            assert observed["filter7"] == qualified
            qualified.remove("filter7")
        assert observed["aai"] == qualified
        expected = [i for i in qualified if i != "high_aai"]
    terminal = pd.read_csv(tmp_path / config["synteny_filter_seqs_csv_file_save_location"])
    assert terminal.id_prompt.tolist() == expected
    fasta = list(SeqIO.parse(tmp_path / config["synteny_filter_seqs_fasta_file_save_location"], "fasta"))
    assert [record.id for record in fasta] == expected
    final_counts = pd.read_csv(tmp_path / config["synteny_filter_counts_file_save_location"])
    assert final_counts.iloc[0, -1] == len(expected)
    if not online and not empty_at:
        assert final_counts.loc[0, "count_sequence_safety_filter"] == 3
        assert final_counts.columns[-1] == "count_average_protein_sequence_identity_filter"


def test_online_measurement_patch_rejects_missing_gbk_conversion_anchor(tmp_path):
    """A drifted GBK conversion anchor must not silently skip its online patch."""
    patched_fragments = [
        value
        for name, value in vars(arc_pipeline).items()
        if name.startswith(("PATCHED_ONLINE_", "PATCHED_REQUIRED_GENE_", "PATCHED_AAI_", "PATCHED_SYNTENY_"))
        and name != "PATCHED_ONLINE_GBK_CONVERSION"
    ]
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text("\n".join(patched_fragments))

    with pytest.raises(ValueError, match="online objective-measurement patches"):
        _apply_online_measurement_patches(tmp_path)


@pytest.mark.parametrize(
    ("header", "expected_start", "expected_end"),
    [
        ("candidate_ORF.1 [567-843](+)", "568", "843"),
        ("candidate_ORF.2 [9-90](-)", "10", "90"),
    ],
)
def test_prepared_arc_writes_orfipy_intervals_as_one_based_inclusive_gff(
    tmp_path,
    header,
    expected_start,
    expected_end,
):
    """ORFipy's zero-based half-open interval must not shift GFF translations by one base."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pipeline_source = """import re

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

def render_orfipy_interval_as_gff(header):
    match = re.search(r"\\[(\\d+)-(\\d+)\\]", header)
    if match:
        start, end = match.groups()
        orf_entry = {
            "seq_id": "candidate",
            "feature_type": "CDS",
            "start": start,
            "end": end,
        }
        return (
            f"{orf_entry['seq_id']}\\tcaller\\t{orf_entry['feature_type']}\\t"
            f"{orf_entry['start']}\\t{orf_entry['end']}\\t.\\t+\\t0\\tID=ORF.1"
        )
    raise ValueError("missing ORFipy interval")


def convert_gff_to_gbk(sequence, start, end, output_path, strand=1):
    record = SeqRecord(Seq(sequence), id="candidate", name="candidate")
    record.annotations["molecule_type"] = "DNA"
    feature = SeqFeature(
        location=FeatureLocation(start, end, strand=strand),
        type="CDS",
    )
    feature.qualifiers["translation"] = [str(feature.extract(record.seq).translate())]
    record.features = [feature]
    SeqIO.write(record, output_path, "genbank")
"""
    for filename in ARC_PIPELINE_FILES:
        content = pipeline_source if filename == "genome_design_filtering_pipeline.py" else "print('ok')\n"
        if filename == "genetic_architecture.py":
            content = f'fasta_file = "{ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA}"\n'
        (source_dir / filename).write_text(content)
    reference_fasta = tmp_path / "reference.fna"
    reference_fasta.write_text(">reference\nACGT\n")
    workdir = tmp_path / "prepared"

    workdir = source_dir
    arc_pipeline._apply_orfipy_gff_coordinate_patch(workdir)
    pipeline_path = workdir / "genome_design_filtering_pipeline.py"
    spec = importlib.util.spec_from_file_location("coordinate_corrected_arc_pipeline", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fields = module.render_orfipy_interval_as_gff(header).split("\t")
    assert fields[3:5] == [expected_start, expected_end]
    start = int(expected_start) - 1
    coding_length = int(expected_end) - start
    if header.startswith("candidate_ORF.1"):
        coding_sequence = Seq(
            "ATGGTACGCTGGACTTTGTGGGATACCCTCGCTTTCCTGCTCCTGTTGAGTTTATTGCTGCCGTCATTGCTTATTATGTT"
            "CATCCCGTCAACATTCAAACGGCCTGTCTCATCATGGAAGGCGCTGAATTTACGGAAAACATTATTAATGGCGTCGAGCGT"
            "CCGGTTAAAGCCGCTGAATTGTTCGCGTTTACCTTGCGTGTACGCGCAGGAAACACTGACGTTCTTACTGACGCAGAAGAA"
            "AACGTGCGTCAAAAATTACGTGCGGAAGGAGTGA"
        )
    else:
        coding_sequence = Seq("ATG" + "GCT" * (coding_length // 3 - 2) + "TAA")
    genome = Seq("A" * start) + coding_sequence
    gbk_path = tmp_path / f"{expected_start}.gbk"
    module.convert_gff_to_gbk(str(genome), int(fields[3]), int(fields[4]), gbk_path)

    converted = SeqIO.read(gbk_path, "genbank")
    for feature in (feature for feature in converted.features if feature.type == "CDS"):
        assert (int(feature.location.start), int(feature.location.end)) == (start, int(expected_end))
        assert len(feature) % 3 == 0
        assert "*" not in feature.qualifiers["translation"][0][:-1]


def test_prepare_arc_pipeline_requires_compatible_arc_revision(tmp_path, monkeypatch):
    """The maintained Arc patch should only apply to its compatible Arc source revision."""
    source_dir = tmp_path / "source" / "phage_gen" / "pipelines"
    source_dir.mkdir(parents=True)
    for filename in ARC_PIPELINE_FILES:
        content = "print('ok')\n"
        if filename == "genetic_architecture.py":
            content = f'fasta_file = "{ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA}"\n'
        (source_dir / filename).write_text(content)
    phix174_fasta = tmp_path / "NC_001422_1.fna"
    phix174_fasta.write_text(">NC_001422.1\nACGT\n")
    patch_file = tmp_path / "arc.patch"
    patch_file.write_text("")

    monkeypatch.setattr("bionemo.evo2_phage_gen.arc_pipeline._git_head", lambda path: "wrong-revision")

    with pytest.raises(RuntimeError, match=f"{ARC_EVO2_GIT_URL}@{ARC_EVO2_REV}"):
        prepare_arc_pipeline_workdir(
            source_dir,
            tmp_path / "patched",
            phix174_fasta=phix174_fasta,
            pipeline_patch=patch_file,
        )


@pytest.mark.parametrize("with_hits", [False, True])
def test_prepare_arc_pipeline_workdir_applies_maintained_patch(tmp_path, with_hits):
    """The real Arc source should be patched from the tracked maintained patch."""
    if not DEFAULT_ARC_PIPELINE_SOURCE_DIR.exists() or not DEFAULT_PHIX174_FASTA.exists():
        pytest.skip("Arc source assets are not available")

    _assert_arc_source_revision(DEFAULT_ARC_PIPELINE_SOURCE_DIR, ARC_EVO2_REV)
    workdir = tmp_path / "patched"
    prepare_arc_pipeline_workdir(
        DEFAULT_ARC_PIPELINE_SOURCE_DIR,
        workdir,
        phix174_fasta=DEFAULT_PHIX174_FASTA,
    )

    pipeline_text = (workdir / "genome_design_filtering_pipeline.py").read_text()
    for filename in ARC_PIPELINE_FILES:
        compile((workdir / filename).read_text(), filename, "exec")

    # Exercise the emitted search stage: a successful no-hit must still reach
    # downstream required-function measurement instead of a generic count gate.
    main = next(
        node for node in ast.parse(pipeline_text).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    search = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "run_protein_database_search"
    )
    config = {
        "results_save_dir": str(tmp_path),
        "orfipy_proteins_file_save_location": "proteins.fasta",
        "mmseqs_db_protein_database": "phrogs",
        "mmseqs_protein_database_results_dir_save_location": "phrogs_hits",
        "mmseqs_threads": 1,
        "mmseqs_protein_database_sensitivity": 7.5,
        "homology_filter_counts_file_save_location": "counts.csv",
        "homology_filter_seqs_csv_file_save_location": "sequences.csv",
        "gff_dir_save_location": "gff",
        "gbk_dir_save_location": "gbk",
        "required_gene_families": {"A": ["phrog:1"], "B": ["phrog:2"]},
    }
    sequences = pd.DataFrame({"id_prompt": ["sample"], "genome_id": ["genome_1"], "sequence": ["ACGT"]})
    (tmp_path / "proteins.fasta").write_text(">sample_ORF.1\nMMMM\n>sample_ORF.2\nMMMM\n")
    (tmp_path / "phrogs_hits").mkdir()
    converter = _load_synthetic_mmseqs_pipeline(tmp_path, "search_stage_converter")
    hits = (
        [
            ("sample_ORF.1", "phrog_942", 1e-90, 90, 100, 100, 100, 1, 1),
            ("sample_ORF.1", "phrog_1", 1e-10, 50, 100, 100, 100, 1, 1),
            ("sample_ORF.1", "phrog_2", 1e-8, 40, 100, 100, 100, 1, 1),
            ("sample_ORF.2", "phrog_1", 1e-6, 35, 40, 100, 100, 0.375, 0.375),
        ]
        if with_hits
        else []
    )
    calls = []

    def search_hits(**kwargs):
        calls.append(kwargs)
        return converter.mmseqs_results_to_df(
            hits, kwargs["query_fasta"], kwargs["output_csv"], kwargs["descriptive_prefix"], kwargs["only_top_hits"]
        )

    namespace = {
        "config": config,
        "run_protein_database_search": True,
        "seq_df": sequences.copy(),
        "filtered_df": sequences.copy(),
        "filter_counts": pd.DataFrame({"count_initial_before_homology_metrics": [1]}),
        "run_mmseqs_search_proteins": search_hits,
        "pd": pd,
        "os": os,
        "online_measurement_mode": True,
    }
    exec(compile(ast.Module(body=[search], type_ignores=[]), str(workdir), "exec"), namespace)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "sequences.csv"), sequences)
    all_hits = pd.read_csv(tmp_path / "phrogs_hits/mmseqs2_all_hits.csv")
    best_hits = pd.read_csv(tmp_path / "phrogs_hits/mmseqs2_hits.csv")
    assert len(all_hits) == len(hits)
    assert best_hits["protein_database_mmseqs_target"].tolist() == (["phrog_942", "phrog_1"] if with_hits else [])

    # Exercise the emitted call, including which artifact it supplies. An
    # ambiguous ORF fills B, leaving the second ORF's partial A evidence useful.
    tree = ast.parse(pipeline_text)
    definition = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "valid_gene_annotations"
    )
    required_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "valid_gene_annotations"
    )
    namespace["mmseqs_results_df"] = best_hits
    required_module = ast.fix_missing_locations(
        ast.Module(body=[definition, ast.Expr(required_call)], type_ignores=[])
    )
    exec(compile(required_module, str(workdir), "exec"), namespace)
    measured = pd.read_csv(tmp_path / "qc6_required_genes_metrics.csv")
    assert measured["required_genes_integrity_sum"].tolist() == ([1.5] if with_hits else [0.0])
    assert measured["required_genes_full_length_count"].tolist() == ([1] if with_hits else [0])


@pytest.mark.parametrize(
    ("with_results", "allowed", "expected_ids"),
    [
        (True, ["Complete", "Low-quality"], ["umi1", "low"]),
        (True, ["Not-determined"], ["umi10 description"]),
        (False, ["Complete", "Low-quality"], []),
    ],
)
def test_checkv_quality_filter(tmp_path, monkeypatch, with_results, allowed, expected_ids):
    """Only allowed, exactly identified records survive; missing evidence never passes."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_checkv", monkeypatch)
    sequences = pd.DataFrame(
        {
            "id_prompt": ["umi10 description", "umi1", "unclassified", "low", "missing"],
            "sequence": ["ACGT", "TGCA", "AAAA", "CCCC", "GGGG"],
            "genome_id": ["g1", "g2", "g3", "g4", "g5"],
        }
    )
    before = sequences.copy(deep=True)
    quality = pd.DataFrame(
        {
            "contig_id": ["umi1", "umi10", "low", "unclassified", "unrelated"],
            "checkv_quality": ["Complete", "Not-determined", "Low-quality", None, "Complete"],
        }
    )
    if not with_results:
        quality = quality.iloc[:0]
    quality_path = tmp_path / "quality_summary.tsv"
    quality.to_csv(quality_path, sep="\t", index=False)

    result = module.valid_checkv_quality(str(quality_path), allowed, sequences)

    assert result["id_prompt"].tolist() == expected_ids
    assert result["checkv_quality"].isin(allowed).all()
    pd.testing.assert_frame_equal(
        result[sequences.columns].reset_index(drop=True),
        sequences.loc[sequences["id_prompt"].isin(expected_ids)].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(sequences, before)


def test_patched_arc_required_gene_measurement_does_not_filter_or_delete(tmp_path, monkeypatch):
    """Online rewards should measure required genes without starving later objectives."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_measurement_test", monkeypatch)
    gff_dir = tmp_path / "gff"
    gbk_dir = tmp_path / "gbk"
    gff_dir.mkdir()
    gbk_dir.mkdir()
    (gff_dir / "genome_1.gff").write_text("contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=major capsid protein\n")
    (gbk_dir / "genome_1.gbk").write_text("LOCUS genome_1\n")
    sequences = pd.DataFrame({"id_prompt": ["umi1"], "genome_id": ["genome_1"], "sequence": ["ACGT"]})
    metrics_csv = tmp_path / "required.csv"

    measured = module.valid_gene_annotations(
        input_gff_dir=str(gff_dir),
        input_gbk_dir=str(gbk_dir),
        required_families={"A": ["phrog:1"], "B": ["phrog:2"]},
        sequences_df=sequences,
        metrics_csv=str(metrics_csv),
        filter_results=False,
    )

    assert measured["id_prompt"].tolist() == ["umi1"]
    assert (gff_dir / "genome_1.gff").exists()
    assert (gbk_dir / "genome_1.gbk").exists()
    metrics = pd.read_csv(metrics_csv)
    assert metrics["required_genes_matched_count"].tolist() == [0]


def test_patched_arc_required_gene_evidence_is_fractional_duplicate_safe_and_shared_with_hard_qc(tmp_path):
    """Online required-family credit and offline rejection must use the same reciprocal-coverage evidence."""
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import os
import shutil

import pandas as pd


def valid_gene_annotations(input_gff_dir, input_gbk_dir, required_products, sequences_df, metrics_csv=None, filter_results=True):
    return sequences_df


##############################
### RUN FILTERING PIPELINE ###
##############################
if __name__ == "__main__":
    print(f"Required genes: {config['required_genes_list']}")
    valid_gene_annotations(input_gff_dir="gff", input_gbk_dir="gbk",
                           required_products=config["required_genes_list"], sequences_df=sequences)
"""
    )
    arc_pipeline._apply_required_gene_evidence_patch(tmp_path)
    assert "required_genes_list" not in pipeline_path.read_text()
    assert 'required_families=config["required_gene_families"]' in pipeline_path.read_text()
    spec = importlib.util.spec_from_file_location("patched_required_gene_evidence", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gff_dir = tmp_path / "gff"
    gbk_dir = tmp_path / "gbk"
    gff_dir.mkdir()
    gbk_dir.mkdir()
    (gff_dir / "genome_1.gff").write_text("##gff-version 3\n")
    (gbk_dir / "genome_1.gbk").write_text("LOCUS genome_1\n")
    sequences = pd.DataFrame({"id_prompt": ["umi1"], "genome_id": ["genome_1"], "sequence": ["ACGT"]})
    hits = pd.DataFrame(
        {
            "id_prompt": ["umi1_ORF.1", "umi1_ORF.2", "umi1_ORF.3"],
            "annot": ["gene A", "gene A", "gene B"],
            "protein_database_mmseqs_target": ["phrog_1", "phrog_1", "phrog_2"],
            "protein_database_mmseqs_percent_identity": [100.0, 100.0, 100.0],
            "protein_database_mmseqs_alignment_length": [100, 100, 50],
            "protein_database_mmseqs_query_length": [100, 100, 50],
            "protein_database_mmseqs_target_length": [100, 100, 100],
            "protein_database_mmseqs_query_coverage": [1, 1, 1],
            "protein_database_mmseqs_target_coverage": [1, 1, 0.5],
        }
    )
    metrics_csv = tmp_path / "required.csv"

    measured = module.valid_gene_annotations(
        input_gff_dir=str(gff_dir),
        input_gbk_dir=str(gbk_dir),
        required_families={"A": ["phrog:1"], "B": ["phrog:2"]},
        sequences_df=sequences,
        metrics_csv=str(metrics_csv),
        filter_results=False,
        protein_database_hits_df=hits,
        minimum_reciprocal_coverage=0.95,
    )

    assert measured["id_prompt"].tolist() == ["umi1"]
    metrics = pd.read_csv(metrics_csv)
    assert metrics["required_genes_matched_count"].tolist() == [2]
    assert metrics["required_genes_integrity_sum"].tolist() == pytest.approx([1.0 + 0.5 / 0.95])
    assert metrics["required_genes_full_length_count"].tolist() == [1]
    assert (gff_dir / "genome_1.gff").exists()
    assert (gbk_dir / "genome_1.gbk").exists()

    # An explicit family profile admits this demonstrated shortened function;
    # without it the unchanged generic coverage gate below still rejects it.
    accepted = module.valid_gene_annotations(
        str(gff_dir),
        str(gbk_dir),
        {"A": ["phrog:1"], "B": ["phrog:2"]},
        sequences,
        metrics_csv=str(metrics_csv),
        protein_database_hits_df=hits,
        minimum_reciprocal_coverage=0.95,
        family_coverage_thresholds={"phrog:2": (0.95, 0.50)},
    )
    assert accepted["id_prompt"].tolist() == ["umi1"]
    assert pd.read_csv(metrics_csv)["required_genes_full_length_count"].tolist() == [2]

    filtered = module.valid_gene_annotations(
        input_gff_dir=str(gff_dir),
        input_gbk_dir=str(gbk_dir),
        required_families={"A": ["phrog:1"], "B": ["phrog:2"]},
        sequences_df=sequences,
        metrics_csv=str(metrics_csv),
        filter_results=True,
        protein_database_hits_df=hits,
        minimum_reciprocal_coverage=0.95,
    )

    assert filtered.empty
    assert not (gff_dir / "genome_1.gff").exists()
    assert not (gbk_dir / "genome_1.gbk").exists()

    # An empty required-function list supplies no evidence, not a vacuous pass.
    assert module.valid_gene_annotations(
        str(gff_dir),
        str(gbk_dir),
        {},
        sequences,
        protein_database_hits_df=hits,
    ).empty


def test_required_gene_patch_preserves_composed_hard_gate_imports(tmp_path):
    """The production patch chain must leave every injected protein gate bound."""
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import os
import shutil

import pandas as pd


def valid_gene_annotations(input_gff_dir, input_gbk_dir, required_products, sequences_df, metrics_csv=None):
    return sequences_df


##############################
### RUN FILTERING PIPELINE ###
##############################
"""
    )
    arc_pipeline._apply_protein_hard_gate_patch(tmp_path)
    arc_pipeline._apply_required_gene_evidence_patch(tmp_path)

    spec = importlib.util.spec_from_file_location("composed_protein_gate_patches", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.valid_coverage_aware_mmseqs_pident)


def test_patched_arc_aai_target_semantics(tmp_path):
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import os

import pandas as pd


def valid_average_protein_percent_identity(gff_directory, gbk_directory, results_csv, output_csv, identity_range):
    pass


def count_total_num_genes(gff_directory, results_csv):
    pass
"""
    )
    arc_pipeline._apply_aai_evidence_patch(tmp_path)
    spec = importlib.util.spec_from_file_location("patched_aai_evidence", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "id_prompt": ["truncated", "complete"],
            "genome_id": ["genome_1", "genome_2"],
        }
    ).to_csv(input_csv, index=False)
    hits = pd.DataFrame(
        {
            "id_prompt": ["truncated_ORF.1", "complete_ORF.1"],
            "protein_database_mmseqs_target": ["family_A", "family_A"],
            "protein_database_mmseqs_percent_identity": [100.0, 80.0],
            "protein_database_mmseqs_e_value": [1e-20, 1e-25],
            "protein_database_mmseqs_alignment_length": [50, 100],
            "protein_database_mmseqs_query_length": [50, 100],
            "protein_database_mmseqs_target_length": [100, 100],
            "protein_database_mmseqs_query_coverage": [1, 1],
            "protein_database_mmseqs_target_coverage": [0.5, 1],
        }
    )

    def search(**kwargs):
        assert kwargs["mmseqs_db"] == "individual-proteins"
        assert kwargs["query_fasta"] == "called-orfs.faa"
        return hits

    module.run_mmseqs_search_proteins = search
    member_options = dict(identity_database="individual-proteins", query_fasta="called-orfs.faa")

    module.valid_average_protein_percent_identity(
        str(tmp_path / "gff"),
        str(tmp_path / "gbk"),
        str(input_csv),
        str(output_csv),
        (0, 95),
        filter_results=False,
        metrics_csv=str(metrics_csv),
        **member_options,
    )

    metrics = pd.read_csv(metrics_csv)
    assert metrics["average_protein_percent_identity"].tolist() == [100.0, 80.0]
    assert metrics["average_protein_identity_gene_count"].tolist() == [1, 1]


def test_arc_function_synteny_uses_shared_family_evidence(tmp_path):
    """The emitted Arc reader must admit alternate J and reject its extra copy."""
    namespace = {"pd": pd, "os": os}
    exec(arc_pipeline.PATCHED_SYNTENY_METRICS_FUNCTION, namespace)
    reference = tmp_path / "reference.gff"
    reference.write_text(
        "##gff-version 3\n"
        "ref\ttest\tCDS\t1\t9\t.\t+\t0\tID=A\n"
        "ref\ttest\tCDS\t10\t18\t.\t+\t0\tID=J\n"
        "##FASTA\n>ref\nATGAAATAAATGCCCTAA\n"
    )
    rows, orfs = [], []
    for genome, families in [("single", [713, 3780]), ("duplicate", [713, 3780, 3780])]:
        for index, family in enumerate(families):
            identifier = f"{genome}_ORF.{index}"
            orfs.append(f">{identifier} [{index * 9}-{index * 9 + 9}](+)\nATGCCCTAA\n")
            rows.append(
                {
                    "id_prompt": identifier,
                    "protein_database_mmseqs_target": f"phrog_{family}",
                    "protein_database_mmseqs_percent_identity": 50,
                    "protein_database_mmseqs_alignment_length": 80,
                    "protein_database_mmseqs_query_length": 100,
                    "protein_database_mmseqs_target_length": 100,
                    "protein_database_mmseqs_query_coverage": 0.8,
                    "protein_database_mmseqs_target_coverage": 0.8,
                }
            )
    (tmp_path / "orfs.fasta").write_text("".join(orfs))
    (tmp_path / "phrogs").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "phrogs/mmseqs2_all_hits.csv", index=False)
    # The annotation winner is unrelated; it must not hide admitted family hits.
    annotations = pd.DataFrame(rows).assign(protein_database_mmseqs_target="phrog_942")
    annotations.to_csv(tmp_path / "phrogs/mmseqs2_hits.csv", index=False)
    input_csv, output_csv = tmp_path / "input.csv", tmp_path / "output.csv"
    pd.DataFrame({"id_prompt": ["single", "duplicate"], "genome_id": ["genome_1", "genome_2"]}).to_csv(
        input_csv, index=False
    )
    namespace["count_syntenic_genes_all"](
        "unused-lovis",
        "unused-gff",
        input_csv,
        output_csv,
        reference_gff_path=reference,
        config={
            "results_save_dir": str(tmp_path),
            "orfipy_orfs_file_save_location": "orfs.fasta",
            "mmseqs_protein_database_results_dir_save_location": "phrogs",
            "required_gene_families": {"A": ["phrog:713"], "J": ["phrog:2354", "phrog:3780"]},
            "core_gene_reference_functions": {"A": "A", "J": "J"},
        },
    )
    measured = pd.read_csv(output_csv)
    assert measured["num_syntenic_genes"].tolist() == [2, 2]
    assert measured["duplicate_reference_gene_count"].tolist() == [0, 1]
    assert core_gene_ordered_conservation_pass_mask(measured).tolist() == [True, False]


def test_arc_tropism_gate_rejects_fragments(tmp_path):
    """Final Arc tropism gate must reject high-identity fragments."""
    pipeline_path = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_path.write_text(
        """import pandas as pd


##############################
### RUN FILTERING PIPELINE ###
##############################
"""
    )
    arc_pipeline._apply_protein_hard_gate_patch(tmp_path)
    spec = importlib.util.spec_from_file_location("patched_protein_hard_gates", pipeline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sequences = pd.DataFrame({"id_prompt": ["duplicate", "complete"], "sequence": ["ACGT", "ACGT"]})
    tropism_hits = pd.DataFrame(
        {
            "id_prompt": ["duplicate_ORF.1", "complete_ORF.1"],
            "tropism_protein_mmseqs_percent_identity": [100.0, 60.0],
            "tropism_protein_mmseqs_alignment_length": [50, 100],
            "tropism_protein_mmseqs_query_length": [50, 100],
            "tropism_protein_mmseqs_target_length": [100, 100],
            "tropism_protein_mmseqs_query_coverage": [1, 1],
            "tropism_protein_mmseqs_target_coverage": [0.5, 1],
        }
    )

    tropism_pass = module.valid_coverage_aware_mmseqs_pident(
        tropism_hits,
        "tropism_protein",
        [60, 100],
        sequences,
        minimum_reciprocal_coverage=0.95,
    )

    assert tropism_pass["id_prompt"].tolist() == ["complete"]
    assert tropism_pass["tropism_protein_mmseqs_percent_identity"].tolist() == [60.0]


def test_patched_arc_synteny_missing_lovis4u_output_receives_zero_credit(tmp_path, monkeypatch):
    """Missing LoVis4u files should zero synteny metrics instead of aborting RL reward scoring."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_for_test", monkeypatch)

    metadata_dir = tmp_path / "metadata"
    gff_dir = tmp_path / "gff"
    (metadata_dir / "genome_1").mkdir(parents=True)
    gff_dir.mkdir()
    (gff_dir / "genome_1.gff").write_text("contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=major spike protein\n")
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame({"id_prompt": ["umi1"], "genome_id": ["genome_1"], "total_num_genes": [1]}).to_csv(
        input_csv,
        index=False,
    )

    module.count_syntenic_genes_all(
        root_dir=str(metadata_dir),
        gff_dir=str(gff_dir),
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        reference_gff_path=None,
    )

    output = pd.read_csv(output_csv)
    assert output["num_syntenic_genes"].tolist() == [0]
    assert output["non_syntenic_genes"].fillna("").tolist() == [""]
    assert output["missing_synteny_output"].tolist() == [True]


def test_patched_arc_synteny_producer_consumer_contract_tracks_positive_and_missing_outputs(tmp_path, monkeypatch):
    """LoVis4u consumer should score real clustering output and mark missing artifacts per input."""
    module = _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_contract_test", monkeypatch)

    metadata_dir = tmp_path / "metadata"
    gff_dir = tmp_path / "gff"
    positive_mmseqs_dir = metadata_dir / "genome_1" / "mmseqs"
    (metadata_dir / "genome_2").mkdir(parents=True)
    positive_mmseqs_dir.mkdir(parents=True)
    gff_dir.mkdir()

    positive_mmseqs = positive_mmseqs_dir / "mmseqs_clustering.tsv"
    positive_mmseqs.write_text("genome_1-ORF.1\treference-ORF.1\ngenome_1-ORF.2\treference-ORF.2\n")
    assert positive_mmseqs.exists()

    (gff_dir / "genome_1.gff").write_text(
        "contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=major spike protein\n"
        "contig\ttool\tCDS\t100\t180\t.\t+\t0\tID=ORF.2;product=minor capsid protein\n"
    )
    (gff_dir / "genome_2.gff").write_text(
        "contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=negative control protein\n"
    )
    (gff_dir / "genome_3.gff").write_text(
        "contig\ttool\tCDS\t1\t90\t.\t+\t0\tID=ORF.1;product=no lovis output protein\n"
    )
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "id_prompt": ["genome_1", "genome_2", "genome_3"],
            "genome_id": ["genome_1", "genome_2", "genome_3"],
        }
    ).to_csv(input_csv, index=False)

    module.count_syntenic_genes_all(
        root_dir=str(metadata_dir),
        gff_dir=str(gff_dir),
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        reference_gff_path=None,
    )
    module.count_total_num_genes(str(gff_dir), str(output_csv))

    output = pd.read_csv(output_csv)
    assert output["id_prompt"].tolist() == ["genome_1", "genome_2", "genome_3"]
    assert output["num_syntenic_genes"].tolist() == [2, 0, 0]
    assert output["total_num_genes"].tolist() == [2, 1, 1]
    assert output["missing_synteny_output"].tolist() == [False, True, True]


@pytest.mark.parametrize("real_assets", [False, True])
def test_patched_arc_mmseqs_protein_search_rejects_missing_output(tmp_path, monkeypatch, real_assets):
    """Execution failure must not masquerade as a successfully measured no-hit genome."""
    module = (
        _load_prepared_arc_pipeline(tmp_path, "patched_arc_pipeline_mmseqs_test", monkeypatch)
        if real_assets
        else _load_synthetic_mmseqs_pipeline(tmp_path, "synthetic_arc_pipeline_failure_test")
    )

    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(">umi1_ORF.1\nM\n")
    mmseqs_db = tmp_path / "mmseqs_db"
    mmseqs_db.mkdir()
    output_csv = tmp_path / "hits.csv"

    def fail_mmseqs(*_args, **_kwargs):
        raise module.subprocess.CalledProcessError(returncode=1, cmd="mmseqs")

    monkeypatch.setattr(module.subprocess, "run", fail_mmseqs)

    with pytest.raises(module.subprocess.CalledProcessError):
        module.run_mmseqs_search_proteins(
            query_fasta=str(query_fasta),
            mmseqs_db=str(mmseqs_db),
            results_dir=str(tmp_path / "mmseqs_results"),
            output_csv=str(output_csv),
            descriptive_prefix="protein_database",
        )
    assert not output_csv.exists()


@pytest.mark.parametrize("stale_output", [False, True])
def test_empty_protein_query_has_zero_hits(tmp_path, monkeypatch, stale_output):
    """No called ORFs is zero evidence, not a failed search or stale hit reuse."""
    module = _load_synthetic_mmseqs_pipeline(tmp_path, "arc_empty_protein_query")
    query = tmp_path / "proteins.fasta"
    query.write_text("")
    results = tmp_path / "search"
    results.mkdir()
    if stale_output:
        (results / "mmseqs_result.m8").write_text("stale hit that must not be reused\n")

    def reject_empty_search(*_args, **_kwargs):
        raise module.subprocess.CalledProcessError(1, "mmseqs easy-search empty.fasta")

    monkeypatch.setattr(module.subprocess, "run", reject_empty_search)
    output = tmp_path / "hits.csv"
    hits = module.run_mmseqs_search_proteins(str(query), "db", str(results), str(output), "protein_database")

    assert hits.empty
    assert pd.read_csv(output).empty
    assert (results / "mmseqs_result.m8").read_text() == ""
    assert list(hits.columns) == [
        "id_prompt",
        "sequence",
        "protein_database_mmseqs_target",
        "protein_database_mmseqs_e_value",
        "protein_database_mmseqs_percent_identity",
        "protein_database_mmseqs_alignment_length",
        "protein_database_mmseqs_query_length",
        "protein_database_mmseqs_target_length",
        "protein_database_mmseqs_query_coverage",
        "protein_database_mmseqs_target_coverage",
    ]


def test_patched_arc_mmseqs_protein_search_allows_successful_empty_hits(tmp_path, monkeypatch):
    """A successful MMseqs run with no hits should still produce an empty hit table."""
    module = _load_synthetic_mmseqs_pipeline(tmp_path, "patched_arc_pipeline_empty_mmseqs_test")
    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(">umi1_ORF.1\nM\n")
    mmseqs_db = tmp_path / "mmseqs_db"
    mmseqs_db.mkdir()
    output_csv = tmp_path / "hits.csv"
    mmseqs_out = tmp_path / "mmseqs_results" / "mmseqs_out.tsv"
    mmseqs_out.parent.mkdir()
    mmseqs_out.write_text("")

    monkeypatch.setattr(module, "mmseqs_search_proteins", lambda *_args, **_kwargs: str(mmseqs_out))

    hits = module.run_mmseqs_search_proteins(
        query_fasta=str(query_fasta),
        mmseqs_db=str(mmseqs_db),
        results_dir=str(tmp_path / "mmseqs_results"),
        output_csv=str(output_csv),
        descriptive_prefix="protein_database",
    )

    assert hits.empty
    assert list(hits.columns) == [
        "id_prompt",
        "sequence",
        "protein_database_mmseqs_target",
        "protein_database_mmseqs_e_value",
        "protein_database_mmseqs_percent_identity",
        "protein_database_mmseqs_alignment_length",
        "protein_database_mmseqs_query_length",
        "protein_database_mmseqs_target_length",
        "protein_database_mmseqs_query_coverage",
        "protein_database_mmseqs_target_coverage",
    ]
    assert output_csv.exists()


def test_patched_arc_mmseqs_protein_search_carries_alignment_lengths(tmp_path, monkeypatch):
    """Arc must retain the MMseqs lengths needed to distinguish fragments from intact proteins."""
    module = _load_synthetic_mmseqs_pipeline(tmp_path, "patched_arc_pipeline_coverage_test")
    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(">umi1_ORF.1\n" + "M" * 80 + "\n")
    mmseqs_db = tmp_path / "mmseqs_db"
    mmseqs_db.mkdir()
    output_csv = tmp_path / "hits.csv"
    mmseqs_out = tmp_path / "mmseqs_results" / "mmseqs_out.tsv"
    mmseqs_out.parent.mkdir()
    mmseqs_out.write_text("umi1_ORF.1\tphrog_1\t1e-20\t75.0\t40\t80\t100\t0.375\t0.4\n")

    monkeypatch.setattr(module, "mmseqs_search_proteins", lambda *_args, **_kwargs: str(mmseqs_out))

    hits = module.run_mmseqs_search_proteins(
        query_fasta=str(query_fasta),
        mmseqs_db=str(mmseqs_db),
        results_dir=str(tmp_path / "mmseqs_results"),
        output_csv=str(output_csv),
        descriptive_prefix="protein_database",
    )

    assert hits.loc[0, "protein_database_mmseqs_alignment_length"] == 40
    assert hits.loc[0, "protein_database_mmseqs_query_length"] == 80
    assert hits.loc[0, "protein_database_mmseqs_target_length"] == 100
    # Ten query-gap columns in a 40-column alignment do not cover 40 query residues.
    assert hits.loc[0, "protein_database_mmseqs_query_coverage"] == 0.375
    assert hits.loc[0, "protein_database_mmseqs_target_coverage"] == 0.4


def test_patched_arc_mmseqs_search_requests_alignment_length_fields(tmp_path, monkeypatch):
    """The MMseqs command contract must request reciprocal-coverage inputs explicitly."""
    module = _load_synthetic_mmseqs_pipeline(tmp_path, "patched_arc_pipeline_command_test")
    query_fasta = tmp_path / "query.fasta"
    query_fasta.write_text(">umi1_ORF.1\nM\n")
    mmseqs_db = tmp_path / "mmseqs_db"
    mmseqs_db.mkdir()
    results_dir = tmp_path / "mmseqs_results"

    def fake_run(cmd, **_kwargs):
        assert "--format-output 'query,target,evalue,pident,alnlen,qlen,tlen,qcov,tcov'" in cmd
        (results_dir / "mmseqs_result.m8").write_text("")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    observed_path = module.mmseqs_search_proteins(
        query_fasta=str(query_fasta),
        mmseqs_db=str(mmseqs_db),
        results_dir=str(results_dir),
    )

    assert observed_path == str(results_dir / "mmseqs_result.m8")
