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
"""Independent ARC final filters; RL measurements must never substitute for these gates.

The caller supplies physical-AAI/nine-core/safety-qualified observations. This module reapplies
ARC filters 1-6, 8 and 9, with an explicitly versioned AAI definition, then returns
per-observation passers. Original ARC synteny always uses its original annotation inventory. V3 separately admits
a structured K+X add-one-gene category without changing the original count rule. Set-level exact deduplication and 99%-identity clustering belong after QC.
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


CONTRACT = "arc-1-6-8-9-plus-nine-core-safety-v1"
CIRCULAR_AAI_CONTRACT = "arc-1-6-8-9-circular-aai-plus-nine-core-safety-v2"
ADD_ONE_GENE_CONTRACT = "arc-circular-aai-syntenic-diversity-plus-nine-core-safety-v3"
REFERENCE_SHA256 = "16a63389bfce4e32121c7e91b1d2936a5461a4e61eb80a9bef1fc54d517cbb7a"
UPSTREAM_SHA256 = "ab1a5d388dcd2085ad79909ea15f3666740d76486f8e4b43d25fdf6fb5553091"
RECIPE_ROOT = Path(__file__).resolve().parents[3]
GATE_NAMES = (
    "valid_nt_chars",
    "valid_genome_len",
    "valid_gc_content",
    "valid_nt_homopolymer_len",
    "arc_seven_protein_hits",
    "arc_tropism",
    "arc_aai",
    "arc_synteny",
)


def arc_synteny_pass_mask(metrics: pd.DataFrame) -> pd.Series:
    """ARC's count-pair gate, independent of nine-core conservation."""
    counts = metrics[["num_syntenic_genes", "total_num_genes"]].apply(pd.to_numeric, errors="raise")
    if counts.isna().any().any() or not (counts % 1 == 0).all().all():
        raise ValueError("ARC synteny needs measured integer counts")
    return (counts.ge(10) & counts.le(12)).all(axis=1) & ~counts.eq(11).all(axis=1)


def classify_syntenic_diversity(
    synteny: pd.DataFrame,
    accessory: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    core_families: set[str],
    k_families: set[str],
) -> pd.DataFrame:
    """Label original ARC and structured add-one-gene routes independently.

    The caller has already enforced physical AAI, nine-core order/copies, safety,
    and the common ARC gates. Counts alone never activate the added route.
    Accessory evidence must come from the maintained native scorer, not telemetry.
    """
    prefix = "syntenic_diversity_"
    required = [
        "accessory_k_credit",
        "accessory_x_credit",
        "accessory_distinct_type_mass",
        "accessory_duplicate_mass",
        "accessory_gene_diversification_measurement_available",
    ]
    if accessory.id_prompt.duplicated().any() or set(accessory.id_prompt) != set(synteny.id_prompt):
        raise ValueError("Accessory observations must match the synteny cohort exactly")
    fields = {"id_prompt", "genome_id", "family", "accessory_type", "credit", "eligible"}
    if not fields <= set(assignments) or assignments.id_prompt.duplicated().any():
        raise ValueError("Missing or duplicate native accessory ORF assignments")
    if not assignments.eligible.isin([True, False]).all():
        raise ValueError("Accessory ORF eligibility must be measured booleans")
    evidence = accessory.set_index("id_prompt")[required].apply(pd.to_numeric, errors="raise")
    result = synteny.copy()
    result[prefix + "arc_count_pass"] = arc_synteny_pass_mask(result)
    result[prefix + "add_one_gene_pass"] = False
    for column in ["k_orf_id", "k_family", "x_orf_id", "x_family"]:
        result[prefix + column] = ""
    for source, target in zip(
        required,
        ["k_credit", "x_credit", "distinct_type_mass", "duplicate_mass", "measurement_available"],
        strict=True,
    ):
        result[prefix + target] = result.id_prompt.map(evidence[source])
    groups = dict(iter(assignments.loc[assignments.eligible].groupby("genome_id")))
    tolerance = 1e-8
    for index, row in result.iterrows():
        values = evidence.loc[row.id_prompt]
        if not (
            values.notna().all()
            and values.accessory_gene_diversification_measurement_available == 1
            and abs(values.accessory_k_credit - 1) <= tolerance
            and abs(values.accessory_x_credit - 1) <= tolerance
            and abs(values.accessory_distinct_type_mass - 2) <= tolerance
            and 0 <= values.accessory_duplicate_mass <= tolerance
        ):
            continue
        orfs = groups.get(row.id_prompt)
        if orfs is None or len(orfs) != 2:
            continue
        k = orfs.loc[orfs.accessory_type.eq("K")]
        x = orfs.loc[orfs.accessory_type.ne("K")]
        if len(k) != 1 or len(x) != 1:
            continue
        k, x = k.iloc[0], x.iloc[0]
        if (
            k.family not in k_families
            or x.family in core_families | k_families
            or k.family == x.family
            or x.accessory_type != x.family
            or not str(x.family).startswith("phrog:")
            or not abs(float(k.credit) - 1) <= tolerance
            or not abs(float(x.credit) - 1) <= tolerance
        ):
            continue
        result.at[index, prefix + "add_one_gene_pass"] = True
        for kind, hit in [("k", k), ("x", x)]:
            result.at[index, prefix + kind + "_orf_id"] = hit.id_prompt
            result.at[index, prefix + kind + "_family"] = hit.family
    arc = result[prefix + "arc_count_pass"]
    extra = result[prefix + "add_one_gene_pass"]
    result[prefix + "pass"] = arc | extra
    result[prefix + "category"] = "fail"
    result.loc[arc, prefix + "category"] = "original_arc_count"
    result.loc[extra, prefix + "category"] = "add_one_gene"
    result.loc[arc & extra, prefix + "category"] = "both"
    return result


def _measure_add_one_gene_evidence(qualified: pd.DataFrame, config: dict, output: Path):
    """Replay physical-ORF PHROG evidence in a separate output directory."""
    from Bio import SeqIO

    from bionemo.evo2_phage_gen.accessory_genes import measure_accessory_gene_artifacts

    source = Path(config["results_save_dir"])
    output.mkdir(parents=True, exist_ok=True)
    ids = set(qualified.id_prompt)
    SeqIO.write(
        (
            r
            for r in SeqIO.parse(source / config["orfipy_proteins_file_save_location"], "fasta")
            if r.id.rsplit("_", 1)[0] in ids
        ),
        output / "physical-proteins.fasta",
        "fasta",
    )
    hits = pd.read_csv(source / config["mmseqs_protein_database_results_dir_save_location"] / "mmseqs2_all_hits.csv")
    hits = hits.loc[hits.id_prompt.str.rsplit("_", n=1).str[0].isin(ids)]
    (output / "family-hits").mkdir(exist_ok=True)
    hits.to_csv(output / "family-hits/mmseqs2_all_hits.csv", index=False)
    staged = dict(
        config,
        results_save_dir=str(output),
        orfipy_proteins_file_save_location="physical-proteins.fasta",
        mmseqs_protein_database_results_dir_save_location="family-hits",
        accessory_gene_metrics_file_save_location="metrics.csv",
    )
    metrics = measure_accessory_gene_artifacts(staged, qualified, env=os.environ.copy())
    assignments = pd.read_csv(output / "qc6_accessory_genes_assignments.csv")
    return metrics, assignments


def load_arc_functions(source_path: Path) -> dict:
    """Load pinned upstream functions without architecture's import-time side effects."""
    source = Path(source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("ARC source differs from the audited pinned revision")
    text = source.decode()
    anchor = "start, end = match.groups()"
    if text.count(anchor) != 1:
        raise ValueError("Unexpected ARC GFF coordinate implementation")
    # orfipy uses zero-based starts; GFF uses one-based starts. Without this
    # correction LoVis translates the wrong frame. Filter formulas are untouched.
    text = text.replace(anchor, anchor + "\n                start = str(int(start) + 1)")
    tree = ast.parse(text)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        and not (isinstance(node, ast.Import) and any(a.name == "genetic_architecture" for a in node.names))
        and not (isinstance(node, ast.FunctionDef) and node.name == "main")
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


def stage_arc_reference(output: Path) -> Path:
    """Preserve all 11 ARC features; normalize coordinates without deleting the prefix feature."""
    from bionemo.evo2_phage_gen.protein_evidence import stage_coordinate_normalized_reference_gff

    source = RECIPE_ROOT / "data/external/arc_evo2/phage_gen/data/NC_001422.1_pseudocircular.gff"
    if hashlib.sha256(source.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("ARC reference differs from the audited original 11-feature inventory")
    stage_coordinate_normalized_reference_gff(source, output, circular_genome_length=None)
    if sum("\tCDS\t" in line for line in output.read_text().splitlines()) != 11:
        raise ValueError("Original ARC reference must retain all 11 features")
    return output


def validate_core_safety(df: pd.DataFrame, config: dict) -> None:
    """Fail closed if final replay is fed online scores or incomplete safety/core evidence."""
    from bionemo.evo2_phage_gen.protein_evidence import core_gene_ordered_conservation_pass_mask
    from bionemo.evo2_phage_gen.rollout_evidence import load_safety_pass_sequences

    if config.get("online_measurement_mode") or config.get("final_arc_contract") not in {
        CONTRACT,
        CIRCULAR_AAI_CONTRACT,
        ADD_ONE_GENE_CONTRACT,
    }:
        raise ValueError("An explicit offline ARC final contract is required")
    if len(config.get("core_gene_reference_functions", {})) != 9 or set(
        config.get("core_gene_reference_functions", {}).values()
    ) != set("ABCDEFGHJ"):
        raise ValueError("Final acceptance requires exactly the nine PhiX core functions")
    if not config.get("sequence_safety_manifest"):
        raise ValueError("Final acceptance requires a reconciled sequence-safety manifest")
    if df.id_prompt.duplicated().any():
        raise ValueError("Input observation IDs must be unique")
    if not df.empty:
        aai_columns = {"average_protein_percent_identity", "average_protein_identity_gene_count"}
        if (
            not aai_columns <= set(df)
            or not (
                df.average_protein_percent_identity.between(0, 95) & df.average_protein_identity_gene_count.gt(0)
            ).all()
        ):
            raise ValueError("Existing physical-ORF AAI must also pass before the original ARC replay")
        required = {
            "num_syntenic_genes",
            "reference_num_genes",
            "duplicate_reference_gene_count",
            "reference_order_violation_count",
            "missing_synteny_output",
        }
        if not required <= set(df):
            raise ValueError("Missing nine-core order/copy evidence")
        if not df.reference_num_genes.eq(9).all() or not core_gene_ordered_conservation_pass_mask(df, 0).all():
            raise ValueError("Input contains a record that fails the nine-core gate")
    safety = load_safety_pass_sequences(
        config["evo_gen_seqs_fasta_file_save_location"],
        config["sequence_safety_manifest"],
        safety_input_fasta=config.get("sequence_safety_input_fasta"),
    )
    if not df.empty and not df.sequence.str.upper().isin(safety).all():
        raise ValueError("Input contains a record without exact safety PASS")


def _cluster_one(args: tuple[str, str, str]) -> tuple[str, str]:
    from bionemo.evo2_phage_gen import lovis4u_metrics

    gff, reference, output = map(Path, args)
    inputs = output / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy(gff, inputs / gff.name)
    shutil.copy(reference, inputs / reference.name)
    table = output / "native/mmseqs/mmseqs_clustering.tsv"
    if not table.exists():
        sys.argv = [
            "lovis4u",
            "-gff",
            str(inputs),
            "-hl",
            "--set-category-colour",
            "-c",
            "A4p2",
            "-o",
            str(output / "native"),
            "-alip",
        ]
        with (output / "log.txt").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            lovis4u_metrics.main()
    if not table.is_file() or not table.stat().st_size:
        raise RuntimeError(f"Missing LoVis clustering for {gff}")
    # LoVis may skip malformed GFFs without raising. Both loci must be represented.
    pairs = pd.read_csv(table, sep="\t", header=None)
    if pairs.shape[1] != 2 or pairs.isna().any().any() or pairs.iloc[:, 1].duplicated().any():
        raise RuntimeError(f"Malformed LoVis cluster memberships for {gff}")
    candidate_prefix = gff.stem + "-"
    tokens = pairs.astype(str).to_numpy().ravel()
    if not any(t.startswith(candidate_prefix) for t in tokens) or not any(
        not t.startswith(candidate_prefix) for t in tokens
    ):
        raise RuntimeError(f"Incomplete candidate/reference clustering for {gff}")

    def cds_count(path):
        return sum(len(parts := line.split("\t")) >= 9 and parts[2] == "CDS" for line in path.read_text().splitlines())

    candidate_count = pairs.iloc[:, 1].str.startswith(candidate_prefix).sum()
    if candidate_count != cds_count(gff) or len(pairs) - candidate_count != cds_count(reference):
        raise RuntimeError(f"LoVis skipped annotated CDSs for {gff}")
    return gff.stem, str(table)


def apply_arc_final_filters(
    qualified: pd.DataFrame, config: dict, *, output_dir: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Apply the original ARC gates in addition to previously measured core and safety.

    Natural-member annotation uses ARC's 90-1800 nt ORF range and sensitivity 4.
    Only the GFF coordinate convention is corrected. ARC retains its original ORF
    inventory and 11-feature reference. Physical-ORF AAI is an additional gate,
    never a replacement for the original ARC count definition. The v2 contract uses
    physical AAI for acceptance and retains original-inventory AAI as a diagnostic;
    the v1 contract requires both. V3 adds a separately evidenced K+X syntenic-diversity
    route; a count pair such as 11/13 alone never qualifies. No contract explicitly adds A*.
    """
    validate_core_safety(qualified, config)
    source_path = RECIPE_ROOT / "data/external/arc_evo2/phage_gen/pipelines/genome_design_filtering_pipeline.py"
    api = load_arc_functions(source_path)
    out = Path(output_dir or (Path(config["results_save_dir"]) / "arc_reference_filters"))
    out.mkdir(parents=True, exist_ok=True)
    natural_db = str(config.get("mmseqs_db_aai_database", ""))
    if not natural_db or not Path(natural_db).exists():
        raise ValueError("A prepared natural-member AAI database is required")
    tropism_db = str(config.get("mmseqs_db_tropism_protein", ""))
    if not tropism_db or not Path(tropism_db).exists():
        raise ValueError("A prepared tropism protein database is required")
    database_files = sorted(
        {
            asset
            for database in (natural_db, tropism_db)
            for asset in Path(database).parent.glob(Path(database).name + "*")
        }
    )
    database_hashes = {}
    for asset in database_files:
        if asset.is_file():
            digest = hashlib.sha256()
            with asset.open("rb") as handle:
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            database_hashes[str(asset)] = digest.hexdigest()
    versions = {
        "mmseqs": subprocess.run(["mmseqs", "version"], capture_output=True, text=True, check=True).stdout.strip(),
        "orfipy": importlib.metadata.version("orfipy"),
        "lovis4u": importlib.metadata.version("lovis4u"),
    }
    annotation_file = Path(config["protein_annotation_file"]).resolve()
    reference = stage_arc_reference(out / "arc-original-reference.coordinate-only.gff")
    accessory_hashes = {}
    if config["final_arc_contract"] == ADD_ONE_GENE_CONTRACT:
        from bionemo.evo2_phage_gen import accessory_genes, protein_evidence, reference_search

        source = Path(config["results_save_dir"])
        assets = [
            source / config["orfipy_proteins_file_save_location"],
            source / config["mmseqs_protein_database_results_dir_save_location"] / "mmseqs2_all_hits.csv",
            Path(config.get("smooth_reference_genome_gff_file") or config["reference_genome_gff_file_save_location"]),
            *[Path(m.__file__) for m in (accessory_genes, protein_evidence, reference_search)],
        ]
        for asset in assets:
            accessory_hashes[str(asset)] = hashlib.sha256(asset.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(
        (
            qualified.to_csv(index=False)
            + json.dumps(config, sort_keys=True)
            + json.dumps(database_hashes, sort_keys=True)
            + json.dumps(versions, sort_keys=True)
            + json.dumps(accessory_hashes, sort_keys=True)
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            + hashlib.sha256(reference.read_bytes()).hexdigest()
            + hashlib.sha256(annotation_file.read_bytes()).hexdigest()
        ).encode()
    ).hexdigest()
    receipt = out / "input-fingerprint.txt"
    if receipt.exists() and receipt.read_text() != fingerprint:
        raise ValueError("Final-filter output contains a different input/configuration; use a new output directory")
    receipt.write_text(fingerprint)
    add_one_gene = config["final_arc_contract"] == ADD_ONE_GENE_CONTRACT
    circular_aai = config["final_arc_contract"] in {CIRCULAR_AAI_CONTRACT, ADD_ONE_GENE_CONTRACT}
    aai_gate = "physical_aai_reconfirmed" if circular_aai else "arc_aai"
    gate_names = tuple(
        aai_gate if name == "arc_aai" else "syntenic_diversity" if add_one_gene and name == "arc_synteny" else name
        for name in GATE_NAMES
    )
    summary = {
        "contract": config["final_arc_contract"],
        "aai_acceptance": "physical_circular_orfs" if circular_aai else "physical_and_original_inventory",
        "original_inventory_aai_role": "diagnostic" if circular_aai else "additional_gate",
        "aai_shared_limitations": [
            "A* is not explicitly modeled separately",
            "ORF start selection is not experimentally verified",
        ],
        "upstream_sha256": UPSTREAM_SHA256,
        "core_safety_and_physical_aai_qualified": len(qualified),
        "architecture_diversification": bool(config.get("genetic_architecture_remove_filter")),
        "annotation_corrections": ["one-based GFF starts only; original ARC feature inventory retained"],
        "arc_reference_feature_count": 11,
        "arc_reference_source_sha256": REFERENCE_SHA256,
        "orf_length_range_nt": [90, 1800],
        "natural_member_database": natural_db,
        "database_sha256": database_hashes,
        "tool_versions": versions,
        "accessory_evidence_sha256": accessory_hashes,
        "gate_order": [],
        "state": "running",
        "syntenic_diversity_definition": "original_arc_count_or_structured_add_one_gene"
        if add_one_gene
        else "original_arc_count",
        "syntenic_diversity_counts": {
            "original_arc_count": 0,
            "add_one_gene": 0,
            "both": 0,
            "add_one_gene_only": 0,
            "union": 0,
        },
    }
    threads = int(config.get("mmseqs_threads", 8))
    workers = int(config.get("final_arc_workers", min(threads, 8)))
    if min(threads, workers) < 1:
        raise ValueError("Thread and worker counts must be positive")
    # Independent annotation fields must not overwrite the separate core evidence.
    df = qualified[["id_prompt", "sequence"]].copy()

    def save(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        frame.to_csv(out / f"{name}.csv", index=False)
        summary["gate_order"].append({"gate": name, "remaining": len(frame)})
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return frame

    def finish(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        seen = {row["gate"] for row in summary["gate_order"]}
        for name in gate_names:
            if name not in seen:
                save(name, frame.iloc[:0].copy())
        # Preserve the full existing evidence under its original columns, and
        # prefix ARC's independent values so core=9 cannot masquerade as ARC synteny.
        fields = frame.drop(columns="sequence").rename(
            columns=lambda c: c if c == "id_prompt" or c.startswith("syntenic_diversity_") else "arc_" + c
        )
        result = qualified.merge(fields, on="id_prompt", how="inner", validate="one_to_one")
        summary.update(state="succeeded", passed_observations=len(result))
        result.to_csv(out / "accepted-metrics.csv", index=False)
        api["save_df_as_fasta"](result, str(out / "accepted-observations.fasta"))
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return result, summary

    if df.empty:
        return finish(df)

    for name, args in [
        ("valid_nt_chars", ()),
        ("valid_genome_len", ([4000, 6000],)),
        ("valid_gc_content", ([30, 65],)),
        ("valid_nt_homopolymer_len", ([0, 10],)),
    ]:
        df = save(name, api[name](df, *args))
        if df.empty:
            return finish(df)
    api["save_df_as_fasta"](df, str(out / "input.fasta"))
    if not (out / "orf.done").exists():
        api["append_upstream_of_last_frame_stop"](str(out / "input.fasta"), str(out / "circular.fasta"))
        api["run_orfipy"](
            str(out / "circular.fasta"),
            threads,
            "ATG",
            "TAA,TGA,TAG",
            "f",
            90,
            1800,
            str(out),
            "orfs.fasta",
            "proteins-tmp.fasta",
            "proteins.fasta",
        )
        (out / "orf.done").touch()

    def search(name: str, database: str, prefix: str, top: bool) -> pd.DataFrame:
        path = out / f"{name}.csv"
        if not path.exists():
            temporary = out / f"{name}.partial.csv"
            api["run_mmseqs_search_proteins"](
                str(out / "proteins.fasta"),
                database,
                str(out / name),
                str(temporary),
                prefix,
                threads=threads,
                sensitivity=4.0,
                only_top_hits=top,
            )
            temporary.replace(path)
        hits = pd.read_csv(path)
        return hits

    hits = search("natural-hits", natural_db, "protein_database", True)
    if hits.empty:
        return finish(save("arc_seven_protein_hits", df.iloc[:0].copy()))
    if not hits.protein_database_mmseqs_target.str.fullmatch(r"phrog_\d+\|.+").all():
        raise ValueError("AAI/ARC annotation targets are not individual natural PHROG members")
    df = save(
        "arc_seven_protein_hits", api["valid_protein_database_hit_count"](hits, df, id_column="id_prompt", min_hits=7)
    )
    if df.empty:
        return finish(df)
    g_hits = search("g-hits", str(Path(tropism_db).resolve()), "tropism_protein", False)
    if g_hits.empty:
        return finish(save("arc_tropism", df.iloc[:0].copy()))
    df = save("arc_tropism", api["valid_mmseqs_pident"](g_hits, "tropism_protein", [60, 100], df))
    if df.empty:
        return finish(df)
    annotations = pd.read_csv(annotation_file, sep="\t")
    annotations["phrog_number"] = annotations.phrog.astype(str).str.extract(r"(\d+)")
    hits["phrog_number"] = hits.protein_database_mmseqs_target.str.extract(r"phrog_(\d+)")
    hits = hits.merge(
        annotations[["phrog_number", "annot", "category"]], on="phrog_number", how="left", validate="many_to_one"
    )
    api["save_df_as_fasta"](df, str(out / "tropism.fasta"))
    api["batch_create_gff_files"](
        str(out / "tropism.fasta"),
        str(out / "circular.fasta"),
        df,
        str(out / "orfs.fasta"),
        hits,
        str(out / "gff"),
        str(out / "mapping"),
    )
    mapping = pd.read_csv(out / "mapping/genome_id_map.csv")
    df = df.merge(mapping, left_on="id_prompt", right_on="genome_name", validate="one_to_one").drop(
        columns="genome_name"
    )
    df.to_csv(out / "before-aai.csv", index=False)
    api["calculate_average_protein_percent_identity"](
        str(out / "gff"), str(out / "before-aai.csv"), str(out / "aai.csv")
    )
    df = pd.read_csv(out / "aai.csv")
    df["original_inventory_aai_pass"] = df.average_protein_percent_identity.between(0, 95)
    df.to_csv(out / "original-inventory-aai-diagnostic.csv", index=False)
    summary["original_inventory_aai_passers"] = int(df.original_inventory_aai_pass.sum())
    # Physical AAI was independently validated before entry. Its original-inventory
    # counterpart is always measured, but only the explicit legacy contract gates on it.
    df = save(aai_gate, df.copy() if circular_aai else df.loc[df.original_inventory_aai_pass].copy())
    if df.empty:
        return finish(df)
    df.to_csv(out / "before-synteny.csv", index=False)
    api["count_total_num_genes"](str(out / "gff"), str(out / "before-synteny.csv"))
    os.environ["LOVIS4U_MMSEQS_THREADS"] = "1"
    os.environ["LOVIS4U_MMSEQS_BINARY"] = shutil.which("mmseqs") or "mmseqs"
    jobs = [
        (str(out / "gff" / f"{gid}.gff"), str(reference), str(out / "clustering" / gid))
        for gid in sorted(df.genome_id)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for gid, table in pool.map(_cluster_one, jobs, chunksize=1):
            target = out / "metadata" / gid / "mmseqs"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy(table, target / "mmseqs_clustering.tsv")
    api["count_syntenic_genes_all"](
        str(out / "metadata"), str(out / "gff"), str(out / "before-synteny.csv"), str(out / "synteny.csv")
    )
    df = pd.read_csv(out / "synteny.csv")
    if add_one_gene:
        metrics, assignments = _measure_add_one_gene_evidence(df, config, out / "accessory-evidence")
        df = classify_syntenic_diversity(
            df,
            metrics,
            assignments,
            core_families={family for allowed in config["required_gene_families"].values() for family in allowed},
            k_families=set(config["accessory_gene_k_families"]),
        )
        df.to_csv(out / "syntenic-diversity-decisions.csv", index=False)
        original = df.syntenic_diversity_arc_count_pass
        added = df.syntenic_diversity_add_one_gene_pass
        summary["syntenic_diversity_counts"] = {
            "original_arc_count": int(original.sum()),
            "add_one_gene": int(added.sum()),
            "both": int((original & added).sum()),
            "add_one_gene_only": int((added & ~original).sum()),
            "union": int((original | added).sum()),
        }
        df = save("syntenic_diversity", df.loc[df.syntenic_diversity_pass].copy())
    else:
        df = save("arc_synteny", df.loc[arc_synteny_pass_mask(df)].copy())
        summary["syntenic_diversity_counts"].update(original_arc_count=len(df), union=len(df))
    return finish(df)
