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

"""Prepare a runnable local copy of Arc's phage filtering pipeline."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from bionemo.evo2_phage_gen.external_qc import ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARC_PIPELINE_SOURCE_DIR = RECIPE_ROOT / "data" / "external" / "arc_evo2" / "phage_gen" / "pipelines"
DEFAULT_ARC_PIPELINE_WORKDIR = RECIPE_ROOT / "data" / "arc_pipeline_patched"
DEFAULT_PHIX174_FASTA = RECIPE_ROOT / "data" / "external" / "arc_evo2" / "phage_gen" / "data" / "NC_001422_1.fna"
DEFAULT_ARC_PIPELINE_PATCH = RECIPE_ROOT / "patches" / "arc-evo2-genome-design-filtering.patch"
ARC_EVO2_GIT_URL = "https://github.com/ArcInstitute/evo2.git"
ARC_EVO2_REV = "53f195997257c56c00e5ef8d33a54f5baad143a6"
ARC_LEGACY_GENETIC_ARCHITECTURE_IMPORT_FASTA = (
    "/large_storage/hielab/samuelking/phage_design/data/phix174_only/microviridae_genomes_NC_001422_1.fna"
)
ARC_LEGACY_LOVIS4U_PARALLEL_CONFIG = """        # Get parallelization settings from config if available
        max_workers = config.get("n_parallel_jobs", None)
        chunk_size = config.get("chunk_size", 10)
"""
PATCHED_LOVIS4U_PARALLEL_CONFIG = """        # Get parallelization settings from config if available
        max_workers = config.get("lovis4u_parallel_jobs", config.get("n_parallel_jobs", None))
        chunk_size = config.get("lovis4u_chunk_size", config.get("chunk_size", 10))
"""
ARC_LEGACY_LOVIS4U_COMMAND = "    command = [\n        'lovis4u', \n"
PATCHED_LOVIS4U_COMMAND = """    executable = ['lovis4u']
    if os.environ.get("LOVIS4U_METRICS_ONLY") == "1":
        executable = [sys.executable, "-m", "bionemo.evo2_phage_gen.lovis4u_metrics"]
    command = executable + [
"""
ARC_LEGACY_LOVIS4U_RUNTIME_CONFIG = """    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

        # Get parallelization settings from config if available
        max_workers = config.get("lovis4u_parallel_jobs", config.get("n_parallel_jobs", None))
        chunk_size = config.get("lovis4u_chunk_size", config.get("chunk_size", 10))
"""
PATCHED_LOVIS4U_RUNTIME_CONFIG = """    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

        if config.get("lovis4u_metrics_only", False):
            os.environ["LOVIS4U_METRICS_ONLY"] = "1"
        else:
            os.environ.pop("LOVIS4U_METRICS_ONLY", None)
        mmseqs_threads = config.get("lovis4u_mmseqs_threads")
        if mmseqs_threads is None:
            os.environ.pop("LOVIS4U_MMSEQS_THREADS", None)
        else:
            mmseqs_threads = int(mmseqs_threads)
            if mmseqs_threads < 1:
                raise ValueError("lovis4u_mmseqs_threads must be positive")
            os.environ["LOVIS4U_MMSEQS_THREADS"] = str(mmseqs_threads)

        # Get parallelization settings from config if available
        max_workers = config.get("lovis4u_parallel_jobs", config.get("n_parallel_jobs", None))
        chunk_size = config.get("lovis4u_chunk_size", config.get("chunk_size", 10))
"""
ARC_ONLINE_GBK_CONVERSION = """        ### Save GBK files ###
        print("Creating gbk files...")
        batch_convert_gff_to_gbk(input_dir=f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
                                 output_dir=f'{config["results_save_dir"]}/{config["gbk_dir_save_location"]}')
"""
PATCHED_ONLINE_GBK_CONVERSION = """        ### Save GBK files only for offline filtering, where rejected artifacts may be deleted. ###
        if online_measurement_mode:
            print("Skipping unconsumed GBK conversion during online measurement.")
        else:
            print("Creating gbk files...")
            batch_convert_gff_to_gbk(input_dir=f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
                                     output_dir=f'{config["results_save_dir"]}/{config["gbk_dir_save_location"]}')
"""
ORFIPY_GFF_COORDINATE_ASSIGNMENT = re.compile(
    r"^(?P<indent>[ \t]*)start, end = match\.groups\(\)[ \t]*$",
    flags=re.MULTILINE,
)
PATCHED_ORFIPY_GFF_START = "start = str(int(start) + 1)"
ARC_GFF_TO_GBK_LOCATION = "location=FeatureLocation(start, end, strand=strand),"
PATCHED_GFF_TO_GBK_LOCATION = "location=FeatureLocation(start - 1, end, strand=strand),"
ORFIPY_CALL_PATTERN = re.compile(
    r'^(?P<indent>[ ]*)run_orfipy\(.*?config\["orfipy_proteins_file_save_location"\]\)',
    flags=re.MULTILINE | re.DOTALL,
)
PATCHED_PSEUDOCIRCULAR_ORF_CALL = "remove_pseudocircular_extension_orfs("
ARC_MMSEQS_PROTEIN_FORMAT_OUTPUT = "--format-output 'query,target,evalue,pident'"
PATCHED_MMSEQS_PROTEIN_FORMAT_OUTPUT = "--format-output 'query,target,evalue,pident,alnlen,qlen,tlen,qcov,tcov'"
ARC_MMSEQS_PROTEIN_PARSE_FIELDS = "query, target, evalue, pident = line.strip().split('\\t')"
PATCHED_MMSEQS_PROTEIN_PARSE_FIELDS = "query, target, evalue, pident, alignment_length, query_length, target_length, query_coverage, target_coverage = line.strip().split('\\t')"
ARC_MMSEQS_PROTEIN_HIT_TUPLE = "hits.append((query, target, float(evalue), float(pident)))"
PATCHED_MMSEQS_PROTEIN_HIT_TUPLE = (
    "hits.append((query, target, float(evalue), float(pident), int(alignment_length), "
    "int(query_length), int(target_length), float(query_coverage), float(target_coverage)))"
)
ARC_MMSEQS_PROTEIN_HIT_LOOP = "for query, target, evalue, pident in hits:"
PATCHED_MMSEQS_PROTEIN_HIT_LOOP = "for query, target, evalue, pident, alignment_length, query_length, target_length, query_coverage, target_coverage in hits:"
ARC_MMSEQS_PROTEIN_DATA_ROW = "data.append([query, sequences[query], target, evalue, pident])"
PATCHED_MMSEQS_PROTEIN_DATA_ROW = "data.append([query, sequences[query], target, evalue, pident, alignment_length, query_length, target_length, query_coverage, target_coverage])"
ARC_MMSEQS_PROTEIN_DATAFRAME = (
    'df = pd.DataFrame(data, columns=["id_prompt", "sequence", f"{descriptive_prefix}_mmseqs_target", '
    'f"{descriptive_prefix}_mmseqs_e_value", f"{descriptive_prefix}_mmseqs_percent_identity"])'
)
PATCHED_MMSEQS_PROTEIN_DATAFRAME = (
    'df = pd.DataFrame(data, columns=["id_prompt", "sequence", f"{descriptive_prefix}_mmseqs_target", '
    'f"{descriptive_prefix}_mmseqs_e_value", f"{descriptive_prefix}_mmseqs_percent_identity", '
    'f"{descriptive_prefix}_mmseqs_alignment_length", f"{descriptive_prefix}_mmseqs_query_length", '
    'f"{descriptive_prefix}_mmseqs_target_length", f"{descriptive_prefix}_mmseqs_query_coverage", '
    'f"{descriptive_prefix}_mmseqs_target_coverage"])'
)
ARC_MMSEQS_PROTEIN_EMPTY_COLUMNS = """                f"{descriptive_prefix}_mmseqs_percent_identity",
            ]"""
PATCHED_MMSEQS_PROTEIN_EMPTY_COLUMNS = """                f"{descriptive_prefix}_mmseqs_percent_identity",
                f"{descriptive_prefix}_mmseqs_alignment_length",
                f"{descriptive_prefix}_mmseqs_query_length",
                f"{descriptive_prefix}_mmseqs_target_length",
                f"{descriptive_prefix}_mmseqs_query_coverage",
                f"{descriptive_prefix}_mmseqs_target_coverage",
            ]"""
REFERENCE_CLUSTER_FUNCTION_PATTERN = re.compile(
    r"^def count_syntenic_genes_all\(.*?(?=^def valid_syntenic_gene_count\()",
    flags=re.MULTILINE | re.DOTALL,
)
PATCHED_REFERENCE_CLUSTER_FUNCTION = '''def count_syntenic_genes_all(
    root_dir: str,
    gff_dir: str,
    input_csv: str,
    output_csv: str,
    reference_gff_path=None,
) -> None:
    """Measure distinct reference loci without counting duplicate cluster edges."""
    from bionemo.evo2_phage_gen.protein_evidence import measure_reference_cluster_architecture

    measure_reference_cluster_architecture(
        root_dir,
        gff_dir,
        input_csv,
        output_csv,
        reference_gff_path,
    )


'''
ARC_ONLINE_MODE_CONFIG_ANCHOR = """    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
"""
PATCHED_ONLINE_MODE_CONFIG = """    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    online_measurement_mode = bool(config.get("online_measurement_mode", False))
"""
ARC_ONLINE_PROTEIN_FILTER = """            filtered_df = valid_protein_database_hit_count(mmseqs_results_df, seq_df, 'id_prompt', config["protein_database_hit_count"])
"""
PATCHED_ONLINE_PROTEIN_FILTER = """            if online_measurement_mode:
                hit_genome_ids = mmseqs_results_df["id_prompt"].astype(str).str.rsplit("_", n=1).str[0]
                hit_counts = hit_genome_ids.value_counts()
                filtered_df = seq_df.copy()
                filtered_df["protein_database_hit_count"] = (
                    filtered_df["id_prompt"].map(hit_counts).fillna(0).astype(int)
                )
            else:
                filtered_df = valid_coverage_aware_protein_database_hit_count(
                    mmseqs_results_df,
                    seq_df,
                    'id_prompt',
                    config["protein_database_hit_count"],
                    config.get("protein_match_min_reciprocal_coverage", 0.75),
                )
"""
ARC_ONLINE_TROPISM_FILTER = """            filtered_df = valid_mmseqs_pident(mmseqs_results_df, "tropism_protein", config["tropism_protein_sequence_identity_range"], filtered_df)
"""
PATCHED_ONLINE_TROPISM_FILTER = """            save_mmseqs_pident_metrics(
                mmseqs_results_df,
                "tropism_protein",
                filtered_df,
                f'{config["results_save_dir"]}/{config.get("tropism_protein_sequence_identity_metrics_file_save_location", "qc4_tropism_protein_sequence_identity_metrics.csv")}',
            )
            if not online_measurement_mode:
                filtered_df = valid_coverage_aware_mmseqs_pident(
                    mmseqs_results_df,
                    "tropism_protein",
                    config["tropism_protein_sequence_identity_range"],
                    filtered_df,
                    config.get("tropism_match_min_reciprocal_coverage", 0.95),
                )
"""
ARC_REQUIRED_GENE_CALL_SUFFIX = """                                   sequences_df=filtered_df,
                                   metrics_csv=f'{config["results_save_dir"]}/{config.get("required_genes_metrics_file_save_location", "qc6_required_genes_metrics.csv")}')
"""
PATCHED_REQUIRED_GENE_CALL_SUFFIX = """                                   sequences_df=filtered_df,
                                   metrics_csv=f'{config["results_save_dir"]}/{config.get("required_genes_metrics_file_save_location", "qc6_required_genes_metrics.csv")}',
                                   filter_results=not online_measurement_mode,
                                   protein_database_hits_df=mmseqs_results_df,
                                   minimum_reciprocal_coverage=config.get("protein_match_min_reciprocal_coverage", 0.75),
                                   family_coverage_thresholds=config.get("required_gene_family_coverage"))
"""
AAI_FUNCTION_PATTERN = re.compile(
    r"^def valid_average_protein_percent_identity\(.*?(?=^def count_total_num_genes\()",
    flags=re.MULTILINE | re.DOTALL,
)
PATCHED_AAI_FUNCTION = '''def valid_average_protein_percent_identity(
    gff_directory: str,
    gbk_directory: str,
    results_csv: str,
    output_csv: str,
    identity_range: tuple,
    filter_results: bool = True,
    metrics_csv: str = None,
    identity_database: str = None,
    query_fasta: str = None,
    threads: int = 2,
) -> None:
    """Measure natural-protein best-hit AAI using the configured reference database."""
    from bionemo.evo2_phage_gen.protein_evidence import summarize_best_hit_aai

    sequences_df = pd.read_csv(results_csv)
    if not identity_database or not query_fasta:
        raise ValueError("Natural-protein AAI requires a reference database and called ORF proteins")
    search_dir = os.path.join(os.path.dirname(output_csv), "qc6_mmseqs_aai")
    hits_df = run_mmseqs_search_proteins(
        query_fasta=query_fasta, mmseqs_db=identity_database, results_dir=search_dir,
        output_csv=os.path.join(search_dir, "mmseqs2_hits.csv"),
        descriptive_prefix="protein_database", threads=threads, split=0,
        sensitivity=4.0, only_top_hits=True,
    )
    metrics_df = summarize_best_hit_aai(hits_df)
    metrics_df = sequences_df[["id_prompt"]].merge(metrics_df, on="id_prompt", how="left")
    metrics_df["average_protein_percent_identity"] = metrics_df["average_protein_percent_identity"].fillna(0.0)
    metrics_df["average_protein_identity_gene_count"] = metrics_df["average_protein_identity_gene_count"].fillna(0)
    if metrics_csv is not None:
        metrics_df.to_csv(metrics_csv, index=False)

    merged_df = sequences_df.merge(metrics_df, on="id_prompt", how="left")
    lower, upper = identity_range
    passing = (
        (merged_df["average_protein_identity_gene_count"] > 0)
        & merged_df["average_protein_percent_identity"].between(lower, upper)
    )
    filtered_df = merged_df.loc[passing].copy() if filter_results else merged_df
    filtered_df.to_csv(output_csv, index=False)
    if not filter_results:
        return

    passing_genome_ids = set(filtered_df["genome_id"].astype(str))
    for genome_id in set(sequences_df["genome_id"].astype(str)) - passing_genome_ids:
        for path in (
            os.path.join(gff_directory, f"{genome_id}.gff"),
            os.path.join(gbk_directory, f"{genome_id}.gbk"),
        ):
            if os.path.exists(path):
                os.remove(path)


'''
ARC_AAI_CALL_SUFFIX = """                                                   f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}',
                                                   config["average_protein_sequence_identity_range"])
"""
PATCHED_AAI_CALL_SUFFIX = """                                                   f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}',
                                                   config["average_protein_sequence_identity_range"],
                                                   filter_results=not online_measurement_mode,
                                                   identity_database=config.get("mmseqs_db_aai_database"),
                                                   query_fasta=f'{config["results_save_dir"]}/{config["orfipy_proteins_file_save_location"]}',
                                                   threads=config["mmseqs_threads"],
                                                   metrics_csv=f'{config["results_save_dir"]}/{config.get("average_protein_sequence_identity_metrics_file_save_location", "qc6_average_protein_sequence_identity_metrics.csv")}' )
"""
ARC_SYNTENY_SIGNATURE = """def valid_syntenic_gene_count(input_csv: str, output_csv: str,
                              syntenic_gene_count_range: list, total_gene_count_range: list, syntenic_total_gene_count_remove: set,
                              gff_dir: str, gbk_dir: str, pdf_dir: str, metadata_dir: str) -> None:
"""
PATCHED_SYNTENY_SIGNATURE = """def valid_syntenic_gene_count(input_csv: str, output_csv: str,
                              syntenic_gene_count_range: list, total_gene_count_range: list, syntenic_total_gene_count_remove: set,
                              gff_dir: str, gbk_dir: str, pdf_dir: str, metadata_dir: str,
                              filter_results: bool = True, max_missing_reference_genes: int = 0) -> None:
"""
ARC_SYNTENY_FILTER_RESULT = """    filtered_df = df[df[['num_syntenic_genes', 'total_num_genes']].apply(tuple, axis=1).isin(valid_combinations)]
    removed_ids = set(df["genome_id"]) - set(filtered_df["genome_id"])
"""
PATCHED_SYNTENY_FILTER_RESULT = """    if filter_results:
        from bionemo.evo2_phage_gen.protein_evidence import reference_synteny_pass_mask
        filtered_df = df[reference_synteny_pass_mask(df, max_missing_reference_genes)]
        removed_ids = set(df["genome_id"]) - set(filtered_df["genome_id"])
    else:
        filtered_df = df
        removed_ids = set()
"""
ARC_SYNTENY_CALL_SUFFIX = """                                      pdf_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_pdf_dir_save_location"]}',
                                      metadata_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_dir_save_location"]}')
"""
PATCHED_SYNTENY_CALL_SUFFIX = """                                      pdf_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_pdf_dir_save_location"]}',
                                      metadata_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_dir_save_location"]}',
                                      filter_results=not online_measurement_mode,
                                      max_missing_reference_genes=config.get("synteny_max_missing_reference_genes", 0))
"""
ARC_PIPELINE_FILES = (
    "genome_design_filtering_pipeline.py",
    "genetic_architecture.py",
    "genetic_architecture_visualization.py",
)


def _git_head(path: Path) -> str | None:
    """Return the Git HEAD for ``path`` when it is inside a Git checkout."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _assert_arc_source_revision(source_dir: Path, expected_revision: str) -> None:
    """Fail when the Arc source is incompatible with the maintained patch."""
    actual_revision = _git_head(source_dir)
    if actual_revision is None:
        raise RuntimeError(
            f"Arc pipeline source {source_dir} is not in a Git checkout; "
            f"the maintained patch expects {ARC_EVO2_GIT_URL}@{expected_revision}."
        )
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"Arc pipeline source revision mismatch for {source_dir}: expected "
            f"{ARC_EVO2_GIT_URL}@{expected_revision}, found {actual_revision}."
        )


def _apply_arc_pipeline_patch(output_dir: Path, patch_path: Path) -> None:
    """Apply the maintained Arc pipeline patch to a freshly copied workdir."""
    result = subprocess.run(
        ["patch", "--batch", "--forward", "--ignore-whitespace", "-p0", "-i", str(patch_path)],
        cwd=output_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to apply Arc pipeline patch {patch_path}:\n{result.stdout}")


def _apply_online_measurement_patches(output_dir: Path) -> None:
    """Keep enabled online objectives observable without changing final-QC filtering."""
    pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = pipeline_path.read_text()
    if ARC_ONLINE_MODE_CONFIG_ANCHOR not in text and PATCHED_ONLINE_MODE_CONFIG not in text:
        return
    replacements = (
        (ARC_ONLINE_MODE_CONFIG_ANCHOR, PATCHED_ONLINE_MODE_CONFIG),
        (ARC_ONLINE_PROTEIN_FILTER, PATCHED_ONLINE_PROTEIN_FILTER),
        (ARC_ONLINE_TROPISM_FILTER, PATCHED_ONLINE_TROPISM_FILTER),
        (ARC_REQUIRED_GENE_CALL_SUFFIX, PATCHED_REQUIRED_GENE_CALL_SUFFIX),
        (ARC_AAI_CALL_SUFFIX, PATCHED_AAI_CALL_SUFFIX),
        (ARC_SYNTENY_SIGNATURE, PATCHED_SYNTENY_SIGNATURE),
        (ARC_SYNTENY_FILTER_RESULT, PATCHED_SYNTENY_FILTER_RESULT),
        (ARC_SYNTENY_CALL_SUFFIX, PATCHED_SYNTENY_CALL_SUFFIX),
    )
    missing = [replacement for anchor, replacement in replacements if anchor not in text and replacement not in text]
    if ARC_ONLINE_GBK_CONVERSION not in text and PATCHED_ONLINE_GBK_CONVERSION not in text:
        missing.append(PATCHED_ONLINE_GBK_CONVERSION)
    if missing:
        raise ValueError(f"Failed to apply {len(missing)} online objective-measurement patches")
    for anchor, replacement in replacements:
        text = text.replace(anchor, replacement)
    text = text.replace(ARC_ONLINE_GBK_CONVERSION, PATCHED_ONLINE_GBK_CONVERSION)
    pipeline_path.write_text(text)


def _apply_orfipy_gff_coordinate_patch(output_dir: Path) -> None:
    """Patch the paired ORFipy-to-GFF and GFF-to-GenBank coordinate conversions."""
    pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = pipeline_path.read_text()
    has_orfipy_conversion = (
        ORFIPY_GFF_COORDINATE_ASSIGNMENT.search(text) is not None
        or PATCHED_ORFIPY_GFF_START in text
        or "def extract_orf_positions_from_protein_database_hits" in text
    )
    if not has_orfipy_conversion:
        return

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return f"{indent}start, end = match.groups()\n{indent}{PATCHED_ORFIPY_GFF_START}"

    if PATCHED_ORFIPY_GFF_START in text:
        patched_text, gff_replacement_count = text, 1
    else:
        patched_text, gff_replacement_count = ORFIPY_GFF_COORDINATE_ASSIGNMENT.subn(replacement, text)
    if gff_replacement_count != 1:
        raise ValueError(
            "Expected exactly one ORFipy-to-GFF coordinate assignment in "
            f"{pipeline_path}, found {gff_replacement_count}."
        )
    if PATCHED_GFF_TO_GBK_LOCATION not in patched_text:
        gbk_replacement_count = patched_text.count(ARC_GFF_TO_GBK_LOCATION)
        if gbk_replacement_count != 1:
            raise ValueError(
                "Expected exactly one GFF-to-GenBank coordinate assignment in "
                f"{pipeline_path}, found {gbk_replacement_count}."
            )
        patched_text = patched_text.replace(ARC_GFF_TO_GBK_LOCATION, PATCHED_GFF_TO_GBK_LOCATION, 1)
    pipeline_path.write_text(patched_text)


def _apply_pseudocircular_orf_filter_patch(output_dir: Path) -> None:
    """Exclude ORFipy calls that begin only in the appended circular extension."""
    pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = pipeline_path.read_text()
    if PATCHED_PSEUDOCIRCULAR_ORF_CALL in text or "run_orfipy(" not in text:
        return

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{match.group(0)}\n"
            f"{indent}from bionemo.evo2_phage_gen.protein_evidence import "
            "remove_pseudocircular_extension_orfs\n"
            f"{indent}remove_pseudocircular_extension_orfs(\n"
            f"{indent}    seq_fasta,\n"
            f"""{indent}    f'{{config["results_save_dir"]}}/{{config["orfipy_orfs_file_save_location"]}}',\n"""
            f"""{indent}    f'{{config["results_save_dir"]}}/{{config["orfipy_proteins_file_save_location"]}}',\n"""
            f"{indent})"
        )

    patched_text, replacement_count = ORFIPY_CALL_PATTERN.subn(replacement, text)
    if replacement_count != 1:
        raise ValueError(f"Expected exactly one ORFipy call in {pipeline_path}, found {replacement_count}.")
    pipeline_path.write_text(patched_text)


def _apply_mmseqs_protein_evidence_patch(output_dir: Path) -> None:
    """Retain MMseqs protein alignment lengths needed for reciprocal-coverage evidence."""
    pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = pipeline_path.read_text()
    if "def mmseqs_search_proteins(" not in text:
        return

    replacements = (
        (ARC_MMSEQS_PROTEIN_FORMAT_OUTPUT, PATCHED_MMSEQS_PROTEIN_FORMAT_OUTPUT),
        (ARC_MMSEQS_PROTEIN_PARSE_FIELDS, PATCHED_MMSEQS_PROTEIN_PARSE_FIELDS),
        (ARC_MMSEQS_PROTEIN_HIT_TUPLE, PATCHED_MMSEQS_PROTEIN_HIT_TUPLE),
        (ARC_MMSEQS_PROTEIN_HIT_LOOP, PATCHED_MMSEQS_PROTEIN_HIT_LOOP),
        (ARC_MMSEQS_PROTEIN_DATA_ROW, PATCHED_MMSEQS_PROTEIN_DATA_ROW),
        (ARC_MMSEQS_PROTEIN_DATAFRAME, PATCHED_MMSEQS_PROTEIN_DATAFRAME),
    )
    missing = [anchor for anchor, replacement in replacements if anchor not in text and replacement not in text]
    if missing:
        raise ValueError(
            f"Expected {len(replacements)} MMseqs protein-evidence anchors in {pipeline_path}; "
            f"{len(missing)} are missing."
        )
    for anchor, replacement in replacements:
        text = text.replace(anchor, replacement)
    if ARC_MMSEQS_PROTEIN_EMPTY_COLUMNS in text:
        text = text.replace(ARC_MMSEQS_PROTEIN_EMPTY_COLUMNS, PATCHED_MMSEQS_PROTEIN_EMPTY_COLUMNS)
    pipeline_path.write_text(text)


def _apply_required_gene_evidence_patch(output_dir: Path) -> None:
    """Use fixed, duplicate-safe required-family evidence online and in final Arc filtering."""
    pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = pipeline_path.read_text()
    if "def valid_gene_annotations(" not in text:
        return
    if "required_genes_integrity_sum" in text:
        return

    function_pattern = re.compile(
        r"^def valid_gene_annotations\(.*?(?=^##############################\n### RUN FILTERING PIPELINE ###)",
        flags=re.MULTILINE | re.DOTALL,
    )
    replacement = '''def valid_gene_annotations(
    input_gff_dir: str,
    input_gbk_dir: str,
    required_products: tuple,
    sequences_df: pd.DataFrame,
    metrics_csv: str = None,
    filter_results: bool = True,
    protein_database_hits_df: pd.DataFrame = None,
    minimum_reciprocal_coverage: float = 0.75,
    family_coverage_thresholds: dict = None,
) -> pd.DataFrame:
    """Measure fixed required families and optionally apply the same hard gate."""
    from bionemo.evo2_phage_gen.protein_evidence import summarize_required_gene_evidence

    hits_df = pd.DataFrame() if protein_database_hits_df is None else protein_database_hits_df
    metrics_df = summarize_required_gene_evidence(
        hits_df,
        sequences_df,
        required_products,
        minimum_reciprocal_coverage,
        family_coverage_thresholds,
    )
    if metrics_csv is not None:
        metrics_df.to_csv(metrics_csv, index=False)
    if not filter_results:
        return sequences_df.copy()

    passing = metrics_df.loc[
        (metrics_df["required_genes_total_count"] > 0)
        & (metrics_df["required_genes_full_length_count"] == metrics_df["required_genes_total_count"]),
        "genome_id",
    ]
    surviving_genome_ids = set(passing.astype(str))
    for genome_id in set(sequences_df["genome_id"].astype(str)) - surviving_genome_ids:
        for path in (
            os.path.join(input_gff_dir, f"{genome_id}.gff"),
            os.path.join(input_gbk_dir, f"{genome_id}.gbk"),
        ):
            if os.path.exists(path):
                os.remove(path)
        genome_dir = os.path.join(input_gff_dir, genome_id)
        if os.path.exists(genome_dir):
            shutil.rmtree(genome_dir)
    return sequences_df[sequences_df["genome_id"].astype(str).isin(surviving_genome_ids)].copy()


'''
    patched_text, replacement_count = function_pattern.subn(replacement, text)
    if replacement_count != 1:
        raise ValueError(f"Expected exactly one required-gene function in {pipeline_path}, found {replacement_count}.")
    pipeline_path.write_text(patched_text)


def _apply_reference_cluster_evidence_patch(output_dir: Path) -> None:
    """Replace Arc's edge count with one-to-one reference-locus matching."""
    pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = pipeline_path.read_text()
    if PATCHED_REFERENCE_CLUSTER_FUNCTION in text or "def count_syntenic_genes_all(" not in text:
        return
    patched_text, replacement_count = REFERENCE_CLUSTER_FUNCTION_PATTERN.subn(PATCHED_REFERENCE_CLUSTER_FUNCTION, text)
    if replacement_count != 1:
        raise ValueError(f"Expected exactly one synteny-count function in {pipeline_path}, found {replacement_count}.")
    pipeline_path.write_text(patched_text)


def _apply_aai_evidence_patch(output_dir: Path) -> None:
    """Use the same configured AAI measurement online and in final Arc filtering."""
    pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = pipeline_path.read_text()
    if PATCHED_AAI_FUNCTION in text or "def valid_average_protein_percent_identity(" not in text:
        return
    patched_text, replacement_count = AAI_FUNCTION_PATTERN.subn(PATCHED_AAI_FUNCTION, text)
    if replacement_count != 1:
        raise ValueError(f"Expected exactly one AAI function in {pipeline_path}, found {replacement_count}.")
    pipeline_path.write_text(patched_text)


def _apply_protein_hard_gate_patch(output_dir: Path) -> None:
    """Expose the recipe's shared protein-evidence gates to the copied Arc script."""
    pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = pipeline_path.read_text()
    import_source = """from bionemo.evo2_phage_gen.protein_evidence import (
    valid_coverage_aware_mmseqs_pident,
    valid_coverage_aware_protein_database_hit_count,
)


"""
    if import_source in text:
        return
    import_anchor = "import pandas as pd\n"
    if import_anchor not in text:
        return
    pipeline_path.write_text(text.replace(import_anchor, import_anchor + "\n" + import_source, 1))


def _apply_lovis4u_runtime_patches(output_dir: Path) -> None:
    """Expose scorer-only LoVis4u work and nested MMseqs threads in copied Arc code."""
    visualization_path = output_dir / "genetic_architecture_visualization.py"
    text = visualization_path.read_text()
    replacements = (
        (ARC_LEGACY_LOVIS4U_PARALLEL_CONFIG, PATCHED_LOVIS4U_PARALLEL_CONFIG),
        (ARC_LEGACY_LOVIS4U_COMMAND, PATCHED_LOVIS4U_COMMAND),
        (ARC_LEGACY_LOVIS4U_RUNTIME_CONFIG, PATCHED_LOVIS4U_RUNTIME_CONFIG),
    )
    for anchor, replacement in replacements:
        if replacement not in text and anchor in text:
            text = text.replace(anchor, replacement)
    visualization_path.write_text(text)


def prepare_arc_pipeline_workdir(
    source_dir: Path = DEFAULT_ARC_PIPELINE_SOURCE_DIR,
    output_dir: Path = DEFAULT_ARC_PIPELINE_WORKDIR,
    *,
    phix174_fasta: Path = DEFAULT_PHIX174_FASTA,
    pipeline_patch: Path = DEFAULT_ARC_PIPELINE_PATCH,
    arc_revision: str | None = ARC_EVO2_REV,
    overwrite: bool = False,
) -> list[Path]:
    """Copy Arc pipeline files and patch the import-time PhiX174 FASTA path."""
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    phix174_fasta = Path(phix174_fasta).resolve()
    pipeline_patch = Path(pipeline_patch).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Arc pipeline source directory not found: {source_dir}")
    if not phix174_fasta.exists():
        raise FileNotFoundError(f"PhiX174 FASTA not found: {phix174_fasta}")
    if not pipeline_patch.exists():
        raise FileNotFoundError(f"Arc pipeline patch not found: {pipeline_patch}")
    if arc_revision:
        _assert_arc_source_revision(source_dir, arc_revision)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for filename in ARC_PIPELINE_FILES:
        src = source_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Required Arc pipeline file not found: {src}")
        dst = output_dir / filename
        shutil.copy2(src, dst)
        written_paths.append(dst)

    genetic_architecture_path = output_dir / "genetic_architecture.py"
    text = genetic_architecture_path.read_text()
    patched_text = text
    for source_path in (
        ARC_LEGACY_GENETIC_ARCHITECTURE_IMPORT_FASTA,
        ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA,
    ):
        patched_text = patched_text.replace(source_path, str(phix174_fasta))
    if patched_text == text:
        raise ValueError(
            f"Did not find expected legacy PhiX174 path in {genetic_architecture_path}: "
            f"{ARC_LEGACY_GENETIC_ARCHITECTURE_IMPORT_FASTA}"
        )
    genetic_architecture_path.write_text(patched_text)

    _apply_arc_pipeline_patch(output_dir, pipeline_patch)
    _apply_orfipy_gff_coordinate_patch(output_dir)
    _apply_pseudocircular_orf_filter_patch(output_dir)
    _apply_online_measurement_patches(output_dir)
    _apply_mmseqs_protein_evidence_patch(output_dir)
    _apply_protein_hard_gate_patch(output_dir)
    _apply_required_gene_evidence_patch(output_dir)
    _apply_reference_cluster_evidence_patch(output_dir)
    _apply_aai_evidence_patch(output_dir)
    _apply_lovis4u_runtime_patches(output_dir)
    return written_paths


def main() -> None:
    """CLI entry point for preparing Arc's local pipeline workdir."""
    parser = argparse.ArgumentParser(description="Prepare a patched local copy of Arc's phage filtering pipeline")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_ARC_PIPELINE_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARC_PIPELINE_WORKDIR)
    parser.add_argument("--phix174-fasta", type=Path, default=DEFAULT_PHIX174_FASTA)
    parser.add_argument("--patch", type=Path, default=DEFAULT_ARC_PIPELINE_PATCH)
    parser.add_argument("--arc-revision", type=str, default=ARC_EVO2_REV)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in prepare_arc_pipeline_workdir(
        args.source_dir,
        args.output_dir,
        phix174_fasta=args.phix174_fasta,
        pipeline_patch=args.patch,
        arc_revision=args.arc_revision,
        overwrite=args.overwrite,
    ):
        print(path)


if __name__ == "__main__":
    main()
