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

"""Pluggable reward functions for Evo2 phage design RL."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import pandas as pd
import yaml

from bionemo.evo2_phage_gen import sequence_safety_cli
from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence
from bionemo.evo2_phage_gen.protein_evidence import (
    add_protein_alignment_evidence as _add_protein_alignment_evidence,
)
from bionemo.evo2_phage_gen.protein_evidence import (
    load_candidate_orf_context,
    reference_synteny_pass_mask,
    stage_coordinate_normalized_reference_gff,
    summarize_smooth_reference_evidence,
    write_reference_protein_fasta,
)
from bionemo.evo2_phage_gen.qc import (
    NucleotideQCConfig,
    add_nucleotide_metrics,
    load_fasta_records,
    nucleotide_pass_mask,
    save_fasta,
)
from bionemo.evo2_phage_gen.rollout_evidence import canonical_circular_sequence


RECIPE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = RECIPE_ROOT.parents[1]
ARC_PATH_KEYS = (
    "reference_genome_fasta",
    "genetic_architecture_reference_genome",
    "reference_tropism_protein",
    "mmseqs_db_protein_database",
    "mmseqs_db_aai_database",
    "training_data_genomes_fasta",
    "mmseqs_db_tropism_protein",
    "genetic_architecture_visualization_script",
    "protein_annotation_file",
    "reference_genome_gff_file_save_location",
)
TIMING_COLUMN_PREFIX = "timing/phage_qc/"
SEQUENCE_SAFETY_CLASSES = ("amr", "toxin", "lysogeny")
SAFETY_REVIEW_CREDIT = 0.25


def sequence_safety_reward_fields(
    *,
    class_states: dict[str, str],
    required_by_class: dict[str, bool],
    review_eligible_by_class: dict[str, bool] | None = None,
) -> dict[str, object]:
    """Derive the strict acceptance gate and independent class rewards from measured states."""
    expected_classes = set(SEQUENCE_SAFETY_CLASSES)
    if set(class_states) != expected_classes or set(required_by_class) != expected_classes:
        raise ValueError("sequence-safety reward inputs must define every safety class exactly once")
    if not all(state in {"PASS", "FAIL", "INDETERMINATE"} for state in class_states.values()):
        raise ValueError("sequence-safety reward input contains an invalid state")
    if not all(type(required) is bool for required in required_by_class.values()):
        raise ValueError("sequence-safety reward applicability must be boolean")
    review_eligible = (
        dict.fromkeys(SEQUENCE_SAFETY_CLASSES, False) if review_eligible_by_class is None else review_eligible_by_class
    )
    if set(review_eligible) != expected_classes or not all(type(value) is bool for value in review_eligible.values()):
        raise ValueError("sequence-safety review eligibility must define every class as boolean")
    if any(
        review_eligible[name] and (not required_by_class[name] or class_states[name] != "INDETERMINATE")
        for name in SEQUENCE_SAFETY_CLASSES
    ):
        raise ValueError("sequence-safety review credit requires a required INDETERMINATE class")

    required_states = [class_states[name] for name in SEQUENCE_SAFETY_CLASSES if required_by_class[name]]
    if not required_states:
        gate_state = "INDETERMINATE"
    elif "FAIL" in required_states:
        gate_state = "FAIL"
    elif "INDETERMINATE" in required_states:
        gate_state = "INDETERMINATE"
    else:
        gate_state = "PASS"
    gate_pass = float(gate_state == "PASS")
    telemetry: dict[str, object] = {
        "safety_gate_state": gate_state,
        "safety_gate_pass": gate_pass,
    }
    telemetry.update(
        {
            f"reward_safety_{name}": (
                1.0
                if not required_by_class[name] or class_states[name] == "PASS"
                else SAFETY_REVIEW_CREDIT
                if review_eligible[name]
                else 0.0
            )
            for name in SEQUENCE_SAFETY_CLASSES
        }
    )
    return telemetry


def _attach_timing_columns(scored_df: pd.DataFrame, timings: dict[str, float]) -> pd.DataFrame:
    """Attach batch-level timing values as per-row columns for rollout metric aggregation."""
    for name, value in timings.items():
        scored_df[f"{TIMING_COLUMN_PREFIX}{name}"] = float(value)
    return scored_df


def _record_elapsed(timings: dict[str, float], name: str, start: float) -> None:
    """Record an elapsed perf-counter interval in seconds."""
    timings[name] = time.perf_counter() - start


@dataclass(frozen=True)
class RewardWeights:
    """Weights for phage-design reward components."""

    valid_nt_chars: float = 1.0
    genome_length: float = 1.0
    gc_content: float = 1.0
    nt_homopolymer: float = 1.0
    dustmask_end: float = 0.0
    nucleotide_pass: float = 0.0
    protein_hit_count: float = 0.0
    tropism: float = 0.0
    synteny: float = 0.0
    gene_a_origin: float = 0.0
    average_protein_identity: float = 0.0
    required_genes: float = 0.0
    mmseqs_cluster_diversity: float = 0.0


@dataclass(frozen=True)
class RewardComponent:
    """A swappable 0-1 reward component used by the aggregate RL score."""

    name: str
    weight_attr: str | None
    score_column: str


REWARD_COMPONENTS: tuple[RewardComponent, ...] = (
    RewardComponent("valid_nt_chars", "valid_nt_chars", "reward_valid_nt_chars"),
    RewardComponent("genome_length", "genome_length", "reward_genome_length"),
    RewardComponent("gc_content", "gc_content", "reward_gc_content"),
    RewardComponent("nt_homopolymer", "nt_homopolymer", "reward_nt_homopolymer"),
    RewardComponent("dustmask_end", "dustmask_end", "reward_dustmask_end"),
    RewardComponent("nucleotide_pass", "nucleotide_pass", "reward_nucleotide_pass"),
    RewardComponent("protein_hit_count", "protein_hit_count", "reward_external_protein_hit_count"),
    RewardComponent("tropism", "tropism", "reward_external_tropism"),
    RewardComponent("synteny", "synteny", "reward_external_synteny"),
    RewardComponent(
        "gene_a_origin",
        "gene_a_origin",
        "reward_gene_a_origin",
    ),
    RewardComponent(
        "average_protein_identity",
        "average_protein_identity",
        "reward_external_average_protein_identity",
    ),
    RewardComponent(
        "required_genes",
        "required_genes",
        "reward_external_required_genes",
    ),
    RewardComponent(
        "mmseqs_cluster_diversity",
        "mmseqs_cluster_diversity",
        "reward_mmseqs_cluster_diversity",
    ),
    RewardComponent("safety_amr", None, "reward_safety_amr"),
    RewardComponent("safety_toxin", None, "reward_safety_toxin"),
    RewardComponent("safety_lysogeny", None, "reward_safety_lysogeny"),
)


@dataclass(frozen=True)
class ExternalQCRewardConfig:
    """Configuration for Arc external-QC reward components."""

    enabled: bool = False
    config_path: Path = Path("configs/arc_genome_design_filtering_local.yaml")
    pipeline_script: Path = Path("data/arc_pipeline_patched/genome_design_filtering_pipeline.py")
    work_dir: Path = Path("data/checkpoints/phage_grpo_external_qc")
    tool_bin_dir: Path | None = None
    keep_artifacts: bool = False
    fail_on_error: bool = True
    timeout_seconds: float | None = 1800.0
    enable_orf: bool = False
    enable_coding_density: bool = False
    enable_protein_hit_count: bool = True
    enable_tropism: bool = True
    enable_synteny: bool = False
    enable_average_protein_identity: bool = False
    enable_required_genes: bool = False
    required_genes_evidence_target: float = 10.0
    protein_match_min_reciprocal_coverage: float = 0.75
    tropism_match_min_reciprocal_coverage: float = 0.95
    enable_smooth_reference_rewards: bool = False
    enable_gene_a_origin: bool = False
    synteny_identity_full_credit: float = 0.90
    synteny_reciprocal_coverage_full_credit: float = 0.95
    synteny_integrity_gamma: float = 1.5
    synteny_raw_integrity_min: float = 0.001
    synteny_min_credit: float = 0.01
    synteny_order_weight: float = 0.75
    synteny_duplicate_penalty_weight: float = 0.75
    tropism_identity_full_credit: float = 0.95
    tropism_reciprocal_coverage_full_credit: float = 0.99
    tropism_integrity_gamma: float = 1.5
    tropism_raw_integrity_min: float = 0.001
    tropism_min_credit: float = 0.01
    gene_a_reference_locus: str = "NC_001422.1_ORF.23"
    tropism_reference_locus: str = "NC_001422.1_ORF.3"
    gene_a_origin_motif: str = "CAACTTGATATTAATAACACTATAGACCAC"
    gene_a_origin_offset_nt: int = 345
    gene_a_origin_offset_tolerance_nt: int = 30
    lovis4u_parallel_jobs: int | None = 12
    lovis4u_chunk_size: int | None = None
    lovis4u_mmseqs_threads: int | None = None
    lovis4u_metrics_only: bool = False
    lovis4u_collect_pdfs: bool = False


@dataclass(frozen=True)
class MMseqsClusterDiversityConfig:
    """Configuration for batch-local MMseqs cluster-diversity rewards."""

    enabled: bool = False
    mmseqs_bin: str = "mmseqs"
    work_dir: Path = Path("data/checkpoints/phage_grpo_mmseqs_cluster_diversity")
    keep_artifacts: bool = False
    # Near-whole-genome similarity, not a short conserved local match.
    min_seq_id: float = 0.99
    coverage: float = 0.95
    cov_mode: int = 0
    seq_id_mode: int = 0
    cluster_mode: int = 0
    parallel_jobs: int = 1
    threads: int | None = None
    verbosity: int = 0
    # Generic callers may supply linear genomes. The PhiX profile sets True so
    # rotations of the same circular genome do not receive spurious diversity credit.
    circular: bool = False


@dataclass(frozen=True)
class SequenceSafetyRewardConfig:
    """Configuration for AMR, toxin, and lysogeny sequence-safety rewards."""

    host_domain: HostDomain
    host_evidence: HostEvidence
    asset_manifest_path: Path
    diamond_bin: Path
    mmseqs_bin: Path
    policy_path: Path = Path("configs/phage_safety_policy.yaml")
    work_dir: Path = Path("data/checkpoints/phage_sequence_safety_reward")
    enabled: bool = True
    strict_lysis: bool = False
    circular: bool = True
    threads: int = 1
    batch_size: int = 1
    orf_workers: int = 1
    phrogs_threads: int = 1
    timeout_seconds: float = 300.0


def _sequence_safety_config_is_valid(config: object) -> bool:
    """Validate runtime types and bounds before any safety configuration can influence eligibility."""
    if type(config) is not SequenceSafetyRewardConfig:
        return False
    if type(config.host_domain) is not HostDomain or config.host_domain not in {
        HostDomain.BACTERIA,
        HostDomain.ARCHAEA,
        HostDomain.BACTERIA_AND_ARCHAEA,
    }:
        return False
    if type(config.host_evidence) is not HostEvidence:
        return False
    evidence = config.host_evidence
    if (
        type(evidence.source) is not str
        or (evidence.source_version is not None and type(evidence.source_version) is not str)
        or type(evidence.confirmed) is not bool
        or any(type(domain) is not HostDomain for domain in evidence.replication_host_domains)
    ):
        return False
    path_values = (
        config.asset_manifest_path,
        config.diamond_bin,
        config.mmseqs_bin,
        config.policy_path,
        config.work_dir,
    )
    if not all(isinstance(path, Path) for path in path_values):
        return False
    if any(type(value) is not bool for value in (config.enabled, config.strict_lysis, config.circular)):
        return False
    if any(
        type(value) is not int or value < 1
        for value in (config.threads, config.batch_size, config.orf_workers, config.phrogs_threads)
    ):
        return False
    if (
        not isinstance(config.timeout_seconds, Real)
        or isinstance(config.timeout_seconds, bool)
        or not math.isfinite(float(config.timeout_seconds))
        or config.timeout_seconds <= 0
    ):
        return False
    try:
        json.dumps(config.host_evidence.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _recipe_path(path: str | Path) -> Path:
    """Resolve recipe-relative paths while preserving absolute paths."""
    path = Path(path)
    return path if path.is_absolute() else RECIPE_ROOT / path


def _repo_path(path: str | Path) -> Path:
    """Resolve repo-root-relative config paths while preserving absolute paths."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _external_qc_env(external_qc: ExternalQCRewardConfig) -> dict[str, str]:
    """Build the environment for Arc external-QC subprocesses."""
    env = os.environ.copy()
    if external_qc.tool_bin_dir is not None:
        tool_bin_dir = _recipe_path(external_qc.tool_bin_dir).resolve()
        env["PATH"] = os.pathsep.join((str(tool_bin_dir), env.get("PATH", "")))
        env["LOVIS4U_MMSEQS_BINARY"] = str((tool_bin_dir / "mmseqs").resolve())
    return env


def _smooth_reference_search_command(
    *,
    reference_fasta: Path,
    candidate_fasta: Path,
    output_tsv: Path,
    temporary_dir: Path,
    threads: int,
) -> list[str]:
    """Build the permissive protein search used only for graded online evidence."""
    return [
        "mmseqs",
        "easy-search",
        str(reference_fasta),
        str(candidate_fasta),
        str(output_tsv),
        str(temporary_dir),
        "--min-seq-id",
        "0",
        "-c",
        "0",
        "-e",
        "1",
        "-s",
        "7.5",
        "--max-seqs",
        "100000",
        "--format-output",
        "query,target,evalue,pident,alnlen,qlen,tlen,qcov,tcov",
        "--threads",
        str(max(1, int(threads))),
        "-v",
        "0",
    ]


def _resolve_executable_path(executable: str) -> str:
    """Resolve recipe-relative executable paths while leaving PATH lookups alone."""
    path = Path(executable)
    if path.is_absolute() or len(path.parts) > 1:
        return str(_recipe_path(path))
    return executable


def _basic_feasibility_mask(scored_df: pd.DataFrame, config: NucleotideQCConfig) -> pd.Series:
    """Return the nucleotide feasibility gate used before expensive diversity scoring."""
    return (
        scored_df["valid_nt_chars"].astype(bool)
        & scored_df["genome_length"].between(config.genome_length_min, config.genome_length_max)
        & scored_df["gc_content"].between(config.gc_content_min, config.gc_content_max)
        & (scored_df["max_nt_homopolymer_length"] <= config.homopolymer_max)
    )


def _mmseqs_cluster_command(
    config: MMseqsClusterDiversityConfig,
    input_fasta: Path,
    result_prefix: Path,
    tmp_dir: Path,
) -> list[str]:
    """Build the configured MMseqs easy-cluster command for batch diversity rewards."""
    command = [
        _resolve_executable_path(config.mmseqs_bin),
        "easy-cluster",
        str(input_fasta),
        str(result_prefix),
        str(tmp_dir),
        "--min-seq-id",
        f"{float(config.min_seq_id):.6g}",
        "-c",
        f"{float(config.coverage):.6g}",
        "--cov-mode",
        str(int(config.cov_mode)),
        "--seq-id-mode",
        str(int(config.seq_id_mode)),
        "--cluster-mode",
        str(int(config.cluster_mode)),
        "-v",
        str(int(config.verbosity)),
    ]
    if config.threads is not None:
        command.extend(["--threads", str(int(config.threads))])
    return command


def _parse_mmseqs_cluster_tsv(cluster_tsv: Path) -> dict[str, set[str]]:
    """Read an MMseqs cluster TSV into representative-to-member sets."""
    clusters: dict[str, set[str]] = {}
    with cluster_tsv.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            representative, member = parts[0], parts[1]
            clusters.setdefault(representative, set()).add(member)
    return clusters


def _cluster_valid_sequence_group(
    group_df: pd.DataFrame,
    run_dir: Path,
    group_index: int,
    config: MMseqsClusterDiversityConfig,
) -> tuple[dict[object, tuple[str, int, float]], int, int]:
    """Cluster one prompt group and return row-index rewards plus cluster counts."""
    if group_df.empty:
        return {}, 0, 0
    if len(group_df) == 1:
        row_index = group_df.index[0]
        return {row_index: (f"group{group_index}:seq_0", 1, 1.0)}, 1, 0

    group_dir = run_dir / f"prompt_group_{group_index:04d}"
    group_dir.mkdir(parents=True, exist_ok=True)
    input_fasta = group_dir / "input_sequences.fasta"
    result_prefix = group_dir / "clusters"
    tmp_dir = group_dir / "tmp"
    sequence_ids = [f"seq_{position}" for position in range(len(group_df))]
    row_by_sequence_id = dict(zip(sequence_ids, group_df.index.tolist(), strict=True))
    fasta_df = pd.DataFrame(
        {
            "id_prompt": sequence_ids,
            "sequence": [
                canonical_circular_sequence(sequence) if config.circular else sequence
                for sequence in group_df["sequence"].astype(str).tolist()
            ],
        }
    )
    save_fasta(fasta_df, input_fasta)

    subprocess.run(_mmseqs_cluster_command(config, input_fasta, result_prefix, tmp_dir), check=True)
    cluster_tsv = Path(f"{result_prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs cluster TSV not found: {cluster_tsv}")

    clusters = _parse_mmseqs_cluster_tsv(cluster_tsv)
    rewards_by_row: dict[object, tuple[str, int, float]] = {}
    valid_cluster_count = 0
    for representative, members in clusters.items():
        known_members = sorted(member for member in members if member in row_by_sequence_id)
        cluster_size = len(known_members)
        if cluster_size == 0:
            continue
        valid_cluster_count += 1
        cluster_id = f"group{group_index}:{representative}"
        reward = 1.0 / float(cluster_size)
        for member in known_members:
            rewards_by_row[row_by_sequence_id[member]] = (cluster_id, cluster_size, reward)

    missing_members = set(sequence_ids) - {
        member for members in clusters.values() for member in members if member in row_by_sequence_id
    }
    for member in missing_members:
        row_index = row_by_sequence_id[member]
        rewards_by_row[row_index] = ("", 0, 0.0)
    return rewards_by_row, valid_cluster_count, len(missing_members)


def add_mmseqs_cluster_diversity_rewards(
    scored_df: pd.DataFrame,
    config: NucleotideQCConfig,
    mmseqs_config: MMseqsClusterDiversityConfig,
) -> pd.DataFrame:
    """Add ``1 / cluster_size`` rewards from batch-local MMseqs clustering."""
    df = scored_df.copy()
    df["reward_mmseqs_cluster_diversity"] = 0.0
    df["mmseqs_cluster_id"] = ""
    df["mmseqs_cluster_size"] = 0
    df["mmseqs_cluster_is_singleton"] = 0.0
    df["mmseqs_cluster_valid_for_clustering"] = _basic_feasibility_mask(df, config).astype(float)
    df["mmseqs_cluster_missing_from_output"] = 0.0
    if not mmseqs_config.enabled:
        return df

    valid_df = df[df["mmseqs_cluster_valid_for_clustering"].astype(bool)]
    if valid_df.empty:
        return df

    work_dir = _recipe_path(mmseqs_config.work_dir)
    run_dir = work_dir / f"batch_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        prompt_groups = (
            valid_df["prompt_group"] if "prompt_group" in valid_df else pd.Series("__all__", index=valid_df.index)
        )
        grouped = list(enumerate(valid_df.groupby(prompt_groups, sort=False)))
        max_workers = min(max(1, int(mmseqs_config.parallel_jobs)), len(grouped))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _cluster_valid_sequence_group,
                    group_df,
                    run_dir,
                    group_index,
                    mmseqs_config,
                )
                for group_index, (_prompt_group, group_df) in grouped
            ]

        total_clusters = 0
        total_missing = 0
        for future in futures:
            rewards_by_row, num_clusters, num_missing = future.result()
            total_clusters += num_clusters
            total_missing += num_missing
            for row_index, (cluster_id, cluster_size, reward) in rewards_by_row.items():
                df.loc[row_index, "mmseqs_cluster_id"] = cluster_id
                df.loc[row_index, "mmseqs_cluster_size"] = int(cluster_size)
                df.loc[row_index, "reward_mmseqs_cluster_diversity"] = float(reward)
                df.loc[row_index, "mmseqs_cluster_is_singleton"] = 1.0 if cluster_size == 1 else 0.0
        df["mmseqs_cluster_num_clusters"] = total_clusters
        df["mmseqs_cluster_num_missing_from_output"] = total_missing
        missing_output_mask = df["mmseqs_cluster_valid_for_clustering"].astype(bool) & (
            df["mmseqs_cluster_size"].astype(int) == 0
        )
        df.loc[missing_output_mask, "mmseqs_cluster_missing_from_output"] = 1.0
    finally:
        if not mmseqs_config.keep_artifacts:
            shutil.rmtree(run_dir, ignore_errors=True)
    return df


def _interval_score(value: float, lower: float, upper: float) -> float:
    """Return 1 inside an interval and a smooth bounded penalty outside it."""
    if lower <= value <= upper:
        return 1.0
    distance = lower - value if value < lower else value - upper
    width = max(upper - lower, 1.0)
    return max(0.0, 1.0 - distance / width)


def score_genome_length(value: float, config: NucleotideQCConfig) -> float:
    """Score length against the four-point reward envelope, independently of hard QC."""
    lower_zero, lower_full, upper_full, upper_zero = (
        config.genome_length_reward_lower_zero,
        config.genome_length_reward_lower_full,
        config.genome_length_reward_upper_full,
        config.genome_length_reward_upper_zero,
    )
    if value <= lower_zero or value >= upper_zero:
        return 0.0
    if lower_full <= value <= upper_full:
        return 1.0
    if value < lower_full:
        return (value - lower_zero) / (lower_full - lower_zero)
    return (upper_zero - value) / (upper_zero - upper_full)


def _upper_bound_ratio_score(value: float, upper: float) -> float:
    """Return a dense score for upper-bound-only metrics such as homopolymer length."""
    if value <= upper:
        return 1.0
    if value <= 0.0:
        return 0.0
    return max(0.0, min(1.0, upper / value))


def _lower_bound_ratio_score(value: float, lower: float) -> float:
    """Return a dense capped score for lower-bound thresholds."""
    if value >= lower:
        return 1.0
    if lower <= 0.0:
        return 0.0
    return max(0.0, min(1.0, value / lower))


def score_tropism_identity(identity: float | None, measured_hit: bool, threshold: float = 60.0) -> float:
    """Plateau spike/tropism reward at the paper identity threshold."""
    if not measured_hit:
        return 0.0
    identity = max(0.0, float(identity or 0.0))
    if identity >= threshold:
        return 1.0
    if threshold <= 0.0:
        return 0.0
    return max(0.0, min(1.0, identity / threshold))


def score_aai_novelty(aai: float) -> float:
    """Reward AAI novelty up to 95%, then keep high-similarity genomes fractional."""
    aai = max(0.0, min(100.0, float(aai)))
    if aai <= 95.0:
        return 1.0
    return max(0.25, (100.0 - aai) / 5.0)


def score_aai_evidence(num_aai_entries: float) -> float:
    """Require enough measured proteins before trusting AAI novelty."""
    return max(0.0, min(1.0, float(num_aai_entries) / 10.0))


def score_synteny_counts(
    syntenic_genes: float,
    reference_genes: float,
    duplicate_reference_genes: float,
    reference_order_violations: float = 0.0,
) -> tuple[float, float, float, float]:
    """Score reference coverage, homolog-copy balance, and circular gene order."""
    if (
        reference_genes <= 0
        or syntenic_genes < 0
        or syntenic_genes > reference_genes
        or duplicate_reference_genes < 0
        or reference_order_violations < 0
    ):
        return 0.0, 0.0, 0.0, 0.0
    reference_coverage = float(syntenic_genes) / float(reference_genes)
    copy_balance = 1.0 / (1.0 + float(duplicate_reference_genes))
    order_balance = 1.0 / (1.0 + float(reference_order_violations))
    deficit = (
        float(reference_genes)
        - float(syntenic_genes)
        + float(duplicate_reference_genes)
        + float(reference_order_violations)
    )
    return reference_coverage * copy_balance * order_balance, reference_coverage, copy_balance, deficit


def _active_reward_components(weights: RewardWeights, scored_df: pd.DataFrame) -> list[tuple[float, RewardComponent]]:
    """Return weighted reward components whose score columns are available."""
    active_components = []
    for component in REWARD_COMPONENTS:
        if component.weight_attr is None:
            continue
        weight = float(getattr(weights, component.weight_attr))
        if weight > 0.0 and component.score_column in scored_df:
            active_components.append((weight, component))
    return active_components


def _exact_safety_gate_pass_mask(scored_df: pd.DataFrame) -> pd.Series:
    """Accept only a real numeric scalar equal to one, never bools or numeric strings."""
    values = scored_df.get("safety_gate_pass", pd.Series(0.0, index=scored_df.index))
    return values.map(lambda value: isinstance(value, Real) and not isinstance(value, bool) and value == 1.0)


def aggregate_rewards(scored_df: pd.DataFrame, weights: RewardWeights) -> pd.DataFrame:
    """Aggregate available 0-1 component scores into the scalar RL reward."""
    active_components = _active_reward_components(weights, scored_df)
    if not active_components:
        raise ValueError("At least one available reward weight must be positive.")

    weighted_sum = 0.0
    total_weight = 0.0
    for weight, component in active_components:
        scored_df[component.score_column] = scored_df[component.score_column].astype(float).clip(0.0, 1.0)
        weighted_sum = weighted_sum + weight * scored_df[component.score_column]
        total_weight += weight

    safety_gate_pass = _exact_safety_gate_pass_mask(scored_df)
    scored_df["reward"] = (weighted_sum / total_weight).where(safety_gate_pass, 0.0)
    scored_df["reward_active_components"] = ",".join(component.name for _, component in active_components)
    scored_df["reward_total_weight"] = total_weight
    return scored_df


def binary_cluster_deduplicated_pass_mask(scored_df: pd.DataFrame, pass_mask: pd.Series) -> pd.Series:
    """Return one passing representative per MMseqs cluster when cluster data is available."""
    if len(pass_mask) != len(scored_df):
        raise ValueError("binary pass mask length does not match scored rows")
    positions = pd.RangeIndex(len(scored_df))
    positional_pass = pd.Series(pass_mask.astype(bool).to_numpy(), index=positions)
    deduplicated = pd.Series(False, index=positions)
    if not {"mmseqs_cluster_id", "mmseqs_cluster_size"}.issubset(scored_df.columns):
        deduplicated.loc[positional_pass] = True
        return pd.Series(deduplicated.to_numpy(), index=scored_df.index)

    cluster_sizes = pd.Series(
        pd.to_numeric(scored_df["mmseqs_cluster_size"], errors="coerce").fillna(0).astype(int).to_numpy(),
        index=positions,
    )
    cluster_ids = pd.Series(scored_df["mmseqs_cluster_id"].astype(str).to_numpy(), index=positions)
    clustered_pass = positional_pass & (cluster_sizes > 0) & (cluster_ids != "")
    cluster_rows = pd.DataFrame({"cluster_id": cluster_ids}).loc[clustered_pass]
    for _cluster_id, cluster_df in cluster_rows.groupby("cluster_id", sort=False):
        deduplicated.iloc[cluster_df.index[0]] = True

    if "mmseqs_cluster_valid_for_clustering" in scored_df:
        valid_for_clustering = pd.Series(
            (
                pd.to_numeric(scored_df["mmseqs_cluster_valid_for_clustering"], errors="coerce").fillna(0.0) > 0.0
            ).to_numpy(),
            index=positions,
        )
        nonclusterable_pass = positional_pass & ~valid_for_clustering
    else:
        nonclusterable_pass = positional_pass & ~clustered_pass
    deduplicated.loc[nonclusterable_pass] = True
    return pd.Series(deduplicated.to_numpy(), index=scored_df.index)


def _sequence_safety_required_by_class(config: SequenceSafetyRewardConfig) -> dict[str, bool]:
    """Record Task 4 applicability when a row has no usable scan manifest."""
    lysogeny_required = config.host_domain is not HostDomain.ARCHAEA or config.strict_lysis
    return {"amr": True, "toxin": True, "lysogeny": lysogeny_required}


def _add_unavailable_sequence_safety_rewards(
    scored_df: pd.DataFrame,
    *,
    reason_code: str,
    required_by_class: dict[str, bool] | None = None,
    strict_lysis: bool = False,
) -> pd.DataFrame:
    """Record zero reward and an explicit reason when required safety evidence is unavailable."""
    required_classes = dict.fromkeys(SEQUENCE_SAFETY_CLASSES, True) if required_by_class is None else required_by_class
    reasons_json = json.dumps([reason_code], separators=(",", ":"))
    defaults: dict[str, object] = {
        "safety_gate_state": "INDETERMINATE",
        "safety_gate_pass": 0.0,
        "safety_gate_reason_codes": reasons_json,
        "safety_environment_healthy": 0.0,
        "safety_gate_measurement_available": 0.0,
        "safety_required_class_count": sum(required_classes.values()),
        "safety_required_class_pass_count": 0,
        "safety_scan_record_id": "",
        "safety_scan_input_index": -1,
        "safety_policy_id": "",
        "safety_asset_state_path": "",
        "safety_scan_manifest_path": "",
        "safety_resolved_profile": "",
        "safety_amrfinder_version": "",
        "safety_diamond_version": "",
        "safety_mmseqs_version": "",
        "safety_strict_lysis": strict_lysis,
    }
    for safety_class in SEQUENCE_SAFETY_CLASSES:
        required = required_classes[safety_class]
        prefix = f"safety_{safety_class}"
        defaults.update(
            {
                f"{prefix}_state": "INDETERMINATE",
                f"{prefix}_required": float(required),
                f"{prefix}_reason_codes": reasons_json,
                f"{prefix}_finding_count": 0,
                f"{prefix}_measurement_available": 0.0,
                f"{prefix}_execution_status": "NOT_STARTED",
                f"{prefix}_policy_id": "",
                f"reward_safety_{safety_class}": float(not required),
            }
        )
    original_index = scored_df.index
    base = scored_df.drop(columns=[column for column in defaults if column in scored_df]).reset_index(drop=True)
    telemetry = pd.DataFrame({column: [value] * len(base) for column, value in defaults.items()}, index=base.index)
    combined = pd.concat([base, telemetry], axis=1).copy()
    combined.index = original_index
    return combined


def _sequence_is_scannable(sequence: object) -> bool:
    return isinstance(sequence, str) and re.fullmatch(r"[ACGTNacgtn]+", sequence) is not None


def _set_row_values(scored_df: pd.DataFrame, position: int, values: dict[str, object]) -> None:
    for column, value in values.items():
        if column not in scored_df:
            # pandas 3 infers a strict str dtype from a "" placeholder, which rejects
            # non-string telemetry; keep the fallback column object-dtype instead.
            scored_df[column] = pd.Series("", index=scored_df.index, dtype=object)
        scored_df.iloc[position, scored_df.columns.get_loc(column)] = value


def _set_unavailable_reason(scored_df: pd.DataFrame, positions: list[int], reason_code: str) -> None:
    reasons = json.dumps([reason_code], separators=(",", ":"))
    for position in positions:
        values = {"safety_gate_reason_codes": reasons}
        values.update({f"safety_{name}_reason_codes": reasons for name in SEQUENCE_SAFETY_CLASSES})
        _set_row_values(scored_df, position, values)


def _json_reason_codes(value: object) -> str:
    if not isinstance(value, list) or not all(isinstance(reason, str) for reason in value):
        raise ValueError("sequence-safety reason codes must be a string list")
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _manifest_safety_row(
    record: object,
    *,
    expected_input_index: int,
    expected_record_id: str,
    manifest: dict[str, object],
    manifest_path: Path,
) -> dict[str, object]:
    """Convert one validated scan record into compact RL telemetry."""
    if not isinstance(record, dict):
        raise ValueError("sequence-safety record must be a mapping")
    if record.get("input_index") != expected_input_index or record.get("record_id") != expected_record_id:
        raise ValueError("sequence-safety record mapping changed")
    class_results = record.get("class_results")
    attempts = record.get("adapter_attempts")
    if not isinstance(class_results, list) or not isinstance(attempts, list):
        raise ValueError("sequence-safety class telemetry is missing")
    class_by_name = {item.get("safety_class"): item for item in class_results if isinstance(item, dict)}
    attempt_by_name = {item.get("safety_class"): item for item in attempts if isinstance(item, dict)}
    if set(class_by_name) != set(SEQUENCE_SAFETY_CLASSES) or set(attempt_by_name) != set(SEQUENCE_SAFETY_CLASSES):
        raise ValueError("sequence-safety class inventory changed")

    policy = manifest.get("policy")
    assets = manifest.get("asset_state")
    profile = manifest.get("resolved_profile")
    tools = manifest.get("tools")
    if not all(isinstance(value, dict) for value in (policy, assets, profile, tools)):
        raise ValueError("sequence-safety runtime state is incomplete")
    values: dict[str, object] = {
        "safety_scan_record_id": expected_record_id,
        "safety_scan_input_index": expected_input_index,
        "safety_gate_reason_codes": _json_reason_codes(record.get("reason_codes")),
        "safety_policy_id": policy.get("policy_id"),
        "safety_asset_state_path": assets.get("path"),
        "safety_scan_manifest_path": str(manifest_path),
        "safety_resolved_profile": profile.get("host_domain"),
        "safety_strict_lysis": profile.get("strict_lysis"),
        "safety_amrfinder_version": tools.get("amrfinder", {}).get("version", ""),
        "safety_diamond_version": tools.get("diamond", {}).get("version", ""),
        "safety_mmseqs_version": tools.get("mmseqs", {}).get("version", ""),
    }
    required_count = 0
    required_pass_count = 0
    required_measurements: list[bool] = []
    states: dict[str, str] = {}
    required_by_class: dict[str, bool] = {}
    review_eligible: dict[str, bool] = {}
    reasons: list[str] = []
    for safety_class in SEQUENCE_SAFETY_CLASSES:
        result = class_by_name[safety_class]
        attempt = attempt_by_name[safety_class]
        state = result.get("state")
        required = result.get("required")
        findings = result.get("findings")
        status = attempt.get("execution_status")
        class_reasons = result.get("reason_codes")
        if (
            state not in {"PASS", "FAIL", "INDETERMINATE"}
            or type(required) is not bool
            or not isinstance(findings, list)
            or not isinstance(status, str)
        ):
            raise ValueError("sequence-safety class telemetry is invalid")
        reasons.extend(class_reasons)
        measured = status == "COMPLETED_AND_PARSED"
        if required:
            required_count += 1
            required_pass_count += int(state == "PASS")
            required_measurements.append(measured)
        states[safety_class] = state
        required_by_class[safety_class] = required
        review_eligible[safety_class] = bool(required and state == "INDETERMINATE" and measured and findings)
        prefix = f"safety_{safety_class}"
        values.update(
            {
                f"{prefix}_state": state,
                f"{prefix}_required": float(required),
                f"{prefix}_reason_codes": _json_reason_codes(class_reasons),
                f"{prefix}_finding_count": len(findings),
                f"{prefix}_measurement_available": float(measured),
                f"{prefix}_execution_status": status,
                f"{prefix}_policy_id": attempt.get("policy_id", ""),
            }
        )
    safety_fields = sequence_safety_reward_fields(
        class_states=states,
        required_by_class=required_by_class,
        review_eligible_by_class=review_eligible,
    )
    if record.get("state") != safety_fields["safety_gate_state"] or record.get("reason_codes") != list(
        dict.fromkeys(reasons)
    ):
        raise ValueError("sequence-safety record aggregate is inconsistent")
    healthy = bool(required_measurements) and all(required_measurements)
    if safety_fields["safety_gate_state"] == "PASS" and not healthy:
        raise ValueError("sequence-safety PASS lacks completed required measurements")
    values.update(safety_fields)
    values["safety_environment_healthy"] = float(healthy)
    values["safety_gate_measurement_available"] = float(healthy)
    values["safety_required_class_count"] = required_count
    values["safety_required_class_pass_count"] = required_pass_count
    return values


def _sequence_safety_scan_argv(
    *,
    input_fasta: Path,
    output_dir: Path,
    config: SequenceSafetyRewardConfig,
) -> list[str]:
    argv = [
        "scan",
        "--input-fasta",
        str(input_fasta),
        "--output-dir",
        str(output_dir),
        "--policy",
        str(_recipe_path(config.policy_path)),
        "--asset-manifest",
        str(_recipe_path(config.asset_manifest_path)),
        "--host-domain",
        config.host_domain.value,
        "--host-evidence-json",
        json.dumps(config.host_evidence.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False),
        "--diamond-bin",
        str(_recipe_path(config.diamond_bin)),
        "--mmseqs-bin",
        str(_recipe_path(config.mmseqs_bin)),
        "--threads",
        str(config.threads),
        "--batch-size",
        str(config.batch_size),
        "--orf-workers",
        str(config.orf_workers),
        "--phrogs-threads",
        str(config.phrogs_threads),
        "--timeout",
        str(config.timeout_seconds),
    ]
    if config.strict_lysis:
        argv.append("--strict-lysis")
    if not config.circular:
        argv.append("--linear")
    return argv


def add_sequence_safety_rewards(
    scored_df: pd.DataFrame,
    config: SequenceSafetyRewardConfig,
) -> pd.DataFrame:
    """Run Task 4 and map its validated per-record results back to the batch."""
    if not _sequence_safety_config_is_valid(config):
        return _add_unavailable_sequence_safety_rewards(
            scored_df.copy(),
            reason_code="SEQUENCE_SAFETY_CONFIG_INVALID",
        )
    result = _add_unavailable_sequence_safety_rewards(
        scored_df.copy(),
        reason_code="SEQUENCE_SAFETY_RECORD_UNSCANNABLE",
        required_by_class=_sequence_safety_required_by_class(config),
        strict_lysis=config.strict_lysis,
    )
    valid_positions = [
        position for position, sequence in enumerate(result["sequence"].tolist()) if _sequence_is_scannable(sequence)
    ]
    if not valid_positions:
        return result
    _set_unavailable_reason(result, valid_positions, "SEQUENCE_SAFETY_SCAN_UNAVAILABLE")
    try:
        work_dir = _recipe_path(config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = work_dir / f"batch_{uuid.uuid4().hex}"
        run_dir.mkdir()
        input_fasta = run_dir / "input.fasta"
        output_dir = run_dir / "scan"
        record_ids = [f"safety_record_{position:06d}" for position in valid_positions]
        scan_df = pd.DataFrame(
            {
                "id_prompt": record_ids,
                "sequence": [result.iloc[position]["sequence"] for position in valid_positions],
            }
        )
        save_fasta(scan_df, input_fasta)
        exit_code = sequence_safety_cli.main(
            _sequence_safety_scan_argv(
                input_fasta=input_fasta,
                output_dir=output_dir,
                config=config,
            )
        )
        if exit_code not in {0, 2, 3}:
            raise RuntimeError(f"sequence-safety scanner returned unsupported exit code {exit_code}")
        manifest_path = (output_dir / "manifest.json").absolute()
        try:
            manifest = sequence_safety_cli.validate_manifest_file(
                manifest_path,
                expected_type="sequence_safety_scan",
            )
            if not isinstance(manifest, dict) or manifest.get("manifest_type") != "sequence_safety_scan":
                raise ValueError("sequence-safety scan did not produce the expected result manifest")
            records = manifest.get("records")
            if not isinstance(records, list) or len(records) != len(valid_positions):
                raise ValueError("sequence-safety result count does not match the input batch")
            mapped_rows = [
                _manifest_safety_row(
                    record,
                    expected_input_index=scan_index,
                    expected_record_id=record_id,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
                for scan_index, (record_id, record) in enumerate(zip(record_ids, records, strict=True))
            ]
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            _set_unavailable_reason(result, valid_positions, "SEQUENCE_SAFETY_MANIFEST_REJECTED")
            return result
        for position, values in zip(valid_positions, mapped_rows, strict=True):
            _set_row_values(result, position, values)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return result
    return result


def add_nucleotide_rewards(metrics: pd.DataFrame, config: NucleotideQCConfig = NucleotideQCConfig()) -> pd.DataFrame:
    """Copy measured nucleotide columns and add 0-1 reward scores without invoking tools.

    Continuous length reward uses the four-point envelope. The optional nucleotide-pass
    score uses exactly the hard filters in ``qc.nucleotide_pass_mask``.
    """
    df = metrics.copy()
    df["reward_valid_nt_chars"] = df["valid_nt_chars"].astype(float)
    df["reward_genome_length"] = df["genome_length"].map(lambda value: score_genome_length(value, config))
    df["reward_gc_content"] = df["gc_content"].map(
        lambda value: _interval_score(value, config.gc_content_min, config.gc_content_max)
    )
    df["reward_nt_homopolymer"] = df["max_nt_homopolymer_length"].map(
        lambda value: _upper_bound_ratio_score(value, config.homopolymer_max)
    )
    df["reward_dustmask_end"] = df["dustmask_max_end_masked_fraction"].map(
        lambda value: _upper_bound_ratio_score(value, config.dustmask_max_end_fraction)
    )
    df["reward_nucleotide_pass"] = nucleotide_pass_mask(df, config).astype(float)
    return df


def score_sequences(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    external_qc: ExternalQCRewardConfig | None = None,
    mmseqs_cluster_diversity: MMseqsClusterDiversityConfig | None = None,
    sequence_safety: SequenceSafetyRewardConfig | None = None,
) -> pd.DataFrame:
    """Score sequences with nucleotide QC, optional external QC, and optional batch diversity."""
    timings: dict[str, float] = {"reward/begin_unix_s": time.time()}
    reward_start = time.perf_counter()

    phase_start = time.perf_counter()
    df = add_nucleotide_metrics(sequences_df, config=config)
    _record_elapsed(timings, "reward/nucleotide_qc_s", phase_start)

    phase_start = time.perf_counter()
    df = add_nucleotide_rewards(df, config)
    _record_elapsed(timings, "reward/nucleotide_reward_scores_s", phase_start)

    if external_qc and external_qc.enabled:
        phase_start = time.perf_counter()
        df = add_external_qc_rewards(df, external_qc)
        timings.setdefault("reward/external_qc/total_s", time.perf_counter() - phase_start)
    if mmseqs_cluster_diversity and mmseqs_cluster_diversity.enabled:
        phase_start = time.perf_counter()
        df = add_mmseqs_cluster_diversity_rewards(df, config, mmseqs_cluster_diversity)
        _record_elapsed(timings, "reward/mmseqs_cluster_diversity_s", phase_start)

    if sequence_safety is None:
        df = _add_unavailable_sequence_safety_rewards(df, reason_code="SEQUENCE_SAFETY_CONFIG_MISSING")
    elif not _sequence_safety_config_is_valid(sequence_safety):
        df = _add_unavailable_sequence_safety_rewards(df, reason_code="SEQUENCE_SAFETY_CONFIG_INVALID")
    elif not sequence_safety.enabled:
        df = _add_unavailable_sequence_safety_rewards(
            df,
            reason_code="SEQUENCE_SAFETY_DISABLED",
            required_by_class=_sequence_safety_required_by_class(sequence_safety),
            strict_lysis=sequence_safety.strict_lysis,
        )
    else:
        phase_start = time.perf_counter()
        df = add_sequence_safety_rewards(df, sequence_safety)
        _record_elapsed(timings, "reward/sequence_safety_s", phase_start)

    phase_start = time.perf_counter()
    df = aggregate_rewards(df, weights)
    _record_elapsed(timings, "reward/aggregate_s", phase_start)
    timings["reward/end_unix_s"] = time.time()
    timings["reward/total_s"] = time.perf_counter() - reward_start
    return _attach_timing_columns(df, timings)


def _write_external_qc_config(
    base_config_path: Path,
    run_dir: Path,
    input_fasta: Path,
    external_qc: ExternalQCRewardConfig,
) -> Path:
    """Write an Arc pipeline config for one RL reward batch."""
    config = yaml.safe_load(base_config_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = run_dir / "arc_external_qc.yaml"
    config["results_save_dir"] = str(run_dir)
    config["current_config_file"] = str(run_config_path)
    config["evo_gen_seqs_fasta_file_save_location"] = str(input_fasta)
    config["overwrite_sequence_ids"] = True
    config["online_measurement_mode"] = True
    # Length remains an independent online reward and final acceptance gate. Do not
    # suppress otherwise usable protein and architecture evidence for an outlier.
    config["genome_length_filter"] = False
    for key in ARC_PATH_KEYS:
        if config.get(key):
            config[key] = str(_repo_path(config[key]))

    if external_qc.enable_gene_a_origin and not external_qc.enable_smooth_reference_rewards:
        raise ValueError("enable_gene_a_origin requires enable_smooth_reference_rewards")
    full_synteny_enabled = external_qc.enable_synteny
    smooth_reference_enabled = bool(
        external_qc.enable_smooth_reference_rewards
        and (external_qc.enable_synteny or external_qc.enable_tropism or external_qc.enable_gene_a_origin)
    )
    paper_synteny_stage_enabled = bool(
        full_synteny_enabled or external_qc.enable_average_protein_identity or external_qc.enable_required_genes
    )

    orf_enabled = external_qc.enable_orf or external_qc.enable_coding_density
    homology_enabled = (
        external_qc.enable_protein_hit_count
        or external_qc.enable_tropism
        or external_qc.enable_synteny
        or external_qc.enable_average_protein_identity
        or external_qc.enable_required_genes
        or external_qc.enable_gene_a_origin
    )

    config["orf_filtering"] = bool(orf_enabled)
    config["prodigal_based_filters"] = bool(orf_enabled)
    config["orf_count_filter"] = bool(external_qc.enable_orf)
    config["orf_lengths_filter"] = bool(external_qc.enable_orf)
    config["coding_density_filter"] = bool(external_qc.enable_coding_density)
    config["aminoacid_homopolymer_length_filter"] = bool(external_qc.enable_orf)

    config["homology_filtering"] = bool(homology_enabled)
    config["use_orf_filtered_df"] = bool(orf_enabled)
    config["use_nucleotide_filtered_df_instead"] = not bool(orf_enabled)
    config["protein_database_hit_count_filter"] = bool(
        external_qc.enable_protein_hit_count or paper_synteny_stage_enabled
    )
    config["training_data_sequence_identity_filter"] = False
    config["genetic_architecture_filter"] = False
    config["tropism_protein_sequence_identity_filter"] = bool(external_qc.enable_tropism)
    config["checkv_filter"] = False

    config["diversification_filtering"] = False
    config["use_homology_filtered_df"] = True
    config["use_orf_filtered_df_instead"] = False
    config["use_nucleotide_filtered_df_instead_2"] = False
    config["mmseqs_clustering_filter"] = False
    config["mmseqs_reference_genome_sequence_identity_remove_filter"] = False
    config["genetic_architecture_remove_filter"] = False
    config["genetic_architecture_visualization_and_synteny_filtering"] = paper_synteny_stage_enabled
    config["average_protein_sequence_identity_filter"] = bool(external_qc.enable_average_protein_identity)
    config["required_genes_filter"] = bool(external_qc.enable_required_genes)
    config["syntenic_gene_count_filter"] = full_synteny_enabled
    if external_qc.lovis4u_parallel_jobs is not None:
        parallel_jobs = max(1, int(external_qc.lovis4u_parallel_jobs))
        config["lovis4u_parallel_jobs"] = parallel_jobs
        config["n_parallel_jobs"] = parallel_jobs
    if external_qc.lovis4u_chunk_size is not None:
        chunk_size = max(1, int(external_qc.lovis4u_chunk_size))
    elif external_qc.lovis4u_parallel_jobs is not None:
        chunk_size = max(1, int(external_qc.lovis4u_parallel_jobs))
    else:
        chunk_size = int(config.get("chunk_size", 10))
    config["lovis4u_chunk_size"] = chunk_size
    config["chunk_size"] = chunk_size
    if external_qc.lovis4u_mmseqs_threads is not None:
        config["lovis4u_mmseqs_threads"] = max(1, int(external_qc.lovis4u_mmseqs_threads))
    config["lovis4u_metrics_only"] = bool(external_qc.lovis4u_metrics_only)
    config["lovis4u_collect_pdfs"] = bool(external_qc.lovis4u_collect_pdfs)
    config["protein_match_min_reciprocal_coverage"] = float(external_qc.protein_match_min_reciprocal_coverage)
    config["tropism_match_min_reciprocal_coverage"] = float(external_qc.tropism_match_min_reciprocal_coverage)
    if full_synteny_enabled or smooth_reference_enabled:
        reference_gff = Path(config["reference_genome_gff_file_save_location"])
        staged_reference_gff = run_dir / "reference_genome.coordinate_normalized.gff"
        circular_genome_length = None
        reference_fasta_value = config.get("genetic_architecture_reference_genome") or config.get(
            "reference_genome_fasta"
        )
        if smooth_reference_enabled and (not reference_fasta_value or not Path(reference_fasta_value).exists()):
            raise FileNotFoundError("Smooth reference rewards require the configured reference FASTA")
        if reference_fasta_value and Path(reference_fasta_value).exists():
            reference_records = load_fasta_records(Path(reference_fasta_value), keep_only_up_to_first_eos=False)
            if len(reference_records) != 1:
                raise ValueError("The reference FASTA must contain exactly one sequence")
            circular_genome_length = len(str(reference_records.iloc[0]["sequence"]))
        stage_coordinate_normalized_reference_gff(
            reference_gff,
            staged_reference_gff,
            circular_genome_length=circular_genome_length,
        )
        config["smooth_reference_genome_gff_file"] = str(staged_reference_gff)
        if full_synteny_enabled:
            config["reference_genome_gff_file_save_location"] = str(staged_reference_gff)
    if paper_synteny_stage_enabled:
        config["use_reference_genome"] = full_synteny_enabled
        if not full_synteny_enabled and not bool(config.get("allow_gff_product_order_synteny_fallback", False)):
            config["reference_genome_gff_file_save_location"] = None
        config.setdefault(
            "average_protein_sequence_identity_metrics_file_save_location",
            "qc6_average_protein_sequence_identity_metrics.csv",
        )
        config.setdefault("required_genes_metrics_file_save_location", "qc6_required_genes_metrics.csv")
        config.setdefault("synteny_metrics_file_save_location", "qc6_synteny_filter_metrics.csv")
        config["required_genes_evidence_target"] = float(external_qc.required_genes_evidence_target)

    run_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return run_config_path


def _add_smooth_reference_rewards(
    scored_df: pd.DataFrame,
    *,
    run_dir: Path,
    input_fasta: Path,
    config: dict,
    external_qc: ExternalQCRewardConfig,
) -> pd.DataFrame:
    """Run one permissive reference-to-called-ORF search and add graded rewards."""
    reference_gff_value = config.get("smooth_reference_genome_gff_file")
    protein_orfs_value = config.get("orfipy_proteins_file_save_location")
    nucleotide_orfs_value = config.get("orfipy_orfs_file_save_location")
    if not reference_gff_value or not protein_orfs_value or not nucleotide_orfs_value:
        raise ValueError("Smooth reference rewards require a staged reference GFF and both ORFipy FASTAs")
    reference_gff = Path(reference_gff_value)
    protein_orfs = run_dir / str(protein_orfs_value)
    nucleotide_orfs = run_dir / str(nucleotide_orfs_value)
    if not reference_gff.exists() or not protein_orfs.exists() or not nucleotide_orfs.exists():
        raise FileNotFoundError("Smooth reference reward input artifact is missing")

    reference_proteins = run_dir / "smooth_reference_proteins.fasta"
    reference_order = write_reference_protein_fasta(reference_gff, reference_proteins)
    candidate_orf_sequences, candidate_orders = load_candidate_orf_context(nucleotide_orfs)
    protein_orf_ids = set(_fasta_header_ids(protein_orfs))
    if protein_orf_ids != set(candidate_orf_sequences):
        raise ValueError("ORFipy nucleotide and protein FASTAs contain different record IDs")
    required_reference_loci = set()
    if external_qc.enable_tropism:
        required_reference_loci.add(external_qc.tropism_reference_locus)
    if external_qc.enable_gene_a_origin:
        required_reference_loci.add(external_qc.gene_a_reference_locus)
    missing_reference_loci = required_reference_loci - set(reference_order)
    if missing_reference_loci:
        raise ValueError(f"Smooth reference loci are absent from the staged GFF: {sorted(missing_reference_loci)}")

    hits_path = run_dir / "smooth_reference_hits.tsv"
    if protein_orf_ids:
        subprocess.run(
            _smooth_reference_search_command(
                reference_fasta=reference_proteins,
                candidate_fasta=protein_orfs,
                output_tsv=hits_path,
                temporary_dir=run_dir / "smooth_reference_mmseqs_tmp",
                threads=external_qc.lovis4u_mmseqs_threads or 1,
            ),
            check=True,
            env=_external_qc_env(external_qc),
            timeout=external_qc.timeout_seconds,
        )
    hit_columns = ["query", "target", "evalue", "pident", "alnlen", "qlen", "tlen", "qcov", "tcov"]
    if hits_path.exists() and hits_path.stat().st_size:
        hits_df = pd.read_csv(hits_path, sep="\t", header=None, names=hit_columns)
    else:
        hits_df = pd.DataFrame(columns=hit_columns)

    genomes_df = load_fasta_records(input_fasta, keep_only_up_to_first_eos=False)
    genome_sequences = dict(zip(genomes_df["id_prompt"].astype(str), genomes_df["sequence"].astype(str), strict=True))
    summary = summarize_smooth_reference_evidence(
        hits_df,
        genome_sequences=genome_sequences,
        candidate_orf_sequences=candidate_orf_sequences,
        candidate_orders=candidate_orders,
        reference_order=reference_order,
        synteny_match_parameters={
            "identity_full_credit": external_qc.synteny_identity_full_credit,
            "reference_coverage_full_credit": external_qc.synteny_reciprocal_coverage_full_credit,
            "candidate_coverage_full_credit": external_qc.synteny_reciprocal_coverage_full_credit,
            "gamma": external_qc.synteny_integrity_gamma,
            "raw_integrity_min": external_qc.synteny_raw_integrity_min,
            "min_credit": external_qc.synteny_min_credit,
        },
        tropism_match_parameters={
            "identity_full_credit": external_qc.tropism_identity_full_credit,
            "reference_coverage_full_credit": external_qc.tropism_reciprocal_coverage_full_credit,
            "candidate_coverage_full_credit": external_qc.tropism_reciprocal_coverage_full_credit,
            "gamma": external_qc.tropism_integrity_gamma,
            "raw_integrity_min": external_qc.tropism_raw_integrity_min,
            "min_credit": external_qc.tropism_min_credit,
        },
        synteny_order_weight=external_qc.synteny_order_weight,
        synteny_duplicate_penalty_weight=external_qc.synteny_duplicate_penalty_weight,
        gene_a_reference_locus=external_qc.gene_a_reference_locus,
        tropism_reference_locus=external_qc.tropism_reference_locus,
        gene_a_origin_motif=external_qc.gene_a_origin_motif,
        gene_a_origin_offset_nt=external_qc.gene_a_origin_offset_nt,
        gene_a_origin_offset_tolerance_nt=external_qc.gene_a_origin_offset_tolerance_nt,
    ).set_index("id_prompt")
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    row_ids = scored_df[id_column].astype(str)
    reward_columns = set()
    if external_qc.enable_synteny:
        reward_columns.add("reward_external_synteny")
    if external_qc.enable_tropism:
        reward_columns.add("reward_external_tropism")
    if external_qc.enable_gene_a_origin:
        reward_columns.add("reward_gene_a_origin")
    telemetry_columns = set(summary.columns) - {
        "reward_external_synteny",
        "reward_external_tropism",
        "reward_gene_a_origin",
    }
    for column in sorted(reward_columns | telemetry_columns):
        scored_df[column] = row_ids.map(summary[column]).fillna(0.0)
    scored_df["smooth_reference_stage_reached"] = 1.0
    scored_df["smooth_reference_measurement_available"] = 1.0
    scored_df["smooth_reference_missing_artifact"] = 0.0
    if external_qc.enable_gene_a_origin:
        scored_df["reward_gene_a_origin_pass"] = scored_df["reward_gene_a_origin"].eq(1.0).astype(float)
    return scored_df


def _sequence_ids_from_csv(path: Path) -> set[str]:
    """Read Arc output IDs from a staged CSV file."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "id_prompt" not in df:
        return set()
    return set(df["id_prompt"].astype(str))


def _genome_ids_from_orf_hits(hits_df: pd.DataFrame) -> pd.Series:
    """Map Arc ORF-level MMseqs query IDs back to genome IDs."""
    return hits_df["id_prompt"].astype(str).str.split("_").str[:-1].str.join("_")


def _fasta_header_ids(path: Path) -> list[str]:
    """Read FASTA record IDs without loading sequence payloads."""
    if not path.exists():
        return []
    with path.open() as handle:
        return [line[1:].strip().split()[0] for line in handle if line.startswith(">")]


def _as_arc_pass_mask(scored_df: pd.DataFrame, pass_ids: set[str]) -> pd.Series:
    """Return a mask for Arc UMI IDs while preserving original IDs in output."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    return scored_df[id_column].astype(str).isin(pass_ids)


def _add_full_synteny_rewards(scored_df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    """Score complete Arc/LoVis measurements without inventing missing architecture evidence."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    scored_df["synteny_stage_reached"] = 0.0
    scored_df["synteny_measurement_available"] = 0.0
    scored_df["synteny_missing_artifact"] = 1.0
    scored_df["reward_external_synteny"] = 0.0
    scored_df["reward_external_synteny_pass"] = 0.0
    metrics_path = run_dir / config.get("synteny_metrics_file_save_location", "qc6_synteny_filter_metrics.csv")
    if not metrics_path.is_file():
        return scored_df
    metrics_df = pd.read_csv(metrics_path)
    numeric_columns = (
        "num_syntenic_genes",
        "total_num_genes",
        "reference_num_genes",
        "duplicate_reference_gene_count",
        "reference_order_violation_count",
    )
    if not {"id_prompt", "missing_synteny_output", *numeric_columns}.issubset(metrics_df.columns):
        return scored_df

    metrics_by_id = metrics_df.set_index(metrics_df["id_prompt"].astype(str))
    row_ids = scored_df[id_column].astype(str)
    stage_reached = row_ids.isin(metrics_by_id.index)
    measured = stage_reached & row_ids.map(metrics_by_id["missing_synteny_output"]).eq(False)
    for column in numeric_columns:
        scored_df[column] = row_ids.map(pd.to_numeric(metrics_by_id[column], errors="coerce"))
        measured &= scored_df[column].map(lambda value: pd.notna(value) and math.isfinite(value) and value >= 0)
    measured &= scored_df["reference_num_genes"].gt(0)
    scored_df["synteny_stage_reached"] = stage_reached.astype(float)
    scored_df["synteny_measurement_available"] = measured.astype(float)
    scored_df["synteny_missing_artifact"] = (~measured).astype(float)
    scores = [
        score_synteny_counts(float(count), float(reference), float(duplicates), float(order))
        if available
        else (0.0, pd.NA, pd.NA, pd.NA)
        for count, reference, duplicates, order, available in zip(
            scored_df["num_syntenic_genes"],
            scored_df["reference_num_genes"],
            scored_df["duplicate_reference_gene_count"],
            scored_df["reference_order_violation_count"],
            measured,
            strict=True,
        )
    ]
    scored_df["reward_external_synteny"] = [score for score, _, _, _ in scores]
    scored_df["synteny_reference_coverage_score"] = [coverage for _, coverage, _, _ in scores]
    scored_df["synteny_copy_balance_score"] = [balance for _, _, balance, _ in scores]
    scored_df["synteny_reference_deficit"] = [deficit for _, _, _, deficit in scores]
    scored_df["synteny_order_score"] = (1.0 / (1.0 + scored_df["reference_order_violation_count"])).where(measured)
    scored_df["reward_external_synteny_pass"] = (
        measured & reference_synteny_pass_mask(scored_df, config.get("synteny_max_missing_reference_genes", 0))
    ).astype(float)
    return scored_df


def _add_average_protein_identity_rewards(
    scored_df: pd.DataFrame,
    run_dir: Path,
    config: dict,
) -> pd.DataFrame:
    """Add continuous rewards for Arc's average protein percent-identity filter."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    scored_df["average_protein_identity_stage_reached"] = 0.0
    scored_df["average_protein_identity_measurement_available"] = 0.0
    scored_df["average_protein_identity_missing_artifact"] = 0.0
    metrics_path = run_dir / config.get(
        "average_protein_sequence_identity_metrics_file_save_location",
        "qc6_average_protein_sequence_identity_metrics.csv",
    )
    if not metrics_path.is_file():
        scored_df["average_protein_identity_missing_artifact"] = 1.0
        return scored_df
    metrics_df = pd.read_csv(metrics_path)
    if not {"id_prompt", "average_protein_percent_identity", "average_protein_identity_gene_count"}.issubset(
        metrics_df.columns
    ):
        scored_df["average_protein_identity_missing_artifact"] = 1.0
        return scored_df

    metrics_by_id = metrics_df.set_index(metrics_df["id_prompt"].astype(str))
    row_ids = scored_df[id_column].astype(str)
    mapped_identity = row_ids.map(pd.to_numeric(metrics_by_id["average_protein_percent_identity"], errors="coerce"))
    mapped_evidence = row_ids.map(pd.to_numeric(metrics_by_id["average_protein_identity_gene_count"], errors="coerce"))
    has_identity_metric = mapped_identity.between(0.0, 100.0) & mapped_evidence.map(
        lambda value: pd.notna(value) and math.isfinite(value) and value >= 0.0
    )
    scored_df["average_protein_identity_stage_reached"] = row_ids.isin(metrics_by_id.index).astype(float)
    scored_df["average_protein_identity_measurement_available"] = has_identity_metric.astype(float)
    scored_df["average_protein_identity_missing_artifact"] = (~has_identity_metric).astype(float)
    scored_df["average_protein_percent_identity"] = mapped_identity.where(has_identity_metric, 0.0)
    scored_df["average_protein_identity_gene_count"] = mapped_evidence.where(has_identity_metric, 0.0)

    lower, upper = config.get("average_protein_sequence_identity_range", [0, 95])
    novelty_scores = mapped_identity.map(lambda value: score_aai_novelty(float(value)) if pd.notna(value) else 0.0)
    evidence_scores = mapped_evidence.map(lambda value: score_aai_evidence(float(value)) if pd.notna(value) else 0.0)
    scored_df["average_protein_identity_novelty_score"] = novelty_scores
    scored_df["average_protein_identity_evidence_score"] = evidence_scores
    scored_df["reward_external_average_protein_identity"] = (novelty_scores * evidence_scores).where(
        has_identity_metric,
        0.0,
    )
    scored_df["reward_external_average_protein_identity_pass"] = (
        has_identity_metric & (mapped_evidence > 0) & mapped_identity.between(float(lower), float(upper))
    ).astype(float)
    return scored_df


def _add_required_gene_rewards(
    scored_df: pd.DataFrame,
    run_dir: Path,
    config: dict,
    evidence_target: float = 10.0,
) -> pd.DataFrame:
    """Add continuous rewards for Arc's required-gene annotation filter."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    scored_df["required_genes_stage_reached"] = 0.0
    scored_df["required_genes_measurement_available"] = 0.0
    scored_df["required_genes_missing_artifact"] = 0.0
    metrics_path = run_dir / config.get("required_genes_metrics_file_save_location", "qc6_required_genes_metrics.csv")
    if not metrics_path.is_file():
        scored_df["required_genes_missing_artifact"] = 1.0
        return scored_df

    metrics_df = pd.read_csv(metrics_path)
    required_columns = {
        "id_prompt",
        "required_genes_matched_count",
        "required_genes_total_count",
        "required_genes_integrity_sum",
        "required_genes_full_length_count",
    }
    if not required_columns.issubset(metrics_df.columns):
        scored_df["required_genes_missing_artifact"] = 1.0
        return scored_df

    metrics_df = metrics_df.copy()
    for column in [
        "required_genes_matched_count",
        "required_genes_total_count",
        "required_genes_integrity_sum",
        "required_genes_full_length_count",
    ]:
        metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce").fillna(0.0)
    metrics_by_id = metrics_df.set_index(metrics_df["id_prompt"].astype(str))
    mapped_matched = scored_df[id_column].astype(str).map(metrics_by_id["required_genes_matched_count"])
    mapped_total = scored_df[id_column].astype(str).map(metrics_by_id["required_genes_total_count"])
    mapped_integrity = scored_df[id_column].astype(str).map(metrics_by_id["required_genes_integrity_sum"])
    mapped_full_length = scored_df[id_column].astype(str).map(metrics_by_id["required_genes_full_length_count"])
    has_required_gene_metric = (
        mapped_matched.notna() & mapped_total.notna() & mapped_integrity.notna() & mapped_full_length.notna()
    )
    scored_df["required_genes_stage_reached"] = (
        scored_df[id_column].astype(str).isin(metrics_by_id.index).astype(float)
    )
    scored_df["required_genes_measurement_available"] = has_required_gene_metric.astype(float)
    scored_df["required_genes_matched_count"] = mapped_matched.fillna(0.0)
    scored_df["required_genes_total_count"] = mapped_total.fillna(0.0)
    scored_df["required_genes_integrity_sum"] = mapped_integrity.fillna(0.0)
    scored_df["required_genes_full_length_count"] = mapped_full_length.fillna(0.0)
    scored_df["required_genes_raw_score"] = [
        0.0 if (not has_metric or total <= 0) else max(0.0, min(1.0, integrity / total))
        for integrity, total, has_metric in zip(
            scored_df["required_genes_integrity_sum"],
            scored_df["required_genes_total_count"],
            has_required_gene_metric,
            strict=False,
        )
    ]
    scored_df["required_genes_evidence_score"] = (
        scored_df["required_genes_total_count"]
        .map(lambda total: max(0.0, min(1.0, float(total) / max(float(evidence_target), 1.0))))
        .where(has_required_gene_metric & (scored_df["required_genes_total_count"] > 0), 0.0)
    )
    scored_df["reward_external_required_genes"] = (
        scored_df["required_genes_raw_score"] * scored_df["required_genes_evidence_score"]
    )
    scored_df["reward_external_required_genes_pass"] = (
        has_required_gene_metric
        & (scored_df["required_genes_total_count"] > 0)
        & (scored_df["required_genes_full_length_count"] >= scored_df["required_genes_total_count"])
    ).astype(float)
    return scored_df


def _add_mmseqs_hit_rewards(scored_df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    """Add protein-hit-count and tropism rewards from Arc MMseqs outputs."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    minimum_reciprocal_coverage = float(config.get("protein_match_min_reciprocal_coverage", 0.75))
    phrogs_dir = config.get("mmseqs_protein_database_results_dir_save_location")
    phrogs_hits_path = run_dir / phrogs_dir / "mmseqs2_hits.csv" if phrogs_dir else None
    scored_df["protein_database_hit_count_stage_reached"] = 0.0
    scored_df["protein_database_hit_count_measurement_available"] = 0.0
    scored_df["protein_database_hit_count_missing_artifact"] = 0.0
    scored_df["protein_database_hit_count_hit_present"] = 0.0
    scored_df["protein_database_alignment_evidence_available"] = 0.0
    scored_df["protein_database_hit_count"] = 0
    scored_df["protein_database_unique_family_count"] = 0
    scored_df["protein_database_effective_family_count"] = 0.0
    scored_df["protein_database_full_length_family_count"] = 0
    scored_df["reward_external_protein_hit_count"] = 0.0
    scored_df["reward_external_protein_hit_count_pass"] = 0.0
    if phrogs_hits_path and phrogs_hits_path.exists():
        scored_df["protein_database_hit_count_stage_reached"] = 1.0
        hits_df = pd.read_csv(phrogs_hits_path)
        if {"id_prompt", "protein_database_mmseqs_target"}.issubset(hits_df.columns):
            genome_counts = _genome_ids_from_orf_hits(hits_df).value_counts()
            scored_df["protein_database_hit_count"] = scored_df[id_column].map(genome_counts).fillna(0).astype(int)
            scored_df["protein_database_hit_count_hit_present"] = (scored_df["protein_database_hit_count"] > 0).astype(
                float
            )
            hits_df, alignment_evidence_available = _add_protein_alignment_evidence(hits_df, "protein_database")
            alignment_evidence_available = bool(hits_df.empty or alignment_evidence_available)
            scored_df["protein_database_alignment_evidence_available"] = float(alignment_evidence_available)
            scored_df["protein_database_hit_count_measurement_available"] = float(alignment_evidence_available)
            if alignment_evidence_available and not hits_df.empty:
                hits_df["genome_id"] = _genome_ids_from_orf_hits(hits_df)
                hits_df["protein_database_full_length_hit"] = (
                    hits_df["protein_database_mmseqs_query_coverage"] >= minimum_reciprocal_coverage
                ) & (hits_df["protein_database_mmseqs_target_coverage"] >= minimum_reciprocal_coverage)
                family_hits = (
                    hits_df.groupby(["genome_id", "protein_database_mmseqs_target"], as_index=False)
                    .agg(
                        protein_database_alignment_integrity=("protein_database_alignment_integrity", "max"),
                        protein_database_full_length_hit=("protein_database_full_length_hit", "max"),
                    )
                    .reset_index(drop=True)
                )
                family_metrics = family_hits.groupby("genome_id").agg(
                    protein_database_unique_family_count=("protein_database_mmseqs_target", "nunique"),
                    protein_database_effective_family_count=("protein_database_alignment_integrity", "sum"),
                    protein_database_full_length_family_count=("protein_database_full_length_hit", "sum"),
                )
                scored_df["protein_database_unique_family_count"] = (
                    scored_df[id_column]
                    .map(family_metrics["protein_database_unique_family_count"])
                    .fillna(0)
                    .astype(int)
                )
                scored_df["protein_database_effective_family_count"] = (
                    scored_df[id_column].map(family_metrics["protein_database_effective_family_count"]).fillna(0.0)
                )
                scored_df["protein_database_full_length_family_count"] = (
                    scored_df[id_column]
                    .map(family_metrics["protein_database_full_length_family_count"])
                    .fillna(0)
                    .astype(int)
                )
            min_hits = int(config.get("protein_database_hit_count", 7))
            scored_df["reward_external_protein_hit_count"] = scored_df["protein_database_effective_family_count"].map(
                lambda value: _lower_bound_ratio_score(float(value), float(min_hits))
            )
            scored_df["reward_external_protein_hit_count_pass"] = (
                scored_df["protein_database_full_length_family_count"] >= min_hits
            ).astype(float)
    elif phrogs_hits_path:
        scored_df["protein_database_hit_count_missing_artifact"] = 1.0

    tropism_dir = config.get("mmseqs_tropism_protein_results_dir_save_location")
    tropism_hits_path = run_dir / tropism_dir / "mmseqs2_hits.csv" if tropism_dir else None
    scored_df["tropism_stage_reached"] = 0.0
    scored_df["tropism_measurement_available"] = 0.0
    scored_df["tropism_missing_artifact"] = 0.0
    scored_df["tropism_hit_present"] = 0.0
    scored_df["tropism_alignment_evidence_available"] = 0.0
    scored_df["tropism_protein_mmseqs_percent_identity"] = 0.0
    scored_df["tropism_protein_min_reciprocal_coverage"] = 0.0
    scored_df["tropism_protein_alignment_integrity"] = 0.0
    scored_df["tropism_protein_measured_hit"] = 0.0
    scored_df["reward_external_tropism"] = 0.0
    scored_df["reward_external_tropism_pass"] = 0.0
    if tropism_hits_path and tropism_hits_path.exists():
        scored_df["tropism_stage_reached"] = 1.0
        hits_df = pd.read_csv(tropism_hits_path)
        if {"id_prompt", "tropism_protein_mmseqs_percent_identity"}.issubset(hits_df.columns):
            hits_df = hits_df.copy()
            hits_df["genome_id"] = _genome_ids_from_orf_hits(hits_df)
            hits_df["tropism_protein_mmseqs_percent_identity"] = pd.to_numeric(
                hits_df["tropism_protein_mmseqs_percent_identity"], errors="coerce"
            ).fillna(0.0)
            lower, _upper = config.get("tropism_protein_sequence_identity_range", [60, 100])
            tropism_minimum_coverage = float(config.get("tropism_match_min_reciprocal_coverage", 0.95))
            hits_df, alignment_evidence_available = _add_protein_alignment_evidence(hits_df, "tropism_protein")
            alignment_evidence_available = bool(hits_df.empty or alignment_evidence_available)
            scored_df["tropism_alignment_evidence_available"] = float(alignment_evidence_available)
            scored_df["tropism_measurement_available"] = float(alignment_evidence_available)
            measured_genomes = set(hits_df["genome_id"].astype(str))
            measured_hit = scored_df[id_column].astype(str).isin(measured_genomes)
            scored_df["tropism_protein_measured_hit"] = measured_hit.astype(float)
            scored_df["tropism_hit_present"] = measured_hit.astype(float)
            if alignment_evidence_available and not hits_df.empty:
                hits_df["tropism_reward"] = [
                    score_tropism_identity(identity, True, float(lower)) * coverage
                    for identity, coverage in zip(
                        hits_df["tropism_protein_mmseqs_percent_identity"],
                        hits_df["tropism_protein_min_reciprocal_coverage"],
                        strict=False,
                    )
                ]
                best_hit_indices = hits_df.groupby("genome_id")["tropism_reward"].idxmax()
                best_hits = hits_df.loc[best_hit_indices].set_index("genome_id")
                scored_df["tropism_protein_mmseqs_percent_identity"] = (
                    scored_df[id_column].map(best_hits["tropism_protein_mmseqs_percent_identity"]).fillna(0.0)
                )
                scored_df["tropism_protein_min_reciprocal_coverage"] = (
                    scored_df[id_column].map(best_hits["tropism_protein_min_reciprocal_coverage"]).fillna(0.0)
                )
                scored_df["tropism_protein_alignment_integrity"] = (
                    scored_df[id_column].map(best_hits["tropism_protein_alignment_integrity"]).fillna(0.0)
                )
                scored_df["reward_external_tropism"] = (
                    scored_df[id_column].map(best_hits["tropism_reward"]).fillna(0.0)
                )
                hits_df["tropism_full_length_pass"] = (
                    (hits_df["tropism_protein_mmseqs_percent_identity"] >= float(lower))
                    & (hits_df["tropism_protein_mmseqs_query_coverage"] >= tropism_minimum_coverage)
                    & (hits_df["tropism_protein_mmseqs_target_coverage"] >= tropism_minimum_coverage)
                )
                hard_pass = hits_df.groupby("genome_id")["tropism_full_length_pass"].max()
                scored_df["reward_external_tropism_pass"] = (
                    scored_df[id_column].map(hard_pass).fillna(False).astype(float)
                )
    elif tropism_hits_path:
        scored_df["tropism_missing_artifact"] = 1.0
    return scored_df


def add_external_qc_rewards(
    scored_df: pd.DataFrame,
    external_qc: ExternalQCRewardConfig,
) -> pd.DataFrame:
    """Run Arc external QC on a batch and add binary staged reward columns."""
    timings: dict[str, float] = {"reward/external_qc/begin_unix_s": time.time()}
    external_start = time.perf_counter()

    def finish_timing(df: pd.DataFrame) -> pd.DataFrame:
        timings["reward/external_qc/end_unix_s"] = time.time()
        timings["reward/external_qc/total_s"] = time.perf_counter() - external_start
        return _attach_timing_columns(df, timings)

    base_config_path = _recipe_path(external_qc.config_path)
    pipeline_script = _recipe_path(external_qc.pipeline_script)
    work_dir = _recipe_path(external_qc.work_dir)
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Arc pipeline script not found: {pipeline_script}")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Arc external-QC config not found: {base_config_path}")

    run_dir = work_dir / f"batch_{uuid.uuid4().hex}"
    input_fasta = run_dir / "input_sequences.fasta"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = scored_df.copy()
    for column in [
        "reward_external_orf",
        "reward_external_coding_density",
        "reward_external_protein_hit_count",
        "reward_external_tropism",
        "reward_external_synteny",
        "reward_gene_a_origin",
        "reward_external_average_protein_identity",
        "reward_external_required_genes",
    ]:
        df[column] = 0.0
    if external_qc.enable_synteny:
        df["reward_external_synteny_pass"] = 0.0
    if external_qc.enable_gene_a_origin:
        df["reward_gene_a_origin_pass"] = 0.0
    if external_qc.enable_average_protein_identity:
        df["reward_external_average_protein_identity_pass"] = 0.0
    if external_qc.enable_required_genes:
        df["reward_external_required_genes_pass"] = 0.0
    df["external_qc_tool_succeeded"] = 0.0
    df["external_qc_measurement_available"] = 0.0

    external_qc_failed = False
    try:
        phase_start = time.perf_counter()
        df["arc_qc_id"] = [f"umi{i + 1}" for i in range(len(df))]
        save_fasta(
            df.rename(columns={"id_prompt": "original_id_prompt", "arc_qc_id": "id_prompt"})[
                ["id_prompt", "sequence"]
            ],
            input_fasta,
        )
        run_config_path = _write_external_qc_config(base_config_path, run_dir, input_fasta, external_qc)
        _record_elapsed(timings, "reward/external_qc/prepare_inputs_s", phase_start)
        try:
            timings["reward/external_qc/subprocess_begin_unix_s"] = time.time()
            phase_start = time.perf_counter()
            try:
                subprocess.run(
                    [sys.executable, str(pipeline_script), str(run_config_path)],
                    check=True,
                    cwd=str(pipeline_script.parent),
                    env=_external_qc_env(external_qc),
                    timeout=external_qc.timeout_seconds,
                )
            finally:
                _record_elapsed(timings, "reward/external_qc/subprocess_s", phase_start)
                timings["reward/external_qc/subprocess_end_unix_s"] = time.time()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            external_qc_failed = True
            message = (
                f"Arc external QC failed for {run_dir}; failed artifacts were retained and "
                "external reward columns remain at 0.0"
            )
            if external_qc.fail_on_error:
                raise RuntimeError(message) from exc
            warnings.warn(f"{message}: {exc}", RuntimeWarning, stacklevel=2)
            return finish_timing(df)
        df["external_qc_tool_succeeded"] = 1.0
        df["external_qc_measurement_available"] = 1.0

        phase_start = time.perf_counter()
        config = yaml.safe_load(run_config_path.read_text())
        orf_csv_name = config.get("orf_filter_seqs_csv_file_save_location")
        orf_pass_ids = _sequence_ids_from_csv(run_dir / orf_csv_name) if orf_csv_name else set()
        if external_qc.enable_orf:
            df["reward_external_orf"] = df["arc_qc_id"].astype(str).isin(orf_pass_ids).astype(float)
        if external_qc.enable_coding_density:
            df["reward_external_coding_density"] = df["arc_qc_id"].astype(str).isin(orf_pass_ids).astype(float)
        _record_elapsed(timings, "reward/external_qc/parse_orf_s", phase_start)

        phase_start = time.perf_counter()
        df = _add_mmseqs_hit_rewards(df, run_dir, config)
        _record_elapsed(timings, "reward/external_qc/parse_protein_hit_count_tropism_s", phase_start)
        if external_qc.enable_synteny:
            phase_start = time.perf_counter()
            df = _add_full_synteny_rewards(df, run_dir, config)
            _record_elapsed(timings, "reward/external_qc/parse_synteny_s", phase_start)
        if external_qc.enable_smooth_reference_rewards and (
            external_qc.enable_synteny or external_qc.enable_tropism or external_qc.enable_gene_a_origin
        ):
            phase_start = time.perf_counter()
            try:
                df = _add_smooth_reference_rewards(
                    df,
                    run_dir=run_dir,
                    input_fasta=input_fasta,
                    config=config,
                    external_qc=external_qc,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
                external_qc_failed = True
                df["external_qc_measurement_available"] = 0.0
                df["smooth_reference_stage_reached"] = 1.0
                df["smooth_reference_measurement_available"] = 0.0
                df["smooth_reference_missing_artifact"] = 1.0
                if external_qc.enable_synteny:
                    df["reward_external_synteny"] = 0.0
                if external_qc.enable_tropism:
                    df["reward_external_tropism"] = 0.0
                if external_qc.enable_gene_a_origin:
                    df["reward_gene_a_origin"] = 0.0
                message = f"Smooth reference scoring failed for {run_dir}; artifacts were retained"
                if external_qc.fail_on_error:
                    raise RuntimeError(message) from exc
                warnings.warn(f"{message}: {exc}", RuntimeWarning, stacklevel=2)
            _record_elapsed(timings, "reward/external_qc/smooth_reference_s", phase_start)
        if external_qc.enable_average_protein_identity:
            phase_start = time.perf_counter()
            df = _add_average_protein_identity_rewards(
                df,
                run_dir,
                config,
            )
            _record_elapsed(timings, "reward/external_qc/parse_average_protein_identity_s", phase_start)
        if external_qc.enable_required_genes:
            phase_start = time.perf_counter()
            df = _add_required_gene_rewards(
                df,
                run_dir,
                config,
                external_qc.required_genes_evidence_target,
            )
            _record_elapsed(timings, "reward/external_qc/parse_required_genes_s", phase_start)
    finally:
        phase_start = time.perf_counter()
        if not external_qc.keep_artifacts and not external_qc_failed:
            shutil.rmtree(run_dir, ignore_errors=True)
        _record_elapsed(timings, "reward/external_qc/cleanup_s", phase_start)
    return finish_timing(df)


def score_fasta(
    input_fasta: Path,
    output_csv: Path,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    mmseqs_cluster_diversity: MMseqsClusterDiversityConfig | None = None,
    sequence_safety: SequenceSafetyRewardConfig | None = None,
) -> Path:
    """Score a FASTA file and write per-sequence reward diagnostics."""
    sequences_df = load_fasta_records(input_fasta)
    scored_df = score_sequences(
        sequences_df,
        config=config,
        weights=weights,
        mmseqs_cluster_diversity=mmseqs_cluster_diversity,
        sequence_safety=sequence_safety,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    """CLI entry point for scoring FASTA files with the online reward."""
    parser = argparse.ArgumentParser(description="Score Evo2 phage FASTA sequences with online-safe reward components")
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--genome-length-min", type=int, default=4000, help="Hard length-QC minimum; does not shape reward"
    )
    parser.add_argument(
        "--genome-length-max", type=int, default=6000, help="Hard length-QC maximum; does not shape reward"
    )
    for bound in ("lower_zero", "lower_full", "upper_full", "upper_zero"):
        parser.add_argument(
            f"--genome-length-reward-{bound.replace('_', '-')}",
            type=float,
            default=getattr(NucleotideQCConfig, f"genome_length_reward_{bound}"),
        )
    parser.add_argument("--gc-content-min", type=float, default=30.0)
    parser.add_argument("--gc-content-max", type=float, default=65.0)
    parser.add_argument("--homopolymer-max", type=int, default=10)
    parser.add_argument("--dustmask-filter", action="store_true")
    parser.add_argument("--dustmasker-bin", default="dustmasker")
    parser.add_argument("--dustmask-window", type=int, default=64)
    parser.add_argument("--dustmask-level", type=float, default=20.0)
    parser.add_argument("--dustmask-end-window", type=int, default=200)
    parser.add_argument("--dustmask-max-end-fraction", type=float, default=0.9)
    args = parser.parse_args()

    output = score_fasta(
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        config=NucleotideQCConfig(
            genome_length_min=args.genome_length_min,
            genome_length_max=args.genome_length_max,
            genome_length_reward_lower_zero=args.genome_length_reward_lower_zero,
            genome_length_reward_lower_full=args.genome_length_reward_lower_full,
            genome_length_reward_upper_full=args.genome_length_reward_upper_full,
            genome_length_reward_upper_zero=args.genome_length_reward_upper_zero,
            gc_content_min=args.gc_content_min,
            gc_content_max=args.gc_content_max,
            homopolymer_max=args.homopolymer_max,
            dustmask_filter=args.dustmask_filter,
            dustmasker_bin=args.dustmasker_bin,
            dustmask_window=args.dustmask_window,
            dustmask_level=args.dustmask_level,
            dustmask_end_window=args.dustmask_end_window,
            dustmask_max_end_fraction=args.dustmask_max_end_fraction,
        ),
    )
    print(f"reward_csv: {output}")


if __name__ == "__main__":
    main()
