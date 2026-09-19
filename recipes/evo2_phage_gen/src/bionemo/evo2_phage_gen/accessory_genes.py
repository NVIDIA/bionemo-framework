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

"""Accessory repertoire diversification, separate from core completeness and order.

K may be retained or lost. A supported non-core protein or divergent K homolog
provides additional novelty; redundant copies and more than two accessory types
reduce credit. See configs/accessory_genes.md for the state surface and limits.
"""

import math
import os
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner

from bionemo.evo2_phage_gen.protein_evidence import smooth_protein_match_integrity, write_reference_protein_fasta
from bionemo.evo2_phage_gen.reference_search import run_reference_protein_search


PREFIX = "protein_database_mmseqs_"
ACCESSORY_METRIC_COLUMNS = (
    "accessory_k_credit",
    "accessory_x_credit",
    "accessory_distinct_type_mass",
    "accessory_duplicate_mass",
    "accessory_gene_diversification_base_score",
    "reward_external_accessory_gene_diversification",
    "accessory_gene_diversification_measurement_available",
)


def score_accessory_diversification(k: float, x: float) -> float:
    """Score normalized K completeness and supported accessory novelty in [0, 1].

    With no X, removing K earns linear progress from 0.50 to 0.75. With admitted
    X, either K endpoint is equally desirable. The shallow V has no flat middle:
    moving toward either endpoint helps, and increasing X always helps. At full
    X, endpoints score 1 and half-complete K scores 0.8. The first admitted X
    intentionally changes state, with a nonnegative jump on the K-retaining side.
    These are reward-shaping choices, not estimates of viability.

    Call only with available evidence. k=0 means no admitted K evidence after a
    successful search, not a missing measurement. x can include a supported K
    swap, whose completeness and sequence divergence must belong to the same ORF.
    """
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (k, x)):
        raise ValueError("Accessory completeness credits must be finite values in [0, 1]")
    if x == 0:
        return 0.75 - 0.25 * k
    distance = min(k, 1 - k)
    return (1 - x) * (0.75 - 0.25 * distance) + x * (1 - 0.40 * distance)


def _family(value: object) -> str:
    match = re.fullmatch(r"(?:phrog[:_-]?)?([1-9]\d*)(?:\.0)?", str(value).strip())
    if match is None:
        raise ValueError(f"Accessory evidence requires explicit PHROG families, found {value!r}")
    return f"phrog:{match[1]}"


def _k_divergence(candidate: str, reference: str, full_novelty_identity: float) -> float:
    # Compare to the actual K reference, never to a PHROG consensus or the source
    # organism's name. Gaps alone earn no novelty: an identical truncated K is
    # still non-novel. Family coverage separately supplies completeness credit.
    aligner = PairwiseAligner(
        mode="global", match_score=2, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5
    )
    counts = aligner.align(reference.rstrip("*"), candidate.rstrip("*"))[0].counts()
    paired = counts.identities + counts.mismatches
    if not paired:
        return 0.0
    identity = counts.identities / paired
    return min(1.0, (1.0 - identity) / (1.0 - full_novelty_identity))


def summarize_accessory_gene_evidence(
    hits_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    *,
    reserved_core_orfs: set[str],
    core_families: set[str],
    k_families: set[str],
    candidate_proteins: dict[str, str],
    reference_k_protein: str,
    minimum_reciprocal_coverage: float = 0.75,
    k_full_novelty_identity: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-genome metrics and per-ORF assignments from all admitted PHROGs hits.

    Any positive core-family evidence reserves its ORF, in addition to the caller's
    direct-reference core reservations. Reservations use IDs, not overlapping
    coordinates. Each ORF takes its lowest-E-value family independently of other
    ORFs, preventing weaker alternate labels from hiding duplicate copies. A
    weak secondary K hit does not collapse a distinct family into K. Best-family
    K evidence is retained before core exclusions, so reserving a K-like ORF
    cannot manufacture K absence.

    Each type contributes its strongest completeness q to distinct mass N; other
    copies contribute their q to excess mass D. Final reward is base / (1 +
    max(0, N-2) + D). Unknown ORFs without admitted hits contribute neither novelty
    nor copy mass. Empty ORF cohorts are unavailable, not K deletions.
    """
    if not 0 < minimum_reciprocal_coverage <= 1 or not 0 <= k_full_novelty_identity < 1:
        raise ValueError("Invalid accessory coverage or K identity setting")
    if not reference_k_protein.rstrip("*"):
        raise ValueError("Accessory scoring requires an explicit reference K protein")
    core_families = {_family(value) for value in core_families}
    k_families = {_family(value) for value in k_families}
    if not k_families or core_families & k_families:
        raise ValueError("K families must be nonempty and disjoint from core families")
    numeric = [
        PREFIX + suffix
        for suffix in (
            "e_value",
            "percent_identity",
            "alignment_length",
            "query_length",
            "target_length",
            "query_coverage",
            "target_coverage",
        )
    ]
    required = {"id_prompt", PREFIX + "target", *numeric}
    if not required.issubset(hits_df):
        raise ValueError("Accessory evidence requires native PHROGs alignment coverage and E-values")
    hits = hits_df.copy()
    hits[numeric] = hits[numeric].apply(pd.to_numeric, errors="coerce")
    valid = (
        hits[numeric].notna().all(axis=1)
        & hits[numeric].abs().lt(math.inf).all(axis=1)
        & hits[PREFIX + "e_value"].ge(0)
        & hits[PREFIX + "percent_identity"].between(0, 100)
        & hits[[PREFIX + "alignment_length", PREFIX + "query_length", PREFIX + "target_length"]].gt(0).all(axis=1)
        & hits[PREFIX + "query_coverage"].between(0, 1)
        & hits[PREFIX + "target_coverage"].between(0, 1)
    )
    if not valid.all():
        raise ValueError("Accessory evidence contains invalid alignment measurements")
    hits["family"] = hits[PREFIX + "target"].map(_family)
    hits["credit"] = (
        hits[[PREFIX + "query_coverage", PREFIX + "target_coverage"]].min(axis=1) / minimum_reciprocal_coverage
    ).clip(upper=1.0)
    hits = hits.loc[hits.credit > 0].copy()
    hits["id_prompt"] = hits.id_prompt.astype(str)
    if not set(hits.id_prompt).issubset(candidate_proteins):
        raise ValueError("Accessory hits refer to missing called-protein evidence")
    core_orfs = reserved_core_orfs | set(hits.loc[hits.family.isin(core_families), "id_prompt"])
    # Type follows the strongest match, not a distant shared role: two distinct
    # K-like families remain different extras even if one weakly matches the other.
    chosen = hits.sort_values([PREFIX + "e_value", "family"], kind="stable").drop_duplicates("id_prompt").copy()
    chosen["eligible"] = ~chosen.id_prompt.isin(core_orfs)
    chosen["accessory_type"] = chosen.family.where(~chosen.family.isin(k_families), "K")
    chosen["genome_id"] = chosen.id_prompt.str.rsplit("_", n=1).str[0]
    chosen["k_novelty_credit"] = 0.0
    for index, hit in chosen.loc[chosen.eligible & chosen.accessory_type.eq("K")].iterrows():
        candidate = candidate_proteins[hit.id_prompt]
        if not candidate.rstrip("*"):
            raise ValueError("Accessory K protein sequence is empty")
        chosen.loc[index, "k_novelty_credit"] = hit.credit * _k_divergence(
            candidate, reference_k_protein, k_full_novelty_identity
        )
    measured_genomes = {
        identifier.rsplit("_", 1)[0] for identifier, protein in candidate_proteins.items() if protein.rstrip("*")
    }
    rows = []
    for genome_id in sequences_df.id_prompt.astype(str):
        genome_hits = chosen.loc[chosen.genome_id == genome_id]
        eligible = genome_hits.loc[genome_hits.eligible]
        k = (
            float(genome_hits.loc[genome_hits.accessory_type.eq("K"), "credit"].max())
            if genome_hits.accessory_type.eq("K").any()
            else 0.0
        )
        x = max(
            eligible.loc[eligible.accessory_type.ne("K"), "credit"].max()
            if eligible.accessory_type.ne("K").any()
            else 0.0,
            eligible.k_novelty_credit.max() if not eligible.empty else 0.0,
        )
        types = eligible.groupby("accessory_type").credit.max()
        distinct_mass = float(types.sum())
        duplicate_mass = max(0.0, float(eligible.credit.sum()) - distinct_mass)
        available = genome_id in measured_genomes
        base = score_accessory_diversification(k, float(x)) if available else 0.0
        score = base / (1 + max(0.0, distinct_mass - 2) + duplicate_mass)
        rows.append(
            {
                "id_prompt": genome_id,
                "accessory_k_credit": k,
                "accessory_x_credit": float(x),
                "accessory_distinct_type_mass": distinct_mass,
                "accessory_duplicate_mass": duplicate_mass,
                "accessory_gene_diversification_base_score": base,
                "reward_external_accessory_gene_diversification": score,
                # Match numeric support telemetry consumed by RL readiness.
                "accessory_gene_diversification_measurement_available": float(available),
            }
        )
    return pd.DataFrame(rows, columns=["id_prompt", *ACCESSORY_METRIC_COLUMNS]), chosen


def measure_accessory_gene_artifacts(
    config: dict, sequences_df: pd.DataFrame, *, env: dict[str, str] | None = None
) -> pd.DataFrame:
    """Measure accessory rewards or replay saved evidence without filtering candidates.

    Reuse the batch's exhaustive direct-reference search, also consumed by the
    core/tropism/origin rewards. Search failure propagates rather than becoming K
    absence. All protein evidence comes from the unfiltered called-ORF pool, even
    when final biological/safety screening has reduced the candidate genomes.
    """
    run_dir = Path(config["results_save_dir"])
    metrics_path = run_dir / config.get("accessory_gene_metrics_file_save_location", "qc6_accessory_genes_metrics.csv")
    assignments_path = run_dir / "qc6_accessory_genes_assignments.csv"
    if sequences_df.empty:
        metrics = pd.DataFrame(columns=["id_prompt", *ACCESSORY_METRIC_COLUMNS])
        metrics.to_csv(metrics_path, index=False)
        pd.DataFrame(columns=["id_prompt", "genome_id", "family", "credit", "eligible", "accessory_type"]).to_csv(
            assignments_path, index=False
        )
        return metrics
    functions = config.get("core_gene_reference_functions")
    families = config.get("required_gene_families")
    if not functions or not families or not set(functions.values()).issubset(families):
        raise ValueError("Accessory evidence requires an explicit core reference/function profile")
    reference_gff = Path(
        config.get("smooth_reference_genome_gff_file") or config["reference_genome_gff_file_save_location"]
    )
    reference_fasta = run_dir / "smooth_reference_proteins.fasta"
    write_reference_protein_fasta(reference_gff, reference_fasta)
    reference_proteins = {record.id: str(record.seq) for record in SeqIO.parse(reference_fasta, "fasta")}
    k_reference = config["accessory_gene_k_reference_locus"]
    if k_reference not in reference_proteins:
        raise ValueError(f"Reference K locus {k_reference!r} is absent from the reference annotation")
    protein_fasta = run_dir / config["orfipy_proteins_file_save_location"]
    proteins = {record.id: str(record.seq) for record in SeqIO.parse(protein_fasta, "fasta")}
    family_hits = pd.read_csv(
        run_dir / config["mmseqs_protein_database_results_dir_save_location"] / "mmseqs2_all_hits.csv"
    )
    reference_hits_path = run_dir / "smooth_reference_hits.tsv"
    if proteins and not reference_hits_path.exists():
        run_reference_protein_search(
            reference_fasta=reference_fasta,
            candidate_fasta=protein_fasta,
            output_tsv=reference_hits_path,
            temporary_dir=run_dir / "smooth_reference_mmseqs_tmp",
            threads=config.get("lovis4u_mmseqs_threads") or config.get("mmseqs_threads", 1),
            env=os.environ.copy() if env is None else env,
            timeout=float(config.get("reference_search_timeout_seconds", 1800)),
        )
    reserved = set()
    if reference_hits_path.exists() and reference_hits_path.stat().st_size:
        reference_hits = pd.read_csv(
            reference_hits_path,
            sep="\t",
            header=None,
            names=["query", "target", "evalue", "pident", "alnlen", "qlen", "tlen", "qcov", "tcov"],
        )
        for hit in reference_hits.itertuples(index=False):
            if (
                hit.query in functions
                and smooth_protein_match_integrity(
                    hit.pident,
                    hit.evalue,
                    hit.alnlen,
                    hit.qlen,
                    hit.tlen,
                    reference_coverage=hit.qcov,
                    candidate_coverage=hit.tcov,
                    identity_zero_credit=float(config.get("core_gene_identity_zero_credit", 0.05)),
                    identity_full_credit=float(config.get("core_gene_identity_full_credit", 0.90)),
                    reference_coverage_full_credit=float(
                        config.get("core_gene_reciprocal_coverage_full_credit", 0.95)
                    ),
                    candidate_coverage_full_credit=float(
                        config.get("core_gene_reciprocal_coverage_full_credit", 0.95)
                    ),
                )
                > 0
            ):
                reserved.add(str(hit.target))
    metrics, assignments = summarize_accessory_gene_evidence(
        family_hits,
        sequences_df,
        reserved_core_orfs=reserved,
        core_families={family for allowed in families.values() for family in allowed},
        k_families=set(config["accessory_gene_k_families"]),
        candidate_proteins=proteins,
        reference_k_protein=reference_proteins[k_reference],
        minimum_reciprocal_coverage=float(config.get("accessory_gene_min_reciprocal_coverage", 0.75)),
        k_full_novelty_identity=float(config.get("accessory_gene_k_full_novelty_identity", 0.95)),
    )
    metrics.to_csv(metrics_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    return metrics
