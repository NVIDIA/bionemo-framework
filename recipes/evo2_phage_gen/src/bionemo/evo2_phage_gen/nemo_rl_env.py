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

"""NeMo-RL environment wrapper for online phage sequence rewards."""

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence
from bionemo.evo2_phage_gen.qc import NucleotideQCConfig, trim_at_first_eos
from bionemo.evo2_phage_gen.reward import (
    REWARD_COMPONENTS,
    SAFETY_REVIEW_CREDIT,
    SEQUENCE_SAFETY_CLASSES,
    TIMING_COLUMN_PREFIX,
    ExternalQCRewardConfig,
    MMseqsClusterDiversityConfig,
    RewardWeights,
    SequenceSafetyRewardConfig,
    score_sequences,
)


@dataclass(frozen=True)
class GDPOObjective:
    """One positional GDPO reward objective derived from reward dataframe columns."""

    name: str
    columns: tuple[str, ...]
    reducer: str = "mean"
    requires_safety_eligibility: bool = True

    def __post_init__(self) -> None:
        """Reject bool lookalikes that could silently invert a mandatory gate."""
        if type(self.requires_safety_eligibility) is not bool:
            raise TypeError("GDPO objective requires_safety_eligibility must be a boolean.")


TIMING_COLUMN_ROOT_PREFIX = TIMING_COLUMN_PREFIX.partition("/")[0] + "/"
TIMING_METRIC_MARKER_PREFIX = "__timing__/"


DEFAULT_GDPO_OBJECTIVES: tuple[GDPOObjective, ...] = (
    GDPOObjective(
        name="feasibility",
        columns=(
            "reward_valid_nt_chars",
            "reward_genome_length",
            "reward_gc_content",
            "reward_nt_homopolymer",
            "reward_dustmask_end",
        ),
    ),
    GDPOObjective(
        name="function",
        columns=(
            "reward_external_protein_hit_count",
            "reward_external_tropism",
            "reward_external_required_genes",
            "reward_gene_a_origin",
        ),
    ),
    GDPOObjective(name="architecture", columns=("reward_external_synteny",)),
    GDPOObjective(
        name="novelty",
        columns=(
            "reward_external_average_protein_identity",
            "reward_mmseqs_cluster_diversity",
        ),
    ),
    GDPOObjective(name="safety_amr", columns=("reward_safety_amr",), requires_safety_eligibility=False),
    GDPOObjective(name="safety_toxin", columns=("reward_safety_toxin",), requires_safety_eligibility=False),
    GDPOObjective(name="safety_lysogeny", columns=("reward_safety_lysogeny",), requires_safety_eligibility=False),
)


_PROKARYOTIC_HOST_DOMAINS = frozenset({HostDomain.BACTERIA, HostDomain.ARCHAEA, HostDomain.BACTERIA_AND_ARCHAEA})
_SEQUENCE_SAFETY_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "host_domain",
        "host_evidence",
        "asset_manifest_path",
        "diamond_bin",
        "mmseqs_bin",
        "policy_path",
        "work_dir",
        "strict_lysis",
        "circular",
        "threads",
        "timeout_seconds",
    }
)
_HOST_EVIDENCE_KEYS = frozenset({"source", "source_version", "replication_host_domains", "confirmed", "metadata"})


def _require_keys(mapping: Mapping[str, object], required: frozenset[str], *, label: str) -> None:
    """Require fields used by this recipe while allowing additive config metadata."""
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def _nonempty_string(value: object, *, label: str) -> str:
    """Return one unmodified nonempty string from a strict config field."""
    if type(value) is not str or not value.strip():
        raise TypeError(f"{label} must be a nonempty string.")
    return value


def _config_path(value: object, *, label: str) -> Path:
    """Parse a required nonempty YAML path without truthiness coercion."""
    return Path(_nonempty_string(value, label=label))


def _coerce_sequence_safety_config(raw_config: Any) -> SequenceSafetyRewardConfig | None:
    """Parse the required sequence-safety settings for Task 6."""
    if raw_config is None:
        return None
    if not isinstance(raw_config, Mapping):
        raise TypeError("sequence_safety must be a mapping.")
    _require_keys(raw_config, _SEQUENCE_SAFETY_CONFIG_KEYS, label="sequence_safety")

    for key in ("enabled", "strict_lysis", "circular"):
        if type(raw_config[key]) is not bool:
            raise TypeError(f"sequence_safety {key} must be a boolean.")
    resource_values = {
        "threads": raw_config["threads"],
        "batch_size": raw_config.get("batch_size", 1),
        "orf_workers": raw_config.get("orf_workers", 1),
        "phrogs_threads": raw_config.get("phrogs_threads", raw_config["threads"]),
    }
    for key, value in resource_values.items():
        if type(value) is not int or value < 1:
            raise TypeError(f"sequence_safety {key} must be a positive integer.")
    timeout = raw_config["timeout_seconds"]
    if not isinstance(timeout, Real) or isinstance(timeout, bool) or not math.isfinite(float(timeout)) or timeout <= 0:
        raise TypeError("sequence_safety timeout_seconds must be a positive finite number.")

    try:
        host_domain = HostDomain(_nonempty_string(raw_config["host_domain"], label="sequence_safety host_domain"))
    except ValueError as error:
        raise ValueError("sequence_safety host_domain is unsupported.") from error
    if host_domain not in _PROKARYOTIC_HOST_DOMAINS:
        raise ValueError("sequence_safety host_domain must be prokaryotic.")

    raw_evidence = raw_config["host_evidence"]
    if not isinstance(raw_evidence, Mapping):
        raise TypeError("sequence_safety host_evidence must be a mapping.")
    _require_keys(raw_evidence, _HOST_EVIDENCE_KEYS, label="sequence_safety host_evidence")
    if type(raw_evidence["confirmed"]) is not bool:
        raise TypeError("sequence_safety host_evidence confirmed must be a boolean.")
    if raw_evidence["confirmed"] is not True:
        raise ValueError("sequence_safety host_evidence must be confirmed.")
    raw_domains = raw_evidence["replication_host_domains"]
    if not isinstance(raw_domains, list) or not raw_domains:
        raise TypeError("sequence_safety host_evidence replication_host_domains must be a nonempty list.")
    try:
        evidence_domains = tuple(
            HostDomain(_nonempty_string(domain, label="sequence_safety host evidence domain"))
            for domain in raw_domains
        )
    except ValueError as error:
        raise ValueError("sequence_safety host evidence must contain only prokaryotic domains.") from error
    if len(set(evidence_domains)) != len(evidence_domains) or any(
        domain not in _PROKARYOTIC_HOST_DOMAINS for domain in evidence_domains
    ):
        raise ValueError("sequence_safety host evidence must contain unique prokaryotic domains.")
    if frozenset(evidence_domains) != frozenset({host_domain}):
        raise ValueError("sequence_safety host evidence must be consistent with host_domain.")
    metadata = raw_evidence["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("sequence_safety host_evidence metadata must be a mapping.")

    evidence = HostEvidence(
        source=_nonempty_string(raw_evidence["source"], label="sequence_safety host_evidence source"),
        source_version=_nonempty_string(
            raw_evidence["source_version"], label="sequence_safety host_evidence source_version"
        ),
        replication_host_domains=frozenset(evidence_domains),
        confirmed=True,
        metadata=dict(metadata),
    )
    return SequenceSafetyRewardConfig(
        host_domain=host_domain,
        host_evidence=evidence,
        asset_manifest_path=_config_path(
            raw_config["asset_manifest_path"], label="sequence_safety asset_manifest_path"
        ),
        diamond_bin=_config_path(raw_config["diamond_bin"], label="sequence_safety diamond_bin"),
        mmseqs_bin=_config_path(raw_config["mmseqs_bin"], label="sequence_safety mmseqs_bin"),
        policy_path=_config_path(raw_config["policy_path"], label="sequence_safety policy_path"),
        work_dir=_config_path(raw_config["work_dir"], label="sequence_safety work_dir"),
        enabled=raw_config["enabled"],
        strict_lysis=raw_config["strict_lysis"],
        circular=raw_config["circular"],
        threads=resource_values["threads"],
        batch_size=resource_values["batch_size"],
        orf_workers=resource_values["orf_workers"],
        phrogs_threads=resource_values["phrogs_threads"],
        timeout_seconds=float(timeout),
    )


def _coerce_gdpo_objectives(raw_objectives: Any) -> tuple[GDPOObjective, ...]:
    """Parse GDPO objective config into a stable positional objective list."""
    if raw_objectives is None:
        return DEFAULT_GDPO_OBJECTIVES
    if not isinstance(raw_objectives, list) or not raw_objectives:
        raise TypeError("gdpo_objectives must be a nonempty list.")

    objectives: list[GDPOObjective] = []
    for raw in raw_objectives:
        if not isinstance(raw, Mapping):
            raise TypeError("Each gdpo_objectives entry must be a mapping with name and columns.")
        allowed_keys = {"name", "columns", "reducer", "requires_safety_eligibility"}
        if not set(raw) <= allowed_keys:
            raise ValueError("GDPO objective contains unknown keys.")
        name = raw.get("name")
        raw_columns = raw.get("columns")
        reducer = raw.get("reducer", "mean")
        requires_safety_eligibility = raw.get("requires_safety_eligibility", True)
        if type(name) is not str or not name or name != name.strip():
            raise ValueError("Each gdpo_objectives entry must define a non-empty name and columns list.")
        if not isinstance(raw_columns, list) or not raw_columns:
            raise ValueError("Each gdpo_objectives entry must define a non-empty name and columns list.")
        if any(type(column) is not str or not column or column != column.strip() for column in raw_columns):
            raise ValueError("GDPO objective columns must be nonempty strings.")
        columns = tuple(raw_columns)
        if len(set(columns)) != len(columns):
            raise ValueError("GDPO objective columns must be unique.")
        if type(reducer) is not str or reducer not in {"mean", "product", "min"}:
            raise ValueError("GDPO objective reducer must be 'mean', 'product', or 'min'.")
        if type(requires_safety_eligibility) is not bool:
            raise TypeError("GDPO objective requires_safety_eligibility must be a boolean.")
        objectives.append(
            GDPOObjective(
                name=name,
                columns=columns,
                reducer=reducer,
                requires_safety_eligibility=requires_safety_eligibility,
            )
        )
    names = [objective.name for objective in objectives]
    if len(set(names)) != len(names):
        raise ValueError("GDPO objective names must be unique.")
    return tuple(objectives)


def _exact_safety_eligibility(scored: pd.DataFrame) -> pd.Series:
    """Accept only reconciled PASS plus real, non-bool numeric gate value one."""
    values = scored.get("safety_gate_pass", pd.Series(0.0, index=scored.index))
    states = scored.get("safety_gate_state", pd.Series(None, index=scored.index, dtype=object))
    return states.eq("PASS") & values.map(_is_exact_one)


def _is_exact_finite_real(value: object) -> bool:
    """Return whether a value is a finite real number without bool/string coercion."""
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _is_exact_one(value: object) -> bool:
    """Return whether a value is exactly numeric one without accepting bool lookalikes."""
    return _is_exact_finite_real(value) and value == 1.0


def _is_exact_binary(value: object) -> bool:
    """Return whether a value is exactly numeric zero or one without coercion."""
    return _is_exact_finite_real(value) and value in {0.0, 1.0}


def _is_positive_integer(value: object) -> bool:
    """Return whether a value is a positive integer without bool/string coercion."""
    return _is_exact_finite_real(value) and float(value).is_integer() and value > 0


def _eod_reward_eligibility(scored: pd.DataFrame, *, required: bool) -> pd.Series:
    """Return token-derived EOD eligibility, failing closed when the optional gate is active."""
    if not required:
        return pd.Series(True, index=scored.index, dtype=bool)
    required_columns = {"generation_stopped_on_eod", "generation_capped_without_eod"}
    if not required_columns.issubset(scored.columns):
        raise ValueError("zero_reward_without_eod requires token-derived EOD and generation-cap evidence.")
    stopped = scored["generation_stopped_on_eod"]
    capped = scored["generation_capped_without_eod"]
    if not stopped.map(pd.api.types.is_bool).all() or not capped.map(pd.api.types.is_bool).all():
        raise ValueError("zero_reward_without_eod requires boolean token-derived EOD and generation-cap evidence.")
    stopped = stopped.astype(bool)
    capped = capped.astype(bool)
    if bool((stopped & capped).any()):
        raise ValueError("A generation cannot both stop on EOD and exhaust its token cap.")
    return stopped


def _qualified_scalar_rewards(
    scored: pd.DataFrame,
    *,
    zero_reward_without_eod: bool = False,
) -> pd.Series:
    """Return finite [0, 1] scalar rewards backed by exact safety and optional EOD eligibility."""
    raw_rewards = scored.get("reward", pd.Series(0.0, index=scored.index))
    bounded_rewards = raw_rewards.map(
        lambda value: float(value) if _is_exact_finite_real(value) and 0.0 <= value <= 1.0 else 0.0
    )
    rewards = bounded_rewards.where(_exact_safety_eligibility(scored), 0.0)
    return rewards.where(_eod_reward_eligibility(scored, required=zero_reward_without_eod), 0.0)


def _exact_safety_class_support(scored: pd.DataFrame, reward_column: str) -> pd.Series:
    """Reconcile one safety reward with its raw class state and required flag."""
    safety_class = reward_column.removeprefix("reward_safety_")
    if safety_class not in SEQUENCE_SAFETY_CLASSES:
        return pd.Series(True, index=scored.index, dtype=bool)
    state_column = f"safety_{safety_class}_state"
    required_column = f"safety_{safety_class}_required"
    if state_column not in scored or required_column not in scored:
        return pd.Series(False, index=scored.index, dtype=bool)

    reward_values = scored[reward_column]
    states = scored[state_column]
    required_values = scored[required_column]
    valid_required = required_values.map(_is_exact_binary)
    valid_state = states.isin(("PASS", "FAIL", "INDETERMINATE"))
    expected_rewards = pd.Series(0.0, index=scored.index)
    expected_rewards.loc[valid_required] = (
        (required_values.loc[valid_required] == 0.0) | states.loc[valid_required].eq("PASS")
    ).astype(float)
    measurement_column = f"safety_{safety_class}_measurement_available"
    finding_count_column = f"safety_{safety_class}_finding_count"
    if measurement_column in scored and finding_count_column in scored:
        measured_review = (
            valid_required
            & required_values.eq(1.0)
            & states.eq("INDETERMINATE")
            & scored[measurement_column].map(_is_exact_one)
            & scored[finding_count_column].map(_is_positive_integer)
        )
        expected_rewards.loc[measured_review] = SAFETY_REVIEW_CREDIT
    return reward_values.map(_is_exact_finite_real) & valid_required & valid_state & reward_values.eq(expected_rewards)


def gdpo_objective_scores_from_scored(
    scored: pd.DataFrame,
    objectives: tuple[GDPOObjective, ...],
    *,
    zero_reward_without_eod: bool = False,
) -> pd.DataFrame:
    """Build a positional GDPO reward matrix from scored reward columns."""
    if scored.empty:
        return pd.DataFrame(
            0.0,
            index=scored.index,
            columns=[objective.name for objective in objectives],
            dtype=float,
        )

    objective_scores = pd.DataFrame(index=scored.index)
    eod_eligibility = _eod_reward_eligibility(scored, required=zero_reward_without_eod)
    for objective in objectives:
        missing_columns = [column for column in objective.columns if column not in scored]
        if missing_columns:
            raise ValueError(
                f"GDPO objective {objective.name!r} missing reward column(s): {', '.join(missing_columns)}"
            )
        raw_values = scored[list(objective.columns)]
        valid_components = raw_values.apply(lambda column: column.map(_is_exact_finite_real))
        valid_rows = valid_components.all(axis=1)
        for column in objective.columns:
            valid_rows &= _exact_safety_class_support(scored, column)
        values = raw_values.apply(
            lambda column: column.map(lambda value: float(value) if _is_exact_finite_real(value) else 0.0)
        ).clip(0.0, 1.0)
        if objective.reducer == "mean":
            objective_scores[objective.name] = values.mean(axis=1)
        elif objective.reducer == "product":
            objective_scores[objective.name] = values.prod(axis=1)
        elif objective.reducer == "min":
            objective_scores[objective.name] = values.min(axis=1)
        else:
            raise ValueError(
                f"Unsupported GDPO reducer {objective.reducer!r} for objective {objective.name!r}; "
                "expected 'mean', 'product', or 'min'."
            )
        objective_scores[objective.name] = objective_scores[objective.name].where(valid_rows, 0.0)
        if objective.requires_safety_eligibility:
            objective_scores[objective.name] = objective_scores[objective.name].where(
                _exact_safety_eligibility(scored),
                0.0,
            )
        # Unlike safety eligibility, an explicit no-EOD gate invalidates the whole generated
        # genome. Keep raw safety/QC evidence in ``scored`` for diagnostics, but do not let any
        # objective reward a candidate that never emitted its terminal token.
        objective_scores[objective.name] = objective_scores[objective.name].where(eod_eligibility, 0.0)
    return objective_scores


def _validate_gdpo_safety_objectives(objectives: tuple[GDPOObjective, ...]) -> None:
    """Require exactly the three unmasked safety signals and mask every other objective."""
    required = {
        "safety_amr": ("reward_safety_amr",),
        "safety_toxin": ("reward_safety_toxin",),
        "safety_lysogeny": ("reward_safety_lysogeny",),
    }
    objective_by_name = {objective.name: objective for objective in objectives}
    missing = sorted(set(required) - set(objective_by_name))
    if missing:
        raise ValueError(f"GDPO objectives are missing mandatory safety objectives: {missing}")
    for name, objective in objective_by_name.items():
        if name in required:
            if objective.columns != required[name] or objective.requires_safety_eligibility is not False:
                raise ValueError(f"GDPO safety objective {name!r} must expose its one unmasked safety reward column.")
        elif objective.requires_safety_eligibility is not True:
            raise ValueError(f"GDPO biological objective {name!r} must require safety eligibility.")


try:  # pragma: no cover - exercised only when NeMo-RL is installed.
    import ray
    import torch
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
except ModuleNotFoundError as exc:  # pragma: no cover
    _NEMO_RL_IMPORT_ERROR = exc
else:  # pragma: no cover
    _NEMO_RL_IMPORT_ERROR = None


def extract_assistant_sequence(message_log: list[dict[str, Any]]) -> str:
    """Concatenate assistant messages into the generated DNA sequence."""
    return "".join(str(message["content"]) for message in message_log if message.get("role") == "assistant")


def _prompt_nucleotides(message_log: list[dict[str, Any]]) -> str:
    """Extract DNA bases from user prompts while dropping fine-tuning soft tokens."""
    prompt = "".join(str(message["content"]) for message in message_log if message.get("role") == "user")
    return "".join(char for char in prompt.upper() if char in {"A", "C", "G", "T"})


def extract_scored_sequence(message_log: list[dict[str, Any]]) -> str:
    """Build pure generated DNA for QC while leaving terminal actions in the RL trajectory."""
    completion = trim_at_first_eos(extract_assistant_sequence(message_log))
    return _prompt_nucleotides(message_log) + completion


def score_message_logs(
    message_log_batch: list[list[dict[str, Any]]],
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    external_qc: ExternalQCRewardConfig | None = None,
    mmseqs_cluster_diversity: MMseqsClusterDiversityConfig | None = None,
    sequence_safety: SequenceSafetyRewardConfig | None = None,
) -> pd.DataFrame:
    """Score a NeMo-RL message-log batch with the dependency-light phage reward."""
    sequences_df = pd.DataFrame(
        {
            "id_prompt": [str(i) for i in range(len(message_log_batch))],
            "prompt_nt_length": [len(_prompt_nucleotides(message_log)) for message_log in message_log_batch],
            "prompt_group": [_prompt_nucleotides(message_log) for message_log in message_log_batch],
            "sequence": [extract_scored_sequence(message_log) for message_log in message_log_batch],
        }
    )
    return score_sequences(
        sequences_df,
        config=config,
        weights=weights,
        external_qc=external_qc,
        mmseqs_cluster_diversity=mmseqs_cluster_diversity,
        sequence_safety=sequence_safety,
    )


def _attach_generation_stop_columns(scored: pd.DataFrame, metadata: list[dict[str, Any]]) -> None:
    """Attach compact token-level stop evidence when every rollout row provides it."""
    if len(scored) != len(metadata) or not all(
        isinstance(item, dict)
        and isinstance(item.get("_generation_stopped_on_eod"), bool)
        and isinstance(item.get("_generation_capped_without_eod"), bool)
        for item in metadata
    ):
        return
    stopped_on_eod = [item["_generation_stopped_on_eod"] for item in metadata]
    capped_without_eod = [item["_generation_capped_without_eod"] for item in metadata]
    if any(stopped and capped for stopped, capped in zip(stopped_on_eod, capped_without_eod, strict=True)):
        raise ValueError("A generation cannot both stop on EOD and exhaust its token cap.")
    scored["generation_stopped_on_eod"] = stopped_on_eod
    scored["generation_capped_without_eod"] = capped_without_eod
    prompt_indices = [item.get("prompt_index") for item in metadata]
    if all(isinstance(value, Integral) and not isinstance(value, bool) and value >= 0 for value in prompt_indices):
        scored["generation_prompt_index"] = [int(value) for value in prompt_indices]


def _mean_numeric(scored: pd.DataFrame, column: str) -> float | None:
    """Return a mean only for a completely measured finite column."""
    if column not in scored:
        return None
    values = pd.to_numeric(scored[column], errors="coerce")
    if values.empty or not values.notna().all() or not values.map(math.isfinite).all():
        return None
    return float(values.mean())


def _reward_prompt_group_keys(scored: pd.DataFrame) -> pd.Series | None:
    """Identify one sampled rollout group without merging duplicate prompt records or DP-local indices."""
    if {"generation_prompt_index", "prompt_group"}.issubset(scored.columns):
        return pd.Series(
            list(zip(scored["generation_prompt_index"], scored["prompt_group"], strict=True)),
            index=scored.index,
        )
    if "generation_prompt_index" in scored:
        return scored["generation_prompt_index"]
    if "prompt_group" in scored:
        return scored["prompt_group"]
    return None


def _add_generation_termination_metrics(
    metrics: dict[str, float | int | list[int]],
    scored: pd.DataFrame,
    config: NucleotideQCConfig | None,
) -> None:
    """Add exact stop-class counts and authentic-EOD length placement."""
    required_columns = {"generation_stopped_on_eod", "generation_capped_without_eod"}
    if not required_columns.issubset(scored.columns):
        return

    stopped_on_eod = scored["generation_stopped_on_eod"].astype(bool)
    capped_without_eod = scored["generation_capped_without_eod"].astype(bool)
    if bool((stopped_on_eod & capped_without_eod).any()):
        raise ValueError("A generation cannot both stop on EOD and exhaust its token cap.")
    non_eod_below_cap = ~(stopped_on_eod | capped_without_eod)
    batch_size = len(scored)
    for name, mask in (
        ("authentic_eod", stopped_on_eod),
        ("capped_without_eod", capped_without_eod),
        ("non_eod_below_cap", non_eod_below_cap),
    ):
        metrics[f"termination/{name}_rate"] = float(mask.mean()) if batch_size else 0.0

    group_keys = _reward_prompt_group_keys(scored)
    if group_keys is not None:
        group_has_eod = stopped_on_eod.groupby(group_keys, sort=False).any()
        no_eod_group_count = int((~group_has_eod).sum())
        group_count = len(group_has_eod)
        metrics["termination/no_authentic_eod_prompt_group_rate"] = (
            no_eod_group_count / group_count if group_count else 0.0
        )

    bounds = (
        None
        if config is None
        else (
            config.genome_length_reward_lower_zero,
            config.genome_length_reward_lower_full,
            config.genome_length_reward_upper_full,
            config.genome_length_reward_upper_zero,
        )
    )
    if bounds is None or "genome_length" not in scored:
        return
    lower_zero, lower_full, upper_full, upper_zero = (float(bound) for bound in bounds)
    eod_lengths = pd.to_numeric(scored.loc[stopped_on_eod, "genome_length"], errors="coerce")
    if not eod_lengths.notna().all():
        return

    length_bins = {
        "below_lower_zero": eod_lengths < lower_zero,
        "lower_taper": (eod_lengths >= lower_zero) & (eod_lengths < lower_full),
        "full_credit": (eod_lengths >= lower_full) & (eod_lengths <= upper_full),
        "upper_taper": (eod_lengths > upper_full) & (eod_lengths < upper_zero),
        "at_or_above_upper_zero": eod_lengths >= upper_zero,
    }
    eod_count = int(stopped_on_eod.sum())
    for name, mask in length_bins.items():
        count = int(mask.sum())
        metrics[f"termination/authentic_eod_length/{name}_rate"] = count / eod_count if eod_count else 0.0


def phage_qc_metrics_from_scored(
    scored: pd.DataFrame,
    weights: RewardWeights,
    *,
    config: NucleotideQCConfig | None = None,
    zero_reward_without_eod: bool = False,
    objectives: tuple[GDPOObjective, ...] = (),
    log_by_prompt_nt_length: bool = False,
) -> dict[str, float | int | list[int]]:
    """Summarize configured rewards after safety/EOD gates; never infer final QC acceptance.

    Max-score rates use the fixed reward ceiling of 1, not the observed batch maximum.
    Per-sequence measurements remain in scored artifacts; only inputs to configured
    objectives and the support needed to interpret them are logged here.
    """
    metrics: dict[str, float | int | list[int]] = {
        "num_sequences": len(scored),
        "all_objectives_max_score_rate": 0.0,
    }
    if scored.empty:
        return metrics

    namespace = "gdpo" if objectives else "component"
    if not objectives:
        objectives = tuple(
            GDPOObjective(component.name, (component.score_column,))
            for component in REWARD_COMPONENTS
            if component.weight_attr is not None
            and getattr(weights, component.weight_attr) > 0
            and component.score_column in scored
        )
    scores = gdpo_objective_scores_from_scored(scored, objectives, zero_reward_without_eod=zero_reward_without_eod)
    # Match the float32 tensor returned by step, including rounding at the reward ceiling.
    scores = scores.astype("float32")

    def add_objective_metrics(rows: pd.DataFrame, prefix: str = "") -> None:
        metrics[f"{prefix}all_objectives_max_score_rate"] = (
            float(rows.eq(1.0).all(axis=1).mean()) if len(rows.columns) else 0.0
        )
        for name in rows.columns:
            values = rows[name]
            metrics[f"{prefix}{namespace}/{name}_mean"] = float(values.mean())
            metrics[f"{prefix}{namespace}/{name}_std"] = float(values.std(ddof=0))
            metrics[f"{prefix}{namespace}/{name}_nonzero_rate"] = float(values.ne(0.0).mean())
            metrics[f"{prefix}{namespace}/{name}_max_score_rate"] = float(values.eq(1.0).mean())

    add_objective_metrics(scores)
    if log_by_prompt_nt_length and "prompt_nt_length" in scored:
        lengths = pd.to_numeric(scored["prompt_nt_length"], errors="coerce")
        for length in sorted(lengths.dropna().unique()):
            mask = lengths.eq(length)
            prefix = f"by_prompt_nt_length/{int(length)}/"
            metrics[f"{prefix}num_sequences"] = int(mask.sum())
            add_objective_metrics(scores.loc[mask], prefix)

    safety_states = scored.get("safety_gate_state", pd.Series(None, index=scored.index, dtype=object))
    metrics["safety_gate_pass_rate"] = float(_exact_safety_eligibility(scored).mean())
    metrics["safety_gate_indeterminate_rate"] = float(safety_states.eq("INDETERMINATE").mean())
    for safety_class in SEQUENCE_SAFETY_CLASSES:
        column = f"safety_{safety_class}_state"
        if column in scored:
            metrics[f"safety_{safety_class}_indeterminate_rate"] = float(scored[column].eq("INDETERMINATE").mean())

    for column in sorted(str(column) for column in scored.columns if str(column).startswith(TIMING_COLUMN_PREFIX)):
        mean_value = _mean_numeric(scored, column)
        if mean_value is not None:
            timing_name = column.removeprefix(TIMING_COLUMN_ROOT_PREFIX)
            metrics[f"{TIMING_METRIC_MARKER_PREFIX}{timing_name}"] = mean_value

    reward_columns = {column for objective in objectives for column in objective.columns}
    # These explain reward shaping in physical units or its constituent scores.
    inputs = {
        "reward_genome_length": ("genome_length",),
        "reward_gc_content": ("gc_content",),
        "reward_nt_homopolymer": ("max_nt_homopolymer_length",),
        "reward_dustmask_end": ("dustmask_max_end_masked_fraction",),
        "reward_external_protein_hit_count": ("protein_database_effective_family_count",),
        "reward_external_tropism": ("tropism_protein_mmseqs_percent_identity",),
        "reward_external_synteny": (
            "synteny_smooth_content_score",
            "synteny_smooth_ordered_score",
            "synteny_smooth_duplicate_score",
        ),
        "reward_gene_a_origin": ("gene_a_origin_motif_score", "gene_a_origin_position_score"),
        "reward_external_average_protein_identity": (
            "average_protein_percent_identity",
            "average_protein_identity_novelty_score",
            "average_protein_identity_evidence_score",
        ),
        "reward_external_required_genes": ("required_genes_integrity_sum", "required_genes_evidence_score"),
        "reward_mmseqs_cluster_diversity": (
            "mmseqs_cluster_valid_for_clustering",
            "mmseqs_cluster_missing_from_output",
        ),
    }
    if "synteny_smooth_content_score" not in scored:
        inputs["reward_external_synteny"] = (
            "synteny_reference_coverage_score",
            "synteny_copy_balance_score",
            "synteny_order_score",
        )
    for reward_column in sorted(reward_columns):
        for column in inputs.get(reward_column, ()):
            mean_value = _mean_numeric(scored, column)
            if mean_value is not None:
                metrics[f"{column}_mean"] = mean_value

    support_prefixes = {
        "reward_external_protein_hit_count": "protein_database_hit_count",
        "reward_external_tropism": "tropism",
        "reward_external_synteny": "synteny",
        "reward_external_average_protein_identity": "average_protein_identity",
        "reward_external_required_genes": "required_genes",
    }
    for reward_column in sorted(reward_columns):
        prefix = support_prefixes.get(reward_column)
        if prefix and f"{prefix}_measurement_available" in scored:
            available = pd.to_numeric(scored[f"{prefix}_measurement_available"], errors="coerce").fillna(0.0).gt(0.0)
            metrics[f"{prefix}_measurement_available_rate"] = float(available.mean())
    if "external_qc_tool_succeeded" in scored:
        values = pd.to_numeric(scored["external_qc_tool_succeeded"], errors="coerce").fillna(0.0)
        metrics["external_qc_tool_succeeded_rate"] = float(values.gt(0.0).mean())

    _add_generation_termination_metrics(metrics, scored, config)
    if "reward_mmseqs_cluster_diversity" in reward_columns and {"mmseqs_cluster_id", "mmseqs_cluster_size"}.issubset(
        scored.columns
    ):
        cluster_sizes = pd.to_numeric(scored["mmseqs_cluster_size"], errors="coerce").fillna(0).astype(int)
        # One observation per cluster; invalid/unclustered genomes contribute none.
        cluster_rows = scored.loc[cluster_sizes > 0, ["mmseqs_cluster_id", "mmseqs_cluster_size"]].drop_duplicates()
        metrics["__histogram__/mmseqs_cluster_size"] = cluster_rows["mmseqs_cluster_size"].astype(int).tolist()
    return metrics


def _scored_records(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert per-sequence scores into metadata-safe scalar/status dictionaries."""
    records: list[dict[str, Any]] = []
    for row in scored.to_dict("records"):
        record: dict[str, Any] = {}
        for key, value in row.items():
            if type(key) is not str:
                continue
            if isinstance(value, bool):
                record[key] = value
            elif isinstance(value, Integral) and _is_exact_finite_real(value):
                record[key] = int(value)
            elif _is_exact_finite_real(value):
                record[key] = float(value)
            elif (
                type(value) is str
                and _is_bounded_utf8(value)
                and (
                    key in {"mmseqs_cluster_id", "prompt_group"}
                    or key.startswith("safety_")
                    or key.endswith(("_pass", "_available", "_artifact"))
                )
            ):
                record[key] = value
        records.append(record)
    return records


def _is_bounded_utf8(value: str) -> bool:
    """Accept only well-formed UTF-8 strings within the rollout metadata bound."""
    try:
        return len(value.encode("utf-8")) <= 4096
    except UnicodeEncodeError:
        return False


def _scored_from_batch_metadata(batch: Any) -> pd.DataFrame:
    """Recover scored rows only when every rollout position has one metadata mapping."""
    extra_env_info = batch.get("extra_env_info", [])
    if not isinstance(extra_env_info, list):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for info in extra_env_info:
        if not isinstance(info, dict) or not isinstance(info.get("_phage_qc_scored"), dict):
            return pd.DataFrame()
        rows.append(info["_phage_qc_scored"])
    return pd.DataFrame(rows)


def _batch_metadata_is_complete(
    scored: pd.DataFrame,
    expected_rows: int,
    objectives: tuple[GDPOObjective, ...],
) -> bool:
    """Require one nonmissing aggregate/objective record for every assembled rollout row."""
    if len(scored) != expected_rows:
        return False
    required_columns = {"reward", "safety_gate_state", "safety_gate_pass"}
    for objective in objectives:
        required_columns.update(objective.columns)
        for column in objective.columns:
            safety_class = column.removeprefix("reward_safety_")
            if safety_class in SEQUENCE_SAFETY_CLASSES:
                required_columns.add(f"safety_{safety_class}_state")
                required_columns.add(f"safety_{safety_class}_required")
    if not required_columns.issubset(scored.columns):
        return expected_rows == 0
    return bool(scored[list(required_columns)].notna().to_numpy().all())


if _NEMO_RL_IMPORT_ERROR is None:  # pragma: no cover
    # NeMo-RL sends one already-batched call per task type. A large threaded-actor
    # pool creates hundreds of glibc arenas and retains their high-water RSS across
    # rollout steps without adding phage-QC parallelism.
    @ray.remote(
        max_restarts=3,
        max_task_retries=3,
        max_concurrency=1,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "",
                "PHAGEHOSTLEARN_CUDA_VISIBLE_DEVICES": "",
                "PHAGEHOSTLEARN_ESM_DEVICE": "cpu",
            }
        },
    )
    class PhageQCEnvironment(EnvironmentInterface[dict[str, Any]]):
        """Single-turn NeMo-RL environment for phage sequence QC reward."""

        def __init__(self, cfg: dict[str, Any]):
            """Create a phage QC environment from NeMo-RL environment config."""
            self.cfg = cfg
            self.config = NucleotideQCConfig(
                genome_length_min=int(cfg.get("genome_length_min", 4000)),
                genome_length_max=int(cfg.get("genome_length_max", 6000)),
                genome_length_reward_lower_zero=float(
                    cfg.get("genome_length_reward_lower_zero", NucleotideQCConfig.genome_length_reward_lower_zero)
                ),
                genome_length_reward_lower_full=float(
                    cfg.get("genome_length_reward_lower_full", NucleotideQCConfig.genome_length_reward_lower_full)
                ),
                genome_length_reward_upper_full=float(
                    cfg.get("genome_length_reward_upper_full", NucleotideQCConfig.genome_length_reward_upper_full)
                ),
                genome_length_reward_upper_zero=float(
                    cfg.get("genome_length_reward_upper_zero", NucleotideQCConfig.genome_length_reward_upper_zero)
                ),
                gc_content_min=float(cfg.get("gc_content_min", 30.0)),
                gc_content_max=float(cfg.get("gc_content_max", 65.0)),
                homopolymer_max=int(cfg.get("homopolymer_max", 10)),
                dustmask_filter=bool(cfg.get("dustmask_filter", False)),
                dustmasker_bin=str(cfg.get("dustmasker_bin", "dustmasker")),
                dustmask_window=int(cfg.get("dustmask_window", 64)),
                dustmask_level=float(cfg.get("dustmask_level", 20.0)),
                dustmask_end_window=int(cfg.get("dustmask_end_window", 200)),
                dustmask_max_end_fraction=float(cfg.get("dustmask_max_end_fraction", 0.9)),
            )
            self.weights = RewardWeights(
                valid_nt_chars=float(cfg.get("weight_valid_nt_chars", 1.0)),
                genome_length=float(cfg.get("weight_genome_length", 1.0)),
                gc_content=float(cfg.get("weight_gc_content", 1.0)),
                nt_homopolymer=float(cfg.get("weight_nt_homopolymer", 1.0)),
                dustmask_end=float(cfg.get("weight_dustmask_end", 0.0)),
                nucleotide_pass=float(cfg.get("weight_nucleotide_pass", 0.0)),
                protein_hit_count=float(cfg.get("weight_protein_hit_count", 0.0)),
                tropism=float(cfg.get("weight_tropism", 0.0)),
                synteny=float(cfg.get("weight_synteny", 0.0)),
                gene_a_origin=float(cfg.get("weight_gene_a_origin", 0.0)),
                average_protein_identity=float(cfg.get("weight_average_protein_identity", 0.0)),
                required_genes=float(cfg.get("weight_required_genes", 0.0)),
                mmseqs_cluster_diversity=float(cfg.get("weight_mmseqs_cluster_diversity", 0.0)),
            )
            reward_output_mode = str(cfg.get("reward_output_mode", "scalar")).strip().lower()
            if reward_output_mode == "grpo":
                reward_output_mode = "scalar"
            if reward_output_mode not in {"scalar", "gdpo"}:
                raise ValueError("reward_output_mode must be 'scalar', 'grpo', or 'gdpo'.")
            self.reward_output_mode = reward_output_mode
            zero_reward_without_eod = cfg.get("zero_reward_without_eod", False)
            if type(zero_reward_without_eod) is not bool:
                raise TypeError("zero_reward_without_eod must be a boolean.")
            self.zero_reward_without_eod = zero_reward_without_eod
            self.gdpo_objectives = _coerce_gdpo_objectives(cfg.get("gdpo_objectives"))
            if self.reward_output_mode == "gdpo":
                _validate_gdpo_safety_objectives(self.gdpo_objectives)
            self.sequence_safety = _coerce_sequence_safety_config(cfg.get("sequence_safety"))
            if self.sequence_safety is None:
                raise ValueError("sequence_safety configuration is required for phage RL training.")
            if self.sequence_safety.enabled is not True:
                raise ValueError("sequence_safety must be enabled for phage RL training.")
            external_qc_cfg = cfg.get("external_qc", {}) or {}
            self.external_qc = ExternalQCRewardConfig(
                enabled=bool(external_qc_cfg.get("enabled", False)),
                config_path=external_qc_cfg.get("config_path", "configs/arc_genome_design_filtering_local.yaml"),
                pipeline_script=external_qc_cfg.get(
                    "pipeline_script", "data/arc_pipeline_patched/genome_design_filtering_pipeline.py"
                ),
                work_dir=external_qc_cfg.get("work_dir", "data/checkpoints/phage_grpo_external_qc"),
                tool_bin_dir=external_qc_cfg.get("tool_bin_dir"),
                keep_artifacts=bool(external_qc_cfg.get("keep_artifacts", False)),
                fail_on_error=bool(external_qc_cfg.get("fail_on_error", True)),
                timeout_seconds=external_qc_cfg.get("timeout_seconds", 1800.0),
                enable_orf=bool(external_qc_cfg.get("enable_orf", False)),
                enable_coding_density=bool(external_qc_cfg.get("enable_coding_density", False)),
                enable_protein_hit_count=bool(external_qc_cfg.get("enable_protein_hit_count", True)),
                enable_tropism=bool(external_qc_cfg.get("enable_tropism", True)),
                enable_synteny=bool(external_qc_cfg.get("enable_synteny", False)),
                enable_average_protein_identity=bool(external_qc_cfg.get("enable_average_protein_identity", False)),
                enable_required_genes=bool(external_qc_cfg.get("enable_required_genes", False)),
                required_genes_evidence_target=float(external_qc_cfg.get("required_genes_evidence_target", 10.0)),
                protein_match_min_reciprocal_coverage=float(
                    external_qc_cfg.get("protein_match_min_reciprocal_coverage", 0.75)
                ),
                tropism_match_min_reciprocal_coverage=float(
                    external_qc_cfg.get("tropism_match_min_reciprocal_coverage", 0.95)
                ),
                enable_smooth_reference_rewards=bool(external_qc_cfg.get("enable_smooth_reference_rewards", False)),
                enable_gene_a_origin=bool(external_qc_cfg.get("enable_gene_a_origin", False)),
                synteny_identity_full_credit=float(external_qc_cfg.get("synteny_identity_full_credit", 0.90)),
                synteny_reciprocal_coverage_full_credit=float(
                    external_qc_cfg.get("synteny_reciprocal_coverage_full_credit", 0.95)
                ),
                synteny_integrity_gamma=float(external_qc_cfg.get("synteny_integrity_gamma", 1.5)),
                synteny_raw_integrity_min=float(external_qc_cfg.get("synteny_raw_integrity_min", 0.001)),
                synteny_min_credit=float(external_qc_cfg.get("synteny_min_credit", 0.01)),
                synteny_order_weight=float(external_qc_cfg.get("synteny_order_weight", 0.75)),
                synteny_duplicate_penalty_weight=float(external_qc_cfg.get("synteny_duplicate_penalty_weight", 0.75)),
                tropism_identity_full_credit=float(external_qc_cfg.get("tropism_identity_full_credit", 0.95)),
                tropism_reciprocal_coverage_full_credit=float(
                    external_qc_cfg.get("tropism_reciprocal_coverage_full_credit", 0.99)
                ),
                tropism_integrity_gamma=float(external_qc_cfg.get("tropism_integrity_gamma", 1.5)),
                tropism_raw_integrity_min=float(external_qc_cfg.get("tropism_raw_integrity_min", 0.001)),
                tropism_min_credit=float(external_qc_cfg.get("tropism_min_credit", 0.01)),
                gene_a_reference_locus=str(external_qc_cfg.get("gene_a_reference_locus", "NC_001422.1_ORF.23")),
                tropism_reference_locus=str(external_qc_cfg.get("tropism_reference_locus", "NC_001422.1_ORF.3")),
                gene_a_origin_motif=str(external_qc_cfg.get("gene_a_origin_motif", "CAACTTGATATTAATAACACTATAGACCAC")),
                gene_a_origin_offset_nt=int(external_qc_cfg.get("gene_a_origin_offset_nt", 345)),
                gene_a_origin_offset_tolerance_nt=int(external_qc_cfg.get("gene_a_origin_offset_tolerance_nt", 30)),
                lovis4u_parallel_jobs=external_qc_cfg.get("lovis4u_parallel_jobs", 12),
                lovis4u_chunk_size=external_qc_cfg.get("lovis4u_chunk_size"),
                lovis4u_mmseqs_threads=external_qc_cfg.get("lovis4u_mmseqs_threads"),
                lovis4u_metrics_only=bool(external_qc_cfg.get("lovis4u_metrics_only", False)),
                lovis4u_collect_pdfs=bool(external_qc_cfg.get("lovis4u_collect_pdfs", False)),
            )
            mmseqs_cfg = cfg.get("mmseqs_cluster_diversity", {}) or {}
            self.mmseqs_cluster_diversity = MMseqsClusterDiversityConfig(
                enabled=bool(mmseqs_cfg.get("enabled", False)),
                circular=bool(mmseqs_cfg.get("circular", False)),
                mmseqs_bin=str(mmseqs_cfg.get("mmseqs_bin", "mmseqs")),
                work_dir=mmseqs_cfg.get("work_dir", "data/checkpoints/phage_grpo_mmseqs_cluster_diversity"),
                keep_artifacts=bool(mmseqs_cfg.get("keep_artifacts", False)),
                min_seq_id=float(mmseqs_cfg.get("min_seq_id", 0.99)),
                coverage=float(mmseqs_cfg.get("coverage", 0.95)),
                cov_mode=int(mmseqs_cfg.get("cov_mode", 0)),
                seq_id_mode=int(mmseqs_cfg.get("seq_id_mode", 0)),
                cluster_mode=int(mmseqs_cfg.get("cluster_mode", 0)),
                parallel_jobs=int(mmseqs_cfg.get("parallel_jobs", 1)),
                threads=mmseqs_cfg.get("threads"),
                verbosity=int(mmseqs_cfg.get("verbosity", 0)),
            )

        def step(
            self,
            message_log_batch: list[list[dict[str, Any]]],
            metadata: list[dict[str, Any]],
        ) -> EnvironmentReturn[dict[str, Any]]:
            """Score generated assistant sequences and terminate each rollout."""
            env_step_begin_unix_s = time.time()
            env_step_start = time.perf_counter()
            phase_start = time.perf_counter()
            scored = score_message_logs(
                message_log_batch,
                config=self.config,
                weights=self.weights,
                external_qc=self.external_qc,
                mmseqs_cluster_diversity=self.mmseqs_cluster_diversity,
                sequence_safety=self.sequence_safety,
            )
            _attach_generation_stop_columns(scored, metadata)
            zero_reward_without_eod = getattr(self, "zero_reward_without_eod", False)
            if zero_reward_without_eod:
                scored["eod_reward_gate_pass"] = _eod_reward_eligibility(scored, required=True)
            reward_scoring_s = time.perf_counter() - phase_start
            qualified_scalar_rewards = _qualified_scalar_rewards(
                scored,
                zero_reward_without_eod=zero_reward_without_eod,
            )
            if self.reward_output_mode == "gdpo":
                phase_start = time.perf_counter()
                objective_scores = gdpo_objective_scores_from_scored(
                    scored,
                    self.gdpo_objectives,
                    zero_reward_without_eod=zero_reward_without_eod,
                )
                gdpo_objectives_s = time.perf_counter() - phase_start
                rewards = torch.tensor(objective_scores.to_numpy(dtype=float), dtype=torch.float32).cpu()
            else:
                gdpo_objectives_s = 0.0
                rewards = torch.tensor(qualified_scalar_rewards.tolist(), dtype=torch.float32).cpu()

            env_step_end_unix_s = time.time()
            env_step_timings = {
                "env_step/begin_unix_s": env_step_begin_unix_s,
                "env_step/end_unix_s": env_step_end_unix_s,
                "env_step/total_s": time.perf_counter() - env_step_start,
                "env_step/reward_scoring_s": reward_scoring_s,
                "env_step/gdpo_objectives_s": gdpo_objectives_s,
            }
            for name, value in env_step_timings.items():
                scored[f"{TIMING_COLUMN_PREFIX}{name}"] = float(value)

            scored_records = _scored_records(scored)
            returned_metadata = []
            for item, scored_record in zip(metadata, scored_records, strict=True):
                item_dict = dict(item or {})
                item_dict["_phage_qc_scored"] = scored_record
                returned_metadata.append(item_dict)
            observations = [
                {"role": "environment", "content": f"phage_qc_reward={reward:.6f}"}
                for reward in qualified_scalar_rewards.tolist()
            ]
            return EnvironmentReturn(
                observations=observations,
                metadata=returned_metadata,
                next_stop_strings=[None] * len(message_log_batch),
                rewards=rewards,
                terminateds=torch.ones(rewards.shape[0], dtype=torch.bool).cpu(),
                answers=scored["sequence"].tolist(),
            )

        def global_post_process_and_metrics(
            self, batch: BatchedDataDict
        ) -> tuple[BatchedDataDict, dict[str, float | int | list[int]]]:
            """Report rollout-level reward metrics."""
            reward_tensor = batch["rewards"] if "rewards" in batch else batch["total_reward"]
            rewards = reward_tensor if reward_tensor.ndim == 1 else reward_tensor.float().mean(dim=1)
            batch_scored = _scored_from_batch_metadata(batch)
            reward_output_mode = getattr(self, "reward_output_mode", "scalar")
            zero_reward_without_eod = getattr(self, "zero_reward_without_eod", False)
            gdpo_objectives = self.gdpo_objectives if reward_output_mode == "gdpo" else ()
            if not _batch_metadata_is_complete(batch_scored, int(rewards.shape[0]), gdpo_objectives):
                batch_scored = pd.DataFrame()
            metrics = phage_qc_metrics_from_scored(
                batch_scored,
                self.weights,
                config=getattr(self, "config", None),
                zero_reward_without_eod=zero_reward_without_eod,
                objectives=gdpo_objectives,
                log_by_prompt_nt_length=self.cfg.get("log_by_prompt_nt_length", False),
            )
            has_rewards = rewards.numel() > 0
            metrics["mean_reward"] = rewards.float().mean().item() if has_rewards else 0.0
            metrics["num_sequences"] = int(rewards.shape[0])
            group_keys = _reward_prompt_group_keys(batch_scored)
            if not batch_scored.empty and group_keys is not None:
                reward_values = reward_tensor.detach().float().cpu()
                if reward_values.ndim == 1:
                    reward_values = reward_values.unsqueeze(1)
                grouped_rewards = pd.DataFrame(reward_values.numpy(), index=batch_scored.index).groupby(
                    group_keys, sort=False
                )
                group_count = int(grouped_rewards.ngroups)
                zero_variance_count = sum(
                    bool(((group.max(axis=0) - group.min(axis=0)) == 0.0).all()) for _name, group in grouped_rewards
                )
                metrics["reward_prompt_group_count"] = group_count
                metrics["reward_zero_variance_prompt_group_rate"] = (
                    zero_variance_count / group_count if group_count else 0.0
                )
            return batch, metrics

else:

    class PhageQCEnvironment:  # pragma: no cover
        """Placeholder that explains how to enable the NeMo-RL integration."""

        def __init__(self, *_args, **_kwargs):
            """Raise a clear error when NeMo-RL is not installed."""
            raise ModuleNotFoundError(
                "PhageQCEnvironment requires NeMo-RL. Install the recipe environment with nemo-rl available."
            ) from _NEMO_RL_IMPORT_ERROR
