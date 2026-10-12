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

"""Exact trainability policy for the two training stages.

The application supplies parameter patterns for each family. Sidecar-held
projectors are enumerated from their attribute rather than the module tree.

Families: ``projector`` (every parameter of every
named projector), ``extra`` (named project parameters registered by the
consumer), ``lora`` (PEFT parameters — the only "adapter"). Stage 1 trains
``projector`` (plus ``extra`` when enabled); stage 2 adds ``lora``. A consumer
may freeze the projector in stage 2 (``train_projector=False``) after loading
its stage-1 artifact. Alternatively,
``train_decoder=True`` selects explicitly named decoder parameters for full
stage-2 training while retaining upstream-required freezes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECTOR = "projector"
EXTRA = "extra"
LORA = "lora"
DECODER = "decoder"


@dataclass(frozen=True)
class TrainabilityPolicy:
    """How a consumer's parameter names map to trainability families.

    ``decoder_patterns`` explicitly selects full-training decoder parameters by
    name substring after projector/extra/LoRA classification. It never overrides
    framework-required freezes and is unused by the default LoRA recipe.

    ``projector_patterns`` covers the in-tree (module-ownership) case by
    substring match on the dotted parameter name. ``sidecar_attribute`` covers
    sidecar ownership: the attribute holds the projector registry outside the
    module tree, and every parameter under it is the projector family.
    """

    projector_patterns: tuple[str, ...] = ()
    extra_patterns: tuple[str, ...] = ()
    lora_marker: str = "lora_"
    sidecar_attribute: str | None = None
    decoder_patterns: tuple[str, ...] = ()


def _parameter_family(name: str, policy: TrainabilityPolicy) -> str | None:
    if any(pattern in name for pattern in policy.projector_patterns):
        return PROJECTOR
    if any(pattern in name for pattern in policy.extra_patterns):
        return EXTRA
    if policy.lora_marker in name:
        return LORA
    if any(pattern in name for pattern in policy.decoder_patterns):
        return DECODER
    return None


def _iter_family_parameters(model: nn.Module, policy: TrainabilityPolicy):
    """Yield ``(name, parameter, family)`` for in-tree and sidecar parameters."""
    seen = set()
    for name, parameter in model.named_parameters():
        seen.add(id(parameter))
        yield name, parameter, _parameter_family(name, policy)
    if policy.sidecar_attribute is not None:
        registry = getattr(model, policy.sidecar_attribute, None)
        if registry is None:
            raise ValueError(f"policy names a sidecar attribute the model lacks: {policy.sidecar_attribute!r}")
        for name, parameter in registry.named_parameters():
            if id(parameter) in seen:
                raise ValueError(
                    f"{policy.sidecar_attribute!r} is registered in the module tree; "
                    "sidecar policy requires sidecar-held parameters"
                )
            yield f"{policy.sidecar_attribute}.{name}", parameter, PROJECTOR


# Stages are plain ints, not a Literal: the value arrives through YAML config
# plumbing, and the runtime check below is the enforcement point regardless.
def _allowed_families(stage: int, train_projector: bool, train_extra: bool, train_decoder: bool = False) -> set[str]:
    if stage not in (1, 2):
        raise ValueError(f"training stage must be 1 or 2, got {stage!r}")
    if train_decoder and stage != 2:
        raise ValueError("Full decoder training requires stage 2")
    allowed = set()
    if train_projector:
        allowed.add(PROJECTOR)
    if train_extra:
        allowed.add(EXTRA)
    if stage == 2:
        allowed.add(DECODER if train_decoder else LORA)
    return allowed


def extra_parameter_names(model: nn.Module, policy: TrainabilityPolicy) -> list[str]:
    """The model's EXTRA-family parameter names, in deterministic order."""
    return sorted(name for name, _, family in _iter_family_parameters(model, policy) if family == EXTRA)


def collect_extra_state(model: nn.Module, policy: TrainabilityPolicy) -> dict[str, torch.Tensor]:
    """Collect the EXTRA family's full tensors, keyed by dotted parameter name.

    DTensor-safe: ``full_tensor()`` is a collective, so this runs on every
    rank and the caller writes on rank zero only. Keys are the model's own
    parameter names, so a load restores by name with no mapping.
    """
    state = {}
    for name, parameter, family in _iter_family_parameters(model, policy):
        if family != EXTRA:
            continue
        tensor = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        state[name] = tensor.detach().cpu()
    return state


def validate_trainable_manifest(
    model: nn.Module,
    stage: int,
    *,
    policy: TrainabilityPolicy,
    train_projector: bool = True,
    train_extra: bool = False,
    train_decoder: bool = False,
) -> dict[str, Any]:
    """Validate the complete trainable-name set and return stable metadata."""
    allowed = _allowed_families(stage, train_projector, train_extra, train_decoder)
    trainable = [
        (name, parameter, family)
        for name, parameter, family in _iter_family_parameters(model, policy)
        if parameter.requires_grad
    ]
    unexpected = [name for name, _, family in trainable if family not in allowed]
    if unexpected:
        raise ValueError(f"unexpected trainable parameters: {', '.join(sorted(unexpected))}")

    families = {family for _, _, family in trainable}
    if train_projector and PROJECTOR not in families:
        raise ValueError("the projector family is enabled but no projector parameters are trainable")
    if stage == 2:
        required = DECODER if train_decoder else LORA
        if required not in families:
            raise ValueError("Stage 2 has no decoder parameters" if train_decoder else "Stage 2 has no LoRA parameters")
    if train_extra and EXTRA not in families:
        raise ValueError("extra parameters are enabled but none are trainable")

    parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype).removeprefix("torch."),
            "numel": parameter.numel(),
        }
        for name, parameter, _ in sorted(trainable, key=lambda item: item[0])
    ]
    return {
        "stage": stage,
        "train_projector": train_projector,
        "train_extra": train_extra,
        "train_decoder": train_decoder,
        "total_trainable_parameters": sum(entry["numel"] for entry in parameters),
        "parameters": parameters,
    }


def configure_trainable_parameters(
    model: nn.Module,
    stage: int,
    *,
    policy: TrainabilityPolicy,
    train_projector: bool = True,
    train_extra: bool = False,
    train_decoder: bool = False,
) -> dict[str, Any]:
    """Apply projector policy before optimization, preserving native decoder freezes.

    Full stage-2 training requires explicit decoder name substrings and no PEFT
    parameters. Matched decoder parameters retain the upstream requires_grad
    state; unnamed parameters stay frozen, including external encoder modules.
    """
    allowed = _allowed_families(stage, train_projector, train_extra, train_decoder)
    if train_decoder:
        if not policy.decoder_patterns or any(not pattern for pattern in policy.decoder_patterns):
            raise ValueError("Full decoder training requires nonempty decoder_patterns")
        if any(family == LORA for _, _, family in _iter_family_parameters(model, policy)):
            raise ValueError("Full decoder training cannot contain LoRA parameters")
        names = [name for name, _, family in _iter_family_parameters(model, policy) if family == DECODER]
        for pattern in policy.decoder_patterns:
            if not any(pattern in name for name in names):
                raise ValueError(f"Decoder pattern matched no parameters: {pattern!r}")
    for _, parameter, family in _iter_family_parameters(model, policy):
        # Retain native full-model trainability, including protected freezes.
        # AutoModel 1814c6c9 lacks freeze_config recipe passthrough. Delete
        # this selection seam after adopting the native freeze selectors.
        if train_decoder and family == DECODER:
            continue
        parameter.requires_grad_(family in allowed)
    return validate_trainable_manifest(
        model,
        stage,
        policy=policy,
        train_projector=train_projector,
        train_extra=train_extra,
        train_decoder=train_decoder,
    )


def write_manifest_rank_zero(manifest: dict[str, object], path: str | None) -> None:
    """Write a deterministic manifest only on distributed rank zero."""
    if path is None:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
