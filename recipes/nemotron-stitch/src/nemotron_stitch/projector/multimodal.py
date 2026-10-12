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

"""MultimodalProjector: the named projector registry.

Validation and dispatch use the flat index contract.
One behavior is deliberately tightened: the old router's mixed mode (explicit
indices for some adapters, placeholder derivation for others) dereferenced a
None and crashed; here indices are either explicit for every projector or
derived for all of them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from nemotron_stitch.contracts import OWNERSHIP_MODES, OWNERSHIP_MODULE, OWNERSHIP_SIDECAR
from nemotron_stitch.projector.projectors import Projector, build_projector
from nemotron_stitch.projector.scatter import (
    dense_to_flat,
    derive_flat_indices,
    derive_indices,
    explicit_to_flat,
    scatter_flat,
)


class MultimodalProjector(nn.Module):
    """The named projector registry, in either ownership mode.

    ``projector_ownership="sidecar"`` (default): the host holds the registry
    outside its module tree (``object.__setattr__``), so framework state-dict
    conversion, FSDP2 wrapping, and PEFT freezing never see it; the registry
    therefore owns device placement and coerces incoming features to its
    parameters' device and dtype at the forward boundary. Fails closed above
    one rank — that guard lives in the consumer's recipe.

    ``projector_ownership="module"``: the host registers the registry as an
    ordinary child module, so FSDP2/DTensor wrapping, the host's
    ``initialize_weights`` extension, and the trainability policy all see it.
    Device placement is then owned by that wrapping (and by the consumer's TP
    plumbing for a replicated projector), so a feature payload on the wrong
    device is a wiring bug: module mode fails closed on device mismatch rather
    than silently moving data — a ``.to()`` cannot repair DTensor placements
    anyway. For a single projector this is a one-entry ``ModuleDict``.
    """

    def __init__(self, projectors: dict[str, Projector], *, projector_ownership: str = OWNERSHIP_SIDECAR):
        super().__init__()
        if projector_ownership not in OWNERSHIP_MODES:
            raise ValueError(
                f"unknown projector_ownership: {projector_ownership!r} (expected one of {sorted(OWNERSHIP_MODES)})"
            )
        if not projectors:
            raise ValueError("at least one projector is required")
        self.projector_ownership = projector_ownership
        self.projectors = nn.ModuleDict(projectors)

    def reset_parameters(self) -> None:
        for projector in self.projectors.values():
            reset = getattr(projector, "reset_parameters", None)
            if reset is None:
                raise TypeError(f"projector has no reset_parameters method: {type(projector).__name__}")
            reset()

    @classmethod
    def from_config(
        cls,
        configs: Iterable[dict[str, Any]],
        output_size: int,
        *,
        projector_ownership: str = OWNERSHIP_SIDECAR,
    ) -> MultimodalProjector:
        projectors: dict[str, Projector] = {}
        for raw in configs:
            config = dict(raw)
            name = str(config.pop("name"))
            if name in projectors:
                raise ValueError(f"duplicate projector name: {name!r}")
            projectors[name] = build_projector(config, output_size)
        return cls(projectors, projector_ownership=projector_ownership)

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        features_by_projector: dict[str, torch.Tensor],
        *,
        placeholder_token_ids: dict[str, int],
        token_indices_by_projector: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Project features and scatter the soft tokens into explicit LM slots.

        Flat per-projector feature fields are the primary contract: features
        are either flat ``[N, mm_hidden]`` (the canonical geometry) or dense
        ``[B, T, mm_hidden]``. Each projector owns a distinct
        placeholder token id: target positions are derived per projector from
        ``input_ids`` and ``placeholder_token_ids``, routing by token
        identity, so segments of different projectors interleaved in one
        prompt cannot be confused by position. Derived counts are validated
        against each projector's emitted soft-token count — per row (uniform)
        for dense features, in total for flat ones. Callers that need exact
        control may supply ``token_indices_by_projector`` as an explicit
        override — for every projector or none of them, flat ``[N]`` indices
        paired with flat features or dense ``[B, T]`` indices with dense
        features; each index must then point at its own projector's
        placeholder token.
        """
        if not features_by_projector:
            raise ValueError("no projector features were provided")
        unknown_projectors = sorted(set(features_by_projector) - set(self.projectors))
        if unknown_projectors:
            raise KeyError(f"unknown projector: {unknown_projectors[0]!r}")
        if placeholder_token_ids is None:
            raise ValueError("placeholder_token_ids are required to derive and validate soft-token slots")
        unknown = sorted(set(placeholder_token_ids) - set(self.projectors))
        if unknown:
            raise ValueError(f"placeholder_token_ids name unregistered projectors: {unknown}")
        missing = sorted(set(features_by_projector) - set(placeholder_token_ids))
        if missing:
            raise ValueError(f"no placeholder token id configured for projectors: {missing}")
        placeholder_token_ids = {name: int(token_id) for name, token_id in placeholder_token_ids.items()}
        active_ids = [placeholder_token_ids[name] for name in features_by_projector]
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("placeholder token ids must be distinct across projectors")
        override = token_indices_by_projector or {}
        if override and set(override) != set(features_by_projector):
            raise ValueError("explicit token indices must cover every projector or none")
        thd = input_ids.ndim == 1
        if thd:
            if inputs_embeds.ndim != 2:
                raise ValueError("THD input_ids [T] require inputs_embeds [T,C]")
            input_ids = input_ids.unsqueeze(0)
            inputs_embeds = inputs_embeds.unsqueeze(0)

        soft_tokens: dict[str, torch.Tensor] = {}
        for name in sorted(features_by_projector):
            projector = self.projectors[name]
            parameter = next(projector.parameters())
            features = features_by_projector[name]
            if self.projector_ownership == OWNERSHIP_MODULE:
                if features.device != parameter.device:
                    raise ValueError(
                        f"features for projector {name!r} are on device {features.device}, "
                        f"but the module-owned parameters are on {parameter.device}"
                    )
                features = features.to(dtype=parameter.dtype)
            else:
                features = features.to(device=parameter.device, dtype=parameter.dtype)
            tokens = projector(features)
            if tokens.ndim not in (2, 3):
                raise ValueError(
                    f"projector {name!r} emitted {tokens.ndim}D soft tokens; expected flat [N, H] or dense [B, T, H]"
                )
            soft_tokens[name] = tokens

        sequence_length = input_ids.shape[1]
        flat: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if override:
            for name in sorted(soft_tokens):
                flat[name] = explicit_to_flat(override[name], soft_tokens[name], sequence_length=sequence_length)
        else:
            dense = {name: tokens for name, tokens in soft_tokens.items() if tokens.ndim == 3}
            if dense:
                derived = derive_indices(
                    input_ids,
                    {name: placeholder_token_ids[name] for name in dense},
                    {name: tokens.shape[1] for name, tokens in dense.items()},
                )
                for name in sorted(dense):
                    flat[name] = dense_to_flat(derived[name], dense[name], sequence_length=sequence_length)
            ragged = {name: tokens for name, tokens in soft_tokens.items() if tokens.ndim == 2}
            if ragged:
                derived_flat = derive_flat_indices(
                    input_ids,
                    {name: placeholder_token_ids[name] for name in ragged},
                    {name: tokens.shape[0] for name, tokens in ragged.items()},
                )
                for name in sorted(ragged):
                    flat[name] = (derived_flat[name], ragged[name])

        # A prompt carrying placeholder slots for a registered projector that
        # received no features would feed the LM raw placeholder embeddings —
        # the same plausible-loss-curve failure as a misrouted scatter.
        for name in sorted(set(self.projectors) - set(soft_tokens)):
            token_id = placeholder_token_ids.get(name)
            if token_id is not None and bool(input_ids.eq(token_id).any()):
                raise ValueError(f"placeholder slots for projector {name!r} were provided no features")

        result = inputs_embeds
        occupied = torch.zeros(input_ids.shape[0] * input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        for name in sorted(soft_tokens):
            flat_indices, flat_tokens = flat[name]
            # Negative entries are dropped by the scatter; keep
            # them out of the cross-projector overlap bookkeeping as well.
            active_indices = flat_indices.to(device=input_ids.device, dtype=torch.long)
            active_indices = active_indices[active_indices.ge(0)]
            if active_indices.numel():
                if occupied[active_indices].any():
                    raise ValueError("projector payloads target overlapping LM positions")
                occupied[active_indices] = True
            result = scatter_flat(
                input_ids,
                result,
                flat_tokens,
                flat_indices,
                placeholder_token_id=placeholder_token_ids[name],
            )
        return result.squeeze(0) if thd else result
