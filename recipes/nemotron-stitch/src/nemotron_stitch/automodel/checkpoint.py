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

"""DTensor-safe parameter copies, state-dict adapter helpers, and the
native-load conservation audit.

Projector prefixes, extra parameter names, and the flag that includes
projector state in HF exports are supplied by the application.
"""

from __future__ import annotations

from typing import Any, cast

import torch


def normalized_checkpoint_name(name: str) -> str:
    """Strip activation-checkpoint wrapper segments from a state-dict FQN.

    PyTorch strips ``_checkpoint_wrapped_module.`` from ``state_dict`` keys
    while ``named_modules`` retains it; consumers that join the two namespaces
    (or stream weights to a serving engine expecting HF names) normalize
    through here.
    """
    normalized = name.replace("_checkpoint_wrapped_module.", "")
    if normalized == "_checkpoint_wrapped_module":
        return ""
    return normalized.removesuffix("._checkpoint_wrapped_module")


def is_projector_state_key(name: str, prefixes: tuple[str, ...]) -> bool:
    """Whether a state-dict FQN belongs to the projector/extra/encoder families."""
    return name.startswith(tuple(prefixes))


def prefix_hf_checkpoint_keys(
    converted: list[tuple[str, torch.Tensor]],
    *,
    prefix: str,
    is_projector_state: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Address a text backbone nested inside a multimodal HF checkpoint."""
    if not prefix or is_projector_state:
        return converted
    return [(f"{prefix}{name}", tensor) for name, tensor in converted]


def select_hf_checkpoint_subtree(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Select and unnest a text backbone from a multimodal HF checkpoint."""
    if not prefix:
        return state_dict
    return {name.removeprefix(prefix): tensor for name, tensor in state_dict.items() if name.startswith(prefix)}


def copy_full_parameter(parameter: Any, tensor: torch.Tensor) -> None:
    """Copy a full (unsharded) tensor into a possibly DTensor-sharded parameter."""
    try:
        from torch.distributed.tensor import DTensor, Replicate
    except ImportError:
        # A torch build without distributed has no DTensors; the plain copy applies.
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
        return
    if isinstance(parameter.data, DTensor):
        full = tensor.to(parameter.data.to_local().device)
        replicated = DTensor.from_local(
            full,
            device_mesh=parameter.data.device_mesh,
            placements=[Replicate()] * parameter.data.device_mesh.ndim,
        )
        local = replicated.redistribute(placements=parameter.data.placements).to_local()
        parameter.data.to_local().copy_(local)
    else:
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def collect_portable_states(
    model,
    *,
    projector_prefix: str,
    extra_names: tuple[str, ...] = (),
    lora_marker: str = "lora_",
) -> tuple[dict, dict, dict]:
    """Collect full projector, extra, and LoRA tensors on every rank.

    Projector keys are returned stripped of ``projector_prefix`` (portable
    sidecar layout); extra and LoRA keys keep their full names.
    """
    projector: dict[str, torch.Tensor] = {}
    extra: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}
    for raw_name, parameter in model.named_parameters():
        name = normalized_checkpoint_name(raw_name)
        tensor = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        if name.startswith(projector_prefix):
            projector[name.removeprefix(projector_prefix)] = tensor.detach().cpu()
        elif name in extra_names:
            extra[name] = tensor.detach().cpu()
        elif lora_marker in name:
            adapter[name] = tensor.detach().cpu()
    return projector, extra, adapter


class ProjectorStateDictAdapterMixin:
    """Keep projector tensors out of HF export until a resume needs them.

    Mix over an AutoModel model-family state-dict adapter (cooperative
    ``__init__``; the base provides ``self.config``). The consumer subclass
    sets ``PROJECTOR_STATE_PREFIXES`` — the projector, extra, and frozen
    encoder prefixes that the base model's HF checkpoint never carries — and
    may override ``LOAD_PROJECTOR_STATE_ATTR``, the config flag that requests
    projector state in the export (set by resume tooling).
    """

    # Provided by the AutoModel adapter base this mixin is composed with.
    # Annotation-only: nothing is assigned, so MRO still reaches the base.
    config: Any

    PROJECTOR_STATE_PREFIXES: tuple[str, ...] = ()
    LOAD_PROJECTOR_STATE_ATTR: str = "load_projector_state_from_hf"
    #: Model-side → export-side prefix rewrites, applied so the HF/serving
    #: contract survives a model-internal projector rename (a consumer that
    #: moved its projector into a package registry keeps exporting the
    #: historical key names). Keys and values are dotted prefixes.
    PROJECTOR_EXPORT_RENAMES: dict[str, str] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.include_projector_state = bool(getattr(self.config, self.LOAD_PROJECTOR_STATE_ATTR, False))

    def set_include_projector_state(self, include: bool) -> None:
        self.include_projector_state = bool(include)

    def _export_name(self, fqn: str) -> str:
        for model_prefix, export_prefix in self.PROJECTOR_EXPORT_RENAMES.items():
            if fqn.startswith(model_prefix):
                return export_prefix + fqn[len(model_prefix) :]
        return fqn

    def _model_name(self, fqn: str) -> str:
        for model_prefix, export_prefix in self.PROJECTOR_EXPORT_RENAMES.items():
            if fqn.startswith(export_prefix):
                return model_prefix + fqn[len(export_prefix) :]
        return fqn

    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: torch.Tensor,
        **kwargs: Any,
    ) -> list[tuple[str, torch.Tensor]]:
        # Activation checkpoint wrappers become part of state_dict FQNs.
        # Serving-engine refit expects the original HF names; leaving the
        # wrapper segment intact silently skips those weights and leaves the
        # rollout engine's dummy initialization in place.
        fqn = normalized_checkpoint_name(fqn)
        is_projector = is_projector_state_key(fqn, self.PROJECTOR_STATE_PREFIXES)
        if not self.include_projector_state and is_projector:
            return []
        converted = cast(Any, super()).convert_single_tensor_to_hf(self._export_name(fqn), tensor, **kwargs)
        return prefix_hf_checkpoint_keys(
            converted,
            prefix=str(getattr(self.config, "hf_checkpoint_prefix", "")),
            is_projector_state=is_projector,
        )

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        hf_state_dict = select_hf_checkpoint_subtree(
            hf_state_dict,
            prefix=str(getattr(self.config, "hf_checkpoint_prefix", "")),
        )
        if self.PROJECTOR_EXPORT_RENAMES:
            hf_state_dict = {self._model_name(name): tensor for name, tensor in hf_state_dict.items()}
        return cast(Any, super()).from_hf(hf_state_dict, *args, **kwargs)
