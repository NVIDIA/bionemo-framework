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

"""Modality-neutral callback helpers for the NeMo RL encoder transport."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nemotron_stitch.features.cache import open_feature_cache
from nemotron_stitch.projector import MultimodalProjector
from nemotron_stitch.projector.artifact import read_projector_artifact


@lru_cache(maxsize=32)
def resolve_callback(fqn: str) -> Callable:
    module_name, separator, attribute = fqn.rpartition(".")
    if not separator:
        raise ValueError(f"callback must be a fully qualified name: {fqn!r}")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError(f"configured callback is not callable: {fqn}")
    return callback


@lru_cache(maxsize=4)
def frozen_projector(artifact_dir: str) -> MultimodalProjector:
    """Load a frozen sidecar used to project raw features for rollout."""
    artifact = read_projector_artifact(Path(artifact_dir).resolve())
    registry = MultimodalProjector.from_config(
        artifact.manifest.projector_configs, output_size=artifact.manifest.output_size
    )
    registry.to(dtype=torch.bfloat16)
    registry.load_state_dict(artifact.state, strict=True)
    registry.eval().requires_grad_(False)
    return registry


def project_features(features: torch.Tensor, artifact_dir: str, adapter: str) -> torch.Tensor:
    """Project one raw feature tensor to the frozen rollout token contract."""
    registry = frozen_projector(str(Path(artifact_dir).resolve()))
    module = registry.projectors[adapter]
    parameter = next(module.parameters())
    with torch.inference_mode():
        projected = module(features.unsqueeze(0).to(dtype=parameter.dtype)).cpu()
    if projected.ndim != 3 or projected.shape[0] != 1 or projected.shape[-1] != module.output_size:
        raise RuntimeError(f"frozen projector emitted an invalid shape: {tuple(projected.shape)}")
    return projected.squeeze(0).contiguous()


def load_cached_features(
    datum: dict[str, Any],
    adapter: str,
    *,
    cache_root: str,
    artifact_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load raw cached features and project the identical tensor for rollout."""
    cache = open_feature_cache(cache_root)
    features = torch.from_numpy(np.asarray(cache.get(datum["feature_key"])))
    return features, project_features(features, artifact_dir, adapter)


def load_cached_policy_features(datum: dict[str, Any], adapter: str, *, cache_root: str) -> torch.Tensor:
    """Read raw features for a policy-only preference pair."""
    return torch.from_numpy(np.asarray(open_feature_cache(cache_root).get(datum["feature_key"])))


def ground_truth_metadata(datum: dict[str, Any]) -> dict[str, Any]:
    """Expose a manifest row's ground truth to an exact-match reward."""
    return {"ground_truth": str(datum["ground_truth"])}


def load_encoder_payload(
    datum: dict[str, Any],
    adapter: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invoke the configured raw-loader/projector callback."""
    callback = resolve_callback(str(datum["encoder_loader"]))
    result = callback(datum, adapter, **dict(datum.get("encoder_loader_kwargs") or {}))
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("encoder loader must return (raw_features, projected_tokens)")
    raw, projected = result
    if not isinstance(raw, torch.Tensor) or not isinstance(projected, torch.Tensor):
        raise TypeError("encoder loader outputs must be torch tensors")
    if projected.ndim != 2:
        raise ValueError(f"projected encoder payload must be [tokens, hidden], got {tuple(projected.shape)}")
    return raw, projected


def task_reward_metadata(datum: dict[str, Any]) -> dict[str, Any]:
    """Invoke the optional task callback that selects environment metadata."""
    fqn = datum.get("task_metadata_callback")
    if not fqn:
        return {}
    value = resolve_callback(str(fqn))(datum)
    if not isinstance(value, dict):
        raise TypeError("task metadata callback must return a mapping")
    return value
