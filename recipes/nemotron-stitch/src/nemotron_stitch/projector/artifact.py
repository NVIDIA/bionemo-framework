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

"""The projector sidecar artifact.

One format with an explicit format discriminator: readers dispatch on
``MANIFEST_FORMAT``, never on a bare integer, because a legacy consumer schema
(gr-v1) also wrote ``"schema_version": 1``. Unknown top-level fields are
allowed so newer writers stay readable; known fields validate strictly and
fail closed.

On-disk layout: ``mm-projector-manifest.json`` plus
``mm-projector.safetensors`` (plus ``extra/<name>.safetensors`` for named
project tensors).

``load_projector_artifact`` loads an artifact into a host model's registered
``mm_projector``; the GRPO data plane needs the same registry detached from
any model, to project cached encoder features into the rollout soft tokens.
``FrozenProjector`` is that detached, frozen form.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file

from nemotron_stitch.automodel.checkpoint import copy_full_parameter
from nemotron_stitch.contracts import (
    EXTRA_STATE_DIR,
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    OWNERSHIP_SIDECAR,
    PROJECTOR_FILENAME,
    SCHEMA_VERSION,
)
from nemotron_stitch.projector import MultimodalProjector
from nemotron_stitch.projector.trainability import TrainabilityPolicy, extra_parameter_names
from nemotron_stitch.provenance import canonical_json, require_provenance, sha256_file


@dataclass
class ProjectorManifest:
    output_size: int
    projector_configs: list[dict[str, Any]]
    projectors: dict[str, dict[str, Any]]
    provenance: dict[str, Any]
    stage: int | None = None
    extra_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    peft: dict[str, Any] | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": MANIFEST_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "output_size": self.output_size,
            "provenance": self.provenance,
            "projector_configs": self.projector_configs,
            "projectors": self.projectors,
            "extra_state": self.extra_state,
            "config_snapshot": self.config_snapshot,
            "checksums": self.checksums,
        }
        if self.stage is not None:
            payload["stage"] = self.stage
        if self.peft is not None:
            payload["peft"] = self.peft
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectorManifest:
        if not isinstance(payload, dict):
            raise TypeError("manifest must be a mapping")
        schema = payload.get("schema")
        if schema != MANIFEST_FORMAT:
            raise ValueError(f"unsupported manifest format: {schema!r}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema_version: {payload.get('schema_version')!r}")
        # The casts keep the parse total: a malformed manifest fails closed in
        # validate() below with a field-level error, not with a KeyError here.
        manifest = cls(
            output_size=cast(int, payload.get("output_size")),
            projector_configs=cast(list[dict[str, Any]], payload.get("projector_configs")),
            projectors=cast(dict[str, dict[str, Any]], payload.get("projectors")),
            provenance=cast(dict[str, Any], payload.get("provenance")),
            stage=payload.get("stage"),
            extra_state=payload.get("extra_state") or {},
            peft=payload.get("peft"),
            config_snapshot=payload.get("config_snapshot") or {},
            checksums=payload.get("checksums") or {},
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not isinstance(self.output_size, int) or self.output_size <= 0:
            raise ValueError(f"output_size must be a positive integer, got {self.output_size!r}")
        if not isinstance(self.projector_configs, list) or not self.projector_configs:
            raise ValueError("projector_configs must be a nonempty list")
        names = set()
        for config in self.projector_configs:
            well_formed = (
                isinstance(config, dict) and isinstance(config.get("name"), str) and isinstance(config.get("kind"), str)
            )
            if not well_formed:
                raise ValueError(f"projector configs must carry string name and kind: {config!r}")
            names.add(config["name"])
        if not isinstance(self.projectors, dict) or set(self.projectors) != names:
            raise ValueError("projectors keys must match the projector_configs names")
        for name, entry in self.projectors.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("class"), str):
                raise ValueError(f"projector {name!r} must record a class name")
            if not isinstance(entry.get("parameters"), int) or entry["parameters"] < 0:
                raise ValueError(f"projector {name!r} must record a nonnegative parameter count")
            if entry.get("num_tokens") is not None and not isinstance(entry["num_tokens"], int):
                raise ValueError(f"projector {name!r} num_tokens must be an integer or null")
        if self.stage is not None and not isinstance(self.stage, int):
            raise ValueError(f"stage must be an integer or null, got {self.stage!r}")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a mapping")
        for name, entry in self.extra_state.items():
            if not isinstance(entry, dict) or "shape" not in entry or "dtype" not in entry:
                raise ValueError(f"extra_state {name!r} must record shape and dtype")
        if not isinstance(self.checksums, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.checksums.items()
        ):
            raise ValueError("checksums must map filenames to sha256 hex digests")


@dataclass
class ProjectorArtifact:
    """A verified artifact read from disk."""

    manifest: ProjectorManifest
    state: dict[str, torch.Tensor]
    extra_state: dict[str, torch.Tensor]


def _distributed_rank() -> int | None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return None


def _portable_registry_state(registry) -> dict[str, torch.Tensor]:
    """Collect the registry's full (unsharded) state on every rank.

    Under FSDP2 the registry's state_dict values are DTensors;
    ``full_tensor()`` is a collective, so this runs on all ranks and the
    caller writes on rank zero only.
    """
    state = {}
    for key, value in registry.state_dict().items():
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        state[key] = value.detach().cpu().contiguous()
    return state


def save_projector_artifact(
    model,
    destination: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
    stage: int | None = None,
    peft: dict[str, Any] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    extra_state: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Write the package-format sidecar for a model's projector registry.

    DTensor-safe: every rank participates in the full-tensor collection and
    rank zero writes, so the same call serves sidecar (one-rank) and
    module-ownership (FSDP2-sharded) exports.
    """
    registry = model.mm_projector
    state = _portable_registry_state(registry)
    projector_configs = getattr(getattr(model, "config", None), "mm_projectors", [])
    output_sizes = {int(projector.output_size) for projector in registry.projectors.values()}
    if len(output_sizes) != 1:
        raise ValueError(f"projectors must share one LM width, got {sorted(output_sizes)}")
    manifest = ProjectorManifest(
        output_size=output_sizes.pop(),
        projector_configs=[dict(config) for config in projector_configs],
        projectors={
            name: {
                "class": type(projector).__name__,
                "parameters": sum(parameter.numel() for parameter in projector.parameters()),
                "num_tokens": getattr(projector, "num_tokens", None),
            }
            for name, projector in registry.projectors.items()
        },
        provenance=dict(provenance or {}),
        stage=stage,
        peft=peft,
        config_snapshot=config_snapshot or {},
    )
    if _distributed_rank() not in (None, 0):
        torch.distributed.barrier()
        return Path(destination)
    write_artifact_files(destination, manifest, state, extra_state)
    if _distributed_rank() == 0:
        torch.distributed.barrier()
    return Path(destination)


def write_artifact_files(
    destination: str | Path,
    manifest: ProjectorManifest,
    state: dict[str, torch.Tensor],
    extra_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Write state files, then the manifest with their checksums."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(state, destination / PROJECTOR_FILENAME)

    extra_state = extra_state or {}
    extra_manifest = {}
    if extra_state:
        (destination / EXTRA_STATE_DIR).mkdir(exist_ok=True)
    for name, tensor in extra_state.items():
        tensor = tensor.detach().cpu().contiguous()
        save_file({name: tensor}, destination / EXTRA_STATE_DIR / f"{name}.safetensors")
        extra_manifest[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    manifest.extra_state = extra_manifest

    written = [PROJECTOR_FILENAME] + [f"{EXTRA_STATE_DIR}/{name}.safetensors" for name in extra_state]
    manifest.checksums = {filename: sha256_file(destination / filename) for filename in written}
    manifest.validate()
    manifest_path = destination / MANIFEST_FILENAME
    manifest_path.write_text(canonical_json(manifest.to_dict()) + "\n")
    manifest_path.chmod(0o644)
    (destination / PROJECTOR_FILENAME).chmod(0o644)


def read_projector_artifact(source: str | Path, *, expected: dict[str, Any] | None = None) -> ProjectorArtifact:
    """Read a package sidecar, checksum-verified and provenance-checked."""
    source = Path(source)
    manifest_path = source / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"no {MANIFEST_FILENAME} at {source}")
    manifest = ProjectorManifest.from_dict(json.loads(manifest_path.read_text()))
    required_checksums = {
        PROJECTOR_FILENAME,
        *(f"{EXTRA_STATE_DIR}/{name}.safetensors" for name in manifest.extra_state),
    }
    missing_checksums = sorted(required_checksums - set(manifest.checksums))
    if missing_checksums:
        raise ValueError(f"artifact is missing checksums for: {', '.join(missing_checksums)}")
    for filename, digest in manifest.checksums.items():
        observed = sha256_file(source / filename)
        if observed != digest:
            raise ValueError(f"checksum mismatch for {filename} in {source}: {observed} != {digest}")
    state = load_file(source / PROJECTOR_FILENAME, device="cpu")
    extra_state = {
        name: load_file(source / EXTRA_STATE_DIR / f"{name}.safetensors", device="cpu")[name]
        for name in manifest.extra_state
    }
    artifact = ProjectorArtifact(manifest, state, extra_state)
    if expected is not None:
        if not expected:
            raise ValueError("expected provenance must not be empty")
        require_provenance(artifact.manifest.provenance, expected, context="projector sidecar")
    return artifact


def load_projector_artifact(
    model,
    source: str | Path,
    expected: dict[str, Any] | None = None,
    *,
    extra_patterns: tuple[str, ...] = (),
) -> ProjectorManifest:
    """Load a sidecar into a model's projector registry and extra project state.

    DTensor-safe: a verified artifact holds full tensors, which
    ``copy_full_parameter`` redistributes into possibly FSDP2-sharded
    parameters, so the same call serves sidecar (one-rank) and
    module-ownership warm starts. Key-exact, like ``load_state_dict(strict=True)``.

    Extra state (the trainability policy's EXTRA family — learned marker
    embeddings and kin) restores by dotted parameter/buffer name on the
    model: every artifact entry must resolve to a same-shaped model tensor.
    When ``extra_patterns`` are given, the artifact's extra state must also
    cover the model's extra family exactly — an artifact missing the
    configured markers is a wrong artifact, not a fresh start.
    """
    artifact = read_projector_artifact(source, expected=expected)
    registry = model.mm_projector
    targets: dict[str, torch.Tensor] = dict(registry.named_parameters())
    targets.update(registry.named_buffers())
    missing = sorted(set(targets) - set(artifact.state))
    unexpected = sorted(set(artifact.state) - set(targets))
    if missing or unexpected:
        raise ValueError(
            f"projector artifact keys do not match the registry: missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    for name, tensor in artifact.state.items():
        copy_full_parameter(targets[name], tensor)
    if extra_patterns:
        family = set(extra_parameter_names(model, TrainabilityPolicy(extra_patterns=extra_patterns)))
        if family != set(artifact.extra_state):
            raise ValueError(
                f"extra state does not match the configured extra family: model expects "
                f"{sorted(family)}, artifact carries {sorted(artifact.extra_state)}"
            )
    if artifact.extra_state:
        named: dict[str, torch.Tensor] = dict(model.named_parameters())
        named.update(model.named_buffers())
        unknown = sorted(set(artifact.extra_state) - set(named))
        if unknown:
            raise ValueError(f"artifact extra state has no counterpart on the model: {unknown[:4]}")
        for name, tensor in artifact.extra_state.items():
            if tuple(named[name].shape) != tuple(tensor.shape):
                raise ValueError(
                    f"extra state shape mismatch for {name!r}: "
                    f"artifact {tuple(tensor.shape)} != model {tuple(named[name].shape)}"
                )
            copy_full_parameter(named[name], tensor)
    return artifact.manifest


class FrozenProjector:
    """A frozen, model-detached projector registry backed by a package artifact."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        projector_config: dict[str, Any],
        output_size: int,
        expected_provenance: dict[str, Any] | None = None,
    ) -> None:
        name = str(projector_config.get("name", ""))
        if not name:
            raise ValueError("projector_config requires a nonempty name")
        artifact = read_projector_artifact(Path(artifact_dir), expected=expected_provenance)
        if artifact.manifest.output_size != int(output_size):
            raise ValueError(f"projector output width {artifact.manifest.output_size} does not match {output_size}")
        if artifact.manifest.projector_configs != [projector_config]:
            raise ValueError("projector artifact config does not match the requested projector")

        self.registry = MultimodalProjector.from_config(
            [projector_config],
            int(output_size),
            projector_ownership=OWNERSHIP_SIDECAR,
        )
        self.registry.load_state_dict(artifact.state, strict=True)
        self.registry.requires_grad_(False).eval()
        self.manifest = artifact.manifest
