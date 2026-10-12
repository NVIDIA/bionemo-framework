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

"""Validated, deterministic encoder registry.

An :class:`EncoderSpec` pins one encoder configuration by an immutable
``(repository, revision, implementation_revision, layer)`` identity; an
:class:`EncoderRegistry` is a locked tuple of specs plus a policy mapping.
Only structural rules live here (immutable revisions, positive geometry,
consistent soft-token caps). Locked consumer policies — a fixed config count,
a required overlap — are supplied by the consumer as *validators*: callables
passed to ``from_dict``/``load`` (or defaulted on a subclass) that raise on a
violation.

Consumers that keep ``source_kind``/``unit`` as enums subclass
:class:`EncoderSpec`, narrow those fields, and pre-convert in ``from_dict``;
``to_dict`` serializes either form identically, so registry digests do not
depend on which side constructed the spec.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from nemotron_stitch.provenance import sha256_json

RegistryValidator = Callable[["EncoderRegistry"], None]


def _serialized(value: Any) -> str:
    # source_kind/unit may be plain strings or consumer str-enums; both must
    # serialize (and hash) identically.
    return str(getattr(value, "value", value))


@dataclass(frozen=True)
class EncoderSpec:
    config_id: str
    family: str
    repository: str
    revision: str
    implementation_revision: str
    tokenizer_revision: str
    source_kind: str
    unit: str
    parameter_count: int
    hidden_size: int
    context_units: int
    candidate_layers: tuple[int, ...]
    chosen_layer: int | None
    overlap: int
    pooling_width: int
    max_soft_tokens: int | None
    dtype: str
    license: str
    license_decision: str
    third_party_notices: tuple[str, ...] = ()
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EncoderSpec:
        data = dict(value)
        data["candidate_layers"] = tuple(int(x) for x in data["candidate_layers"])
        data["third_party_notices"] = tuple(data.get("third_party_notices", ()))
        return cls(**data)

    def validate(self) -> None:
        if not self.config_id or self.config_id != self.config_id.lower():
            raise ValueError("config_id must be non-empty lowercase")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError(f"{self.config_id}: revision must be an immutable 40-char SHA")
        if self.tokenizer_revision != self.revision:
            raise ValueError(f"{self.config_id}: tokenizer revision must equal checkpoint revision")
        for name in ("parameter_count", "hidden_size", "context_units", "pooling_width"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{self.config_id}: {name} must be positive")
        if self.overlap < 0 or self.overlap >= self.context_units:
            raise ValueError(f"{self.config_id}: invalid overlap")
        if self.max_soft_tokens is not None and self.max_soft_tokens <= 0:
            raise ValueError(f"{self.config_id}: max_soft_tokens must be positive or null")
        if not self.candidate_layers:
            raise ValueError(f"{self.config_id}: candidate_layers must not be empty")
        if self.chosen_layer is not None and self.chosen_layer not in self.candidate_layers:
            raise ValueError(f"{self.config_id}: chosen layer is not a candidate")
        if self.license_decision != "approved":
            raise ValueError(f"{self.config_id}: license is not approved")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_kind"] = _serialized(self.source_kind)
        value["unit"] = _serialized(self.unit)
        value["candidate_layers"] = list(self.candidate_layers)
        value["third_party_notices"] = list(self.third_party_notices)
        return value


@dataclass(frozen=True)
class EncoderRegistry:
    spec_type: ClassVar[type[EncoderSpec]] = EncoderSpec

    schema_version: int
    policy: Mapping[str, Any]
    encoders: tuple[EncoderSpec, ...]
    # Consumer policy predicates; excluded from equality so two
    # loads of the same payload compare equal regardless of validator binding.
    validators: tuple[RegistryValidator, ...] = field(default=(), compare=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, validators: Iterable[RegistryValidator] = ()) -> EncoderRegistry:
        registry = cls(
            schema_version=int(value["schema_version"]),
            policy=dict(value["policy"]),
            encoders=tuple(cls.spec_type.from_dict(row) for row in value["encoders"]),
            validators=tuple(validators),
        )
        registry.validate()
        return registry

    @classmethod
    def load(cls, path: str | Path, *, validators: Iterable[RegistryValidator] = ()) -> EncoderRegistry:
        # Registry locks are deliberately JSON-compatible YAML so the base wheel has no YAML dependency.
        return cls.from_dict(json.loads(Path(path).read_text()), validators=validators)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported registry schema")
        ids = [row.config_id for row in self.encoders]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate config_id")
        for row in self.encoders:
            row.validate()
        policy_cap = self.policy.get("max_soft_tokens")
        if policy_cap is not None and (not isinstance(policy_cap, int) or policy_cap <= 0):
            raise ValueError("registry max_soft_tokens policy must be positive or null")
        inconsistent = [row.config_id for row in self.encoders if row.max_soft_tokens != policy_cap]
        if inconsistent:
            raise ValueError(f"encoder max_soft_tokens values differ from registry policy: {inconsistent[:3]}")
        for validator in self.validators:
            validator(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": dict(self.policy),
            "encoders": [row.to_dict() for row in self.encoders],
        }

    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def by_id(self, config_id: str) -> EncoderSpec:
        return next(row for row in self.encoders if row.config_id == config_id)

    def families(self) -> Iterable[str]:
        return sorted({row.family for row in self.encoders})
