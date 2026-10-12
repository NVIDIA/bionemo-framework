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

"""Encoder registry: structural validation, consumer predicates, frozen digests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import pytest

from nemotron_stitch.features.registry import EncoderRegistry, EncoderSpec


def _spec_dict(**overrides):
    data = {
        "config_id": "toy-1b",
        "family": "toy",
        "repository": "org/toy-encoder-1b",
        "revision": "a" * 40,
        "implementation_revision": "org/impl@1.0.0",
        "tokenizer_revision": "a" * 40,
        "source_kind": "code",
        "unit": "token",
        "parameter_count": 1_000_000_000,
        "hidden_size": 512,
        "context_units": 1022,
        "candidate_layers": [10, 12],
        "chosen_layer": 12,
        "overlap": 128,
        "pooling_width": 21,
        "max_soft_tokens": 128,
        "dtype": "bfloat16",
        "license": "Apache-2.0",
        "license_decision": "approved",
    }
    data.update(overrides)
    return data


def _registry_dict(count=2, **policy):
    policy = {"overlap": 128, "max_soft_tokens": 128, **policy}
    encoders = []
    for index in range(count):
        row = _spec_dict()
        row["config_id"] = f"toy-{index}"
        encoders.append(row)
    return {"schema_version": 1, "policy": policy, "encoders": encoders}


def test_spec_validation_is_fail_closed():
    with pytest.raises(ValueError, match="immutable 40-char SHA"):
        EncoderSpec.from_dict(_spec_dict(revision="main")).validate()
    with pytest.raises(ValueError, match="lowercase"):
        EncoderSpec.from_dict(_spec_dict(config_id="Toy")).validate()
    with pytest.raises(ValueError, match="tokenizer revision must equal"):
        EncoderSpec.from_dict(_spec_dict(tokenizer_revision="b" * 40)).validate()
    with pytest.raises(ValueError, match="invalid overlap"):
        EncoderSpec.from_dict(_spec_dict(overlap=1022)).validate()
    with pytest.raises(ValueError, match="not a candidate"):
        EncoderSpec.from_dict(_spec_dict(chosen_layer=3)).validate()
    with pytest.raises(ValueError, match="license is not approved"):
        EncoderSpec.from_dict(_spec_dict(license_decision="pending")).validate()


def test_registry_round_trip_and_digest_are_deterministic():
    registry = EncoderRegistry.from_dict(_registry_dict())
    assert registry.digest() == EncoderRegistry.from_dict(registry.to_dict()).digest()
    assert registry.by_id("toy-1").config_id == "toy-1"
    assert list(registry.families()) == ["toy"]


def test_registry_rejects_duplicates_and_foreign_schema():
    value = _registry_dict()
    value["encoders"].append(value["encoders"][0])
    with pytest.raises(ValueError, match="duplicate config_id"):
        EncoderRegistry.from_dict(value)
    with pytest.raises(ValueError, match="unsupported registry schema"):
        EncoderRegistry.from_dict({**_registry_dict(), "schema_version": 2})


def test_registry_policy_cap_must_match_every_encoder():
    value = _registry_dict()
    value["encoders"][1]["max_soft_tokens"] = None
    with pytest.raises(ValueError, match="differ from registry policy"):
        EncoderRegistry.from_dict(value)
    with pytest.raises(ValueError, match="positive or null"):
        EncoderRegistry.from_dict(_registry_dict(max_soft_tokens=0))


def test_consumer_predicates_are_the_only_config_count_or_overlap_locks():
    # The package itself imposes no config-count or overlap policy...
    registry = EncoderRegistry.from_dict(_registry_dict(count=1, overlap=64))
    assert len(registry.encoders) == 1

    # ...consumers supply them as validators.
    def locked_policy(candidate: EncoderRegistry) -> None:
        if len(candidate.encoders) != 23:
            raise ValueError(f"primary registry must contain 23 configs, got {len(candidate.encoders)}")
        if candidate.policy.get("overlap") != 128:
            raise ValueError("registry violates locked overlap policy")

    with pytest.raises(ValueError, match="23 configs"):
        EncoderRegistry.from_dict(_registry_dict(count=1), validators=[locked_policy])
    with pytest.raises(ValueError, match="locked overlap policy"):
        EncoderRegistry.from_dict(_registry_dict(count=23, overlap=64), validators=[locked_policy])
    assert len(EncoderRegistry.from_dict(_registry_dict(count=23), validators=[locked_policy]).encoders) == 23


def test_registry_load_reads_json_compatible_yaml(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(json.dumps(_registry_dict()))
    registry = EncoderRegistry.load(path)
    assert [row.config_id for row in registry.encoders] == ["toy-0", "toy-1"]


def test_enum_typed_subclass_serializes_and_hashes_identically():
    class Kind(str, Enum):
        CODE = "code"

    class Unit(str, Enum):
        TOKEN = "token"

    @dataclass(frozen=True)
    class ConsumerSpec(EncoderSpec):
        source_kind: Kind
        unit: Unit

        @classmethod
        def from_dict(cls, value):
            data = dict(value)
            data["source_kind"] = Kind(data["source_kind"])
            data["unit"] = Unit(data["unit"])
            return super().from_dict(data)

    @dataclass(frozen=True)
    class ConsumerRegistry(EncoderRegistry):
        spec_type = ConsumerSpec

    plain = EncoderRegistry.from_dict(_registry_dict())
    specialized = ConsumerRegistry.from_dict(_registry_dict())
    assert specialized.encoders[0].source_kind is Kind.CODE
    assert specialized.to_dict() == plain.to_dict()
    assert specialized.digest() == plain.digest()
