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

"""Exported conformance suite.

A shared library that only shares *code* drifts. Consumers subclass
``ProjectorContractSuite`` in their own ``tests/`` with their projector config,
and the suite pins the contract the port phases rely on: scatter semantics,
projector construction/init, sidecar round-trip with provenance enforcement,
and trainability-manifest exactness across both stages.

The base class name deliberately lacks the ``Test`` prefix so pytest does not
collect it here; consumer subclasses are named ``Test*Conformance``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from nemotron_stitch.contracts import OWNERSHIP_MODES, OWNERSHIP_SIDECAR
from nemotron_stitch.projector import MultimodalProjector
from nemotron_stitch.projector.artifact import (
    load_projector_artifact,
    read_projector_artifact,
    save_projector_artifact,
)
from nemotron_stitch.projector.scatter import (
    dense_to_flat,
    derive_indices,
    scatter_flat,
)
from nemotron_stitch.projector.trainability import (
    TrainabilityPolicy,
    configure_trainable_parameters,
)
from nemotron_stitch.testing.fixtures import ragged_batch

_PLACEHOLDER = 99


class ProjectorContractSuite:
    """Subclass with ``projector_config``, ``output_size`` (and optionally ``provenance``) set.

    ``projector_ownership`` selects the host wiring: both modes
    run the same contract — scatter semantics, construction/init, artifact
    round-trip with provenance enforcement, and trainability exactness.
    """

    projector_config: dict[str, Any] = {}
    output_size: int = 0
    provenance: dict[str, Any] = {"base_model": {"repo_id": "example/base", "revision": "abc"}}
    projector_ownership: str = OWNERSHIP_SIDECAR

    @classmethod
    def _name(cls) -> str:
        return str(cls.projector_config["name"])

    @classmethod
    def _registry(cls) -> MultimodalProjector:
        if not cls.projector_config:
            raise NotImplementedError("conformance subclasses must set projector_config")
        if cls.projector_ownership not in OWNERSHIP_MODES:
            raise ValueError(f"unknown projector_ownership: {cls.projector_ownership!r}")
        return MultimodalProjector.from_config(
            [cls.projector_config], cls.output_size, projector_ownership=cls.projector_ownership
        )

    @classmethod
    def _trainability_policy(cls) -> TrainabilityPolicy:
        if cls.projector_ownership == OWNERSHIP_SIDECAR:
            return TrainabilityPolicy(sidecar_attribute="mm_projector")
        return TrainabilityPolicy(projector_patterns=("mm_projector",))

    @classmethod
    def _host(cls):
        registry = cls._registry()
        ownership = cls.projector_ownership

        class _Host(nn.Module):
            def __init__(self, projector_registry):
                super().__init__()
                if ownership == OWNERSHIP_SIDECAR:
                    # Sidecar ownership: outside the module tree.
                    object.__setattr__(self, "mm_projector", projector_registry)
                else:
                    # Module ownership: a registered, FSDP2-visible child.
                    self.mm_projector = projector_registry
                self.config = type("Cfg", (), {"mm_projectors": [cls.projector_config]})()

        return _Host(registry)

    # -- scatter contract (modality-free, shared fixtures) --------------------

    def test_scatter_matches_hand_built_expectation(self):
        batch = ragged_batch()
        out = scatter_flat(
            batch["input_ids"],
            batch["inputs_embeds"],
            batch["soft_tokens"],
            batch["flat_indices"],
            placeholder_token_id=batch["placeholder_token_id"],
        )
        torch.testing.assert_close(out, batch["expected"], rtol=0, atol=0)

    def test_placeholder_derived_equals_explicit_indices(self):
        # derive_indices requires uniform slot counts; the ragged fixture is
        # exercised above via the explicit flat path.
        input_ids = torch.tensor([[7, 99, 99, 99, 8], [9, 99, 99, 99, 10]])
        name = self._name()
        derived = derive_indices(input_ids, {name: _PLACEHOLDER}, {name: 3})[name]
        expected_dense = torch.tensor([[1, 2, 3], [1, 2, 3]])
        torch.testing.assert_close(derived, expected_dense, rtol=0, atol=0)

        soft = torch.randn(2, 3, self.output_size)
        flat_indices, flat_tokens = dense_to_flat(expected_dense, soft, sequence_length=5)
        target = torch.randn(2, 5, self.output_size)
        out = scatter_flat(input_ids, target, flat_tokens, flat_indices, placeholder_token_id=_PLACEHOLDER)
        expected = target.clone()
        expected[:, 1:4] = soft
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_scatter_rejects_overlap_and_out_of_range(self):
        batch = ragged_batch()
        with pytest.raises(ValueError):
            scatter_flat(
                batch["input_ids"],
                batch["inputs_embeds"],
                torch.cat([batch["soft_tokens"], batch["soft_tokens"][:1]]),
                torch.cat([batch["flat_indices"], batch["flat_indices"][:1]]),
            )
        with pytest.raises(ValueError):
            scatter_flat(
                batch["input_ids"],
                batch["inputs_embeds"],
                torch.cat([batch["soft_tokens"], batch["soft_tokens"][:1]]),
                torch.cat([batch["flat_indices"], torch.tensor([10_000])]),
            )

    # -- projector construction ----------------------------------------------

    def test_meta_construction_materializes_deterministically(self):
        # nn.Linear.__init__ consumes RNG at construction, and differently on
        # meta than on a real device; the contract is that reset_parameters
        # after meta materialization reproduces the seeded eager reset.
        eager = self._registry()
        torch.manual_seed(0)
        eager.reset_parameters()
        with torch.device("meta"):
            meta = MultimodalProjector.from_config([self.projector_config], self.output_size)
        assert all(p.is_meta for p in meta.parameters())
        materialized = meta.to_empty(device="cpu")
        torch.manual_seed(0)
        materialized.reset_parameters()
        for key, value in eager.state_dict().items():
            torch.testing.assert_close(materialized.state_dict()[key], value, rtol=0, atol=0)

    @classmethod
    def _features(cls, batch: int) -> torch.Tensor:
        """One batch of raw per-projector payloads. Per-projector payloads are
        batched per row; the token-wise kinds take [B, tokens, mm_hidden_size]
        and kinds over raw encoder geometry (perceiver3d) override this."""
        width = int(cls.projector_config["mm_hidden_size"])
        token_count = int(cls.projector_config.get("num_tokens", 3))
        return torch.randn(batch, token_count, width)

    def test_registry_forward_projects_and_scatters(self):
        registry = self._registry()
        name = self._name()
        token_count = int(self.projector_config.get("num_tokens", 3))
        input_ids = torch.full((1, token_count + 2), 7)
        input_ids[0, 1 : token_count + 1] = _PLACEHOLDER
        target = torch.randn(1, token_count + 2, self.output_size)
        features = self._features(1)
        out = registry(input_ids, target, {name: features}, placeholder_token_ids={name: _PLACEHOLDER})
        assert out.shape == target.shape
        # The scatter casts soft tokens to the embeds' dtype; a projector whose
        # parameters differ (mlp2x_gelu_norm is bf16 by construction) does not
        # change the output dtype.
        assert out.dtype == target.dtype
        out.sum().backward()
        assert any(p.grad is not None for p in registry.parameters())

    def test_distinct_placeholder_ids_route_by_token_identity(self):
        """Two projectors, distinct placeholder ids, segments laid out against
        sorted-name order: routing must follow token identity, not position.
        This is the multi-modality prompt contract."""
        base = dict(self.projector_config)
        configs = []
        for suffix in ("a", "b"):
            entry = dict(base)
            entry["name"] = f"{base['name']}_{suffix}"
            configs.append(entry)
        registry = MultimodalProjector.from_config(configs, self.output_size)
        name_a, name_b = (str(config["name"]) for config in configs)
        assert sorted([name_a, name_b]) == [name_a, name_b]  # the test is meaningless otherwise
        token_count = int(base.get("num_tokens", 3))
        sequence = 2 * token_count + 3
        # b's segment precedes a's in the prompt; a single shared placeholder id
        # would route b's features to a's slots by sorted-name order.
        input_ids = torch.full((2, sequence), 7)
        input_ids[:, 1 : token_count + 1] = 99  # name_b
        input_ids[:, token_count + 2 : 2 * token_count + 2] = 98  # name_a
        placeholder_ids = {name_a: 98, name_b: 99}
        features_a, features_b = self._features(2), self._features(2)
        target = torch.randn(2, sequence, self.output_size)
        out = registry(
            input_ids,
            target,
            {name_a: features_a, name_b: features_b},
            placeholder_token_ids=placeholder_ids,
        )
        param_dtype = next(registry.projectors[name_a].parameters()).dtype
        soft_a = registry.projectors[name_a](features_a.to(dtype=param_dtype))
        soft_b = registry.projectors[name_b](features_b.to(dtype=param_dtype))
        torch.testing.assert_close(out[:, 1 : token_count + 1], soft_b.to(dtype=out.dtype))
        torch.testing.assert_close(out[:, token_count + 2 : 2 * token_count + 2], soft_a.to(dtype=out.dtype))
        # An explicit index pointing at the other projector's slots misroutes;
        # the per-projector placeholder check must catch it.
        wrong_slots = torch.arange(1, token_count + 1).unsqueeze(0).expand(2, token_count)
        with pytest.raises(ValueError, match="non-placeholder"):
            registry(
                input_ids,
                target,
                {name_a: features_a, name_b: features_b},
                placeholder_token_ids=placeholder_ids,
                # name_a's indices point at name_b's (99) slots.
                token_indices_by_projector={name_a: wrong_slots, name_b: wrong_slots},
            )

    # -- artifact round-trip --------------------------------------------------

    def test_artifact_round_trip_and_provenance_enforcement(self, tmp_path):
        host = self._host()
        expected = {k: v.clone() for k, v in host.mm_projector.state_dict().items()}
        save_projector_artifact(host, tmp_path, provenance=dict(self.provenance), stage=1)
        for parameter in host.mm_projector.parameters():
            nn.init.zeros_(parameter)
        load_projector_artifact(host, tmp_path, expected=dict(self.provenance))
        for name, value in host.mm_projector.state_dict().items():
            torch.testing.assert_close(value, expected[name])
        with pytest.raises(ValueError, match="provenance"):
            read_projector_artifact(tmp_path, expected={"never": "matched"})

    # -- trainability manifest ------------------------------------------------

    def test_trainability_manifest_is_exact_across_stages(self):
        host = self._host()
        backbone = nn.Linear(4, 4)
        host.backbone = backbone
        host.lora_A = nn.Parameter(torch.zeros(2, 2))
        policy = self._trainability_policy()

        stage1 = configure_trainable_parameters(host, 1, policy=policy, train_projector=True)
        stage1_names = {entry["name"] for entry in stage1["parameters"]}
        assert stage1_names and all("mm_projector" in name for name in stage1_names)
        assert not backbone.weight.requires_grad
        assert not host.lora_A.requires_grad

        stage2 = configure_trainable_parameters(host, 2, policy=policy, train_projector=False)
        stage2_names = {entry["name"] for entry in stage2["parameters"]}
        assert stage2_names == {"lora_A"}
        assert not any(p.requires_grad for p in host.mm_projector.parameters())
