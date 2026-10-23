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

"""Parameter-family trainability and projector optimizer configuration."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from nemotron_stitch.automodel.optim import ProjectorAdamWConfig
from nemotron_stitch.projector import Mlp2xGeluProjector, MultimodalProjector
from nemotron_stitch.projector.trainability import (
    TrainabilityPolicy,
    collect_extra_state,
    configure_trainable_parameters,
    extra_parameter_names,
    validate_trainable_manifest,
    write_manifest_rank_zero,
)


class _Model(nn.Module):
    """In-tree (module-ownership) shape: base + projector + extra + lora."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.projection = nn.Linear(4, 4)
        self.marker_delta = nn.Parameter(torch.zeros(4))
        self.lora_layer = nn.Linear(4, 4)


POLICY = TrainabilityPolicy(projector_patterns=("projection",), extra_patterns=("marker_delta",))


def test_stage1_trains_only_the_projector():
    model = _Model()
    manifest = configure_trainable_parameters(model, 1, policy=POLICY)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"projection.weight", "projection.bias"}
    assert [entry["name"] for entry in manifest["parameters"]] == sorted(trainable)
    assert manifest["total_trainable_parameters"] == sum(p.numel() for p in model.projection.parameters())


def test_stage2_adds_lora():
    model = _Model()
    configure_trainable_parameters(model, 2, policy=POLICY)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        "projection.weight",
        "projection.bias",
        "lora_layer.weight",
        "lora_layer.bias",
    }


def test_stage2_with_frozen_projector():
    # Stage 2: LoRA only, projector frozen after warm start.
    model = _Model()
    configure_trainable_parameters(model, 2, policy=POLICY, train_projector=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"lora_layer.weight", "lora_layer.bias"}


def test_extra_family_gated_by_flag():
    model = _Model()
    configure_trainable_parameters(model, 1, policy=POLICY, train_extra=True)
    assert model.marker_delta.requires_grad
    model = _Model()
    configure_trainable_parameters(model, 1, policy=POLICY, train_extra=False)
    assert not model.marker_delta.requires_grad


def test_extra_parameter_names_and_collect_extra_state():
    model = _Model()
    assert extra_parameter_names(model, POLICY) == ["marker_delta"]
    with torch.no_grad():
        model.marker_delta.fill_(1.5)
    state = collect_extra_state(model, POLICY)
    assert set(state) == {"marker_delta"}
    torch.testing.assert_close(state["marker_delta"], torch.full((4,), 1.5))
    # The artifact codec owns serialization; collected tensors are detached copies.
    assert not state["marker_delta"].requires_grad


def test_unexpected_trainable_fails_closed():
    # configure_trainable_parameters sets requires_grad per policy, so a stray
    # trainable is only visible to the standalone validation pass (what the
    # recipe runs after other mutations).
    model = _Model()
    configure_trainable_parameters(model, 1, policy=POLICY)
    model.base.weight.requires_grad_(True)  # a stray the policy did not enable
    with pytest.raises(ValueError, match="unexpected trainable parameters: base.weight"):
        validate_trainable_manifest(model, 1, policy=POLICY)


def test_missing_projector_fails_closed():
    model = _Model()
    del model.projection  # the family is absent, not merely frozen
    with pytest.raises(ValueError, match="no projector parameters are trainable"):
        configure_trainable_parameters(model, 1, policy=POLICY)


def test_stage2_without_lora_fails_closed():
    model = _Model()
    del model.lora_layer
    with pytest.raises(ValueError, match="no LoRA parameters"):
        configure_trainable_parameters(model, 2, policy=POLICY)


def test_extra_enabled_but_absent_fails_closed():
    model = _Model()
    del model.marker_delta
    with pytest.raises(ValueError, match="extra parameters are enabled"):
        configure_trainable_parameters(model, 1, policy=POLICY, train_extra=True)


def test_bad_stage_fails_closed():
    with pytest.raises(ValueError, match="must be 1 or 2"):
        configure_trainable_parameters(_Model(), 3, policy=POLICY)


class _SidecarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(4, 8, 4)})
        object.__setattr__(self, "mm_projector", registry)


SIDECAR_POLICY = TrainabilityPolicy(sidecar_attribute="mm_projector")


def test_sidecar_projector_is_enumerated_outside_the_module_tree():
    model = _SidecarModel()
    manifest = configure_trainable_parameters(model, 1, policy=SIDECAR_POLICY)
    names = [entry["name"] for entry in manifest["parameters"]]
    assert names and all(name.startswith("mm_projector.") for name in names)
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    assert all(parameter.requires_grad for parameter in model.mm_projector.parameters())


def test_sidecar_policy_rejects_a_registered_registry():
    model = _SidecarModel()
    model.mm_projector = model.mm_projector  # nn.Module setattr registers it
    with pytest.raises(ValueError, match="registered in the module tree"):
        configure_trainable_parameters(model, 1, policy=SIDECAR_POLICY)


def test_sidecar_policy_requires_the_attribute():
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="sidecar attribute the model lacks"):
        configure_trainable_parameters(model, 1, policy=SIDECAR_POLICY)


def test_manifest_is_deterministic(tmp_path):
    model = _Model()
    first = configure_trainable_parameters(model, 2, policy=POLICY)
    second = configure_trainable_parameters(model, 2, policy=POLICY)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    destination = tmp_path / "manifest.json"
    write_manifest_rank_zero(first, str(destination))
    assert json.loads(destination.read_text()) == json.loads(json.dumps(first))


def test_projector_adamw_applies_policy_then_builds():
    model = _Model()
    config = ProjectorAdamWConfig(policy=POLICY, stage=1)
    optimizers = config.build(model)
    assert len(optimizers) == 1
    assert optimizers[0].param_groups[0]["params"] == list(model.projection.parameters())
    assert not model.base.weight.requires_grad


def test_projector_adamw_requires_a_policy():
    with pytest.raises(ValueError, match="requires a TrainabilityPolicy"):
        ProjectorAdamWConfig().build(_Model())


def test_projector_adamw_stage2_frozen_projector():
    # Stage 2: LoRA only, projector frozen after warm start.
    model = _Model()
    config = ProjectorAdamWConfig(policy=POLICY, stage=2, train_projector=False)
    optimizers = config.build(model)
    assert optimizers[0].param_groups[0]["params"] == list(model.lora_layer.parameters())
    assert not any(parameter.requires_grad for parameter in model.projection.parameters())


@pytest.mark.parametrize("train_projector", [False, True])
def test_full_decoder_update_preserves_encoder_and_upstream_freezes(train_projector):
    """A real optimizer step changes selected decoder weights, never frozen families."""
    model = _Model()
    del model.lora_layer
    model.encoder = nn.Linear(4, 4)
    model.base.bias.requires_grad_(False)  # An upstream protected parameter.
    policy = TrainabilityPolicy(projector_patterns=("projection",), decoder_patterns=("base.",))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    config = ProjectorAdamWConfig(policy=policy, stage=2, train_decoder=True, train_projector=train_projector)
    optimizer = config.build(model)[0]
    model.projection(model.base(torch.ones(2, 4))).sum().backward()
    optimizer.step()
    changed = {name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])}
    expected = {"base.weight"}
    if train_projector:
        expected.update({"projection.weight", "projection.bias"})
    assert changed == expected
    assert not model.base.bias.requires_grad
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    manifest = validate_trainable_manifest(model, 2, policy=policy, train_decoder=True, train_projector=train_projector)
    assert manifest["train_decoder"] is True


def test_full_decoder_requires_explicit_names_and_rejects_peft():
    """Misspelled/full-with-LoRA settings fail before an optimizer can train them."""
    model = _Model()
    with pytest.raises(ValueError, match="requires nonempty decoder_patterns"):
        configure_trainable_parameters(model, 2, policy=POLICY, train_decoder=True)
    policy = TrainabilityPolicy(projector_patterns=("projection",), decoder_patterns=("base.",))
    with pytest.raises(ValueError, match="cannot contain LoRA"):
        configure_trainable_parameters(model, 2, policy=policy, train_decoder=True)
    del model.lora_layer
    missing = TrainabilityPolicy(projector_patterns=("projection",), decoder_patterns=("missing.",))
    with pytest.raises(ValueError, match="Decoder pattern matched no parameters"):
        configure_trainable_parameters(model, 2, policy=missing, train_decoder=True)
    with pytest.raises(ValueError, match="requires stage 2"):
        configure_trainable_parameters(model, 1, policy=policy, train_decoder=True)


def test_full_decoder_rejects_one_missing_selector_even_when_another_matches():
    """An HF export name must not silently exclude the native decoder body."""
    model = _Model()
    del model.lora_layer
    policy = TrainabilityPolicy(projector_patterns=("projection",), decoder_patterns=("base.", "backbone."))
    with pytest.raises(ValueError, match="Decoder pattern matched no parameters: 'backbone.'"):
        configure_trainable_parameters(model, 2, policy=policy, train_decoder=True)
