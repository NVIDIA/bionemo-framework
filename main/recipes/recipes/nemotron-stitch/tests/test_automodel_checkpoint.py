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

"""Checkpoint copying, projector exports, and conservation audits."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from nemotron_stitch.automodel.checkpoint import (
    ProjectorStateDictAdapterMixin,
    collect_portable_states,
    copy_full_parameter,
    is_projector_state_key,
    normalized_checkpoint_name,
    prefix_hf_checkpoint_keys,
    select_hf_checkpoint_subtree,
)

PREFIXES = ("projection.", "marker_delta", "encoder.")


def test_normalized_checkpoint_name_strips_wrapper_segments():
    assert (
        normalized_checkpoint_name("model.layers.0._checkpoint_wrapped_module.mixer.weight")
        == "model.layers.0.mixer.weight"
    )
    assert normalized_checkpoint_name("_checkpoint_wrapped_module") == ""
    assert normalized_checkpoint_name("model._checkpoint_wrapped_module") == "model"
    assert normalized_checkpoint_name("model.layers.0.mixer.weight") == "model.layers.0.mixer.weight"


def test_is_projector_state_key_matches_exact_names_and_prefixes():
    assert is_projector_state_key("projection.fc1.weight", PREFIXES)
    assert is_projector_state_key("marker_delta", PREFIXES)
    assert is_projector_state_key("encoder.blocks.0.weight", PREFIXES)
    assert not is_projector_state_key("backbone.layers.0.weight", PREFIXES)


def test_hf_checkpoint_prefix_is_applied_only_to_base_model_state():
    converted = [("backbone.layers.0.weight", torch.ones(1))]

    prefixed = prefix_hf_checkpoint_keys(converted, prefix="language_model.", is_projector_state=False)
    projector_state = prefix_hf_checkpoint_keys(
        [("projection.fc1.weight", torch.ones(1))],
        prefix="language_model.",
        is_projector_state=True,
    )

    assert [name for name, _ in prefixed] == ["language_model.backbone.layers.0.weight"]
    assert [name for name, _ in projector_state] == ["projection.fc1.weight"]

    selected = select_hf_checkpoint_subtree(
        {
            "language_model.backbone.layers.0.weight": torch.ones(1),
            "vision_model.layers.0.weight": torch.zeros(1),
        },
        prefix="language_model.",
    )
    assert list(selected) == ["backbone.layers.0.weight"]
    # Empty prefix is the identity in both directions.
    assert select_hf_checkpoint_subtree({"a": 1}, prefix="") == {"a": 1}
    assert prefix_hf_checkpoint_keys([("a", 1)], prefix="", is_projector_state=False) == [("a", 1)]


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.projection = nn.Linear(3, 4)
        self.marker_delta = nn.Parameter(torch.zeros(2, 4))
        self.lora_A = nn.Parameter(torch.zeros(2, 4))


def test_collect_portable_states_partitions_the_three_families():
    projector, extra, adapter = collect_portable_states(
        _ToyModel(),
        projector_prefix="projection.",
        extra_names=("marker_delta",),
    )
    assert set(projector) == {"weight", "bias"}
    assert set(extra) == {"marker_delta"}
    assert set(adapter) == {"lora_A"}


def test_copy_full_parameter_round_trips_a_plain_parameter():
    parameter = nn.Parameter(torch.zeros(2, 3))
    tensor = torch.randn(2, 3)
    copy_full_parameter(parameter, tensor)
    assert torch.equal(parameter.data, tensor)


class _BaseAdapter:
    """Stand-in for an AutoModel model-family state-dict adapter."""

    def __init__(self, config):
        self.config = config

    def convert_single_tensor_to_hf(self, fqn, tensor, **_kwargs):
        return [(fqn, tensor)]

    def from_hf(self, hf_state_dict, *_args, **_kwargs):
        return hf_state_dict


class _Adapter(ProjectorStateDictAdapterMixin, _BaseAdapter):
    PROJECTOR_STATE_PREFIXES = PREFIXES
    LOAD_PROJECTOR_STATE_ATTR = "load_projection_state_from_hf"


def _make_adapter(**config):
    return _Adapter(SimpleNamespace(hf_checkpoint_prefix="", **config))


def test_adapter_excludes_projector_state_until_resumes_need_it():
    adapter = _make_adapter()
    assert adapter.include_projector_state is False
    assert adapter.convert_single_tensor_to_hf("projection.fc1.weight", torch.ones(1)) == []
    assert adapter.convert_single_tensor_to_hf("backbone.weight", torch.ones(1)) != []

    adapter.set_include_projector_state(True)
    assert adapter.include_projector_state is True
    converted = adapter.convert_single_tensor_to_hf("projection.fc1.weight", torch.ones(1))
    assert [name for name, _ in converted] == ["projection.fc1.weight"]


def test_adapter_honors_the_config_flag_and_nested_backbone_prefix():
    adapter = _Adapter(
        SimpleNamespace(
            hf_checkpoint_prefix="language_model.",
            load_projection_state_from_hf=True,
        )
    )
    assert adapter.include_projector_state is True

    converted = adapter.convert_single_tensor_to_hf("backbone.weight", torch.ones(1))
    assert [name for name, _ in converted] == ["language_model.backbone.weight"]
    # Projector state stays at the checkpoint root even when included.
    converted = adapter.convert_single_tensor_to_hf("projection.fc1.weight", torch.ones(1))
    assert [name for name, _ in converted] == ["projection.fc1.weight"]

    selected = adapter.from_hf({"language_model.backbone.weight": torch.ones(1)})
    assert list(selected) == ["backbone.weight"]


def test_adapter_strips_activation_checkpoint_wrappers_from_fqns():
    adapter = _make_adapter()
    converted = adapter.convert_single_tensor_to_hf(
        "model.layers.0._checkpoint_wrapped_module.mixer.weight",
        torch.ones(1),
    )
    assert converted[0][0] == "model.layers.0.mixer.weight"


def test_adapter_fails_closed_without_prefixes():
    class _Unconfigured(ProjectorStateDictAdapterMixin, _BaseAdapter):
        pass

    adapter = _Unconfigured(SimpleNamespace(hf_checkpoint_prefix=""))
    # No projector prefixes configured: nothing is excluded.
    assert adapter.convert_single_tensor_to_hf("projection.fc1.weight", torch.ones(1)) != []


def test_adapter_renames_projector_state_on_export_and_import():
    """A model-internal projector rename keeps the HF contract."""

    class _RenamedAdapter(ProjectorStateDictAdapterMixin, _BaseAdapter):
        PROJECTOR_STATE_PREFIXES = ("mm_projector.projectors.demo.", "marker_delta")
        PROJECTOR_EXPORT_RENAMES = {"mm_projector.projectors.demo.": "demo_projection."}

    adapter = _RenamedAdapter(SimpleNamespace(hf_checkpoint_prefix=""))

    # Excluded by default, regardless of rename.
    assert adapter.convert_single_tensor_to_hf("mm_projector.projectors.demo.fc1.weight", 1) == []

    adapter.set_include_projector_state(True)
    converted = adapter.convert_single_tensor_to_hf("mm_projector.projectors.demo.fc1.weight", torch.ones(1))
    assert [name for name, _ in converted] == ["demo_projection.fc1.weight"]

    # Inbound: an export-written checkpoint maps back to the model's names.
    loaded = adapter.from_hf({"demo_projection.fc1.weight": torch.ones(1)})
    assert list(loaded) == ["mm_projector.projectors.demo.fc1.weight"]
