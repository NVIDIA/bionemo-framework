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

"""build_mm_plugin(mode="encode") tests.

vLLM is absent from the package's base CI env, so the class-building cases
skip there and run in a consumer image; the payload-shape helper is pure
torch and runs everywhere.
"""

from __future__ import annotations

import pytest
import torch

from nemotron_stitch.vllm.plugin import build_mm_plugin, split_flat_payload


def test_split_flat_payload_normalizes_the_three_handoff_shapes():
    flat = torch.arange(12).reshape(6, 2)
    sizes = torch.tensor([2, 4], dtype=torch.int32)
    items = split_flat_payload(flat, sizes)
    assert len(items) == 2
    torch.testing.assert_close(items[0], flat[:2])
    torch.testing.assert_close(items[1], flat[2:])

    # Single-item tensor with no (or single-entry) sizes passes through whole.
    assert split_flat_payload(flat, None) == [flat]
    assert split_flat_payload(flat, torch.tensor([6])) == [flat]

    listed = [flat[:2], flat[2:]]
    assert split_flat_payload(listed, None) == listed


def test_encode_mode_requires_its_callbacks():
    with pytest.raises(ValueError, match="encode_geometry_fn"):
        build_mm_plugin(
            modality="toy",
            architecture="ToyArch",
            mode="encode",
            placeholder_text="<toy_placeholder>",
            encode_model_cls="toy.module:ToyModel",
            encode_normalize_item_fn=lambda item: item,
        )
    with pytest.raises(ValueError, match="unknown build_mm_plugin mode"):
        build_mm_plugin(
            modality="toy",
            architecture="ToyArch",
            mode="sideways",
            placeholder_text="<toy_placeholder>",
        )


def _toy_plugin():
    pytest.importorskip("vllm", reason="vLLM is not installed")

    def normalize(item):
        return [int(item)] if not isinstance(item, list) else item

    def geometry(item):
        return len(item)

    def pack(items):
        flat = torch.tensor([value for item in items for value in item], dtype=torch.int8)
        return {
            "toy_packed": flat,
            "toy_size_per_item": torch.tensor([len(item) for item in items], dtype=torch.int32),
        }

    def field_configs(hf_inputs):
        from vllm.multimodal.inputs import MultiModalFieldConfig

        sizes = hf_inputs["toy_size_per_item"]
        return {
            "toy_packed": MultiModalFieldConfig.flat_from_sizes("toy", sizes),
            "toy_size_per_item": MultiModalFieldConfig.batched("toy"),
        }

    def dummy_item(config):
        return [0, 0]

    return build_mm_plugin(
        modality="toy",
        architecture="ToyArch",
        mode="encode",
        placeholder_text="<toy_placeholder>",
        encode_model_cls="vllm.model_executor.models.nemotron_h:NemotronHForCausalLM",
        encode_normalize_item_fn=normalize,
        encode_geometry_fn=geometry,
        encode_pack_items_fn=pack,
        encode_field_configs_fn=field_configs,
        encode_dummy_item_fn=dummy_item,
        encode_dummy_text="Describe this toy: <toy_placeholder>",
        encode_supported_mm_limits={"toy": None},
        encode_max_tokens_per_item=1024,
        encode_sentinel_start="<S0>",
        encode_sentinel_placeholder="<S1>",
        encode_sentinel_end="<S2>",
    )


def test_encode_mode_builds_the_processing_layer():
    plugin = _toy_plugin()

    parser = plugin.data_parser(expected_hidden_size=4)
    items = parser._parse_modality_payload(7)  # one scalar item, not a list
    assert items.get_count() == 1
    assert items.get(0) == [7]
    # A list is many items (real payloads are arrays/bytes/mappings, never lists).
    items = parser._parse_modality_payload([[4], [5, 6]])
    assert items.get_count() == 2
    # The processor-data key stays the singular modality name (no pluralization).
    assert items.get_processor_data() == {"toy": items.get_all()}
    assert parser._parse_modality_payload(None) is None
    assert "toy" in parser._get_subparsers()


def test_encode_mode_prompt_replacement_uses_geometry_and_sentinels():
    plugin = _toy_plugin()
    from vllm.multimodal.processing import PromptUpdateDetails

    processor = object.__new__(plugin.processor)
    items = plugin.processor_items([[1, 2, 3, 4]])

    class _Items:
        def get_items(self, modality, cls):
            return items

    updates = plugin.processor._get_prompt_updates(processor, _Items(), {}, {})
    assert len(updates) == 1
    assert updates[0].modality == "toy"
    assert updates[0].target == "<toy_placeholder>"
    detail = updates[0].replacement(0)
    assert isinstance(detail, PromptUpdateDetails)
    assert detail.full == "<S0><S1><S1><S1><S1><S2>"


def test_encode_mode_dummy_inputs_follow_the_consumer_payload():
    from types import SimpleNamespace

    plugin = _toy_plugin()
    builder = plugin.dummy_inputs(SimpleNamespace(get_hf_config=lambda: SimpleNamespace()))
    assert builder.get_dummy_text({"toy": 1}) == "Describe this toy: <toy_placeholder>"
    assert builder.get_dummy_text({"toy": 0}) == "What is the answer?"
    data = builder.get_dummy_mm_data(64, {"toy": 2}, {})
    assert data == {"toy": [[0, 0], [0, 0]]}
    assert builder.get_dummy_mm_data(64, {"toy": 0}, {}) == {}


def test_encode_register_is_reentrant():
    plugin = _toy_plugin()
    plugin.register()
    plugin.register()
