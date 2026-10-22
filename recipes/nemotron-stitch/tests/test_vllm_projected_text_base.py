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

"""Projected mode over a text-only vLLM base (P4). Needs vLLM installed."""

from __future__ import annotations

import pytest
from test_vllm_encode_plugin import _SentinelTokenizer

from nemotron_stitch.vllm.plugin import SentinelAttrs, build_mm_plugin


def _register():
    pytest.importorskip("vllm", reason="vLLM is not installed")
    return build_mm_plugin(
        modality="toyc",
        architecture="ToyEncoderTextBase",
        placeholder_text="<toyc>",
        mode="projected",
        base_model_cls="vllm.model_executor.models.nemotron_h:NemotronHForCausalLM",
        base_processing_info_cls="vllm.multimodal.processing:BaseProcessingInfo",
        base_processor_cls="vllm.multimodal.processing:BaseMultiModalProcessor",
        base_dummy_inputs_cls="vllm.multimodal.processing:BaseDummyInputsBuilder",
        sentinels=SentinelAttrs(start="toyc_start", placeholder="toyc_placeholder", end="toyc_end"),
        num_tokens_attr="toyc_num_tokens",
    )


def test_text_base_registration_adds_multimodal_surface():
    register = _register()
    register()
    from vllm import ModelRegistry
    from vllm.model_executor.models.interfaces import supports_multimodal

    model_cls = ModelRegistry._try_load_model_cls("ToyEncoderTextBase")
    assert model_cls is not None
    assert supports_multimodal(model_cls)
    assert model_cls.get_placeholder_str("toyc", 0) == "<toyc>"
    assert model_cls.get_placeholder_str("image", 0) is None
    # embed_multimodal is implemented by the plugin, not inherited from nowhere.
    assert "embed_multimodal" in vars(model_cls)
    register()  # idempotent


@pytest.mark.parametrize("start,end", [("", ""), ("<start>", "<end>")])
def test_projected_items_can_leave_boundaries_in_prompt_text(monkeypatch, start, end):
    from types import SimpleNamespace

    pytest.importorskip("vllm", reason="vLLM is not installed")
    from vllm.multimodal import MULTIMODAL_REGISTRY

    captured = []
    original = MULTIMODAL_REGISTRY.register_processor

    def capture(processor, **kwargs):
        captured.append(processor)
        return original(processor, **kwargs)

    monkeypatch.setattr(MULTIMODAL_REGISTRY, "register_processor", capture)
    _register()()
    config = SimpleNamespace(toyc_start=start, toyc_placeholder="<slot>", toyc_end=end, toyc_num_tokens=1)
    tokenizer = _SentinelTokenizer(["<toyc>", "<slot>", "<start>", "<end>"])
    processor = SimpleNamespace(info=SimpleNamespace(get_hf_config=lambda: config, get_tokenizer=lambda: tokenizer))

    class Items(dict):
        def get_items(self, modality, cls):
            return SimpleNamespace(get_feature_size=lambda index: 1)

    updates = captured[-1]._get_prompt_updates(processor, Items(toyc=True), {}, {})
    detail = updates[0].replacement(0)
    assert updates[0].target == [0]
    assert detail.full == ([2, 1, 3] if start else [1])
    assert detail.is_embed(detail.full).tolist() == ([False, True, False] if start else [True])
    config.toyc_placeholder = ""
    with pytest.raises(ValueError, match="vLLM config must define"):
        updates[0].replacement(0)
    config.toyc_placeholder = "<slot>"
    config.toyc_start = None
    with pytest.raises(ValueError, match="vLLM config must define"):
        updates[0].replacement(0)
