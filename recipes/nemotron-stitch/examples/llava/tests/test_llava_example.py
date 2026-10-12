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

"""Contracts owned by the LLaVA application modules."""

from __future__ import annotations

from functools import partial

import llava_example as application
import pytest

from nemotron_stitch.features.preparation import InvalidRow
from nemotron_stitch.prompt import PlaceholderSpec


class _FakeImage:
    def convert(self, mode: str) -> _FakeImage:
        assert mode == "RGB"
        return self


def _row(*, answer: str = "The answer is 5", image=True) -> dict:
    return {
        "id": "sample-1",
        "image": _FakeImage() if image else None,
        "conversations": [
            {"from": "human", "value": "<image>\nHow many objects are there?"},
            {"from": "gpt", "value": answer},
        ],
        "data_source": "fake",
    }


def test_clevr_row_becomes_the_declared_training_contract():
    record, encoder_input = application.normalize_row(
        _row(),
        configuration=application.CONFIG_CLEVR,
        source_index=7,
    )

    assert isinstance(encoder_input, _FakeImage)
    assert record["source_index"] == 7
    assert record["question"] == "How many objects are there?"
    assert record["target"] == "<answer>5</answer>"
    assert record["ground_truth"] == "5"
    assert record["prompt"].count(application.PLACEHOLDER_SPEC.placeholder_token) == application.CLIP_PATCH_TOKENS


def test_row_normalization_distinguishes_schema_drift_from_skippable_rows():
    malformed = _row()
    malformed["conversations"] = malformed["conversations"][:1]
    with pytest.raises(ValueError, match="exactly one human and one assistant"):
        application.normalize_row(malformed, configuration=application.CONFIG_CLEVR, source_index=0)

    with pytest.raises(InvalidRow, match="strict form"):
        application.normalize_row(
            _row(answer="five"),
            configuration=application.CONFIG_CLEVR,
            source_index=0,
        )
    with pytest.raises(InvalidRow, match="image is absent"):
        application.normalize_row(
            _row(image=False),
            configuration=application.CONFIG_CLEVR,
            source_index=0,
        )


def test_prepare_data_binds_the_application_contract(monkeypatch, tmp_path):
    captured = {}

    def prepare(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"records": 0}

    monkeypatch.setattr(application, "prepare_features", prepare)
    streams = {"source": []}

    def encode(value):
        return value

    result = application.prepare_data(
        streams,
        encode,
        cache_root=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.jsonl",
        implementation_revision="transformers==test",
    )

    assert result == {"records": 0}
    assert captured["args"][:2] == (streams, encode)
    normalizer = captured["args"][2]
    assert isinstance(normalizer, partial)
    assert normalizer.func is application.normalize_row
    assert normalizer.keywords == {"placeholder_spec": application.PLACEHOLDER_SPEC}
    assert captured["kwargs"]["partitions"] == application.PARTITIONS
    assert "feature_shape" not in captured["kwargs"]
    assert "context_units" not in captured["kwargs"]
    assert captured["kwargs"]["encoder"] == {
        "repo_id": application.ENCODER_REPO_ID,
        "revision": application.ENCODER_REVISION,
        "implementation_revision": "transformers==test",
        "layer": -1,
    }
    assert captured["kwargs"]["dataset"] == {
        "repo_id": application.DATASET_REPO_ID,
        "revision": application.DATASET_REVISION,
    }


def test_qwen_row_uses_model_specific_sentinels():
    spec = PlaceholderSpec("clip_image", "<|vision_start|>", "<|vision_pad|>", "<|vision_end|>")
    record, _ = application.normalize_row(
        _row(),
        configuration=application.CONFIG_CLEVR,
        source_index=0,
        placeholder_spec=spec,
    )

    assert record["prompt"].startswith("<|vision_start|>")
    assert record["prompt"].count("<|vision_pad|>") == application.CLIP_PATCH_TOKENS
    assert record["prompt"].count(spec.marker()) == 0


def test_model_registry_callback_registers_every_application_host():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel._transformers.registry import ModelRegistry

    application.register_models()
    application.register_models()
    for spec in application.MODEL_SPECS:
        assert ModelRegistry.model_arch_name_to_cls[spec.architecture] is application.build_host_cls(spec)


def test_qwen_registry_decorates_automodel_native_model():
    pytest.importorskip("nemo_automodel")
    import llava_qwen
    from nemo_automodel._transformers.registry import ModelRegistry
    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

    llava_qwen.register_models()
    host_cls = ModelRegistry.model_arch_name_to_cls[llava_qwen.TRAINING_ARCHITECTURE]
    assert host_cls is llava_qwen.build_host_cls()
    assert Qwen3_5ForConditionalGeneration in host_cls.__mro__


def test_qwen_moe_registry_decorates_automodel_native_model():
    pytest.importorskip("nemo_automodel")
    import llava_qwen
    from nemo_automodel._transformers.registry import ModelRegistry
    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration

    llava_qwen.register_moe_models()
    host_cls = ModelRegistry.model_arch_name_to_cls[llava_qwen.MOE_TRAINING_ARCHITECTURE]
    assert host_cls is llava_qwen.build_moe_host_cls()
    assert Qwen3_5MoeForConditionalGeneration in host_cls.__mro__


@pytest.mark.skipif(
    __import__("os").environ.get("LLAVA_EXAMPLE_HUB_TEST") != "1",
    reason="opt-in Hub integration test (LLAVA_EXAMPLE_HUB_TEST=1)",
)
def test_pinned_hub_configuration_schemas():
    datasets = pytest.importorskip("datasets")
    for configuration in (application.CONFIG_ALIGNMENT, application.CONFIG_CLEVR):
        stream = datasets.load_dataset(
            application.DATASET_REPO_ID,
            configuration,
            revision=application.DATASET_REVISION,
            split="train",
            streaming=True,
        )
        row = next(iter(stream))
        assert set(row) == {"id", "image", "conversations", "data_source"}
        application.normalize_row(row, configuration=configuration, source_index=0)

    transformers = pytest.importorskip("transformers")
    for repo_id, revision in (
        ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
        ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, revision=revision)
        assert tokenizer.convert_tokens_to_ids("<|vision_start|>") == 248053
        assert tokenizer.convert_tokens_to_ids("<|vision_pad|>") == 248055
        assert tokenizer.convert_tokens_to_ids("<|vision_end|>") == 248054
