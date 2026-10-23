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

"""Learned token-embedding overrides for projected-mode vLLM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_stitch.projector import MultimodalProjector, scatter_flat
from nemotron_stitch.projector.artifact import save_projector_artifact
from nemotron_stitch.vllm.plugin import (
    SentinelAttrs,
    apply_token_embedding_overrides,
    build_mm_plugin,
    load_token_embedding_overrides,
)


class _ArtifactModel(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        config = {
            "name": "tokens",
            "kind": "mlp2x_gelu",
            "mm_hidden_size": 3,
            "hidden_size": 5,
        }
        self.config = SimpleNamespace(mm_projectors=[config])
        self.mm_projector = MultimodalProjector.from_config([config], output_size=hidden_size)


class _Tokenizer:
    unk_token_id = 0
    unk_token = "<unk>"

    def __init__(self):
        self.vocab = {"<unk>": 0, "text": 1, "<start>": 2, "<slot>": 3, "<end>": 4}

    def __len__(self):
        return len(self.vocab)

    def convert_tokens_to_ids(self, token):
        if token == "two tokens":
            return 1
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, token, add_special_tokens=False):
        assert not add_special_tokens
        if token == "two tokens":
            return [1, 1]
        return [self.convert_tokens_to_ids(token)]


class _ProjectedBase(nn.Module):
    supports_multimodal = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        del prefix
        self.embedding = nn.Embedding(5, vllm_config.model_config.hf_config.hidden_size)

    def embed_multimodal(self, **kwargs):
        return tuple(kwargs.values())

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        inputs_embeds = self.embedding(input_ids)
        if multimodal_embeddings:
            values = torch.cat(tuple(multimodal_embeddings))
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[is_multimodal] = values
        return inputs_embeds


def _artifact(tmp_path, **extra_state):
    save_projector_artifact(_ArtifactModel(), tmp_path, extra_state=extra_state)
    return tmp_path


def test_loads_only_declared_extra_state_as_token_embeddings(tmp_path):
    start = torch.arange(4, dtype=torch.float32)
    end = start + 10
    path = _artifact(tmp_path, learned_start=start, learned_end=end, unrelated=torch.ones(2, 2))

    token_ids, vectors = load_token_embedding_overrides(
        path,
        {"<start>": "learned_start", 4: "learned_end"},
        hidden_size=4,
        tokenizer=_Tokenizer(),
    )

    assert token_ids == (2, 4)
    torch.testing.assert_close(vectors, torch.stack((start, end)))


@pytest.mark.parametrize(
    ("declarations", "extra_state", "message"),
    [
        ({2: "missing"}, {"learned": torch.zeros(4)}, "no declared EXTRA tensor"),
        ({2: "learned"}, {"learned": torch.zeros(2, 4)}, r"must have shape \(4,\)"),
        ({"missing": "learned"}, {"learned": torch.zeros(4)}, "not in the tokenizer vocabulary"),
        ({"two tokens": "learned"}, {"learned": torch.zeros(4)}, "exactly one token"),
        ({9: "learned"}, {"learned": torch.zeros(4)}, "outside tokenizer vocabulary"),
    ],
)
def test_invalid_embedding_override_declarations_fail_closed(tmp_path, declarations, extra_state, message):
    path = _artifact(tmp_path, **extra_state)
    with pytest.raises(ValueError, match=message):
        load_token_embedding_overrides(
            path,
            declarations,
            hidden_size=4,
            tokenizer=_Tokenizer(),
        )


def test_duplicate_resolved_token_fails_closed(tmp_path):
    path = _artifact(tmp_path, learned_start=torch.zeros(4), other=torch.ones(4))
    with pytest.raises(ValueError, match="duplicate token embedding override for token ID 2"):
        load_token_embedding_overrides(
            path,
            {2: "learned_start", "<start>": "other"},
            hidden_size=4,
            tokenizer=_Tokenizer(),
        )


@pytest.mark.parametrize("feature_count", [2, 5])
def test_host_and_projected_vllm_input_embeddings_conform_for_variable_spans(feature_count):
    hidden_size = 4
    start_id, placeholder_id, end_id = 2, 3, 4
    input_ids = torch.tensor([[1, start_id, *([placeholder_id] * feature_count), end_id, 1]])
    table = torch.arange(5 * hidden_size, dtype=torch.float32).reshape(5, hidden_size)
    text_embeddings = table[input_ids]
    projected = torch.arange(feature_count * hidden_size, dtype=torch.float32).reshape(feature_count, hidden_size)
    flat_indices = input_ids.eq(placeholder_id).reshape(-1).nonzero().flatten()
    merged = scatter_flat(
        input_ids,
        text_embeddings,
        projected,
        flat_indices,
        placeholder_token_id=placeholder_id,
    )
    learned_start = torch.full((hidden_size,), 101.0)
    learned_end = torch.full((hidden_size,), 202.0)

    # Independent spelling of the AutoModel consumer's post-scatter marker override.
    host = torch.where(input_ids.eq(start_id).unsqueeze(-1), learned_start, merged)
    host = torch.where(input_ids.eq(end_id).unsqueeze(-1), learned_end, host)
    vllm = apply_token_embedding_overrides(
        input_ids,
        merged,
        (start_id, end_id),
        torch.stack((learned_start, learned_end)),
    )

    torch.testing.assert_close(vllm, host, rtol=0, atol=0)
    torch.testing.assert_close(vllm[0, 0], text_embeddings[0, 0])
    torch.testing.assert_close(vllm[0, -1], text_embeddings[0, -1])
    torch.testing.assert_close(vllm[0, 2 : 2 + feature_count], projected)
    torch.testing.assert_close(vllm[0, 1], learned_start)
    torch.testing.assert_close(vllm[0, -2], learned_end)


def test_embedding_override_application_is_a_noop_without_declarations():
    input_ids = torch.tensor([[1, 2, 3]])
    embeddings = torch.randn(1, 3, 4)
    result = apply_token_embedding_overrides(input_ids, embeddings, (), torch.empty(0, 4))
    assert result is embeddings


def test_additive_embedding_overrides_match_host_marker_deltas():
    input_ids = torch.tensor([[1, 2, 3, 4, 1]])
    embeddings = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    deltas = torch.tensor([[0.5, -1.0, 2.0, 3.0], [4.0, 0.0, -2.0, 1.0]])
    host = embeddings.clone()
    host[0, 1] += deltas[0]
    host[0, 3] += deltas[1]

    result = apply_token_embedding_overrides(input_ids, embeddings, (2, 4), deltas, mode="add")

    torch.testing.assert_close(result, host, rtol=0, atol=0)
    torch.testing.assert_close(embeddings, torch.arange(20, dtype=torch.float32).reshape(1, 5, 4))


def test_unknown_embedding_override_mode_fails_at_construction():
    with pytest.raises(ValueError, match="unknown token embedding override mode"):
        build_mm_plugin(
            modality="override_test",
            architecture="BadOverrideMode",
            placeholder_text="<override_test>",
            token_embedding_override_mode="multiply",
        )


@pytest.mark.parametrize("override_mode", ["replace", "add"])
def test_registered_vllm_model_applies_overrides_after_projected_scatter(tmp_path, monkeypatch, override_mode):
    pytest.importorskip("vllm", reason="vLLM is not installed")
    from vllm import ModelRegistry
    from vllm.tokenizers import registry as tokenizer_registry

    start = torch.full((4,), 101.0)
    end = torch.full((4,), 202.0)
    artifact_path = _artifact(tmp_path, learned_start=start, learned_end=end)
    monkeypatch.setattr(tokenizer_registry, "cached_tokenizer_from_config", lambda _config: _Tokenizer())
    register = build_mm_plugin(
        modality="override_test",
        architecture=f"TokenEmbeddingOverrideProjectedTest{override_mode}",
        placeholder_text="<override_test>",
        mode="projected",
        base_model_cls=f"{__name__}:_ProjectedBase",
        base_processing_info_cls="vllm.multimodal.processing:BaseProcessingInfo",
        base_processor_cls="vllm.multimodal.processing:BaseMultiModalProcessor",
        base_dummy_inputs_cls="vllm.multimodal.processing:BaseDummyInputsBuilder",
        sentinels=SentinelAttrs(start="start", placeholder="placeholder", end="end"),
        num_tokens_attr="max_tokens",
        geometry="variable",
        token_embedding_overrides={2: "learned_start", 4: "learned_end"},
        token_embedding_override_mode=override_mode,
    )
    register()
    model_cls = ModelRegistry._try_load_model_cls(f"TokenEmbeddingOverrideProjectedTest{override_mode}")
    hf_config = SimpleNamespace(
        hidden_size=4,
        start="<start>",
        placeholder="<slot>",
        end="<end>",
        mm_projector_artifact_path=str(artifact_path),
    )
    model_config = SimpleNamespace(hf_config=hf_config, dtype=torch.float32)
    model = model_cls(vllm_config=SimpleNamespace(model_config=model_config))
    with torch.no_grad():
        model.embedding.weight.copy_(torch.arange(20, dtype=torch.float32).reshape(5, 4))

    for feature_count in (2, 4):
        input_ids = torch.tensor([1, 2, *([3] * feature_count), 4, 1])
        mask = input_ids.eq(3)
        projected = torch.arange(feature_count * 4, dtype=torch.float32).reshape(feature_count, 4)
        result = model.embed_input_ids(input_ids, (projected,), is_multimodal=mask)

        torch.testing.assert_close(result[0], model.embedding.weight[1])
        torch.testing.assert_close(result[-1], model.embedding.weight[1])
        expected_start = start if override_mode == "replace" else model.embedding.weight[2] + start
        expected_end = end if override_mode == "replace" else model.embedding.weight[4] + end
        torch.testing.assert_close(result[1], expected_start)
        torch.testing.assert_close(result[-2], expected_end)
        torch.testing.assert_close(result[2:-2], projected)
    assert not any(name.startswith("_stitch_token_embedding_override") for name in model.state_dict())
