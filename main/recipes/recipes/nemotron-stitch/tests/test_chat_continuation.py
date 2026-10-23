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

"""U-34: preserve inference tokens when BPE merges across the chat boundary."""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemotron_stitch.automodel.data import FeatureCollator, PackedManifestFeatureDataset


class _BoundaryTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pieces = {"<eos>": 2, "<feature>": 3, "\n": 198, "\n\n": 271}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prompt = messages[0]["content"] + "<assistant>\n"
        return prompt if add_generation_prompt else prompt + messages[1]["content"] + "<eos>"

    def __call__(self, text, add_special_tokens=False):
        pieces = re.findall(r"<eos>|<feature>|\n\n|[\s\S]", text)
        return SimpleNamespace(input_ids=[self.pieces[p] if p in self.pieces else 1000 + ord(p) for p in pieces])

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        pieces = {v: k for k, v in self.pieces.items()}
        return "".join(pieces[token] if token in pieces else chr(token - 1000) for token in ids)


def _collator(tokenizer=None, **kwargs):
    return FeatureCollator(tokenizer or _BoundaryTokenizer(), projector_name="sample", placeholder_token_id=3, **kwargs)


def _render(collator, row):
    prompt, full = collator._messages(row)
    kwargs = collator.chat_template_kwargs
    return (
        collator.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **kwargs),
        collator.tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False, **kwargs),
    )


def _assert_continuation(collator, row):
    tokenizer = collator.tokenizer
    prompt, full = _render(collator, row)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    canonical_ids = list(tokenizer(full, add_special_tokens=False).input_ids)
    assert canonical_ids[: len(prompt_ids)] != prompt_ids
    ids, labels = collator._encode(row)
    assert ids[: len(prompt_ids)] == prompt_ids
    assert labels == [-100] * len(prompt_ids) + ids[len(prompt_ids) :]
    assert tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == full
    suffix = tokenizer.decode(labels[len(prompt_ids) :], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert suffix == full[len(prompt) :]
    assert tokenizer.eos_token_id in labels[len(prompt_ids) :]
    return ids, labels


@pytest.mark.parametrize("target", ["\n</think>answer", "\n</think> café 中文 🧬"])
def test_boundary_merge_preserves_prompt_text_and_supervises_eos(target):
    _assert_continuation(_collator(), {"prompt": "<feature>Question?", "target": target})


@pytest.mark.parametrize("supervision", ["suffix", "assistant_content"])
def test_prefix_stable_rows_keep_exact_ids_and_labels(supervision):
    collator = _collator(supervision=supervision)
    row = {"prompt": "Question?", "target": "answer"}
    prompt, full = _render(collator, row)
    prompt_ids = collator.tokenizer(prompt).input_ids
    full_ids = collator.tokenizer(full).input_ids
    expected_labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if supervision == "assistant_content":
        expected_labels[-1] = -100
    assert collator._encode(row) == (full_ids, expected_labels)


def test_non_prefix_rendering_still_fails_closed():
    class RewritingTokenizer(_BoundaryTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            rendered = super().apply_chat_template(messages, **kwargs)
            return rendered if kwargs["add_generation_prompt"] else "rewritten" + rendered

    with pytest.raises(ValueError, match="not prefix-stable"):
        _collator(RewritingTokenizer())._encode({"prompt": "question", "target": "\nanswer"})


@pytest.mark.parametrize("lossy_part", ["prompt", "continuation"])
def test_lossy_encoding_still_fails_closed(lossy_part):
    class LossyTokenizer(_BoundaryTokenizer):
        def __call__(self, text, **kwargs):
            if lossy_part == "prompt" and text.endswith("<assistant>\n"):
                text = text.replace("question", "changed")
            if lossy_part == "continuation" and text.startswith("\n"):
                text = text.strip()
            return super().__call__(text, **kwargs)

    with pytest.raises(ValueError, match="does not preserve"):
        _collator(LossyTokenizer())._encode({"prompt": "question", "target": "\nanswer"})


def test_assistant_content_boundary_merge_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="not prefix-stable"):
        _collator(supervision="assistant_content")._encode({"prompt": "question", "target": "\nanswer"})


def test_length_cap_applies_to_continuation_encoding():
    row = {"prompt": "question", "target": "\nanswer"}
    ids, labels = _collator()._encode(row)
    cap = len(ids) - 1
    with pytest.raises(ValueError, match=f"over the {cap} cap"):
        _collator(max_length=cap)._encode(row)
    assert _collator(max_length=cap, truncate=True)._encode(row) == (ids[:cap], labels[:cap])


def _assert_feature_batch(collator, rows):
    batch = collator(rows)
    expected_indices = batch["input_ids"].flatten().eq(collator.placeholder_token_id).nonzero().flatten()
    assert torch.equal(batch["mm_token_indices__sample"], expected_indices)
    np.testing.assert_array_equal(batch["mm_features__sample"], np.concatenate([row["features"] for row in rows]))
    for index, row in enumerate(rows):
        ids, labels = collator._encode(row)
        assert batch["input_ids"][index, : len(ids) - 1].tolist() == ids[:-1]
        assert batch["labels"][index, : len(ids) - 1].tolist() == labels[1:]
        assert batch["labels"][index, len(ids) - 1 :].eq(-100).all()


def test_continuations_preserve_ragged_padded_and_packed_feature_indices(monkeypatch):
    rows = [
        {"prompt": "<feature>" * count + "question", "target": "\nanswer", "features": np.full((count, 4), count)}
        for count in (1, 3, 0)
    ]
    collator = _collator()
    _assert_feature_batch(collator, rows)

    class Dataset:
        def __init__(self, *args, **kwargs):
            self.rows = rows

        def __getitem__(self, index):
            return self.rows[index]

    monkeypatch.setattr("nemotron_stitch.automodel.data.ManifestFeatureDataset", Dataset)
    packed = PackedManifestFeatureDataset(
        "unused",
        tokenizer=collator.tokenizer,
        stage="sft",
        split="train",
        cache_root="unused",
        max_tokens=1024,
        projector_name="sample",
        placeholder_token_id=3,
    )
    assert len(packed) == 1
    batch = packed[0]
    encoded = [collator._encode(row) for row in rows]
    assert batch["input_ids"].tolist() == [token for ids, _ in encoded for token in ids[:-1]]
    assert batch["labels"].tolist() == [token for _, labels in encoded for token in labels[1:]]
    assert batch["seq_lens"].tolist() == [len(ids) - 1 for ids, _ in encoded]
    assert torch.equal(batch["mm_token_indices__sample"], batch["input_ids"].eq(3).nonzero().flatten())
    np.testing.assert_array_equal(batch["mm_features__sample"], np.concatenate([row["features"] for row in rows]))


def test_native_qwen_empty_reasoning_continuation():
    """Optional offline qualification against U-34's pinned tokenizer snapshot."""
    path = os.environ.get("STITCH_TEST_QWEN_TOKENIZER")
    if not path:
        pytest.skip("set STITCH_TEST_QWEN_TOKENIZER to Qwen3.6 tokenizer revision 995ad96e")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    placeholder = tokenizer.convert_ids_to_tokens(248055)
    collator = FeatureCollator(
        tokenizer,
        projector_name="sample",
        placeholder_token_id=248055,
        max_length=2048,
        chat_template_kwargs={"enable_thinking": True},
    )
    row = {"prompt": "How many atoms?", "target": "<think>\n</think>((2))"}
    ids, labels = _assert_continuation(collator, row)
    assert ids[13] == 198
    suffix = tokenizer.decode([token for token in labels if token != -100])
    assert suffix.startswith("\n</think>\n\n((2))")
    rows = [
        {**row, "prompt": placeholder * count + "How many atoms?", "features": np.full((count, 4), count)}
        for count in (1, 3, 0)
    ]
    for item in rows:
        _assert_continuation(collator, item)
    _assert_feature_batch(collator, rows)
    for thinking in (True, False):
        collator.chat_template_kwargs = {"enable_thinking": thinking}
        stable_row = {"prompt": "How many atoms?", "target": "((2))"}
        if thinking:
            stable_row["reasoning_content"] = "Count them."
        prompt, full = _render(collator, stable_row)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(full, add_special_tokens=False).input_ids
        assert full_ids[: len(prompt_ids)] == prompt_ids
        assert collator._encode(stable_row) == (full_ids, [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])
