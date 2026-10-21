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

"""Packed metadata parity and the documented distributed sanity launch."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import test_domain_preparation
import torch
from test_chat_continuation import _BoundaryTokenizer
from test_domain_preparation import _preparation
from test_features_cache import CONTRACT

from nemotron_stitch.automodel.data import ManifestFeatureDataset, PackedManifestFeatureDataset, packed_collate_fn
from nemotron_stitch.features.cache import CacheBuilder

endpoint = test_domain_preparation.endpoint
ROOT = Path(__file__).resolve().parents[1]


class _ChatTokenizer(_BoundaryTokenizer):
    def apply_chat_template(self, messages, *, add_generation_prompt=False, enable_thinking=False, **kwargs):
        text = "".join(
            "<"
            + message["role"]
            + ">"
            + (message.get("reasoning_content", "") if enable_thinking else "")
            + message["content"]
            + "<eos>"
            for message in messages
        )
        return text + ("<assistant>" if add_generation_prompt else "")


def _manifest(tmp_path, *, multi_turn=False):
    rows = [
        dict(
            sample_id="first",
            prompt="<feature>first question",
            target="first answer",
            system_prompt="System context",
            reasoning_content="First reasoning",
            chat_template_kwargs={"enable_thinking": True},
        ),
        dict(
            sample_id="second",
            prompt="unused",
            target="second answer",
            messages=[{"role": "user", "content": "<feature>history question"}],
            chat_template_kwargs={"enable_thinking": False},
        ),
    ]
    if multi_turn:
        rows[0]["turns"] = [{"question": "followup", "answer": "next answer", "reasoning_content": "Next reasoning"}]
    cache = CacheBuilder(tmp_path / "cache", CONTRACT)
    for index, row in enumerate(rows):
        row.update(stage="sft", split="train", feature_key=CONTRACT.entry_key(row["sample_id"], str(index)))
        cache.add(row["feature_key"], np.ones((1, 4), dtype=np.float32) * (index + 1))
    cache.publish()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest, rows


@pytest.mark.parametrize("supervision", ["suffix", "assistant_content"])
def test_packed_metadata_survives_default_field_filter(tmp_path, supervision):
    manifest, rows = _manifest(tmp_path, multi_turn=supervision == "assistant_content")
    settings = dict(stage="sft", split="train", cache_root=str(tmp_path / "cache"))
    standalone = ManifestFeatureDataset(str(manifest), **settings)
    assert set(standalone[0]) == {"prompt", "target", "features"}
    packed = PackedManifestFeatureDataset(
        str(manifest),
        tokenizer=_ChatTokenizer(),
        max_tokens=4096,
        max_length=4096,
        projector_name="sample",
        placeholder_token_id=3,
        supervision=supervision,
        **settings,
    )
    encoded = [packed.collator._encode(row) for row in rows]
    item = packed[0]
    assert item["input_ids"].tolist() == [token for ids, _ in encoded for token in ids[:-1]]
    assert item["labels"].tolist() == [token for _, labels in encoded for token in labels[1:]]
    assert item["seq_lens"].tolist() == [len(ids) - 1 for ids, _ in encoded]
    assert torch.equal(item["mm_token_indices__sample"], item["input_ids"].eq(3).nonzero().flatten())
    decoded = packed.collator.tokenizer.decode(
        [token for token in item["labels"].tolist() if token != -100],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    input_text = packed.collator.tokenizer.decode(
        item["input_ids"].tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert "System context" in input_text
    assert "First reasoning" in input_text
    assert "history question" in input_text
    if supervision == "suffix":
        assert "First reasoning" in decoded
    if supervision == "assistant_content":
        assert "next answer" in decoded


@pytest.mark.parametrize("field", ["input_ids", "labels"])
def test_packed_custom_collator_drift_is_rejected(tmp_path, field):
    manifest, _ = _manifest(tmp_path)
    packed = PackedManifestFeatureDataset(
        str(manifest),
        tokenizer=_ChatTokenizer(),
        max_tokens=4096,
        max_length=4096,
        projector_name="sample",
        placeholder_token_id=3,
        stage="sft",
        split="train",
        cache_root=str(tmp_path / "cache"),
    )

    def divergent(examples):
        batch = packed.collator(examples)
        batch[field][0, -1] += 1
        return batch

    packed.collate_fn = divergent
    with pytest.raises(ValueError, match="collator encoding differs from pack planning"):
        packed[0]


def test_documented_sanity_overrides_reach_native_parser():
    text = (ROOT / "examples/README.md").read_text()
    arguments = re.findall(r"--(?:dataset|step_scheduler)\.[^\s\\]+", text)
    expected = {
        "step_scheduler.max_steps=3",
        "step_scheduler.num_epochs=1000",
        "step_scheduler.ckpt_every_steps=3",
        "step_scheduler.save_checkpoint_every_epoch=false",
        "dataset.manifest_path=artifacts/data/manifest-sanity.jsonl",
    }
    assert {argument[2:] for argument in arguments} == expected
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.config._arg_parser import parse_cli_argv

    _, overrides = parse_cli_argv(default_cfg_path="configs/sft.yaml", argv=arguments)
    assert set(overrides) == expected


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
@pytest.mark.parametrize("stage", ["alignment", "sft"])
def test_documented_sanity_view_feeds_all_ranks(domain, stage, endpoint, tmp_path, monkeypatch):
    pytest.importorskip("nemo_automodel")
    from bionemo_perception import Evo2Client, KERMTClient
    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.components.datasets.loader import ParallelAwareDataloader, make_dataset_config
    from test_domain_examples import _Tokenizer

    url, requests, _ = endpoint
    output = tmp_path / "artifacts/data"
    _preparation(domain).prepare(
        (Evo2Client if domain == "genetics" else KERMTClient)(url),
        ROOT / "examples" / domain / "data/sanity.jsonl",
        output,
    )
    monkeypatch.chdir(tmp_path)
    snippets = re.findall(r"```python\n(.*?)\n```", (ROOT / "examples/README.md").read_text(), re.S)
    (snippet,) = [snippet for snippet in snippets if "manifest-sanity.jsonl" in snippet]
    exec(compile(snippet, "examples/README.md", "exec"), {})
    assert len(requests) == 8  # Repetition reuses cached features, not HTTP requests.
    monkeypatch.syspath_prepend(str(ROOT / "examples" / domain))
    app = __import__(domain + "_example")
    real_tokenizer = os.environ.get("STITCH_TEST_LIGHTNING_TOKENIZER")
    processor = (
        app.load_processor(real_tokenizer)
        if real_tokenizer
        else (app.Evo2Processor if domain == "genetics" else app.KERMTProcessor)(_Tokenizer())
    )
    config = load_yaml_config(str(ROOT / "examples" / domain / "configs" / f"{stage}.yaml"))
    settings = config.dataset.to_dict()
    target = settings.pop("_target_")
    settings.pop("tokenizer")
    settings.update(manifest_path=str(output / "manifest-sanity.jsonl"), cache_root=str(output / "feature-cache"))
    dataset = make_dataset_config(target, settings).build(tokenizer=processor)
    assert len(dataset) >= config.step_scheduler.global_batch_size
    for rank in range(8):
        loader = ParallelAwareDataloader(
            dataset,
            dp_rank=rank,
            dp_world_size=8,
            batch_size=1,
            num_workers=0,
            collate_fn=packed_collate_fn,
            drop_last=True,
            shuffle=True,
        )
        assert len(loader) >= config.step_scheduler.global_batch_size // 8
        assert all(0 <= index < len(dataset) for index in loader.sampler)
        if rank == 0:
            batch = next(iter(loader))
            assert batch["input_ids"].numel() > 0
            assert batch["labels"].ne(-100).any()
