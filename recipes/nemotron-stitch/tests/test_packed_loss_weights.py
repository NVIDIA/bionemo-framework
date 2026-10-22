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

"""Label-aligned loss weights survive feature packing without a second shift."""

from __future__ import annotations

import pytest
import torch
from test_packed_metadata import _ChatTokenizer, _manifest

from nemotron_stitch.automodel.data import (
    PackedManifestFeatureDataset,
    StreamingFeaturePackDataset,
    packed_collate_fn,
)


@pytest.mark.parametrize("packing_format", ["thd", "neat"])
def test_map_packing_preserves_custom_collator_weights(tmp_path, packing_format):
    if packing_format == "neat":
        pytest.importorskip("nemo_automodel")
    manifest, _ = _manifest(tmp_path)
    packed = PackedManifestFeatureDataset(
        str(manifest),
        tokenizer=_ChatTokenizer(),
        stage="sft",
        split="train",
        cache_root=str(tmp_path / "cache"),
        max_tokens=512,
        max_length=512,
        projector_name="sample",
        placeholder_token_id=3,
        packing_strategy=packing_format,
    )
    expected = []

    def weighted_collator(examples):
        batch = packed.collator(examples)
        weights = torch.arange(batch["labels"].numel(), dtype=torch.float32).reshape_as(batch["labels"]) + 0.5
        batch["loss_token_weights"] = weights
        expected.append(weights.masked_fill(batch["labels"] == -100, 0).flatten())
        return batch

    packed.collate_fn = weighted_collator
    item = packed[0]
    weights = item["loss_token_weights"]
    unpadded = torch.cat(expected)
    assert weights.shape == item["labels"].shape
    assert torch.equal(weights[: unpadded.numel()], unpadded)
    assert weights[unpadded.numel() :].eq(0).all()
    assert weights[item["labels"] == -100].eq(0).all()
    pytest.importorskip("nemo_automodel")
    batch = packed_collate_fn([item])
    assert batch["loss_token_weights"].shape == batch["labels"].shape
    assert torch.equal(batch["loss_token_weights"], weights.unsqueeze(0))


@pytest.mark.parametrize("packing_format", ["thd", "neat"])
def test_streaming_packing_preserves_weights_and_zeroes_masked_padding(packing_format):
    samples = [
        dict(
            input_ids=torch.tensor([1, 2]), labels=torch.tensor([-100, 3]), loss_token_weights=torch.tensor([9.0, 10.0])
        ),
        dict(
            input_ids=torch.tensor([4, 5, 6]),
            labels=torch.tensor([5, -100, 7]),
            loss_token_weights=torch.tensor([2.0, 9.0, 4.0]),
        ),
    ]
    packed = StreamingFeaturePackDataset(samples, max_tokens=7, pad_token_id=0, packing_format=packing_format)
    item = next(iter(packed))
    expected = torch.tensor([0.0, 10.0, 2.0, 0.0, 4.0, 0.0, 0.0])
    assert torch.equal(item["loss_token_weights"], expected)
    assert item["labels"].tolist() == [-100, 3, 5, -100, 7, -100, -100]
    if packing_format == "thd":
        pytest.importorskip("nemo_automodel")
    batch = packed_collate_fn([item])
    assert torch.equal(batch["loss_token_weights"], expected.unsqueeze(0))


@pytest.mark.parametrize("bad_weights", [None, [1.0], [1.0, -1.0], [1.0, float("nan")], [1.0, float("inf")]])
def test_weighted_packing_rejects_missing_or_invalid_weights(bad_weights):
    sample = dict(input_ids=torch.tensor([1, 2]), labels=torch.tensor([2, 3]))
    weighted = {**sample, "loss_token_weights": torch.ones(2)}
    other = dict(sample)
    if bad_weights is not None:
        other["loss_token_weights"] = torch.tensor(bad_weights)
    packed = StreamingFeaturePackDataset([weighted, other], max_tokens=4, pad_token_id=0)
    with pytest.raises(ValueError, match="loss_token_weights"):
        next(iter(packed))
