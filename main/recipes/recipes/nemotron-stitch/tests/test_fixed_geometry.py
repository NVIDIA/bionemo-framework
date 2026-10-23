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

"""Shared feature preparation, dataset, and collation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from nemotron_stitch.automodel.data import (
    FeatureCollator,
    ManifestFeatureDataset,
    ManifestFeatureIterableDatasetConfig,
    PackedManifestFeatureDataset,
    StreamingFeaturePackingConfig,
    packed_collate_fn,
    streaming_packed_collate_fn,
)
from nemotron_stitch.features.manifest import load_manifest
from nemotron_stitch.features.preparation import (
    InvalidRow,
    Partition,
    iter_selected_rows,
    prepare_features,
)

PARTITIONS = (
    Partition("align", "train", "source", 0, 2),
    Partition("align", "validation", "source", 2, 3),
)
FEATURE_SHAPE = (2, 3)


def _normalize(row, *, configuration, source_index):
    if row.get("skip"):
        raise InvalidRow("expected bad row")
    return (
        {
            "sample_id": row["id"],
            "configuration": configuration,
            "source_index": source_index,
            "prompt": "<SPECIAL_30><SPECIAL_32><SPECIAL_32><SPECIAL_31>\nQuestion?",
            "target": f"answer-{row['value']}",
            "reasoning_content": "because",
        },
        row["value"],
    )


def _encode(value):
    return np.full(FEATURE_SHAPE, value, dtype=np.float32)


def _publish(tmp_path: Path, name: str):
    return prepare_features(
        {"source": [{"id": "bad", "skip": True}, *({"id": f"row-{i}", "value": i} for i in range(3))]},
        _encode,
        _normalize,
        partitions=PARTITIONS,
        encoder={"repo_id": "org/encoder", "revision": "enc-rev", "implementation_revision": "impl"},
        dataset={"repo_id": "org/data", "revision": "data-rev"},
        cache_root=tmp_path / f"{name}-cache",
        manifest_path=tmp_path / f"{name}.jsonl",
        max_scan_rows=8,
    )


def test_preparation_is_deterministic_and_dataset_reads_the_cache(tmp_path):
    first = _publish(tmp_path, "first")
    second = _publish(tmp_path, "second")

    assert first["records"] == 3
    assert first["cache_id"] == second["cache_id"]
    assert Path(first["manifest"]).read_bytes() == Path(second["manifest"]).read_bytes()
    rows = load_manifest(first["manifest"])
    assert [(row["ordinal"], row["split"]) for row in rows] == [
        (0, "train"),
        (1, "train"),
        (2, "validation"),
    ]
    assert (tmp_path / "first-align-train.jsonl").is_file()

    dataset = ManifestFeatureDataset(
        first["manifest"],
        stage="align",
        split="train",
        cache_root=first["cache_root"],
        extra_fields=["reasoning_content"],
    )
    assert len(dataset) == 2
    np.testing.assert_array_equal(dataset[1]["features"], np.ones(FEATURE_SHAPE, dtype=np.float32))
    assert dataset[0]["reasoning_content"] == "because"


def test_preparation_does_not_require_repository_metadata(tmp_path):
    result = prepare_features(
        {"source": ({"id": f"row-{i}", "value": i} for i in range(3))},
        _encode,
        _normalize,
        partitions=PARTITIONS,
        cache_root=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.jsonl",
        max_scan_rows=8,
    )

    assert result["records"] == 3


def test_preparation_rejects_encoder_dtype_mismatching_the_cache_contract(tmp_path):
    with pytest.raises(ValueError, match="encoder produced dtype float16.*records 'float32'"):
        prepare_features(
            {"source": [{"id": "row", "value": 0}]},
            lambda _value: np.ones((1, 3), dtype=np.float16),
            _normalize,
            partitions=(Partition("align", "train", "source", 0, 1),),
            cache_root=tmp_path / "cache",
            manifest_path=tmp_path / "manifest.jsonl",
            max_scan_rows=1,
        )


def test_preparation_caches_variable_length_feature_rows(tmp_path):
    result = prepare_features(
        {"source": ({"id": f"row-{i}", "value": i} for i in range(3))},
        lambda value: np.ones((value + 1, 3), dtype=np.float32),
        _normalize,
        partitions=PARTITIONS,
        cache_root=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.jsonl",
        max_scan_rows=8,
    )
    dataset = ManifestFeatureDataset(
        result["manifest"],
        stage="align",
        split="train",
        cache_root=result["cache_root"],
    )

    assert [dataset[index]["features"].shape for index in range(2)] == [(1, 3), (2, 3)]


def test_overlapping_selection_fails_closed():
    overlapping = (
        Partition("one", "train", "source", 0, 2),
        Partition("two", "train", "source", 1, 3),
    )
    with pytest.raises(ValueError, match="overlapping"):
        list(
            iter_selected_rows(
                ({"id": f"row-{i}", "value": i} for i in range(3)),
                configuration="source",
                partitions=overlapping,
                normalize_row=_normalize,
                max_scan_rows=4,
            )
        )


def test_selection_skips_duplicates_and_stops_when_the_partition_is_full():
    consumed = []

    def stream():
        for row in (
            {"id": "one", "value": 1},
            {"id": "one", "value": 2},
            {"id": "two", "value": 3},
            {"id": "unused", "value": 4},
        ):
            consumed.append(row["id"])
            yield row

    rows = list(
        iter_selected_rows(
            stream(),
            configuration="source",
            partitions=(Partition("align", "train", "source", 0, 2),),
            normalize_row=_normalize,
            max_scan_rows=8,
        )
    )
    assert [row["sample_id"] for row, _ in rows] == ["one", "two"]
    assert consumed == ["one", "one", "two"]


def test_selection_rejects_a_partial_snapshot():
    with pytest.raises(RuntimeError, match="shortfall"):
        list(
            iter_selected_rows(
                ({"id": "only", "value": 1},),
                configuration="source",
                partitions=(Partition("align", "train", "source", 0, 2),),
                normalize_row=_normalize,
                max_scan_rows=8,
            )
        )


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{message['role'][0]}>" + message["content"] for message in messages)
        return text + ("<a>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        def token_id(piece):
            if re.fullmatch(r"<SPECIAL_\d+>", piece):
                return int(piece[9:-1])
            return 100 + len(piece)

        return type(
            "Encoding", (), {"input_ids": [token_id(piece) for piece in re.findall(r"<[^>]+>|[^<\s]+", text)]}
        )()


def test_collator_emits_one_projector_kwarg_and_masks_the_prompt():
    collator = FeatureCollator(
        _Tokenizer(),
        projector_name="sample",
        placeholder_token_id=32,
    )
    row = {
        "prompt": "<SPECIAL_30><SPECIAL_32><SPECIAL_32><SPECIAL_31>\nQuestion?",
        "target": "answer",
        "features": np.ones(FEATURE_SHAPE, dtype=np.float32),
    }
    batch = collator([row, row])
    assert [key for key in batch if key.startswith("mm_")] == [
        "mm_features__sample",
        "mm_token_indices__sample",
    ]
    assert batch["mm_features__sample"].shape == (4, FEATURE_SHAPE[-1])
    assert batch["mm_token_indices__sample"].shape == (4,)
    assert (batch["input_ids"] == 32).sum().item() == 4
    assert (batch["labels"][batch["labels"] != -100] != 32).all()

    row["prompt"] = "<SPECIAL_30><SPECIAL_32><SPECIAL_31>\nQuestion?"
    with pytest.raises(ValueError, match="placeholder slots"):
        collator([row])


def test_collator_accepts_ragged_feature_and_placeholder_counts():
    collator = FeatureCollator(
        _Tokenizer(),
        projector_name="sample",
        placeholder_token_id=32,
    )
    rows = [
        {
            "prompt": "<SPECIAL_30><SPECIAL_32><SPECIAL_31>\nQuestion?",
            "target": "one",
            "features": np.ones((1, 3), dtype=np.float32),
        },
        {
            "prompt": "<SPECIAL_30><SPECIAL_32><SPECIAL_32><SPECIAL_32><SPECIAL_31>\nQuestion?",
            "target": "three",
            "features": np.ones((3, 3), dtype=np.float32),
        },
    ]

    batch = collator(rows)

    assert batch["mm_features__sample"].shape == (4, 3)
    assert batch["mm_token_indices__sample"].shape == (4,)


def test_packed_dataset_preserves_boundaries_features_and_token_budget(tmp_path):
    prepared = _publish(tmp_path, "packed")
    dataset = PackedManifestFeatureDataset(
        prepared["manifest"],
        tokenizer=_Tokenizer(),
        stage="align",
        split="train",
        cache_root=prepared["cache_root"],
        max_tokens=14,
        projector_name="sample",
        placeholder_token_id=32,
        extra_fields=["reasoning_content"],
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["qkv_format"] == "thd"
    assert item["seq_lens"].tolist() == [7, 7]
    assert item["seq_lens_padded"].tolist() == [7, 7]
    assert item["position_ids"].tolist() == [*range(7), *range(7)]
    assert item["input_ids"].numel() <= 14
    assert item["mm_features__sample"].shape == (4, 3)
    assert item["mm_token_indices__sample"].tolist() == [2, 3, 9, 10]

    pytest.importorskip("nemo_automodel")
    batch = packed_collate_fn([item])
    assert batch["input_ids"].shape == (1, 14)
    assert batch["seq_lens"].shape == (1, 2)
    assert batch["mm_features__sample"].shape == (4, 3)


def test_packed_dataset_rejects_a_sample_over_the_token_budget(tmp_path):
    prepared = _publish(tmp_path, "over-budget")
    with pytest.raises(ValueError, match="over the 6 pack budget"):
        PackedManifestFeatureDataset(
            prepared["manifest"],
            tokenizer=_Tokenizer(),
            stage="align",
            split="train",
            cache_root=prepared["cache_root"],
            max_tokens=6,
            projector_name="sample",
            placeholder_token_id=32,
        )


def test_neat_packed_dataset_emits_one_indexed_token_budget_row(tmp_path):
    pytest.importorskip("nemo_automodel")
    prepared = _publish(tmp_path, "neat-packed")
    dataset = PackedManifestFeatureDataset(
        prepared["manifest"],
        tokenizer=_Tokenizer(),
        stage="align",
        split="train",
        cache_root=prepared["cache_root"],
        max_tokens=16,
        projector_name="sample",
        placeholder_token_id=32,
        packing_strategy="neat",
    )

    item = dataset[0]
    assert item["input_ids"].shape == (16,)
    assert item["attention_mask"].tolist() == [*[1] * 7, *[2] * 7, 0, 0]
    assert item["position_ids"].tolist() == [*range(7), *range(7), 0, 0]
    assert item["labels"][-2:].tolist() == [-100, -100]
    assert "qkv_format" not in item
    assert item["mm_token_indices__sample"].tolist() == [2, 3, 9, 10]

    batch = packed_collate_fn([item])
    assert batch["input_ids"].shape == (1, 16)
    assert batch["attention_mask"].shape == (1, 16)
    assert batch["mm_features__sample"].shape == (4, 3)


def test_packed_collator_requires_one_pack_per_microbatch():
    with pytest.raises(ValueError, match="batch_size=1"):
        packed_collate_fn([{}, {}])


def _streaming_packs(prepared, *, shuffle_buffer_size=1, max_tokens=14, packing_format="thd"):
    source = ManifestFeatureIterableDatasetConfig(
        manifest_path=prepared["manifest"],
        stage="align",
        split="train",
        cache_root=prepared["cache_root"],
        projector_name="sample",
        placeholder_token_id=32,
    ).build(tokenizer=_Tokenizer())
    source.shuffle(buffer_size=shuffle_buffer_size, seed=17)
    packed, _ = StreamingFeaturePackingConfig(
        packed_sequence_size=max_tokens,
        packing_format=packing_format,
    ).build(source)
    return packed


def test_streaming_packer_tokenizes_lazily_and_rebases_feature_indices(tmp_path):
    prepared = _publish(tmp_path, "streaming")
    packed = _streaming_packs(prepared, max_tokens=15)

    assert not hasattr(packed, "packs")
    item = next(iter(packed))
    assert item["seq_lens"].tolist() == [7, 7]
    assert item["seq_lens_padded"].tolist() == [7, 8]
    assert item["input_ids"].numel() == 15
    assert item["labels"][-1].item() == -100
    assert item["position_ids"].tolist() == [*range(7), *range(8)]
    assert item["mm_features__sample"].shape == (4, 3)
    assert item["mm_token_indices__sample"].tolist() == [2, 3, 9, 10]


def test_streaming_packer_emits_neat_indexed_token_mask(tmp_path):
    prepared = _publish(tmp_path, "streaming-neat")
    packed = _streaming_packs(prepared, max_tokens=15, packing_format="neat")

    item = next(iter(packed))
    assert item["input_ids"].shape == (15,)
    assert item["attention_mask"].tolist() == [*[1] * 7, *[2] * 7, 0]
    assert item["position_ids"].tolist() == [*range(7), *range(7), 0]
    assert item["labels"][-1].item() == -100
    assert "qkv_format" not in item
    assert item["mm_features__sample"].shape == (4, 3)
    assert item["mm_token_indices__sample"].tolist() == [2, 3, 9, 10]

    batch = streaming_packed_collate_fn(item)
    assert batch["input_ids"].shape == (1, 15)
    assert batch["attention_mask"].shape == (1, 15)


def test_streaming_packer_resume_restores_partial_pack_shuffle_buffer_and_rng(tmp_path):
    prepared = _publish(tmp_path, "streaming-resume")
    iterator = iter(_streaming_packs(prepared, shuffle_buffer_size=2))
    next(iterator)
    state = iterator.state_dict()
    expected = [next(iterator)["mm_features__sample"].clone() for _ in range(4)]

    restored = iter(_streaming_packs(prepared, shuffle_buffer_size=2))
    restored.load_state_dict(state)
    actual = [next(restored)["mm_features__sample"] for _ in range(4)]

    assert state["samples"]
    assert "buffer" in state["source"]
    assert "rng_state" in state["source"]
    assert all(np.array_equal(left.numpy(), right.numpy()) for left, right in zip(expected, actual, strict=True))


def test_stateful_dataloader_resumes_streaming_packs_without_fast_forward(tmp_path):
    stateful = pytest.importorskip("torchdata.stateful_dataloader")
    prepared = _publish(tmp_path, "streaming-loader-resume")
    loader = stateful.StatefulDataLoader(
        _streaming_packs(prepared, shuffle_buffer_size=2),
        batch_size=None,
        collate_fn=lambda item: item,
    )
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = [next(iterator)["mm_features__sample"].clone() for _ in range(4)]

    restored = stateful.StatefulDataLoader(
        _streaming_packs(prepared, shuffle_buffer_size=2),
        batch_size=None,
        collate_fn=lambda item: item,
    )
    restored.load_state_dict(state)
    restored_iterator = iter(restored)
    actual = [next(restored_iterator)["mm_features__sample"] for _ in range(4)]

    assert all(np.array_equal(left.numpy(), right.numpy()) for left, right in zip(expected, actual, strict=True))
