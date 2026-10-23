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

"""Manifest feature dataset and chat collator for AutoModel."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from nemotron_stitch.features.cache import ImmutableEmbeddingCache, open_feature_cache
from nemotron_stitch.features.manifest import load_manifest

if TYPE_CHECKING:
    from collections.abc import Iterator


# AutoModel 7d36972d5ec37f959fa01dd447d36e42ebe051a2 exposes the public
# PackingConfig target used here. The package owns the application tensor merge
# and index-rebase policy that its stock language-model packers cannot infer.


class ManifestFeatureDataset(torch.utils.data.Dataset):
    """Read one stage/split from a JSONL manifest and its immutable feature cache."""

    def __init__(
        self,
        manifest_path: str,
        *,
        stage: str,
        split: str,
        cache_root: str,
        extra_fields: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.rows = [row for row in load_manifest(manifest_path) if row["stage"] == stage and row["split"] == split]
        if not self.rows:
            raise ValueError(f"no {stage}/{split} rows in {manifest_path}")
        self.cache_root = cache_root
        self.extra_fields = tuple(str(field) for field in extra_fields)
        self._cache: ImmutableEmbeddingCache | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cache is None:
            self._cache = open_feature_cache(self.cache_root)
        row = self.rows[index]
        features = np.asarray(self._cache.get(row["feature_key"]))
        return {
            "prompt": row["prompt"],
            "target": row["target"],
            "features": features,
            **{field: row[field] for field in self.extra_fields if field in row},
        }


# AutoModel 1814c6c93a66b9d59d254960ef6a99a64249b671's stock THD and NEAT
# packers retain only LM fields, dropping external feature tensors and their
# scatter indices. Delete this dataset when packing can preserve and rebase
# arbitrary per-sample tensor payloads.
class PackedManifestFeatureDataset(torch.utils.data.Dataset):
    """Greedily group feature-backed chat samples into token-budget packs."""

    def __init__(
        self,
        manifest_path: str,
        *,
        tokenizer: Any,
        stage: str,
        split: str,
        cache_root: str,
        max_tokens: int,
        projector_name: str,
        placeholder_token_id: int,
        packing_strategy: str = "thd",
        max_length: int = 1024,
        chat_template_kwargs: dict[str, Any] | None = None,
        supervision: str = "suffix",
        truncate: bool = False,
        extra_fields: list[str] | tuple[str, ...] = (),
    ) -> None:
        if tokenizer is None:
            raise ValueError("a tokenizer is required to build packed samples")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if packing_strategy not in ("thd", "neat"):
            raise ValueError(f"unsupported packing strategy: {packing_strategy!r}")
        self.dataset = ManifestFeatureDataset(
            manifest_path,
            stage=stage,
            split=split,
            cache_root=cache_root,
            extra_fields=extra_fields,
        )
        self.collator = FeatureCollator(
            tokenizer,
            projector_name=projector_name,
            placeholder_token_id=placeholder_token_id,
            max_length=max_length,
            chat_template_kwargs=chat_template_kwargs,
            supervision=supervision,
            truncate=truncate,
        )
        self.max_tokens = int(max_tokens)
        self.packing_strategy = packing_strategy
        self.padding_token_id = tokenizer.pad_token_id
        if self.padding_token_id is None:
            self.padding_token_id = tokenizer.eos_token_id
        if packing_strategy == "neat" and self.padding_token_id is None:
            raise ValueError("NEAT packing requires a tokenizer pad or EOS token")
        encoded_rows: list[tuple[int, list[int], list[int]]] = []
        lengths: list[int] = []
        for index, row in enumerate(self.dataset.rows):
            ids, labels = self.collator._encode(row)
            sample_tokens = len(ids) - 1
            if sample_tokens < 1:
                raise ValueError(f"sample {index} has no next-token training positions")
            if sample_tokens > max_tokens:
                raise ValueError(f"sample {index} has {sample_tokens} tokens, over the {max_tokens} pack budget")
            encoded_rows.append((index, ids, labels))
            lengths.append(sample_tokens)

        self.packs: list[list[tuple[int, list[int], list[int]]]] = []
        if packing_strategy == "neat":
            from nemo_automodel.components.datasets.llm.neat_packing import greedy_knapsack

            self.packs = [
                [encoded_rows[index] for index in indices] for indices in greedy_knapsack(lengths, max_tokens)
            ]
            return

        current: list[tuple[int, list[int], list[int]]] = []
        current_tokens = 0
        for encoded, sample_tokens in zip(encoded_rows, lengths, strict=True):
            if current and current_tokens + sample_tokens > max_tokens:
                self.packs.append(current)
                current = []
                current_tokens = 0
            current.append(encoded)
            current_tokens += sample_tokens
        if current:
            self.packs.append(current)

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pack = self.packs[index]
        examples = [self.dataset[row_index] for row_index, _, _ in pack]
        input_ids = torch.tensor([token for _, ids, _ in pack for token in ids[:-1]], dtype=torch.long)
        labels = torch.tensor([token for _, _, row_labels in pack for token in row_labels[1:]], dtype=torch.long)
        seq_lens = [len(ids) - 1 for _, ids, _ in pack]
        position_ids = torch.cat([torch.arange(length) for length in seq_lens])
        positions = input_ids.eq(self.collator.placeholder_token_id).nonzero().flatten()

        attention_mask = None
        if self.packing_strategy == "neat":
            token_count = input_ids.numel()
            pad_tokens = self.max_tokens - token_count
            attention_mask = torch.cat(
                [
                    *(torch.full((length,), sequence, dtype=torch.long) for sequence, length in enumerate(seq_lens, 1)),
                    torch.zeros(pad_tokens, dtype=torch.long),
                ]
            )
            input_ids = torch.cat([input_ids, torch.full((pad_tokens,), int(self.padding_token_id), dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_tokens,), -100, dtype=torch.long)])
            position_ids = torch.cat([position_ids, torch.zeros(pad_tokens, dtype=torch.long)])

        feature_rows = [torch.as_tensor(np.asarray(example["features"]), dtype=torch.float32) for example in examples]
        if not all(features.ndim == 2 for features in feature_rows):
            raise ValueError("feature packing requires rank-2 feature tensors")
        if positions.numel() != sum(features.shape[0] for features in feature_rows):
            raise ValueError(
                f"pack has {positions.numel()} placeholder slots for "
                f"{sum(features.shape[0] for features in feature_rows)} feature rows"
            )
        name = self.collator.projector_name
        result: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            f"mm_features__{name}": torch.cat(feature_rows),
            f"mm_token_indices__{name}": positions,
        }
        if attention_mask is not None:
            result["attention_mask"] = attention_mask
        else:
            result.update(
                seq_lens=torch.tensor(seq_lens, dtype=torch.long),
                seq_lens_padded=torch.tensor(seq_lens, dtype=torch.long),
                qkv_format="thd",
            )
        return result


def _assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return str(item.get("text", ""))
    return ""


def _find_pattern(template: list[int], pattern: list[int], start: int = 0) -> tuple[int, int]:
    if not pattern:
        return -1, -1
    for index in range(start, len(template) - len(pattern) + 1):
        if template[index : index + len(pattern)] == pattern:
            return index, index + len(pattern)
    return -1, -1


class FeatureCollator:
    """Tokenize chat rows and attach one ``mm_features__*`` tensor."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        projector_name: str,
        placeholder_token_id: int,
        max_length: int = 1024,
        chat_template_kwargs: dict[str, Any] | None = None,
        supervision: str = "suffix",
        truncate: bool = False,
    ) -> None:
        if supervision not in ("suffix", "assistant_content"):
            raise ValueError(f"unsupported supervision mode: {supervision!r}")
        self.tokenizer = tokenizer
        self.projector_name = str(projector_name)
        self.placeholder_token_id = int(placeholder_token_id)
        self.max_length = int(max_length)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.supervision = supervision
        self.truncate = bool(truncate)

    @staticmethod
    def _assistant_message(content: str, reasoning: str | None = None) -> dict[str, str]:
        if reasoning:
            return {"role": "assistant", "reasoning_content": reasoning, "content": content}
        return {"role": "assistant", "content": content}

    def _messages(self, row: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        prompt: list[dict[str, str]] = []
        if row.get("system_prompt"):
            prompt.append({"role": "system", "content": str(row["system_prompt"])})
        prompt.append({"role": "user", "content": str(row["prompt"])})
        full = [*prompt, self._assistant_message(str(row["target"]), row.get("reasoning_content"))]
        for turn in row.get("turns", ()):
            full.append({"role": "user", "content": str(turn["question"])})
            full.append(self._assistant_message(str(turn.get("answer", "")), turn.get("reasoning_content")))
        if self.supervision == "suffix" and len(full) != len(prompt) + 1:
            raise ValueError("multi-turn rows require supervision='assistant_content'")
        return prompt, full

    def _encode(self, row: dict[str, Any]) -> tuple[list[int], list[int]]:
        prompt_messages, full_messages = self._messages(row)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            **self.chat_template_kwargs,
        )
        prompt_ids = list(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
        full_ids = list(self.tokenizer(full_text, add_special_tokens=False).input_ids)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            # Transformers 5.12.1 has no prompt-preserving continuation API:
            # full-text tokenization may merge across the generation boundary.
            # Delete this fallback when the pinned public API meets U-34.
            if self.supervision != "suffix" or not full_text.startswith(prompt_text):
                raise ValueError("chat template is not prefix-stable between prompt and full rendering")
            continuation = self.tokenizer(full_text[len(prompt_text) :], add_special_tokens=False).input_ids
            full_ids = prompt_ids + list(continuation)
            for ids, text in ((prompt_ids, prompt_text), (full_ids, full_text)):
                if self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
                    raise ValueError("chat continuation encoding does not preserve the native rendered text")
        if self.supervision == "suffix":
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        else:
            labels = [-100] * len(full_ids)
            search_start = 0
            for message in full_messages:
                if message.get("role") != "assistant":
                    continue
                text = _assistant_text(message)
                tokens = list(self.tokenizer(text, add_special_tokens=False).input_ids)
                start, end = _find_pattern(full_ids, tokens, search_start)
                if start < 0 and text != text.lstrip():
                    tokens = list(self.tokenizer(text.lstrip(), add_special_tokens=False).input_ids)
                    start, end = _find_pattern(full_ids, tokens, search_start)
                if start < 0 and text.strip():
                    raise ValueError(f"assistant content is absent from the rendered chat template: {text!r}")
                if start >= 0:
                    labels[start:end] = full_ids[start:end]
                    search_start = end

        if len(full_ids) > self.max_length:
            if not self.truncate:
                raise ValueError(f"sample is {len(full_ids)} tokens, over the {self.max_length} cap")
            full_ids = full_ids[: self.max_length]
            labels = labels[: self.max_length]
        return full_ids, labels

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(row) for row in examples]
        width = max(len(ids) for ids, _ in encoded)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = torch.full((len(encoded), width), int(pad_id), dtype=torch.long)
        labels = torch.full((len(encoded), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, (ids, row_labels) in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            labels[row, : len(ids)] = torch.tensor(row_labels)
            attention_mask[row, : len(ids)] = 1
        feature_rows = [torch.as_tensor(np.asarray(example["features"]), dtype=torch.float32) for example in examples]
        result = {
            "input_ids": input_ids[:, :-1],
            "labels": labels[:, 1:],
            "attention_mask": attention_mask[:, :-1],
        }
        if all(features.ndim == 2 for features in feature_rows):
            result[f"mm_features__{self.projector_name}"] = torch.cat(feature_rows)
            sequence_length = result["input_ids"].shape[1]
            indices = []
            for row, features in enumerate(feature_rows):
                positions = result["input_ids"][row].eq(self.placeholder_token_id).nonzero().flatten()
                if positions.numel() != features.shape[0]:
                    raise ValueError(
                        f"row {row} has {positions.numel()} placeholder slots for {features.shape[0]} feature rows"
                    )
                indices.append(positions + row * sequence_length)
            result[f"mm_token_indices__{self.projector_name}"] = torch.cat(indices)
        else:
            result[f"mm_features__{self.projector_name}"] = torch.stack(feature_rows)
        return result


class _ManifestFeatureIterator:
    """Stateful JSONL iterator with exact bounded-shuffle resume."""

    def __init__(self, dataset: ManifestFeatureIterableDataset) -> None:
        worker = torch.utils.data.get_worker_info()
        workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        self.dataset = dataset
        self.num_shards = dataset.num_shards * workers
        self.shard_index = dataset.shard_index * workers + worker_id
        self.stream = Path(dataset.manifest_path).open()  # noqa: SIM115 - iterator owns the stream lifetime
        self.offset = 0
        self.selected_rows = 0
        self.epoch = dataset.epoch
        self.buffer: list[dict[str, Any]] = []
        self.source_exhausted = False
        self.matched_in_epoch = False
        self.shard_rows_in_epoch = 0
        self.rng = random.Random(dataset.shuffle_seed + self.epoch)
        self.cache: ImmutableEmbeddingCache | None = None

    def __iter__(self) -> _ManifestFeatureIterator:
        return self

    def _read_row(self) -> dict[str, Any] | None:
        while line := self.stream.readline():
            self.offset = self.stream.tell()
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") != self.dataset.stage or row.get("split") != self.dataset.split:
                continue
            selected = self.selected_rows
            self.selected_rows += 1
            self.matched_in_epoch = True
            if selected % self.num_shards == self.shard_index:
                self.shard_rows_in_epoch += 1
                return row
        self.source_exhausted = True
        return None

    def _reset_source(self) -> None:
        if not self.matched_in_epoch:
            raise ValueError(f"no {self.dataset.stage}/{self.dataset.split} rows in {self.dataset.manifest_path}")
        if not self.shard_rows_in_epoch:
            raise ValueError(f"shard {self.shard_index} of {self.num_shards} has no matching manifest rows")
        self.epoch += 1
        self.stream.seek(0)
        self.offset = 0
        self.selected_rows = 0
        self.source_exhausted = False
        self.matched_in_epoch = False
        self.shard_rows_in_epoch = 0
        self.rng.seed(self.dataset.shuffle_seed + self.epoch)

    def _next_row(self) -> dict[str, Any]:
        while True:
            if self.dataset.shuffle_buffer_size <= 1:
                row = self._read_row()
                if row is not None:
                    return row
                if not self.dataset.repeat_on_exhaustion:
                    raise StopIteration
                self._reset_source()
                continue

            while not self.source_exhausted and len(self.buffer) < self.dataset.shuffle_buffer_size:
                row = self._read_row()
                if row is not None:
                    self.buffer.append(row)
            if self.buffer:
                index = self.rng.randrange(len(self.buffer))
                row = self.buffer[index]
                replacement = None if self.source_exhausted else self._read_row()
                if replacement is None:
                    self.buffer.pop(index)
                else:
                    self.buffer[index] = replacement
                return row
            if not self.dataset.repeat_on_exhaustion:
                raise StopIteration
            self._reset_source()
        raise RuntimeError("unreachable")

    def __next__(self) -> dict[str, Any]:
        row = self._next_row()
        if self.cache is None:
            self.cache = open_feature_cache(self.dataset.cache_root)
        ids, labels = self.dataset.collator._encode(row)
        sample_tokens = len(ids) - 1
        if sample_tokens < 1:
            raise ValueError(f"sample {row.get('sample_id', '<unknown>')} has no next-token training positions")
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        features = torch.as_tensor(np.asarray(self.cache.get(row["feature_key"])), dtype=torch.float32)
        if features.ndim != 2:
            raise ValueError("feature packing requires rank-2 feature tensors")
        positions = input_ids.eq(self.dataset.collator.placeholder_token_id).nonzero().flatten()
        if positions.numel() != features.shape[0]:
            raise ValueError(
                f"sample {row.get('sample_id', '<unknown>')} has {positions.numel()} placeholder slots for "
                f"{features.shape[0]} feature rows"
            )
        name = self.dataset.collator.projector_name
        return {
            "input_ids": input_ids,
            "labels": torch.tensor(labels[1:], dtype=torch.long),
            f"mm_features__{name}": features,
            f"mm_token_indices__{name}": positions,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "selected_rows": self.selected_rows,
            "epoch": self.epoch,
            "buffer": deepcopy(self.buffer),
            "source_exhausted": self.source_exhausted,
            "matched_in_epoch": self.matched_in_epoch,
            "shard_rows_in_epoch": self.shard_rows_in_epoch,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.offset = int(state_dict["offset"])
        self.selected_rows = int(state_dict["selected_rows"])
        self.epoch = int(state_dict["epoch"])
        self.buffer = list(state_dict["buffer"])
        self.source_exhausted = bool(state_dict["source_exhausted"])
        self.matched_in_epoch = bool(state_dict["matched_in_epoch"])
        self.shard_rows_in_epoch = int(state_dict["shard_rows_in_epoch"])
        self.rng.setstate(state_dict["rng_state"])
        self.stream.seek(self.offset)


class ManifestFeatureIterableDataset(torch.utils.data.IterableDataset):
    """Lazily read, tokenize, and attach cached features to manifest rows."""

    def __init__(
        self,
        manifest_path: str,
        *,
        tokenizer: Any,
        stage: str,
        split: str,
        cache_root: str,
        projector_name: str,
        placeholder_token_id: int,
        max_length: int = 1024,
        chat_template_kwargs: dict[str, Any] | None = None,
        supervision: str = "suffix",
        truncate: bool = False,
        repeat_on_exhaustion: bool = True,
    ) -> None:
        if tokenizer is None:
            raise ValueError("a tokenizer is required to stream manifest samples")
        self.manifest_path = str(manifest_path)
        self.stage = str(stage)
        self.split = str(split)
        self.cache_root = str(cache_root)
        self.repeat_on_exhaustion = bool(repeat_on_exhaustion)
        self.num_shards = 1
        self.shard_index = 0
        self.shuffle_buffer_size = 1
        self.shuffle_seed = 0
        self.epoch = 0
        self.collator = FeatureCollator(
            tokenizer,
            projector_name=projector_name,
            placeholder_token_id=placeholder_token_id,
            max_length=max_length,
            chat_template_kwargs=chat_template_kwargs,
            supervision=supervision,
            truncate=truncate,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return _ManifestFeatureIterator(self)

    def shard(self, num_shards: int, index: int) -> ManifestFeatureIterableDataset:
        if num_shards < 1 or not 0 <= index < num_shards:
            raise ValueError(f"invalid shard {index} of {num_shards}")
        self.num_shards = int(num_shards)
        self.shard_index = int(index)
        return self

    def shuffle(self, buffer_size: int = 1000, seed: int | None = None) -> ManifestFeatureIterableDataset:
        if buffer_size < 1:
            raise ValueError(f"shuffle buffer must be positive, got {buffer_size}")
        self.shuffle_buffer_size = int(buffer_size)
        self.shuffle_seed = int(seed or 0)
        return self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


@dataclass
class ManifestFeatureIterableDatasetConfig:
    """AutoModel tokenizer-aware config for lazy manifest feature samples."""

    accepts_tokenizer: ClassVar[bool] = True

    manifest_path: str
    stage: str
    split: str
    cache_root: str
    projector_name: str
    placeholder_token_id: int
    max_length: int = 1024
    chat_template_kwargs: dict[str, Any] | None = None
    supervision: str = "suffix"
    truncate: bool = False
    repeat_on_exhaustion: bool = True

    def build(self, *, tokenizer: Any) -> ManifestFeatureIterableDataset:
        return ManifestFeatureIterableDataset(
            self.manifest_path,
            tokenizer=tokenizer,
            stage=self.stage,
            split=self.split,
            cache_root=self.cache_root,
            projector_name=self.projector_name,
            placeholder_token_id=self.placeholder_token_id,
            max_length=self.max_length,
            chat_template_kwargs=self.chat_template_kwargs,
            supervision=self.supervision,
            truncate=self.truncate,
            repeat_on_exhaustion=self.repeat_on_exhaustion,
        )


def _merge_feature_pack(
    samples: list[dict[str, Any]],
    *,
    max_tokens: int,
    pad_token_id: int,
    packing_format: str,
) -> dict[str, Any]:
    seq_lens = [int(torch.as_tensor(sample["input_ids"]).numel()) for sample in samples]
    pad_tokens = max_tokens - sum(seq_lens)
    feature_keys = {key for key in samples[0] if key.startswith("mm_features__")}
    expected_keys = feature_keys | {key.replace("mm_features__", "mm_token_indices__") for key in feature_keys}
    for sample in samples:
        observed = {key for key in sample if key.startswith(("mm_features__", "mm_token_indices__"))}
        if observed != expected_keys:
            raise ValueError(
                f"inconsistent multimodal payload keys: expected {sorted(expected_keys)}, got {sorted(observed)}"
            )

    result: dict[str, Any] = {
        "input_ids": torch.cat(
            [
                *(torch.as_tensor(sample["input_ids"], dtype=torch.long) for sample in samples),
                torch.full((pad_tokens,), pad_token_id, dtype=torch.long),
            ]
        ),
        "labels": torch.cat(
            [
                *(torch.as_tensor(sample["labels"], dtype=torch.long) for sample in samples),
                torch.full((pad_tokens,), -100, dtype=torch.long),
            ]
        ),
    }
    if packing_format == "neat":
        result.update(
            position_ids=torch.cat(
                [*(torch.arange(length) for length in seq_lens), torch.zeros(pad_tokens, dtype=torch.long)]
            ),
            attention_mask=torch.cat(
                [
                    *(torch.full((length,), sequence, dtype=torch.long) for sequence, length in enumerate(seq_lens, 1)),
                    torch.zeros(pad_tokens, dtype=torch.long),
                ]
            ),
        )
    else:
        result.update(
            position_ids=torch.cat(
                [
                    *(torch.arange(length) for length in seq_lens),
                    torch.arange(seq_lens[-1], seq_lens[-1] + pad_tokens),
                ]
            ),
            seq_lens=torch.tensor(seq_lens, dtype=torch.long),
            seq_lens_padded=torch.tensor([*seq_lens[:-1], seq_lens[-1] + pad_tokens], dtype=torch.long),
            qkv_format="thd",
        )
    offsets = [0]
    for length in seq_lens[:-1]:
        offsets.append(offsets[-1] + length)
    for feature_key in sorted(feature_keys):
        index_key = feature_key.replace("mm_features__", "mm_token_indices__")
        features = [torch.as_tensor(sample[feature_key]) for sample in samples]
        indices = [torch.as_tensor(sample[index_key], dtype=torch.long) for sample in samples]
        if any(feature.shape[0] != index.numel() for feature, index in zip(features, indices, strict=True)):
            raise ValueError(f"{feature_key} rows do not match {index_key} entries")
        for sample_length, index in zip(seq_lens, indices, strict=True):
            if index.numel() and (index.min().item() < 0 or index.max().item() >= sample_length):
                raise ValueError(f"{index_key} contains an out-of-range sample-local token index")
        result[feature_key] = torch.cat(features)
        result[index_key] = torch.cat([index + offset for index, offset in zip(indices, offsets, strict=True)])
    return result


class _StreamingFeaturePackIterator:
    def __init__(self, dataset: StreamingFeaturePackDataset) -> None:
        self.dataset = dataset
        self.source = iter(dataset.dataset)
        self.samples: list[dict[str, Any]] = []
        self.current_tokens = 0
        self.emitted = 0

    def __iter__(self) -> _StreamingFeaturePackIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        if self.dataset.max_packs is not None and self.emitted >= self.dataset.max_packs:
            raise StopIteration
        while True:
            try:
                sample = next(self.source)
            except StopIteration:
                if not self.samples:
                    raise
                pack = _merge_feature_pack(
                    self.samples,
                    max_tokens=self.dataset.max_tokens,
                    pad_token_id=self.dataset.pad_token_id,
                    packing_format=self.dataset.packing_format,
                )
                self.samples = []
                self.current_tokens = 0
                self.emitted += 1
                return pack
            sample_tokens = int(torch.as_tensor(sample["input_ids"]).numel())
            if sample_tokens < 1:
                raise ValueError("sample has no next-token training positions")
            if sample_tokens > self.dataset.max_tokens:
                raise ValueError(f"sample has {sample_tokens} tokens, over the {self.dataset.max_tokens} pack budget")
            if self.samples and self.current_tokens + sample_tokens > self.dataset.max_tokens:
                pack = _merge_feature_pack(
                    self.samples,
                    max_tokens=self.dataset.max_tokens,
                    pad_token_id=self.dataset.pad_token_id,
                    packing_format=self.dataset.packing_format,
                )
                self.samples = [sample]
                self.current_tokens = sample_tokens
                self.emitted += 1
                return pack
            self.samples.append(sample)
            self.current_tokens += sample_tokens
        raise RuntimeError("unreachable")

    def state_dict(self) -> dict[str, Any]:
        if not callable(getattr(self.source, "state_dict", None)):
            raise TypeError(f"{type(self.source).__name__} must implement state_dict for exact resume")
        return {
            "source": self.source.state_dict(),
            "samples": deepcopy(self.samples),
            "current_tokens": self.current_tokens,
            "emitted": self.emitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not callable(getattr(self.source, "load_state_dict", None)):
            raise TypeError(f"{type(self.source).__name__} must implement load_state_dict for exact resume")
        self.source.load_state_dict(state_dict["source"])
        self.samples = list(state_dict["samples"])
        self.current_tokens = int(state_dict["current_tokens"])
        self.emitted = int(state_dict["emitted"])


class StreamingFeaturePackDataset(torch.utils.data.IterableDataset):
    """Greedily pack an iterable sample stream while retaining only one pack."""

    def __init__(
        self,
        dataset: Any,
        *,
        max_tokens: int,
        pad_token_id: int,
        packing_format: str = "thd",
        max_packs: int | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if packing_format not in ("thd", "neat"):
            raise ValueError(f"unsupported packing format: {packing_format!r}")
        self.dataset = dataset
        self.max_tokens = int(max_tokens)
        self.pad_token_id = int(pad_token_id)
        self.packing_format = packing_format
        self.max_packs = int(max_packs) if max_packs is not None else None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return _StreamingFeaturePackIterator(self)

    def shard(self, num_shards: int, index: int) -> StreamingFeaturePackDataset:
        self.dataset = self.dataset.shard(num_shards, index)
        return self

    def shuffle(self, buffer_size: int = 1000, seed: int | None = None) -> StreamingFeaturePackDataset:
        self.dataset = self.dataset.shuffle(buffer_size=buffer_size, seed=seed)
        return self

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


def streaming_packed_collate_fn(item: dict[str, Any]) -> dict[str, Any]:
    """Batch one pack emitted through a dataloader configured with ``batch_size=None``."""
    return packed_collate_fn([item])


@dataclass
class StreamingFeaturePackingConfig:
    """AutoModel packing target for feature-preserving streaming packs."""

    packed_sequence_size: int
    packing_format: str = "thd"
    max_packs: int | None = None
    prepacked: bool = False
    num_proc: int = 1

    def build(
        self,
        dataset: object,
        *,
        split: str | list[str] | None = None,
        seed: int = 42,
        supports_seq_lens: bool = True,
        pad_token_id: int = 0,
        cp_size: int = 1,
        attn_implementation: str | None = None,
    ) -> tuple[object, Any]:
        del split, seed, attn_implementation
        if self.packing_format == "thd" and not supports_seq_lens:
            raise ValueError("streaming THD packing requires a model forward that accepts seq_lens")
        if cp_size != 1:
            raise ValueError("streaming feature packing does not support context parallelism")
        if self.prepacked:
            raise ValueError("StreamingFeaturePackingConfig cannot be marked prepacked")
        if self.num_proc != 1:
            raise ValueError("streaming tokenization does not support num_proc")
        resolved_pad_token_id: Any = pad_token_id
        if resolved_pad_token_id is None:
            tokenizer = getattr(getattr(dataset, "collator", None), "tokenizer", None)
            resolved_pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if resolved_pad_token_id is None:
            raise ValueError("streaming feature packing requires a tokenizer pad or EOS token")
        return (
            StreamingFeaturePackDataset(
                dataset,
                max_tokens=self.packed_sequence_size,
                pad_token_id=resolved_pad_token_id,
                packing_format=self.packing_format,
                max_packs=self.max_packs,
            ),
            streaming_packed_collate_fn,
        )


@lru_cache(maxsize=8)
def _cached_collator(
    tokenizer_name_or_path: str,
    tokenizer_revision: str | None,
    projector_name: str,
    placeholder_token_id: int,
    max_length: int,
    chat_template_kwargs: tuple | None,
    supervision: str,
    truncate: bool,
    trust_remote_code: bool,
) -> FeatureCollator:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        revision=tokenizer_revision,
        trust_remote_code=trust_remote_code,
    )
    return FeatureCollator(
        tokenizer,
        projector_name=projector_name,
        placeholder_token_id=placeholder_token_id,
        max_length=max_length,
        chat_template_kwargs=dict(chat_template_kwargs) if chat_template_kwargs else None,
        supervision=supervision,
        truncate=truncate,
    )


def collate_fn(
    batch: list[dict[str, Any]],
    *,
    tokenizer_name_or_path: str,
    projector_name: str,
    placeholder_token_id: int,
    tokenizer_revision: str | None = None,
    max_length: int = 1024,
    chat_template_kwargs: dict[str, Any] | None = None,
    supervision: str = "suffix",
    truncate: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, torch.Tensor]:
    """Config-friendly, process-local-cached entry point for :class:`FeatureCollator`."""
    template_kwargs = tuple(sorted((chat_template_kwargs or {}).items()))
    if trust_remote_code and not tokenizer_revision:
        raise ValueError("trust_remote_code requires a pinned tokenizer_revision")
    return _cached_collator(
        tokenizer_name_or_path,
        tokenizer_revision,
        projector_name,
        int(placeholder_token_id),
        int(max_length),
        template_kwargs,
        supervision,
        bool(truncate),
        bool(trust_remote_code),
    )(batch)


def packed_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate one feature-preserving THD or indexed pack."""
    if len(batch) != 1:
        raise ValueError(f"feature packs require dataloader batch_size=1, got {len(batch)}")
    item = batch[0]
    if item.get("qkv_format") != "thd":
        result = {
            key: value.unsqueeze(0)
            for key, value in item.items()
            if key in ("input_ids", "labels", "position_ids", "attention_mask")
        }
        result.update({key: value for key, value in item.items() if key.startswith("mm_")})
        return result
    from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

    result = packed_sequence_thd_collater(
        [{key: item[key].tolist() for key in ("input_ids", "labels", "position_ids", "seq_lens", "seq_lens_padded")}]
    )
    result.update({key: value for key, value in item.items() if key.startswith("mm_")})
    return result
