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

"""Bounded preparation into the immutable feature cache.

Applications own source access, row normalization, and encoder construction.
This module owns the shared walk, partitioning, cache publication, and
manifest files used independently by the LLaVA and CT paths.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nemotron_stitch.features.cache import CacheBuilder, CacheContract
from nemotron_stitch.features.manifest import manifest_source_id


@dataclass(frozen=True)
class Partition:
    """One stage/split slice of a source configuration's valid-row ordinals."""

    stage: str
    split: str
    configuration: str
    first: int
    last: int

    def __post_init__(self) -> None:
        if not self.stage or not self.split or not self.configuration:
            raise ValueError("partition stage, split, and configuration must be nonempty")
        if self.first < 0 or self.last <= self.first:
            raise ValueError(f"invalid partition interval [{self.first}, {self.last})")

    def contains(self, ordinal: int) -> bool:
        return self.first <= ordinal < self.last


class InvalidRow(Exception):
    """A per-row quality failure that does not consume a valid-row ordinal."""


def iter_selected_rows(
    stream: Iterable[dict[str, Any]],
    *,
    configuration: str,
    partitions: tuple[Partition, ...],
    normalize_row: Callable[..., tuple[dict[str, Any], Any]],
    max_scan_rows: int,
) -> Iterable[tuple[dict[str, Any], Any]]:
    """Normalize and yield the requested partitions from one bounded stream."""
    wanted = tuple(partition for partition in partitions if partition.configuration == configuration)
    if not wanted:
        raise ValueError(f"no partition covers configuration {configuration!r}")
    ordered = sorted(wanted, key=lambda partition: partition.first)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.first < previous.last:
            raise ValueError(f"overlapping partitions for configuration {configuration!r}")

    counts = {partition: 0 for partition in wanted}
    seen_ids: set[str] = set()
    ordinal = 0
    complete = False
    for source_index, row in enumerate(stream):
        if source_index >= int(max_scan_rows):
            break
        try:
            record, encoder_input = normalize_row(
                row,
                configuration=configuration,
                source_index=source_index,
            )
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"normalized row at source index {source_index} has no sample_id")
            if sample_id in seen_ids:
                raise InvalidRow(f"duplicate id within the snapshot: {sample_id!r}")
        except InvalidRow:
            continue
        seen_ids.add(sample_id)
        partition = next((candidate for candidate in wanted if candidate.contains(ordinal)), None)
        if partition is not None:
            record["ordinal"] = ordinal
            record["stage"] = partition.stage
            record["split"] = partition.split
            counts[partition] += 1
            yield record, encoder_input
        ordinal += 1
        if all(counts[partition] == partition.last - partition.first for partition in wanted):
            complete = True
            break
    if not complete:
        shortfall = {
            f"{partition.stage}/{partition.split}": partition.last - partition.first - counts[partition]
            for partition in wanted
        }
        raise RuntimeError(
            f"configuration {configuration!r}: stream did not yield enough valid, unique rows "
            f"within {max_scan_rows} scanned rows (shortfall: {shortfall})"
        )


def prepare_features(
    streams: Mapping[str, Iterable[dict[str, Any]]],
    encode: Callable[[Any], np.ndarray],
    normalize_row: Callable[..., tuple[dict[str, Any], Any]],
    *,
    partitions: tuple[Partition, ...],
    encoder: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    cache_root: str | Path,
    manifest_path: str | Path,
    max_scan_rows: int,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Publish one bounded feature snapshot and its stage slices."""
    encoder = dict(encoder or {})
    dataset = dict(dataset or {})

    records: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    max_context_units = 0
    configurations = tuple(dict.fromkeys(partition.configuration for partition in partitions))
    for configuration in configurations:
        if configuration not in streams:
            raise ValueError(f"streams is missing configuration {configuration!r}")
        for record, encoder_input in iter_selected_rows(
            streams[configuration],
            configuration=configuration,
            partitions=partitions,
            normalize_row=normalize_row,
            max_scan_rows=max_scan_rows,
        ):
            encoded = np.asarray(encode(encoder_input))
            if encoded.ndim == 0:
                raise ValueError("encoder must produce a non-scalar feature array")
            if str(encoded.dtype) != dtype:
                raise ValueError(f"encoder produced dtype {encoded.dtype}, but the cache contract records {dtype!r}")
            max_context_units = max(max_context_units, encoded.shape[0])
            record["feature_digest"] = hashlib.sha256(encoded.tobytes()).hexdigest()
            records.append(record)
            features.append(encoded)

    source_id = manifest_source_id(records)
    contract = CacheContract.for_feature_cache(
        source_manifest_id=source_id,
        checkpoint_id=str(encoder.get("repo_id", "")),
        revision=str(encoder.get("revision", "")),
        implementation_revision=str(encoder.get("implementation_revision", "")),
        layer=int(encoder.get("layer", -1)),
        max_context_units=max_context_units,
        dtype=dtype,
        source_release=str(dataset.get("revision", "")),
    )
    builder = CacheBuilder(cache_root, contract)
    for record, feature in zip(records, features, strict=True):
        key = contract.entry_key(record["sample_id"], record["feature_digest"])
        if not builder.has(key):
            builder.add(key, feature)
        record["feature_key"] = key
    published = builder.publish()

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "dataset": dict(dataset),
        "encoder": dict(encoder),
        "source_manifest_id": source_id,
        "cache_id": contract.cache_id(),
        "records": len(records),
    }
    manifest_path.with_suffix(".header.json").write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    for stage, split in dict.fromkeys((partition.stage, partition.split) for partition in partitions):
        rows = [record for record in records if record["stage"] == stage and record["split"] == split]
        if rows:
            manifest_path.with_name(f"{manifest_path.stem}-{stage}-{split}.jsonl").write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in rows)
            )
    return {
        "records": len(records),
        "cache_id": contract.cache_id(),
        "retained_bytes": published["total_bytes"] + manifest_path.stat().st_size,
        "manifest": str(manifest_path),
        "cache_root": str(cache_root),
    }
