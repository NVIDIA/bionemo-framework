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

"""Fail-closed immutable pooled-feature cache with atomic publication.

The contract's field set, order, and hashing are frozen:
``cache_id()`` is ``sha256_json(asdict(contract))`` and is the identity of
every published cache directory, so adding, removing, or reordering a field
invalidates existing caches and is forbidden without an explicit refill plan.
Hashing uses ``nemotron_stitch.provenance``;
``tests/test_features_cache.py`` pins the byte stream and digests.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nemotron_stitch.provenance import sha256_json


@dataclass(frozen=True)
class CacheContract:
    source_manifest_id: str
    checkpoint_id: str
    revision: str
    implementation_revision: str
    layer: int
    tokenizer_alignment: str
    context_units: int
    overlap: int
    stitching: str
    pooling: str
    max_soft_tokens: int | None
    dtype: str
    source_release: str
    representative_policy: str

    def cache_id(self) -> str:
        return sha256_json(asdict(self))

    @classmethod
    def for_feature_cache(
        cls,
        *,
        source_manifest_id: str,
        checkpoint_id: str,
        revision: str,
        implementation_revision: str,
        layer: int,
        max_context_units: int,
        dtype: str,
        source_release: str,
        max_soft_tokens: int | None = None,
    ) -> CacheContract:
        """The non-chunked convention for one feature array per source record.

        Encoders whose output is a single array per sample have no
        tokenized-text alignment, chunking, stitching, pooling, or
        representative selection; the sequence-only fields take the package's
        sentinel values so every non-sequence consumer records the same
        honest convention instead of inventing its own. Adds, removes, and
        reorders no field, so ``cache_id()`` for existing caches is untouched.
        """
        if int(max_context_units) <= 0:
            raise ValueError("max_context_units must be positive")
        return cls(
            source_manifest_id=source_manifest_id,
            checkpoint_id=checkpoint_id,
            revision=revision,
            implementation_revision=implementation_revision,
            layer=int(layer),
            tokenizer_alignment="none",
            context_units=int(max_context_units),
            overlap=0,
            stitching="none",
            pooling="none",
            max_soft_tokens=int(max_soft_tokens) if max_soft_tokens is not None else None,
            dtype=dtype,
            source_release=source_release,
            representative_policy="all_units",
        )

    def entry_key(self, source_sample_id: str, source_record_hash: str) -> str:
        return sha256_json(
            {
                "contract": asdict(self),
                "source_sample_id": source_sample_id,
                "source_record_hash": source_record_hash,
            }
        )


class CacheMissError(KeyError):
    pass


class ImmutableEmbeddingCache:
    def __init__(self, root: str | Path, contract: CacheContract):
        self.root = Path(root)
        self.contract = contract
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"cache is not published: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("complete") is not True or self.manifest.get("cache_id") != contract.cache_id():
            raise ValueError("cache manifest is incomplete or incompatible")

    def get(self, key: str) -> np.ndarray:
        record = self.manifest["entries"].get(key)
        if record is None:
            raise CacheMissError(key)
        path = self.root / record["path"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"cache checksum mismatch for {key}")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != record["shape"]:
            raise ValueError(f"cache shape mismatch for {key}")
        return array


def open_feature_cache(cache_root: str | Path) -> ImmutableEmbeddingCache:
    """Open a published cache using the complete contract in its manifest."""
    manifest = json.loads((Path(cache_root) / "manifest.json").read_text())
    return ImmutableEmbeddingCache(cache_root, CacheContract(**manifest["contract"]))


def open_validated_feature_cache(
    cache_root: str | Path,
    expected_contract: Mapping[str, Any],
) -> ImmutableEmbeddingCache:
    """Open an immutable cache only when selected provenance/geometry fields match."""
    cache_manifest = json.loads((Path(cache_root) / "manifest.json").read_text())
    contract = CacheContract(**cache_manifest["contract"])
    expected = dict(expected_contract)
    observed = {name: getattr(contract, name) for name in expected}
    if observed != expected:
        raise ValueError(f"feature-cache contract mismatch: expected {expected}, got {observed}")
    return ImmutableEmbeddingCache(cache_root, contract)


class CacheBuilder:
    def __init__(self, root: str | Path, contract: CacheContract):
        self.root = Path(root)
        self.contract = contract
        self.staging = self.root.with_name(self.root.name + ".incomplete")
        self.entries: dict[str, dict[str, Any]] = {}
        self.staging.mkdir(parents=True, exist_ok=True)
        staging_contract = self.staging / "contract.json"
        expected_contract = asdict(contract)
        if staging_contract.exists():
            if json.loads(staging_contract.read_text()) != expected_contract:
                raise ValueError(f"incomplete cache contract mismatch: {staging_contract}")
        elif any(self.staging.glob("*.npy")):
            raise ValueError(f"legacy incomplete cache has no contract identity: {self.staging}")
        else:
            staging_contract.write_text(json.dumps(expected_contract, indent=2, sort_keys=True) + "\n")
        for temporary in self.staging.glob(".*.npy.tmp"):
            temporary.unlink()
        for path in sorted(self.staging.glob("*.npy")):
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            self.entries[path.stem] = {
                "path": path.name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "nbytes": path.stat().st_size,
            }

    def has(self, key: str) -> bool:
        return key in self.entries

    def add(self, key: str, array: np.ndarray) -> None:
        if key in self.entries:
            raise ValueError(f"duplicate cache key: {key}")
        array = np.ascontiguousarray(array)
        path = self.staging / f"{key}.npy"
        temporary = self.staging / f".{key}.npy.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.entries[key] = {
            "path": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "nbytes": path.stat().st_size,
        }

    def publish(self) -> dict[str, Any]:
        if self.root.exists():
            raise FileExistsError(f"immutable cache already exists: {self.root}")
        manifest = {
            "schema_version": 1,
            "cache_id": self.contract.cache_id(),
            "contract": asdict(self.contract),
            "complete": True,
            "entries": {key: self.entries[key] for key in sorted(self.entries)},
            "entry_count": len(self.entries),
            "total_bytes": sum(r["nbytes"] for r in self.entries.values()),
        }
        (self.staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(self.staging, self.root)
        return manifest
