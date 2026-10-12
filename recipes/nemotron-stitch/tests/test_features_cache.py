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

"""Content-addressed feature cache: frozen contract hash, atomic publish, resume.

``cache_id()`` is ``sha256_json(asdict(contract))``; the pinned digests below
lock the field set, field order, and JSON serialization. A changed digest
invalidates every published cache directory.
"""

from __future__ import annotations

import numpy as np
import pytest

from nemotron_stitch.features.cache import (
    CacheBuilder,
    CacheContract,
    CacheMissError,
    ImmutableEmbeddingCache,
)

CONTRACT = CacheContract(
    source_manifest_id="0" * 64,
    checkpoint_id="org/encoder-1b",
    revision="a" * 40,
    implementation_revision="impl-v1",
    layer=12,
    tokenizer_alignment="offset-v1",
    context_units=1022,
    overlap=128,
    stitching="overlap-mean-fp32-v1",
    pooling="mean21-cap128-v1",
    max_soft_tokens=128,
    dtype="float32",
    source_release="release-2026-01",
    representative_policy="longest-v1",
)

# Frozen cache hash compatibility values.
EXPECTED_CACHE_ID = "84dfe9f4fd94151a77e3924705bfa46ce173bb6a77132a883ca39827252cb268"
EXPECTED_ENTRY_KEY = "df8626e389dcc36b24bebf65e3cb394eca00b1a4defc04adeea97f854baaec6f"  # gitleaks:allow


def test_cache_id_is_byte_identical_to_the_pre_move_writer():
    assert CONTRACT.cache_id() == EXPECTED_CACHE_ID


def test_entry_key_is_byte_identical_to_the_pre_move_writer():
    assert CONTRACT.entry_key("sample::1", "b" * 64) == EXPECTED_ENTRY_KEY


def test_published_cache_loads_and_serves_entries(tmp_path):
    root = tmp_path / "cache"
    builder = CacheBuilder(root, CONTRACT)
    builder.add(EXPECTED_ENTRY_KEY, np.arange(12, dtype=np.float32).reshape(3, 4))
    second_key = CONTRACT.entry_key("sample::2", "c" * 64)
    builder.add(second_key, np.full((2, 4), 7.5, dtype=np.float32))
    builder.publish()
    cache = ImmutableEmbeddingCache(root, CONTRACT)
    assert cache.manifest["cache_id"] == EXPECTED_CACHE_ID
    assert set(cache.manifest["entries"]) == {EXPECTED_ENTRY_KEY, second_key}
    np.testing.assert_array_equal(cache.get(EXPECTED_ENTRY_KEY), np.arange(12, dtype=np.float32).reshape(3, 4))
    np.testing.assert_array_equal(
        cache.get(second_key),
        np.full((2, 4), 7.5, dtype=np.float32),
    )


def test_cache_atomic_publish_checksum_and_fail_closed(tmp_path):
    root = tmp_path / "cache"
    builder = CacheBuilder(root, CONTRACT)
    key = CONTRACT.entry_key("sample", "record-hash")
    builder.add(key, np.arange(12, dtype=np.float32).reshape(3, 4))
    manifest = builder.publish()
    assert manifest["complete"] is True
    cache = ImmutableEmbeddingCache(root, CONTRACT)
    np.testing.assert_array_equal(cache.get(key), np.arange(12, dtype=np.float32).reshape(3, 4))
    with pytest.raises(CacheMissError):
        cache.get("missing")
    with pytest.raises(FileExistsError):
        CacheBuilder(root, CONTRACT).publish()


def test_incomplete_cache_is_bound_to_its_contract(tmp_path):
    root = tmp_path / "cache"
    first = CacheBuilder(root, CONTRACT)
    first.add("entry", np.ones((1, 2), dtype=np.float32))
    interrupted = first.staging / ".interrupted.npy.tmp"
    interrupted.write_bytes(b"partial")
    resumed = CacheBuilder(root, CONTRACT)
    assert resumed.has("entry")
    assert not interrupted.exists()
    incompatible = CacheContract(**{**CONTRACT.__dict__, "tokenizer_alignment": "different-alignment-v1"})
    with pytest.raises(ValueError, match="contract mismatch"):
        CacheBuilder(root, incompatible)


def test_reader_rejects_tampered_payload(tmp_path):
    root = tmp_path / "cache"
    builder = CacheBuilder(root, CONTRACT)
    key = CONTRACT.entry_key("sample", "record-hash")
    builder.add(key, np.ones((2, 2), dtype=np.float32))
    builder.publish()
    (root / f"{key}.npy").write_bytes((root / f"{key}.npy").read_bytes()[:-1] + b"0")
    cache = ImmutableEmbeddingCache(root, CONTRACT)
    with pytest.raises(ValueError, match="checksum mismatch"):
        cache.get(key)


def test_reader_rejects_unpublished_or_incompatible_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="not published"):
        ImmutableEmbeddingCache(tmp_path / "absent", CONTRACT)
    root = tmp_path / "cache"
    CacheBuilder(root, CONTRACT).publish()
    other = CacheContract(**{**CONTRACT.__dict__, "pooling": "mean64-uncapped-v1"})
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        ImmutableEmbeddingCache(root, other)


def test_for_feature_cache_fills_the_sequence_only_sentinels():
    contract = CacheContract.for_feature_cache(
        source_manifest_id="1" * 64,
        checkpoint_id="org/encoder",
        revision="b" * 40,
        implementation_revision="impl-v2",
        layer=-1,
        max_context_units=49,
        dtype="bfloat16",
        source_release="release-2026-08",
    )
    assert contract.tokenizer_alignment == "none"
    assert contract.overlap == 0
    assert contract.stitching == "none"
    assert contract.pooling == "none"
    assert contract.representative_policy == "all_units"
    assert contract.context_units == 49
    assert contract.max_soft_tokens is None
    # Same frozen field set and hashing as the hand-built form.
    assert contract.cache_id() == CacheContract(**contract.__dict__).cache_id()


def test_for_feature_cache_rejects_nonpositive_geometry():
    with pytest.raises(ValueError, match="max_context_units"):
        CacheContract.for_feature_cache(
            source_manifest_id="1" * 64,
            checkpoint_id="org/encoder",
            revision="b" * 40,
            implementation_revision="impl-v2",
            layer=-1,
            max_context_units=0,
            dtype="bfloat16",
            source_release="release-2026-08",
        )
