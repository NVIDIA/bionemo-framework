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

"""Canonical provenance serialization and digest compatibility."""

from __future__ import annotations

import pytest

from nemotron_stitch.provenance import (
    canonical_json,
    load_provenance,
    require_provenance,
    sha256_json,
)


def test_canonical_json_is_deterministic() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_values(value) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"value": value})


def test_sha256_json_is_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_load_provenance_round_trip(tmp_path) -> None:
    lock = tmp_path / "lock.yaml"
    lock.write_text("base_model: x\nnested:\n  revision: abc\n")
    assert load_provenance(lock) == {"base_model": "x", "nested": {"revision": "abc"}}


def test_require_provenance_allows_descriptive_extras() -> None:
    require_provenance({"a": 1, "extra": "ok"}, {"a": 1}, context="test")


def test_require_provenance_recurses() -> None:
    require_provenance({"a": {"b": 2, "c": 3}}, {"a": {"b": 2}}, context="test")


def test_require_provenance_fails_closed_on_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        require_provenance({"a": 1}, {"a": 2}, context="test")


def test_require_provenance_fails_closed_on_missing_key() -> None:
    with pytest.raises(ValueError, match="missing"):
        require_provenance({}, {"a": 1}, context="test")
