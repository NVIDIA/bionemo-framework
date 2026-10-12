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

"""Shared JSONL manifest helpers for cached encoder features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron_stitch.provenance import sha256_json


def manifest_source_id(records: list[dict[str, Any]]) -> str:
    """Return the content identity of records before cache keys are assigned."""
    return sha256_json([{key: value for key, value in record.items() if key != "feature_key"} for record in records])


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a nonempty JSONL manifest and reject duplicate sample ids."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"manifest contains duplicate sample ids: {path}")
    return rows
