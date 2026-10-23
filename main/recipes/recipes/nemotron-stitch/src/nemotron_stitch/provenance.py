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

"""Small, deterministic provenance primitives used by manifests and caches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashes and safetensors metadata."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_stream(stream: BinaryIO, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return sha256_stream(stream)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_provenance(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML/JSON provenance lock."""
    import yaml

    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"provenance lock must be a mapping: {path}")
    return value


def require_provenance(actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str) -> None:
    """Require every locked value, recursively, while allowing descriptive extras."""

    def compare(observed: Any, locked: Any, path: str) -> None:
        if isinstance(locked, Mapping):
            if not isinstance(observed, Mapping):
                raise ValueError(f"{context} provenance mismatch for {path}: expected a mapping")
            for key, value in locked.items():
                if key not in observed:
                    raise ValueError(f"{context} provenance is missing {path}.{key}")
                compare(observed[key], value, f"{path}.{key}")
        elif observed != locked:
            raise ValueError(f"{context} provenance mismatch for {path}: {observed!r} != {locked!r}")

    compare(actual, expected, "provenance")
