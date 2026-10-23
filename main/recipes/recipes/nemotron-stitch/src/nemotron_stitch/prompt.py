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

"""The soft-token prompt renderer: ``start + placeholder * count + end`` runs in a template.

One renderer for every surface that turns soft-token prompts into text — the
training collator and the RL processor must render byte-identically, or the
warm start silently degrades into a mediocre reward curve. Stdlib-only and
importable from the base package (no framework import, no modality name).

Multi-projector by construction: embedding several modalities at once is
the point of the package, and a single-projector renderer is exactly the API
that would need replacing the first time two encoders share one prompt. The
caller supplies a template carrying one marker per projector that received
features — ``{mm:<name>}`` — so interleaved prompts render in text order while
scatter routing stays by token identity. A single projector is
the degenerate case.

The renderer emits text, not token ids: placeholder-id checks stay in the RL
processor and in ``MultimodalProjector``, and ``derive_indices`` re-checks the
run-length invariant at the scatter boundary.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_MARKER = re.compile(r"\{mm:([A-Za-z0-9_.\-]+)\}")


@dataclass(frozen=True)
class PlaceholderSpec:
    """One projector's sentinel run: start token, placeholder token, end token.

    ``name`` keys the spec the same way ``mm_placeholder_token_ids`` keys its
    mapping — the package validates those keys against the registered
    projectors at model construction.
    """

    name: str
    start_token: str
    placeholder_token: str
    end_token: str

    def marker(self) -> str:
        return "{mm:" + self.name + "}"

    def render_run(self, token_count: int) -> str:
        if int(token_count) <= 0:
            raise ValueError(f"projector {self.name!r} must emit a positive soft-token count")
        return self.start_token + self.placeholder_token * int(token_count) + self.end_token


def render_soft_token_prompt(
    template: str,
    specs: Iterable[PlaceholderSpec],
    token_counts: Mapping[str, int],
) -> str:
    """Splice each projector's sentinel run into its template marker.

    Fail-closed at render time: exactly one marker per projector that received
    features, no marker for a projector that did not, distinct placeholder
    tokens across projectors, and a positive run length equal to the
    soft-token count the collator will emit.
    """
    specs = tuple(specs)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate projector names in placeholder specs: {sorted(names)}")
    placeholders = [spec.placeholder_token for spec in specs]
    if len(set(placeholders)) != len(placeholders):
        raise ValueError("placeholder tokens must be distinct across projectors")
    unknown_counts = sorted(set(token_counts) - set(names))
    if unknown_counts:
        raise KeyError(f"soft-token counts name unknown projectors: {unknown_counts}")
    spec_by_name = {spec.name: spec for spec in specs}

    rendered = template
    for name in sorted(token_counts):
        marker = spec_by_name[name].marker()
        occurrences = rendered.count(marker)
        if occurrences != 1:
            raise ValueError(
                f"template must contain exactly one marker {marker!r} for projector {name!r}, found {occurrences}"
            )
        rendered = rendered.replace(marker, spec_by_name[name].render_run(int(token_counts[name])))
    leftover = _MARKER.search(rendered)
    if leftover is not None:
        raise ValueError(f"template carries a marker for projector {leftover.group(1)!r}, which received no features")
    return rendered
