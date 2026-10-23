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

"""One resolver for the language-model width of a host config.

The training path (``automodel/model.py``) and the serving path
(``vllm/plugin.py``) must agree on the LM width of the same model: a
disagreement is a wrong-width projector that the vLLM hidden-size check
rejects only at rollout, after alignment and SFT have already run. Host
configs carry the width in one of three places — plain ``hidden_size``,
``text_config.hidden_size``, or ``llm_config.hidden_size`` — so both
integrations resolve it here and nowhere else. Stdlib-only; imported lazily
by the framework integrations.
"""

from __future__ import annotations

from typing import Any

_COMPOSITE_ATTRS = ("text_config", "llm_config")


def resolve_lm_hidden_size(config: Any) -> int:
    """Resolve the LM width from plain or composite host configs, failing closed.

    Accepts exactly one value: every present spelling must agree. A missing
    width and conflicting widths both raise — either would otherwise surface
    as a scatter or serving failure long after training began.
    """
    candidates: list[tuple[str, int]] = []
    plain = getattr(config, "hidden_size", None)
    if plain is not None:
        candidates.append(("hidden_size", int(plain)))
    for attr in _COMPOSITE_ATTRS:
        sub = getattr(config, attr, None)
        value = getattr(sub, "hidden_size", None) if sub is not None else None
        if value is not None:
            candidates.append((f"{attr}.hidden_size", int(value)))
    if not candidates:
        raise ValueError(
            "host config does not expose an LM hidden size "
            f"(looked for hidden_size, {', '.join(f'{a}.hidden_size' for a in _COMPOSITE_ATTRS)})"
        )
    values = {value for _, value in candidates}
    if len(values) != 1:
        detail = ", ".join(f"{name}={value}" for name, value in candidates)
        raise ValueError(f"host config exposes conflicting LM hidden sizes: {detail}")
    return values.pop()
