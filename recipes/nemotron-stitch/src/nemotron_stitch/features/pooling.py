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

"""Unit-axis mean pooling with an explicit optional soft-token cap.

The pooling *width* (units per soft token) and the *cap* are encoder-policy
inputs supplied by the caller — the package owns the shape of the
computation, the consumer owns the values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Lazy at runtime (the function imports torch inside); annotations only.
    import torch


def pool_positions(
    hidden: torch.Tensor,
    *,
    pooling_width: int,
    max_soft_tokens: int | None = None,
) -> torch.Tensor:
    import torch

    if hidden.ndim != 2:
        raise ValueError("position hidden states must be [units, hidden]")
    n = int(hidden.shape[0])
    if n == 0:
        return hidden
    width = int(pooling_width)
    if width <= 0:
        raise ValueError("pooling_width must be positive")
    if max_soft_tokens is not None and int(max_soft_tokens) <= 0:
        raise ValueError("max_soft_tokens must be positive or null")
    fixed = [hidden[start : min(n, start + width)].to(torch.float32).mean(dim=0) for start in range(0, n, width)]
    pooled = torch.stack(fixed)
    if max_soft_tokens is None or pooled.shape[0] <= int(max_soft_tokens):
        return pooled
    cap = int(max_soft_tokens)
    boundaries = torch.linspace(0, pooled.shape[0], cap + 1, device=pooled.device).floor().to(torch.long)
    return torch.stack([pooled[boundaries[i] : boundaries[i + 1]].mean(dim=0) for i in range(cap)])
