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

"""FeatureRef, EncoderOutput, and the Unit protocol.

An encoder consumes a sequence of consumer-defined *elements* (the consumer
owns the alphabet) addressed in *units*: the encoder-facing granularity that
chunking, coverage, and pooling count in. ``Unit`` is the protocol a consumer
enum implements to expose the element→unit conversion. ``FeatureRef`` is the
content-addressed identity of one source record; ``EncoderOutput`` is the
``[units, hidden]`` result with per-unit coverage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Lazy at runtime (this module is on the framework-free import path);
    # annotations only.
    import numpy as np
    import torch


@runtime_checkable
class Unit(Protocol):
    """Element→unit conversion for a source sequence."""

    @property
    def units_per_element(self) -> Fraction:
        """Units per source element, exact (e.g. a triplet unit is 1/3)."""
        ...


@dataclass(frozen=True)
class FeatureRef:
    source_sample_id: str
    source_manifest_id: str
    source_record_hash: str
    source_kind: str
    missing_reason: str | None = None

    @property
    def is_missing(self) -> bool:
        return self.missing_reason is not None


@dataclass(frozen=True)
class EncoderOutput:
    ref: FeatureRef
    #: The encoder's output matrix, ``[units, hidden_size]`` — torch on the
    #: chunk/pool path, numpy at the cache boundary.
    hidden_states: np.ndarray | torch.Tensor
    unit: Unit
    layer: int
    coverage: tuple[float, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, expected_units: int, expected_hidden_size: int) -> None:
        shape = tuple(int(x) for x in self.hidden_states.shape)
        if shape != (expected_units, expected_hidden_size):
            raise ValueError(f"encoder output shape {shape} != {(expected_units, expected_hidden_size)}")
        if len(self.coverage) != expected_units or any(weight <= 0 for weight in self.coverage):
            raise ValueError("encoder output must have positive coverage for every unit")
