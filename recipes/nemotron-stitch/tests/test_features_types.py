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

"""FeatureRef / EncoderOutput contracts and the Unit protocol."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from nemotron_stitch.features.types import EncoderOutput, FeatureRef, Unit


class TripletUnit:
    """A toy unit: one unit per three source elements."""

    @property
    def units_per_element(self) -> Fraction:
        return Fraction(1, 3)


def _ref(**overrides) -> FeatureRef:
    data = {
        "source_sample_id": "sample::1",
        "source_manifest_id": "0" * 64,
        "source_record_hash": "b" * 64,
        "source_kind": "code",
    }
    data.update(overrides)
    return FeatureRef(**data)


def test_unit_protocol_is_structural():
    assert isinstance(TripletUnit(), Unit)
    assert TripletUnit().units_per_element * 300 == 100  # exact, not float
    assert not isinstance(object(), Unit)


def test_ref_missing_reason_is_the_only_absence_signal():
    assert not _ref().is_missing
    assert _ref(missing_reason="unresolved").is_missing


def test_encoder_output_validate_accepts_exact_geometry():
    output = EncoderOutput(
        ref=_ref(),
        hidden_states=np.ones((6, 4), dtype=np.float32),
        unit=TripletUnit(),
        layer=12,
        coverage=(1.0,) * 6,
    )
    output.validate(expected_units=6, expected_hidden_size=4)


def test_encoder_output_validate_fails_closed():
    output = EncoderOutput(
        ref=_ref(),
        hidden_states=np.ones((6, 4), dtype=np.float32),
        unit=TripletUnit(),
        layer=12,
        coverage=(1.0,) * 6,
    )
    with pytest.raises(ValueError, match="shape"):
        output.validate(expected_units=5, expected_hidden_size=4)
    short_coverage = EncoderOutput(
        ref=_ref(),
        hidden_states=np.ones((6, 4), dtype=np.float32),
        unit=TripletUnit(),
        layer=12,
        coverage=(1.0,) * 5,
    )
    with pytest.raises(ValueError, match="positive coverage"):
        short_coverage.validate(expected_units=6, expected_hidden_size=4)
    zero_coverage = EncoderOutput(
        ref=_ref(),
        hidden_states=np.ones((6, 4), dtype=np.float32),
        unit=TripletUnit(),
        layer=12,
        coverage=(1.0,) * 5 + (0.0,),
    )
    with pytest.raises(ValueError, match="positive coverage"):
        zero_coverage.validate(expected_units=6, expected_hidden_size=4)
