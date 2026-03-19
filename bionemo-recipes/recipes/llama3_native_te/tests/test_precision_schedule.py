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

"""Unit tests for PrecisionSchedule."""

import json

import pytest

from precision_schedule import PrecisionSchedule


NUM_LAYERS = 6


def _write_entries(path, entries):
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in entries)


class TestPrecisionScheduleLoad:
    """Tests for loading and validating schedule files."""

    def test_valid_schedule(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8"] * NUM_LAYERS},
                {"step": 100, "layer_precision": ["fp4"] * NUM_LAYERS},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        assert schedule.get_precision_for_step(0) == ["fp8"] * NUM_LAYERS

    def test_missing_step_0(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 10, "layer_precision": ["fp8"] * NUM_LAYERS},
            ],
        )
        with pytest.raises(ValueError, match="step 0"):
            PrecisionSchedule(str(path), NUM_LAYERS)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="step 0"):
            PrecisionSchedule(str(path), NUM_LAYERS)

    def test_wrong_layer_count(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8"] * 3},  # wrong length
            ],
        )
        with pytest.raises(ValueError, match="length"):
            PrecisionSchedule(str(path), NUM_LAYERS)

    def test_invalid_precision_value(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8", "fp4", "fp16", None, None, None]},
            ],
        )
        with pytest.raises(ValueError, match="fp16"):
            PrecisionSchedule(str(path), NUM_LAYERS)

    def test_missing_keys(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"step": 0}) + "\n")
        with pytest.raises(ValueError, match="layer_precision"):
            PrecisionSchedule(str(path), NUM_LAYERS)

    def test_negative_step(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": -1, "layer_precision": ["fp8"] * NUM_LAYERS},
            ],
        )
        with pytest.raises(ValueError, match="non-negative"):
            PrecisionSchedule(str(path), NUM_LAYERS)


class TestGetPrecisionForStep:
    """Tests for step-based precision lookup."""

    def test_exact_step_match(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8"] * NUM_LAYERS},
                {"step": 100, "layer_precision": ["fp4"] * NUM_LAYERS},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        assert schedule.get_precision_for_step(100) == ["fp4"] * NUM_LAYERS

    def test_between_steps(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8"] * NUM_LAYERS},
                {"step": 100, "layer_precision": ["fp4"] * NUM_LAYERS},
                {"step": 200, "layer_precision": [None] * NUM_LAYERS},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        # Step 50 should use step 0 entry
        assert schedule.get_precision_for_step(50) == ["fp8"] * NUM_LAYERS
        # Step 150 should use step 100 entry
        assert schedule.get_precision_for_step(150) == ["fp4"] * NUM_LAYERS
        # Step 999 should use step 200 entry
        assert schedule.get_precision_for_step(999) == [None] * NUM_LAYERS

    def test_mixed_precision(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        mixed = ["fp8", "fp8", "fp8", "fp4", "fp4", None]
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": mixed},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        assert schedule.get_precision_for_step(0) == mixed


class TestStreamingAppend:
    """Tests for picking up new entries appended to the file mid-training."""

    def test_append_new_entries(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8"] * NUM_LAYERS},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        assert schedule.get_precision_for_step(50) == ["fp8"] * NUM_LAYERS

        # Append a new entry
        with open(path, "a") as f:
            f.write(json.dumps({"step": 50, "layer_precision": ["fp4"] * NUM_LAYERS}) + "\n")

        # Next call should pick it up
        assert schedule.get_precision_for_step(50) == ["fp4"] * NUM_LAYERS

    def test_multiple_appends(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8"] * NUM_LAYERS},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)

        # Append two entries sequentially
        with open(path, "a") as f:
            f.write(json.dumps({"step": 100, "layer_precision": ["fp4"] * NUM_LAYERS}) + "\n")

        assert schedule.get_precision_for_step(100) == ["fp4"] * NUM_LAYERS

        with open(path, "a") as f:
            f.write(json.dumps({"step": 200, "layer_precision": [None] * NUM_LAYERS}) + "\n")

        assert schedule.get_precision_for_step(200) == [None] * NUM_LAYERS
        # Earlier steps still work
        assert schedule.get_precision_for_step(150) == ["fp4"] * NUM_LAYERS


class TestCommentAndBlankHandling:
    """Tests for comment lines and blank lines."""

    def test_comments_and_blanks(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        with open(path, "w") as f:
            f.write("# This is a header comment\n")
            f.write("\n")
            f.write(json.dumps({"step": 0, "layer_precision": ["fp8"] * NUM_LAYERS}) + "\n")
            f.write("# Switch to fp4 at step 100\n")
            f.write("\n")
            f.write(json.dumps({"step": 100, "layer_precision": ["fp4"] * NUM_LAYERS}) + "\n")

        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        assert schedule.get_precision_for_step(0) == ["fp8"] * NUM_LAYERS
        assert schedule.get_precision_for_step(100) == ["fp4"] * NUM_LAYERS


class TestGetActivePrecisions:
    """Tests for get_active_precisions."""

    def test_single_precision(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8"] * NUM_LAYERS},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        assert schedule.get_active_precisions() == {"fp8"}

    def test_mixed_precisions(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": ["fp8", "fp8", "fp4", "fp4", None, None]},
                {"step": 100, "layer_precision": ["fp8"] * NUM_LAYERS},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        assert schedule.get_active_precisions() == {"fp8", "fp4"}

    def test_all_none(self, tmp_path):
        path = tmp_path / "schedule.jsonl"
        _write_entries(
            path,
            [
                {"step": 0, "layer_precision": [None] * NUM_LAYERS},
            ],
        )
        schedule = PrecisionSchedule(str(path), NUM_LAYERS)
        assert schedule.get_active_precisions() == set()
