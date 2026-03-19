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

"""Optional per-layer-per-iteration precision schedule loaded from a JSONL file.

This module is for experimentation: it allows an external agent to dynamically
change layer precision during training by appending to a JSONL schedule file.
When no schedule file is provided, the existing static precision config is used
unchanged.
"""

import json
import logging


logger = logging.getLogger(__name__)

VALID_PRECISIONS = {"fp8", "fp4", None}


class PrecisionSchedule:
    """Reads a JSONL precision schedule and returns per-layer precision for a given training step.

    The JSONL file format is one JSON object per line::

        {"step": 0, "layer_precision": ["fp8", "fp8", "fp4", "fp4", null, null]}
        {"step": 100, "layer_precision": ["fp8", "fp8", "fp8", "fp8", "fp8", "fp8"]}

    An external process can append new lines while training is running;
    ``get_precision_for_step`` picks them up automatically via file seek.
    """

    def __init__(self, schedule_file: str, num_layers: int) -> None:
        """Load initial schedule entries and validate.

        Args:
            schedule_file: Path to JSONL precision schedule file.
            num_layers: Expected number of transformer layers (length of each ``layer_precision`` list).

        Raises:
            FileNotFoundError: If the schedule file does not exist.
            ValueError: If step 0 is missing or entries are invalid.
        """
        self._schedule_file = schedule_file
        self._num_layers = num_layers
        self._entries: list[dict] = []
        self._file = open(schedule_file, "r")
        self._read_new_entries()

        if not self._entries or self._entries[0]["step"] != 0:
            raise ValueError(
                f"Precision schedule must contain a step 0 entry. Found steps: {[e['step'] for e in self._entries]}"
            )

    def _read_new_entries(self) -> None:
        """Read any new lines appended since the last read position."""
        for raw_line in self._file:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            entry = json.loads(stripped)
            self._validate_entry(entry)
            self._entries.append(entry)

    def _validate_entry(self, entry: dict) -> None:
        """Validate a single schedule entry.

        Args:
            entry: Parsed JSON object with ``step`` and ``layer_precision`` keys.

        Raises:
            ValueError: If the entry is malformed.
        """
        if "step" not in entry or "layer_precision" not in entry:
            raise ValueError(f"Schedule entry must have 'step' and 'layer_precision' keys, got: {entry}")

        step = entry["step"]
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"'step' must be a non-negative integer, got: {step!r}")

        lp = entry["layer_precision"]
        if not isinstance(lp, list) or len(lp) != self._num_layers:
            raise ValueError(
                f"'layer_precision' must be a list of length {self._num_layers}, "
                f"got length {len(lp) if isinstance(lp, list) else type(lp).__name__}"
            )

        for i, p in enumerate(lp):
            if p not in VALID_PRECISIONS:
                raise ValueError(f"layer_precision[{i}] must be 'fp8', 'fp4', or null/None, got: {p!r}")

    def get_precision_for_step(self, step: int) -> list[str | None]:
        """Return the layer precision list active at the given training step.

        Picks up any newly appended lines, then returns the ``layer_precision``
        from the entry with the largest ``step <= current_step``.

        Args:
            step: Current training step.

        Returns:
            List of per-layer precision strings (``"fp8"``, ``"fp4"``, or ``None``).
        """
        self._read_new_entries()

        best = None
        for entry in self._entries:
            if entry["step"] <= step:
                if best is None or entry["step"] > best["step"]:
                    best = entry

        if best is None:
            raise ValueError(f"No schedule entry found for step {step}")

        return best["layer_precision"]

    def get_active_precisions(self) -> set[str]:
        """Return the set of precision types referenced across all schedule entries.

        Useful for determining which quantization recipes to instantiate upfront.

        Returns:
            Set of precision strings, e.g. ``{"fp8", "fp4"}``. ``None`` values are excluded.
        """
        self._read_new_entries()
        precisions: set[str] = set()
        for entry in self._entries:
            for p in entry["layer_precision"]:
                if p is not None:
                    precisions.add(p)
        return precisions
