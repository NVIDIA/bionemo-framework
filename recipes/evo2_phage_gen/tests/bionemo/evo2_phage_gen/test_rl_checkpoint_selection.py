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

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from bionemo.evo2_phage_gen import rl_checkpoint_selection


def _event(step: int, aggregate: float, strict: float) -> dict[str, float | int]:
    return {"step": step, "aggregate_reward": aggregate, "all_objectives_max_score_rate": strict}


def _write_checkpoint(root: Path, step: int, payload: bytes = b"weights") -> Path:
    checkpoint = root / f"step_{step}"
    iteration = checkpoint / "policy" / "weights" / "iter_0000000"
    iteration.mkdir(parents=True)
    (checkpoint / "training_info.json").write_text("{}\n")
    (iteration / "metadata.json").write_text("{}\n")
    (iteration / "common.pt").write_bytes(payload)
    return checkpoint


def test_selection_prefers_a_strict_positive_checkpoint() -> None:
    events = [
        _event(10, 9.0, 0.0),
        _event(20, 8.0, 0.25),
        _event(30, 8.5, 0.25),
        _event(40, 9.5, 0.0),
    ]

    selected = rl_checkpoint_selection.select_checkpoint_event(events)

    assert selected == {
        "step": 30,
        "selection_metric": "all_objectives_max_score_rate",
        "has_max_score_sequences": True,
        "value": 0.25,
        "aggregate_reward": 8.5,
    }


def test_selection_falls_back_to_best_interior_aggregate() -> None:
    events = [
        _event(10, 9.5, 0.0),
        _event(20, 8.0, 0.0),
        _event(30, 9.0, 0.0),
        _event(40, 9.6, 0.0),
    ]

    selected = rl_checkpoint_selection.select_checkpoint_event(events)

    assert selected == {
        "step": 30,
        "selection_metric": "aggregate_reward",
        "has_max_score_sequences": False,
        "value": 9.0,
        "aggregate_reward": 9.0,
    }


def test_extract_checkpoint_events_uses_stable_task_namespace(monkeypatch, tmp_path: Path) -> None:
    points = {
        "validation/phage_qc/mean_reward": {10: (1.0, 8.25)},
        "validation/phage_qc/num_sequences": {10: (1.0, 96.0)},
        "validation/phage_qc/all_objectives_max_score_rate": {10: (1.0, 0.125)},
    }
    monkeypatch.setattr(rl_checkpoint_selection, "_load_scalar_points", lambda _root: points)

    assert rl_checkpoint_selection.extract_checkpoint_events(tmp_path) == [
        {"step": 10, "aggregate_reward": 8.25, "all_objectives_max_score_rate": 0.125}
    ]


def test_sync_protects_best_aggregate_and_strict_candidates_with_hardlinks(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    protected_root = tmp_path / "protected"
    first = _write_checkpoint(checkpoint_root, 10, b"first")
    second = _write_checkpoint(checkpoint_root, 20, b"second")
    third = _write_checkpoint(checkpoint_root, 30, b"third")

    report = rl_checkpoint_selection.sync_protected_candidates(
        checkpoint_root,
        protected_root,
        [_event(10, 9.0, 0.0), _event(20, 8.0, 0.5), _event(30, 9.5, 0.25)],
    )

    assert report["aggregate_reward"]["step"] == 30
    assert report["all_objectives_max_score_rate"]["step"] == 20
    aggregate_file = protected_root / "aggregate-reward" / "step_30" / "policy/weights/iter_0000000/common.pt"
    strict_file = (
        protected_root / "all-objectives-max-score-rate" / "step_20" / "policy/weights/iter_0000000/common.pt"
    )
    assert aggregate_file.stat().st_ino == (third / "policy/weights/iter_0000000/common.pt").stat().st_ino
    assert strict_file.stat().st_ino == (second / "policy/weights/iter_0000000/common.pt").stat().st_ino

    _write_checkpoint(checkpoint_root, 40, b"fourth")
    rl_checkpoint_selection.sync_protected_candidates(
        checkpoint_root,
        protected_root,
        [_event(10, 9.0, 0.0), _event(20, 8.0, 0.5), _event(30, 9.5, 0.25), _event(40, 9.6, 0.0)],
    )

    assert not (protected_root / "aggregate-reward" / "step_30").exists()
    assert (protected_root / "aggregate-reward" / "step_40").is_dir()
    assert (protected_root / "all-objectives-max-score-rate" / "step_20").is_dir()
    assert (first / "training_info.json").is_file()


def test_resolve_selected_checkpoint_uses_protected_copy_after_managed_source_is_pruned(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    protected_root = tmp_path / "protected"
    _write_checkpoint(checkpoint_root, 10)
    selected_source = _write_checkpoint(checkpoint_root, 20)
    _write_checkpoint(checkpoint_root, 30)
    events = [_event(10, 8.0, 0.0), _event(20, 9.0, 0.5), _event(30, 8.5, 0.0)]
    rl_checkpoint_selection.sync_protected_candidates(checkpoint_root, protected_root, events)
    shutil.rmtree(selected_source)

    result = rl_checkpoint_selection.resolve_selected_checkpoint(events, checkpoint_root, protected_root)

    assert result["step"] == 20
    assert result["has_max_score_sequences"] is True
    assert (
        Path(result["checkpoint"])
        == (protected_root / "all-objectives-max-score-rate" / "step_20" / "policy/weights/iter_0000000").resolve()
    )
    assert os.path.samefile(
        Path(result["checkpoint"]) / "common.pt",
        protected_root / "aggregate-reward" / "step_20" / "policy/weights/iter_0000000/common.pt",
    )


def test_supervisor_polls_and_performs_a_final_sync(monkeypatch, tmp_path: Path) -> None:
    calls: list[Path] = []

    def record_sync(tensorboard_root: Path, _checkpoint_root: Path, _protected_root: Path) -> None:
        calls.append(tensorboard_root)

    monkeypatch.setattr(rl_checkpoint_selection, "_sync_from_tensorboard", record_sync)
    status = rl_checkpoint_selection.supervise(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        tensorboard_root=tmp_path / "logs",
        checkpoint_root=tmp_path / "checkpoints",
        protected_root=tmp_path / "protected",
        poll_seconds=0.01,
    )

    assert status == 0
    assert len(calls) >= 2
    assert calls[-1] == tmp_path / "logs"


def test_sync_waits_for_an_atomically_published_checkpoint(monkeypatch, tmp_path: Path) -> None:
    events = [_event(10, 8.25, 0.0)]
    monkeypatch.setattr(rl_checkpoint_selection, "extract_checkpoint_events", lambda _root: events)

    with pytest.raises(ValueError, match="aggregate_reward"):
        rl_checkpoint_selection._sync_from_tensorboard(
            tmp_path / "logs",
            tmp_path / "checkpoints",
            tmp_path / "protected",
        )
