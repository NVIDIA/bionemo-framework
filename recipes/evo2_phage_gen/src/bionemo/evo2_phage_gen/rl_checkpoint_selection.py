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

"""Protect and select complementary scientific checkpoints during phage GDPO."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bionemo.evo2_phage_gen.objective_monitor import (
    _load_scalar_points,
    _phage_scalar,
    _scalar,
    _validation_prefix,
)


AGGREGATE_METRIC = "mean_reward"
STRICT_ENDPOINT_METRIC = "binary_safety_qualified_full_qc_cluster_deduplicated_rate"


def _finite_float(value: Any, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _normalized_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, float | int]]:
    normalized: dict[int, dict[str, float | int]] = {}
    for event in events:
        step = int(event["step"])
        if step < 0:
            raise ValueError(f"checkpoint step must be non-negative, got {step}")
        normalized[step] = {
            "step": step,
            "aggregate_reward": _finite_float(event["aggregate_reward"], name="aggregate_reward"),
            "strict_endpoint": _finite_float(event["strict_endpoint"], name="strict_endpoint"),
        }
    return [normalized[step] for step in sorted(normalized)]


def extract_checkpoint_events(tensorboard_root: Path) -> list[dict[str, float | int]]:
    """Read comparable aggregate and strict-endpoint validation events."""
    points = _load_scalar_points(tensorboard_root)
    validation_prefix = _validation_prefix(points)
    reward_tag = f"{validation_prefix}/{AGGREGATE_METRIC}"
    events: list[dict[str, float | int]] = []
    for step in sorted(points.get(reward_tag, {})):
        if _scalar(points, f"{validation_prefix}/num_sequences", step) is None:
            continue
        aggregate = _scalar(points, reward_tag, step)
        strict = _phage_scalar(points, validation_prefix, STRICT_ENDPOINT_METRIC, step)
        if aggregate is None or strict is None:
            raise ValueError(
                f"validation step {step} does not contain both {AGGREGATE_METRIC!r} "
                f"and {STRICT_ENDPOINT_METRIC!r} under {validation_prefix!r}"
            )
        events.append({"step": step, "aggregate_reward": aggregate, "strict_endpoint": strict})
    return _normalized_events(events)


def select_checkpoint_event(events: Sequence[Mapping[str, Any]]) -> dict[str, float | int | bool | str]:
    """Prefer a strict-positive checkpoint, otherwise the best interior aggregate."""
    ordered = _normalized_events(events)
    if len(ordered) < 3:
        raise ValueError("need at least three comparable validation events")

    strict_candidates = [event for event in ordered if float(event["strict_endpoint"]) > 0.0]
    if strict_candidates:
        selected = max(
            strict_candidates,
            key=lambda event: (
                float(event["strict_endpoint"]),
                float(event["aggregate_reward"]),
                int(event["step"]),
            ),
        )
        return {
            "step": int(selected["step"]),
            "selection_metric": "strict_endpoint",
            "strict_endpoint_qualified": True,
            "value": float(selected["strict_endpoint"]),
            "aggregate_reward": float(selected["aggregate_reward"]),
        }

    selected = max(
        ordered[1:-1],
        key=lambda event: (float(event["aggregate_reward"]), int(event["step"])),
    )
    return {
        "step": int(selected["step"]),
        "selection_metric": "aggregate_reward",
        "strict_endpoint_qualified": False,
        "value": float(selected["aggregate_reward"]),
        "aggregate_reward": float(selected["aggregate_reward"]),
    }


def _checkpoint_weights(checkpoint: Path) -> Path:
    return checkpoint / "policy" / "weights" / "iter_0000000"


def _checkpoint_ready(checkpoint: Path) -> bool:
    # NeMo-RL publishes a complete checkpoint by atomically renaming tmp_step_N to step_N.
    return (checkpoint / "training_info.json").is_file() and _checkpoint_weights(checkpoint).is_dir()


def _protect_checkpoint(source: Path, destination: Path) -> None:
    if destination.exists():
        if not _checkpoint_ready(destination):
            raise ValueError(f"protected checkpoint is incomplete: {destination}")
        if _checkpoint_ready(source) and not os.path.samefile(
            source / "training_info.json", destination / "training_info.json"
        ):
            raise ValueError(f"protected checkpoint does not match the managed source: {destination}")
        return
    if not _checkpoint_ready(source):
        raise ValueError(f"managed checkpoint is not complete: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary, copy_function=os.link, symlinks=True)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _replace_protected_candidate(source: Path, kind_root: Path) -> Path:
    destination = kind_root / source.name
    _protect_checkpoint(source, destination)
    for previous in kind_root.glob("step_*"):
        if previous != destination:
            shutil.rmtree(previous)
    return destination


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> bool:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text() == serialized:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(serialized)
    temporary.replace(path)
    return True


def sync_protected_candidates(
    checkpoint_root: Path,
    protected_root: Path,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | str]]:
    """Hard-link the best aggregate and best strict-positive complete checkpoints."""
    ordered = _normalized_events(events)
    if not ordered:
        return {}

    candidates: list[tuple[str, dict[str, float | int]]] = [
        (
            "aggregate_reward",
            max(ordered, key=lambda event: (float(event["aggregate_reward"]), int(event["step"]))),
        )
    ]
    strict = [event for event in ordered if float(event["strict_endpoint"]) > 0.0]
    if strict:
        candidates.append(
            (
                "strict_endpoint",
                max(
                    strict,
                    key=lambda event: (
                        float(event["strict_endpoint"]),
                        float(event["aggregate_reward"]),
                        int(event["step"]),
                    ),
                ),
            )
        )

    report: dict[str, dict[str, float | int | str]] = {}
    for kind, event in candidates:
        step = int(event["step"])
        source = checkpoint_root / f"step_{step}"
        kind_root = protected_root / kind.replace("_", "-")
        destination = kind_root / source.name
        if _checkpoint_ready(source):
            destination = _replace_protected_candidate(source, kind_root)
        elif not _checkpoint_ready(destination):
            continue
        report[kind] = {
            "step": step,
            "aggregate_reward": float(event["aggregate_reward"]),
            "strict_endpoint": float(event["strict_endpoint"]),
            "checkpoint": str(destination.resolve()),
        }
    if report and len(report) == len(candidates):
        _write_json_atomic(protected_root / "candidates.json", report)
    return report


def resolve_selected_checkpoint(
    events: Sequence[Mapping[str, Any]],
    checkpoint_root: Path,
    protected_root: Path,
) -> dict[str, Any]:
    """Select a validation event and resolve its retained Megatron weights."""
    selected = select_checkpoint_event(events)
    step = int(selected["step"])
    preferred_kind = "strict-endpoint" if selected["strict_endpoint_qualified"] else "aggregate-reward"
    roots = (
        protected_root / preferred_kind / f"step_{step}",
        checkpoint_root / f"step_{step}",
        protected_root / "aggregate-reward" / f"step_{step}",
        protected_root / "strict-endpoint" / f"step_{step}",
    )
    checkpoint = next((candidate for candidate in roots if _checkpoint_ready(candidate)), None)
    if checkpoint is None:
        searched = ", ".join(str(candidate) for candidate in roots)
        raise ValueError(f"selected validation step {step} has no complete checkpoint; searched: {searched}")
    result = dict(selected)
    result["checkpoint"] = str(_checkpoint_weights(checkpoint).resolve())
    return result


def _sync_from_tensorboard(tensorboard_root: Path, checkpoint_root: Path, protected_root: Path) -> None:
    manifest = protected_root / "candidates.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
    events = extract_checkpoint_events(tensorboard_root)
    report = sync_protected_candidates(checkpoint_root, protected_root, events)
    expected = {"aggregate_reward"} if events else set()
    if any(float(event["strict_endpoint"]) > 0.0 for event in events):
        expected.add("strict_endpoint")
    missing = sorted(expected - report.keys())
    if missing:
        raise ValueError(f"complete checkpoint not yet available for protected candidate(s): {', '.join(missing)}")
    current = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if current and current != previous:
        print(f"protected RL checkpoint candidates: {json.dumps(current, sort_keys=True)}", flush=True)


def supervise(
    command: Sequence[str],
    *,
    tensorboard_root: Path,
    checkpoint_root: Path,
    protected_root: Path,
    poll_seconds: float,
) -> int:
    """Run GDPO while retaining complementary checkpoint candidates."""
    if not command:
        raise ValueError("missing supervised command")
    process = subprocess.Popen(list(command), start_new_session=True)

    def forward_signal(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    previous_handlers = {
        signum: signal.signal(signum, forward_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    sync_error: Exception | None = None
    reported_sync_error: str | None = None
    try:
        while True:
            try:
                return_code = process.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                try:
                    _sync_from_tensorboard(tensorboard_root, checkpoint_root, protected_root)
                    sync_error = None
                    reported_sync_error = None
                except (OSError, RuntimeError, ValueError) as error:
                    sync_error = error
                    if str(error) != reported_sync_error:
                        print(f"checkpoint protection pending: {error}", file=sys.stderr, flush=True)
                        reported_sync_error = str(error)
        try:
            _sync_from_tensorboard(tensorboard_root, checkpoint_root, protected_root)
            sync_error = None
        except (OSError, RuntimeError, ValueError) as error:
            sync_error = error
            print(f"final checkpoint protection failed: {error}", file=sys.stderr, flush=True)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if return_code == 0 and sync_error is not None:
        return 1
    return return_code


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    supervise_parser = subparsers.add_parser("supervise", help="run GDPO and retain both candidate classes")
    supervise_parser.add_argument("--tensorboard-root", type=Path, required=True)
    supervise_parser.add_argument("--checkpoint-root", type=Path, required=True)
    supervise_parser.add_argument("--protected-root", type=Path, required=True)
    supervise_parser.add_argument("--poll-seconds", type=_positive_float, default=30.0)
    supervise_parser.add_argument("command", nargs=argparse.REMAINDER)

    select_parser = subparsers.add_parser("select", help="select a retained checkpoint after GDPO")
    select_parser.add_argument("--tensorboard-root", type=Path, required=True)
    select_parser.add_argument("--checkpoint-root", type=Path, required=True)
    select_parser.add_argument("--protected-root", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run checkpoint supervision or final scientific selection."""
    args = _parser().parse_args()
    if args.action == "supervise":
        command = list(args.command)
        if command[:1] == ["--"]:
            command = command[1:]
        try:
            status = supervise(
                command,
                tensorboard_root=args.tensorboard_root,
                checkpoint_root=args.checkpoint_root,
                protected_root=args.protected_root,
                poll_seconds=args.poll_seconds,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from None
        raise SystemExit(status)

    try:
        events = extract_checkpoint_events(args.tensorboard_root)
        result = resolve_selected_checkpoint(events, args.checkpoint_root, args.protected_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from None
    _write_json_atomic(args.output, result)
    print(result["checkpoint"])


if __name__ == "__main__":
    main()
