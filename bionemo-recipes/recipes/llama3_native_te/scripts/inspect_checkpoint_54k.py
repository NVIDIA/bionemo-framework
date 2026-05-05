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

"""Inspect the step_54000 checkpoint to debug the loss drop after resume.

Checks:
1. Does dataloader state exist for all ranks?
2. What does the dataloader state contain? (position, worker info)
3. Compare dataloader state across checkpoints (52500 vs 54000)
4. Check model/optimizer state file sizes for anomalies

Run on prenyx (no GPU needed):
    python scripts/inspect_checkpoint_54k.py /lustre/fsw/healthcareeng_bionemo/savithas/checkpoints/lingua_7b_mxfp8_fl1_qinit_8n_prenyx/train_fsdp2
"""

import argparse
import sys
from pathlib import Path

import torch


def inspect_dataloader_state(ckpt_step_dir: Path, label: str) -> None:
    """Inspect dataloader state files in a checkpoint directory."""
    print(f"\n{'=' * 60}")
    print(f"Dataloader state for: {label} ({ckpt_step_dir.name})")
    print(f"{'=' * 60}")

    dl_files = sorted(ckpt_step_dir.glob("dataloader_rank_*.pt"))
    if not dl_files:
        print("  NO DATALOADER STATE FILES FOUND!")
        return

    print(f"  Found {len(dl_files)} dataloader state files")

    for f in dl_files:
        state = torch.load(f, weights_only=True, map_location="cpu")
        rank = f.stem.split("_")[-1]
        print(f"\n  Rank {rank} ({f.name}, {f.stat().st_size / 1024:.1f} KB):")
        print(f"    Keys: {list(state.keys())}")
        if "num_workers" in state:
            print(f"    num_workers: {state['num_workers']}")
        if "num_ranks" in state:
            print(f"    num_ranks: {state['num_ranks']}")

        # Inspect the actual dataloader state (torchdata StatefulDataLoader format)
        for key, val in state.items():
            if key in ("num_workers", "num_ranks"):
                continue
            if isinstance(val, dict):
                print(f"    {key}: dict with keys {list(val.keys())}")
                for k2, v2 in val.items():
                    if isinstance(v2, (int, float, str, bool)):
                        print(f"      {k2}: {v2}")
                    elif isinstance(v2, dict):
                        print(f"      {k2}: dict with {len(v2)} keys: {list(v2.keys())[:5]}...")
                    elif isinstance(v2, (list, tuple)):
                        print(f"      {k2}: {type(v2).__name__} of length {len(v2)}")
                    elif isinstance(v2, torch.Tensor):
                        print(f"      {k2}: tensor shape={v2.shape}, dtype={v2.dtype}")
                    else:
                        print(f"      {k2}: {type(v2).__name__}")
            elif isinstance(val, (int, float, str, bool)):
                print(f"    {key}: {val}")
            else:
                print(f"    {key}: {type(val).__name__}")

        # Only print rank 0 in detail, summarize others
        if rank != "0":
            continue


def compare_dataloader_states(dir_a: Path, dir_b: Path) -> None:
    """Compare dataloader states between two checkpoints."""
    print(f"\n{'=' * 60}")
    print(f"Comparing dataloader states: {dir_a.name} vs {dir_b.name}")
    print(f"{'=' * 60}")

    for rank in range(64):  # Check up to 64 ranks
        fa = dir_a / f"dataloader_rank_{rank}.pt"
        fb = dir_b / f"dataloader_rank_{rank}.pt"
        if not fa.exists() or not fb.exists():
            if rank == 0:
                print(f"  Rank {rank}: missing in {'A' if not fa.exists() else 'B'}")
            break

        sa = torch.load(fa, weights_only=True, map_location="cpu")
        sb = torch.load(fb, weights_only=True, map_location="cpu")

        if rank == 0:
            print("  Rank 0 comparison:")
            all_keys = set(list(sa.keys()) + list(sb.keys()))
            for key in sorted(all_keys):
                va = sa.get(key, "MISSING")
                vb = sb.get(key, "MISSING")
                if isinstance(va, dict) and isinstance(vb, dict):
                    if va == vb:
                        print(f"    {key}: IDENTICAL")
                    else:
                        print(f"    {key}: DIFFERENT")
                        # Show what differs
                        for k in set(list(va.keys()) + list(vb.keys())):
                            a_val = va.get(k, "MISSING")
                            b_val = vb.get(k, "MISSING")
                            if a_val != b_val:
                                print(f"      {k}: {a_val} -> {b_val}")
                elif va == vb:
                    print(f"    {key}: {va} (same)")
                else:
                    print(f"    {key}: {va} -> {vb}")


def inspect_checkpoint_files(ckpt_step_dir: Path, label: str) -> None:
    """List all files in checkpoint with sizes."""
    print(f"\n{'=' * 60}")
    print(f"Checkpoint files for: {label} ({ckpt_step_dir.name})")
    print(f"{'=' * 60}")

    files = sorted(ckpt_step_dir.rglob("*"))
    total_size = 0
    for f in files:
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            rel = f.relative_to(ckpt_step_dir)
            print(f"  {rel}: {size / (1024 * 1024):.1f} MB")
    print(f"  TOTAL: {total_size / (1024 * 1024 * 1024):.2f} GB")


def main():
    """Inspect checkpoint for loss drop debugging."""
    parser = argparse.ArgumentParser(description="Inspect checkpoint for loss drop debugging")
    parser.add_argument("ckpt_base", help="Base checkpoint dir (contains step_XXXXX dirs)")
    parser.add_argument("--step", type=int, default=54000, help="Primary step to inspect")
    parser.add_argument("--compare-step", type=int, default=52500, help="Step to compare against")
    args = parser.parse_args()

    base = Path(args.ckpt_base)
    primary = base / f"step_{args.step}"
    compare = base / f"step_{args.compare_step}"

    # List available checkpoints
    print(f"Available checkpoints in {base}:")
    steps = sorted([d for d in base.iterdir() if d.name.startswith("step_")], key=lambda x: int(x.name.split("_")[1]))
    for s in steps:
        print(f"  {s.name}")

    if not primary.exists():
        print(f"\nERROR: {primary} does not exist!")
        sys.exit(1)

    # Inspect primary checkpoint
    inspect_checkpoint_files(primary, f"step_{args.step}")
    inspect_dataloader_state(primary, f"step_{args.step}")

    # Compare if available
    if compare.exists():
        inspect_dataloader_state(compare, f"step_{args.compare_step}")
        compare_dataloader_states(compare, primary)
    else:
        print(f"\nCompare checkpoint {compare} not found, skipping comparison")
        # Try to find the previous checkpoint
        prev_steps = [s for s in steps if int(s.name.split("_")[1]) < args.step]
        if prev_steps:
            prev = prev_steps[-1]
            print(f"Using previous checkpoint instead: {prev.name}")
            inspect_dataloader_state(prev, prev.name)
            compare_dataloader_states(prev, primary)


if __name__ == "__main__":
    main()
