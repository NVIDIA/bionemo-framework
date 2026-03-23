#!/usr/bin/env python3

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

"""Pull and compare WandB loss curves for FL2 vs FL4 runs.

Usage:
    python3 pull_wandb_comparison.py

Requires: pip install wandb pandas matplotlib
Requires: WANDB_API_KEY set or `wandb login` done.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FL2 and FL4 original run IDs
RUNS = {
    "FL2 (first/last 2 BF16)": {
        "run_id": "98xg86yz",
        "color": "#2196F3",
    },
    "FL4 (first/last 4 BF16)": {
        "run_id": "ww9z3vy4",
        "color": "#F44336",
    },
}

PROJECT = "clara-discovery/llama3-metagenome-7b"
METRICS = ["train_loss", "learning_rate", "train_loss_smoothed"]


def pull_run_history(api, project, run_id, metrics):
    """Pull history for a single run."""
    run = api.run(f"{project}/{run_id}")
    logger.info(f"  Run: {run.name} ({run_id}), state={run.state}, steps={run.lastHistoryStep}")

    # Pull metrics
    history = run.history(keys=metrics, pandas=True)
    config = run.config

    return history, config, run.name


def main():
    """Entry point."""
    import wandb

    api = wandb.Api()
    out_dir = Path("wandb_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PULLING WANDB DATA: FL2 vs FL4 COMPARISON")
    logger.info("=" * 70)

    all_histories = {}
    all_configs = {}

    for label, info in RUNS.items():
        logger.info(f"\nPulling {label}...")
        history, config, name = pull_run_history(api, PROJECT, info["run_id"], METRICS)
        all_histories[label] = history
        all_configs[label] = config
        logger.info(f"  Got {len(history)} data points")

        # Save raw data
        history.to_csv(out_dir / f"history_{info['run_id']}.csv", index=False)

    # Save configs
    with open(out_dir / "configs.json", "w") as f:
        json.dump(all_configs, f, indent=2, default=str)

    # --- Plot 1: Loss curves ---
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, info in RUNS.items():
        history = all_histories[label]
        loss_col = "train_loss" if "train_loss" in history.columns else history.columns[0]
        data = history.dropna(subset=[loss_col])
        ax.plot(data["_step"], data[loss_col], label=label, color=info["color"], alpha=0.7, linewidth=1)

        # Add smoothed if available
        if "train_loss_smoothed" in history.columns:
            smoothed = history.dropna(subset=["train_loss_smoothed"])
            ax.plot(
                smoothed["_step"],
                smoothed["train_loss_smoothed"],
                color=info["color"],
                linewidth=2.5,
                linestyle="-",
            )

    ax.set_xlabel("Training Step", fontsize=14, fontweight="bold")
    ax.set_ylabel("Training Loss", fontsize=14, fontweight="bold")
    ax.set_title("FL2 vs FL4: Training Loss Comparison\n(FP8 Block Scaling, E4M3)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_path = out_dir / "fl2_vs_fl4_loss_curves.png"
    plt.savefig(loss_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"\nSaved: {loss_path}")

    # --- Plot 2: Learning rate ---
    if any("learning_rate" in h.columns for h in all_histories.values()):
        fig, ax = plt.subplots(figsize=(14, 5))
        for label, info in RUNS.items():
            history = all_histories[label]
            if "learning_rate" in history.columns:
                data = history.dropna(subset=["learning_rate"])
                ax.plot(data["_step"], data["learning_rate"], label=label, color=info["color"], linewidth=1.5)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        lr_path = out_dir / "fl2_vs_fl4_learning_rate.png"
        plt.savefig(lr_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Saved: {lr_path}")

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for label, info in RUNS.items():
        history = all_histories[label]
        loss_col = "train_loss" if "train_loss" in history.columns else history.columns[0]
        data = history.dropna(subset=[loss_col])
        if len(data) > 0:
            logger.info(f"\n{label} ({info['run_id']}):")
            logger.info(f"  Steps: {int(data['_step'].min())} - {int(data['_step'].max())}")
            logger.info(f"  Final loss: {data[loss_col].iloc[-1]:.4f}")
            logger.info(
                f"  Min loss: {data[loss_col].min():.4f} (step {int(data.loc[data[loss_col].idxmin(), '_step'])})"
            )
            logger.info(f"  Mean loss (last 1000): {data[loss_col].tail(1000).mean():.4f}")

    # Config diff
    logger.info("\n--- Config differences ---")
    keys_to_compare = ["fp8_layers", "num_train_steps", "learning_rate", "micro_batch_size", "gradient_accumulation"]
    for key in keys_to_compare:
        vals = {label: all_configs[label].get(key, "N/A") for label in RUNS}
        is_diff = len({str(v) for v in vals.values()}) > 1
        marker = " *** DIFFERENT ***" if is_diff else ""
        logger.info(f"  {key}:{marker}")
        for label, val in vals.items():
            logger.info(f"    {label}: {val}")

    logger.info(f"\nOutputs in: {out_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
