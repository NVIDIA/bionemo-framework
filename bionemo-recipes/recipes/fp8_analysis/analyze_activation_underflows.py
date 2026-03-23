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

"""FP8 activation underflow & MSE analyzer with heatmaps and boundary comparison.

Generates:
  1. Activation underflow heatmap (per-layer, over time)
  2. Activation MSE heatmap (per-layer, over time)
  3. Boundary layer time-series (fc2 underflows + MSE at last FP8 layer)
  4. Summary table (CSV) with per-layer mean/max stats

Usage:
    python3 analyze_activation_underflows.py <log_directory> <suffix>

Example:
    python3 analyze_activation_underflows.py /data/.../quant_logs _fl2
    python3 analyze_activation_underflows.py /data/.../quant_logs _fl4
"""

import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sns.set_style("white")


def parse_log_file(log_file_path, max_lines=None):
    """Parse FP8 statistics log file."""
    logger.info(f"Parsing: {log_file_path}")
    pattern = (
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+-\s+INFO\s+-\s+(.+?)\s+iteration=(\d+)\s+value=([\d.e+-]+)$"
    )
    data = []
    for line_num, raw_line in enumerate(open(log_file_path), 1):
        if max_lines and line_num > max_lines:
            break
        match = re.match(pattern, raw_line.strip())
        if match:
            data.append(
                {
                    "metric_name": match.group(1).strip(),
                    "iteration": int(match.group(2)),
                    "value": float(match.group(3)),
                }
            )
        if line_num % 1000000 == 0:
            logger.info(f"  Processed {line_num:,} lines...")
    df = pd.DataFrame(data)
    if len(df) > 0:
        logger.info(f"  Extracted {len(df):,} metrics, iterations {df['iteration'].min()}-{df['iteration'].max()}")
    return df


def extract_layer_num(metric_name):
    """Extract 0-indexed layer number from metric name."""
    match = re.search(r"\.layers\.(\d+)\.", metric_name)
    return int(match.group(1)) if match else None


def extract_component(metric_name):
    """Extract component (e.g., 'fc2', 'layernorm_qkv', 'proj', 'fc1') from metric name."""
    for comp in ["layernorm_qkv", "proj", "fc1", "fc2", "layernorm_mlp"]:
        if comp in metric_name:
            return comp
    return "unknown"


def create_metric_heatmap(df, metric_suffix, title, output_path, vmax=None, cmap="rocket_r"):
    """Create a heatmap for a specific metric across layers and time."""
    metric_df = df[df["metric_name"].str.endswith(metric_suffix)].copy()
    if len(metric_df) == 0:
        logger.warning(f"No data for {metric_suffix}")
        return

    metric_df["layer"] = metric_df["metric_name"].apply(extract_layer_num)
    metric_df = metric_df.dropna(subset=["layer"])
    metric_df["layer"] = metric_df["layer"].astype(int)

    pivot = metric_df.pivot_table(values="value", index="layer", columns="iteration", aggfunc="mean")
    pivot = pivot.sort_index()

    # Sample columns for readability
    if len(pivot.columns) > 150:
        step = max(1, len(pivot.columns) // 150)
        pivot = pivot.iloc[:, ::step]

    fig, ax = plt.subplots(figsize=(20, max(8, len(pivot.index) * 0.35)))

    if vmax is None:
        vmax = min(pivot.values.max(), np.percentile(pivot.values[~np.isnan(pivot.values)], 99))
    if vmax == 0:
        vmax = 1

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_suffix.replace("_", " ").title(), fontsize=12)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Layer {layer_idx}" for layer_idx in pivot.index], fontsize=8)
    ax.set_ylabel("Layer (0-indexed)", fontsize=12, fontweight="bold")

    x_ticks = np.linspace(0, len(pivot.columns) - 1, min(15, len(pivot.columns))).astype(int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(pivot.columns[i])}" for i in x_ticks], fontsize=10)
    ax.set_xlabel("Training Iteration", fontsize=12, fontweight="bold")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # Summary stats box
    max_val = pivot.values.max()
    mean_val = np.nanmean(pivot.values)
    summary = f"Max: {max_val:.4f}\nMean: {mean_val:.4f}\nLayers: {len(pivot.index)}"
    ax.text(
        0.98,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def create_boundary_timeseries(df, output_path, suffix=""):
    """Plot time-series of fc2 activation metrics for the last 3 FP8 layers."""
    fc2_underflows = df[df["metric_name"].str.contains("fc2_activation_underflows")].copy()
    fc2_underflows["layer"] = fc2_underflows["metric_name"].apply(extract_layer_num)
    fc2_underflows = fc2_underflows.dropna(subset=["layer"])

    if len(fc2_underflows) == 0:
        logger.warning("No fc2 activation underflow data")
        return

    # Find the last 3 FP8 layers (highest layer numbers with data)
    layers = sorted(fc2_underflows["layer"].unique())
    boundary_layers = layers[-3:] if len(layers) >= 3 else layers

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot 1: fc2 activation underflows
    ax1 = axes[0]
    for layer in boundary_layers:
        layer_data = fc2_underflows[fc2_underflows["layer"] == layer].sort_values("iteration")
        ax1.plot(layer_data["iteration"], layer_data["value"], label=f"Layer {int(layer)}", linewidth=1.5)
    ax1.set_ylabel("fc2 Activation Underflows %", fontsize=12, fontweight="bold")
    ax1.set_title(f"FP8 Boundary Layer Degradation{suffix}", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("symlog", linthresh=0.1)

    # Plot 2: fc2 activation MSE
    fc2_mse = df[df["metric_name"].str.contains("fc2_activation_mse")].copy()
    fc2_mse["layer"] = fc2_mse["metric_name"].apply(extract_layer_num)
    fc2_mse = fc2_mse.dropna(subset=["layer"])

    ax2 = axes[1]
    for layer in boundary_layers:
        layer_data = fc2_mse[fc2_mse["layer"] == layer].sort_values("iteration")
        if len(layer_data) > 0:
            ax2.plot(layer_data["iteration"], layer_data["value"], label=f"Layer {int(layer)}", linewidth=1.5)
    ax2.set_ylabel("fc2 Activation MSE", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Iteration", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("symlog", linthresh=0.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")


def generate_summary_table(df, output_path):
    """Generate per-layer summary statistics CSV."""
    # Focus on key metrics
    key_suffixes = [
        "fc2_activation_underflows%",
        "fc2_activation_mse",
        "fc2_activation_scale_inv_max",
        "fc2_activation_scale_inv_min",
        "layernorm_qkv_activation_underflows%",
        "proj_activation_underflows%",
        "fc1_activation_underflows%",
        "fc1_gradient_underflows%",
        "fc2_gradient_underflows%",
    ]

    rows = []
    for suffix in key_suffixes:
        metric_df = df[df["metric_name"].str.endswith(suffix)].copy()
        metric_df["layer"] = metric_df["metric_name"].apply(extract_layer_num)
        metric_df = metric_df.dropna(subset=["layer"])

        for layer, group in metric_df.groupby("layer"):
            rows.append(
                {
                    "layer": int(layer),
                    "metric": suffix,
                    "mean": group["value"].mean(),
                    "max": group["value"].max(),
                    "std": group["value"].std(),
                    "last_value": group.sort_values("iteration").iloc[-1]["value"],
                    "last_iteration": int(group["iteration"].max()),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["metric", "layer"])
    summary.to_csv(output_path, index=False)
    logger.info(f"  Saved summary: {output_path} ({len(summary)} rows)")
    return summary


def main():
    """Entry point."""
    if len(sys.argv) < 2:
        logger.error("Usage: python3 analyze_activation_underflows.py <log_directory> [suffix]")
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""

    log_file = log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"
    if not log_file.exists():
        logger.error(f"Log file not found: {log_file}")
        sys.exit(1)

    # Output directories
    out_dir = Path("activation_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"FP8 ACTIVATION UNDERFLOW ANALYSIS{suffix}")
    logger.info("=" * 70)

    # Parse (use full file)
    df = parse_log_file(log_file)

    if len(df) == 0:
        logger.error("No data!")
        sys.exit(1)

    # 1. Activation underflow heatmap (fc2 — the problematic one)
    logger.info("\n--- Generating fc2 activation underflow heatmap ---")
    create_metric_heatmap(
        df,
        "fc2_activation_underflows%",
        f"fc2 Activation Underflows % (FP8 Block Scaling){suffix}",
        out_dir / f"heatmap_fc2_activation_underflows{suffix}.png",
        vmax=20,
    )

    # 2. Activation MSE heatmap
    logger.info("\n--- Generating fc2 activation MSE heatmap ---")
    create_metric_heatmap(
        df,
        "fc2_activation_mse",
        f"fc2 Activation MSE (Quantization Error){suffix}",
        out_dir / f"heatmap_fc2_activation_mse{suffix}.png",
        cmap="inferno",
    )

    # 3. All activation underflows heatmap (layernorm_qkv, proj, fc1, fc2)
    logger.info("\n--- Generating all activation underflows heatmap ---")
    create_metric_heatmap(
        df,
        "activation_underflows%",
        f"All Activation Underflows % (FP8 Block Scaling){suffix}",
        out_dir / f"heatmap_all_activation_underflows{suffix}.png",
        vmax=5,
    )

    # 4. Boundary time-series
    logger.info("\n--- Generating boundary time-series ---")
    create_boundary_timeseries(df, out_dir / f"boundary_timeseries{suffix}.png", suffix)

    # 5. Summary table
    logger.info("\n--- Generating summary table ---")
    summary = generate_summary_table(df, out_dir / f"summary{suffix}.csv")

    # Print key findings
    fc2_summary = summary[summary["metric"] == "fc2_activation_underflows%"].sort_values("layer")
    if len(fc2_summary) > 0:
        logger.info("\n" + "=" * 70)
        logger.info("KEY FINDINGS: fc2 Activation Underflows")
        logger.info("=" * 70)
        for _, row in fc2_summary.iterrows():
            marker = " <<<" if row["last_value"] > 1.0 else ""
            logger.info(
                f"  Layer {int(row['layer']):2d}:  mean={row['mean']:.4f}%  "
                f"max={row['max']:.4f}%  last={row['last_value']:.4f}%{marker}"
            )

    logger.info("\n" + "=" * 70)
    logger.info("OUTPUTS:")
    for f in sorted(out_dir.glob(f"*{suffix}*")):
        logger.info(f"  {f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
