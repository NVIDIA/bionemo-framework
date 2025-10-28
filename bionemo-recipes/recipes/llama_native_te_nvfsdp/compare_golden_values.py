# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Compare golden values between LLAMA3 implementation and John's bionemo implementation.

This script loads golden values from both implementations and computes various
comparison metrics to ensure they match.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np

# Import bionemo collator for loading multi-rank predictions
try:
    from bionemo.llm.lightning import batch_collator
    HAS_BIONEMO = True
except ImportError:
    HAS_BIONEMO = False
    print("Warning: bionemo not installed, will use simple collation for multi-rank files")


def load_golden_values(filepath: Path) -> Dict[str, torch.Tensor]:
    """
    Load golden values from a .pt file or directory.
    
    If a directory is provided, loads all predictions__rank_*.pt files
    and collates them using bionemo's batch_collator (same as John's notebook).
    """
    if filepath.is_dir():
        # Load and collate multiple rank files
        pred_files = list(filepath.glob("predictions__rank_*.pt"))
        if not pred_files:
            raise ValueError(f"No predictions__rank_*.pt files found in {filepath}")
        
        print(f"Loading {len(pred_files)} prediction files from {filepath}")
        preds_list = [torch.load(f, map_location='cpu') for f in sorted(pred_files)]
        
        # Collate using bionemo's batch_collator (same as notebook)
        if HAS_BIONEMO:
            data = batch_collator([p for p in preds_list if p is not None])
            print(f"Collated predictions from {len(pred_files)} ranks using bionemo batch_collator")
        else:
            # Simple collation: concatenate along batch dimension
            print(f"Collating {len(pred_files)} ranks using simple concatenation")
            data = {
                'log_probs_seqs': torch.cat([p['log_probs_seqs'] for p in preds_list if p is not None], dim=0),
                'seq_idx': torch.cat([p['seq_idx'] for p in preds_list if p is not None], dim=0),
            }
            if 'loss_mask' in preds_list[0]:
                data['loss_mask'] = torch.cat([p['loss_mask'] for p in preds_list if p is not None], dim=0)
            print(f"Collated predictions from {len(pred_files)} ranks")
    else:
        # Load single file
        data = torch.load(filepath, map_location='cpu')
        print(f"Loaded from {filepath}")
    
    print(f"Keys: {data.keys()}")
    if 'log_probs_seqs' in data:
        print(f"  log_probs_seqs shape: {data['log_probs_seqs'].shape}")
    if 'loss_mask' in data:
        print(f"  loss_mask shape: {data['loss_mask'].shape}")
    if 'seq_idx' in data:
        print(f"  seq_idx shape: {data['seq_idx'].shape}")
    return data


def compute_differences(
    llama3_values: torch.Tensor,
    bionemo_values: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute various difference metrics between two sets of golden values.
    
    Args:
        llama3_values: Log probabilities from LLAMA3 implementation [N, L]
        bionemo_values: Log probabilities from bionemo implementation [N, L]
        loss_mask: Mask indicating valid positions [N, L]
    
    Returns:
        Dictionary of comparison metrics
    """
    # Only compare valid (non-masked) positions
    mask = loss_mask.bool()
    
    llama3_masked = llama3_values[mask]
    bionemo_masked = bionemo_values[mask]
    
    # Compute differences
    abs_diff = torch.abs(llama3_masked - bionemo_masked)
    rel_diff = abs_diff / (torch.abs(bionemo_masked) + 1e-10)
    
    metrics = {
        'num_compared_tokens': mask.sum().item(),
        'max_absolute_difference': abs_diff.max().item(),
        'mean_absolute_difference': abs_diff.mean().item(),
        'median_absolute_difference': abs_diff.median().item(),
        'std_absolute_difference': abs_diff.std().item(),
        'max_relative_difference': rel_diff.max().item(),
        'mean_relative_difference': rel_diff.mean().item(),
        'median_relative_difference': rel_diff.median().item(),
        # Correlation
        'pearson_correlation': torch.corrcoef(
            torch.stack([llama3_masked, bionemo_masked])
        )[0, 1].item(),
        # Cosine similarity
        'cosine_similarity': torch.nn.functional.cosine_similarity(
            llama3_masked.unsqueeze(0),
            bionemo_masked.unsqueeze(0)
        ).item(),
    }
    
    # Check if values are "close enough"
    # Using PyTorch's allclose with reasonable tolerances
    rtol = 1e-4  # relative tolerance
    atol = 1e-5  # absolute tolerance
    
    metrics['all_close_1e-4'] = torch.allclose(
        llama3_masked, bionemo_masked, rtol=rtol, atol=atol
    )
    metrics['all_close_1e-3'] = torch.allclose(
        llama3_masked, bionemo_masked, rtol=1e-3, atol=1e-4
    )
    metrics['all_close_1e-2'] = torch.allclose(
        llama3_masked, bionemo_masked, rtol=1e-2, atol=1e-3
    )
    
    # Compute percentage of values within different thresholds
    within_1e_2 = (abs_diff < 1e-2).float().mean().item()
    within_1e_3 = (abs_diff < 1e-3).float().mean().item()
    within_1e_4 = (abs_diff < 1e-4).float().mean().item()
    within_1e_5 = (abs_diff < 1e-5).float().mean().item()
    
    metrics['pct_within_1e-2'] = within_1e_2 * 100
    metrics['pct_within_1e-3'] = within_1e_3 * 100
    metrics['pct_within_1e-4'] = within_1e_4 * 100
    metrics['pct_within_1e-5'] = within_1e_5 * 100
    
    return metrics


def align_by_seq_idx(
    llama3_data: Dict[str, torch.Tensor],
    bionemo_data: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align golden values by seq_idx.
    
    Both datasets should have a 'seq_idx' field that identifies which
    sequence each row corresponds to.
    
    Returns:
        Tuple of (llama3_aligned, bionemo_aligned, loss_mask_aligned)
    """
    # Get seq_idx from both
    llama3_seq_idx = llama3_data['seq_idx']
    bionemo_seq_idx = bionemo_data['seq_idx']
    
    # Find common seq_idx values
    llama3_set = set(llama3_seq_idx.tolist())
    bionemo_set = set(bionemo_seq_idx.tolist())
    common_indices = sorted(llama3_set & bionemo_set)

    # breakpoint()
    print(f"\nAlignment info:")
    print(f"  LLAMA3 sequences: {len(llama3_set)}")
    print(f"  BioNeMo sequences: {len(bionemo_set)}")
    print(f"  Common sequences: {len(common_indices)}")
    
    if len(common_indices) == 0:
        raise ValueError("No common seq_idx found between the two datasets!")
    
    # Create masks for common indices
    llama3_mask = torch.isin(llama3_seq_idx, torch.tensor(common_indices))
    bionemo_mask = torch.isin(bionemo_seq_idx, torch.tensor(common_indices))
    
    # Get aligned data
    llama3_aligned = llama3_data['log_probs_seqs'][llama3_mask]
    bionemo_aligned = bionemo_data['log_probs_seqs'][bionemo_mask]
    # breakpoint()
    # Use loss_mask from bionemo (should be the same in both)
    loss_mask_aligned = bionemo_data['loss_mask'][bionemo_mask]
    # breakpoint()
    # Sort both by seq_idx to ensure alignment
    llama3_sorted_idx = torch.argsort(llama3_seq_idx[llama3_mask])
    bionemo_sorted_idx = torch.argsort(bionemo_seq_idx[bionemo_mask])
    # breakpoint()
    
    llama3_aligned = llama3_aligned[llama3_sorted_idx]
    bionemo_aligned = bionemo_aligned[bionemo_sorted_idx]
    loss_mask_aligned = loss_mask_aligned[bionemo_sorted_idx]
    
    print(f"  Aligned shapes: {llama3_aligned.shape}")
    
    return llama3_aligned, bionemo_aligned, loss_mask_aligned


def print_sample_values(
    llama3_values: torch.Tensor,
    bionemo_values: torch.Tensor,
    loss_mask: torch.Tensor,
    num_samples: int = 5,
):
    """Print a few sample values for manual inspection."""
    print(f"\nSample values (first {num_samples} valid tokens):")
    print(f"{'Index':<10} {'LLAMA3':<20} {'BioNeMo':<20} {'Difference':<20}")
    print("-" * 70)
    
    # Find first num_samples valid positions
    mask = loss_mask.bool()
    flat_mask = mask.flatten()
    valid_indices = torch.where(flat_mask)[0][:num_samples]
    
    llama3_flat = llama3_values.flatten()
    bionemo_flat = bionemo_values.flatten()
    
    for idx in valid_indices:
        idx = idx.item()
        l3_val = llama3_flat[idx].item()
        bn_val = bionemo_flat[idx].item()
        diff = abs(l3_val - bn_val)
        print(f"{idx:<10} {l3_val:<20.10f} {bn_val:<20.10f} {diff:<20.10e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare golden values from two implementations"
    )
    parser.add_argument(
        "--llama3-results",
        type=Path,
        required=True,
        help="Path to LLAMA3 golden values (.pt file)"
    )
    parser.add_argument(
        "--bionemo-results",
        type=Path,
        required=True,
        help="Path to BioNeMo golden values (.pt file or directory with predictions__rank_*.pt files)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_results"),
        help="Directory to save comparison results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load both sets of golden values
    print("Loading LLAMA3 golden values...")
    llama3_data = load_golden_values(args.llama3_results)
    
    print("\nLoading BioNeMo golden values...")
    bionemo_data = load_golden_values(args.bionemo_results)
    
    # Align by seq_idx
    print("\nAligning sequences...")
    
    # Breakpoint for debugging - remove or comment out when not debugging
    # breakpoint()  # ← Uncomment this to pause here
    # breakpoint()
    llama3_aligned, bionemo_aligned, loss_mask = align_by_seq_idx(
        llama3_data, bionemo_data
    )
    
    # Check shapes match
    if llama3_aligned.shape != bionemo_aligned.shape:
        raise ValueError(
            f"Shape mismatch after alignment! "
            f"LLAMA3: {llama3_aligned.shape}, BioNeMo: {bionemo_aligned.shape}"
        )
    
    # Print sample values
    # breakpoint()
    print_sample_values(llama3_aligned, bionemo_aligned, loss_mask)
    
    # Compute comparison metrics
    print("\nComputing comparison metrics...")
    metrics = compute_differences(llama3_aligned, bionemo_aligned, loss_mask)
    
    # Print metrics
    print("\n" + "=" * 70)
    print("COMPARISON METRICS")
    print("=" * 70)
    for key, value in metrics.items():
        if isinstance(value, bool):
            print(f"{key:<40}: {value}")
        elif isinstance(value, int):
            print(f"{key:<40}: {value:,}")
        elif isinstance(value, float):
            if 'pct' in key:
                print(f"{key:<40}: {value:.2f}%")
            elif abs(value) < 1e-3 or abs(value) > 1e3:
                print(f"{key:<40}: {value:.6e}")
            else:
                print(f"{key:<40}: {value:.6f}")
    
    # Save metrics to file
    metrics_file = args.output_dir / "comparison_metrics.json"
    # Convert bool to string for JSON serialization
    metrics_json = {
        k: (str(v) if isinstance(v, bool) else v)
        for k, v in metrics.items()
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")
    
    # Provide interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if metrics['all_close_1e-4']:
        print("PASS: Golden values match within tight tolerance (rtol=1e-4, atol=1e-5)")
    elif metrics['all_close_1e-3']:
        print("  CLOSE: Golden values match within reasonable tolerance (rtol=1e-3, atol=1e-4)")
        print("   This is likely acceptable for most applications.")
    elif metrics['all_close_1e-2']:
        print("  LOOSE: Golden values match within loose tolerance (rtol=1e-2, atol=1e-3)")
        print("   You may want to investigate the differences.")
    else:
        print(" FAIL: Golden values do NOT match within acceptable tolerance")
        print("   There may be a significant difference in the implementations.")
    
    print(f"\nMean absolute difference: {metrics['mean_absolute_difference']:.6e}")
    print(f"Pearson correlation: {metrics['pearson_correlation']:.6f}")
    print(f"Cosine similarity: {metrics['cosine_similarity']:.6f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

