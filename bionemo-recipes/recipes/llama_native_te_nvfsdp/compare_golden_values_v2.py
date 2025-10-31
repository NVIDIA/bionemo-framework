#!/usr/bin/env python
"""
Compare golden values with CORRECT alignment by lex-sorted headers.
This version uses the proper alignment method.
"""

import argparse
import json
import torch
from pathlib import Path

def load_data(llama3_file, bionemo_dir):
    """Load both datasets"""
    # Load LLAMA3
    llama3 = torch.load(llama3_file)
    
    # Load BioNeMo (multiple ranks)
    bionemo_files = list(Path(bionemo_dir).glob('predictions__rank_*.pt'))
    if not bionemo_files:
        bionemo_files = [Path(bionemo_dir)]
    
    if len(bionemo_files) > 1:
        preds = [torch.load(f) for f in sorted(bionemo_files)]
        bionemo = {
            'log_probs_seqs': torch.cat([p['log_probs_seqs'] for p in preds], dim=0),
            'loss_mask': torch.cat([p['loss_mask'] for p in preds], dim=0),
            'seq_idx': torch.cat([p['seq_idx'] for p in preds], dim=0),
        }
    else:
        bionemo = torch.load(bionemo_files[0])
    
    return llama3, bionemo

def align_by_headers(llama3, bionemo, llama3_map_file, bionemo_map_file):
    """Align by lex-sorted headers (CORRECT method)"""
    # Load seq_idx maps
    with open(llama3_map_file, 'r') as f:
        llama3_map = json.load(f)
    with open(bionemo_map_file, 'r') as f:
        bionemo_map = json.load(f)
    
    # Lex sort headers
    sorted_headers = sorted(set(llama3_map.keys()) & set(bionemo_map.keys()))
    
    print(f"Aligning {len(sorted_headers)} sequences by lex-sorted headers")
    
    # Get indices for each header in sorted order
    llama3_indices = [llama3_map[h] for h in sorted_headers]
    bionemo_indices = [bionemo_map[h] for h in sorted_headers]
    
    # Extract aligned predictions
    llama3_aligned = llama3['log_probs_seqs'][llama3_indices]
    bionemo_aligned = bionemo['log_probs_seqs'][bionemo_indices]
    loss_mask_aligned = bionemo['loss_mask'][bionemo_indices]
    
    return llama3_aligned, bionemo_aligned, loss_mask_aligned, sorted_headers

# Main
llama3_file = 'golden_values_llama3_nocache/predictions_llama3_native.pt'
bionemo_dir = '/workspaces/bionemo-framework/golden_values_bionemo/'
llama3_map = 'golden_values_llama3_nocache/seq_idx_map.json'
bionemo_map = '/workspaces/bionemo-framework/golden_values_bionemo/seq_idx_map.json'

print("Loading data...")
llama3, bionemo = load_data(llama3_file, bionemo_dir)

print("Aligning by headers...")
llama3_aligned, bionemo_aligned, loss_mask, headers = align_by_headers(
    llama3, bionemo, llama3_map, bionemo_map
)

print(f"\nAligned: {llama3_aligned.shape}")

# Calculate metrics
print("\n" + "=" * 80)
print("METRICS")
print("=" * 80)

# Mean Absolute Difference (per-token, then averaged)
mad = torch.abs(llama3_aligned - bionemo_aligned).mean()
print(f"\nMean Absolute Difference: {mad.item():.6f}")

# Distribution of differences (to understand if MAD is from outliers or systematic)
diff = torch.abs(llama3_aligned - bionemo_aligned)
print(f"\nDifference distribution:")
print(f"  Min:    {diff.min().item():.6f}")
print(f"  Median: {diff.median().item():.6f}")
print(f"  Mean:   {diff.mean().item():.6f}")
print(f"  Max:    {diff.max().item():.6f}")
print(f"  Positions < 0.1: {100.0 * (diff < 0.1).sum().item() / diff.numel():.2f}%")
print(f"  Positions < 1.0: {100.0 * (diff < 1.0).sum().item() / diff.numel():.2f}%")
print(f"  Positions < 2.0: {100.0 * (diff < 2.0).sum().item() / diff.numel():.2f}%")

# LM Loss
llama3_lm_loss = (-llama3_aligned * loss_mask).sum() / loss_mask.sum()
bionemo_lm_loss = (-bionemo_aligned * loss_mask).sum() / loss_mask.sum()

print(f"\nLM Loss:")
print(f"  LLAMA3:  {llama3_lm_loss:.8f}")
print(f"  BioNeMo: {bionemo_lm_loss:.8f}")
print(f"  Diff:    {abs(llama3_lm_loss - bionemo_lm_loss):.8f}")

# Create plots
import matplotlib.pyplot as plt

# Per-position metrics
llama3_nll_per_pos = (-llama3_aligned).mean(dim=0)  # [8191]
bionemo_nll_per_pos = (-bionemo_aligned).mean(dim=0)  # [8191]

# ABSOLUTE difference per position (SAME method as MAD)
abs_diff_per_pos = torch.abs(llama3_aligned - bionemo_aligned).mean(dim=0)  # [8191]

# Also calculate signed for reference
signed_diff_per_pos = (llama3_aligned - bionemo_aligned).mean(dim=0)  # [8191]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: LM Loss (NLL) by position
axes[0].plot(llama3_nll_per_pos.numpy(), label='LLAMA3', alpha=0.8, linewidth=2)
axes[0].plot(bionemo_nll_per_pos.numpy(), label='BioNeMo', alpha=0.8, linewidth=2)
axes[0].set_xlabel('Position')
axes[0].set_ylabel('NLL (Negative Log Likelihood)')
axes[0].set_title('LM Loss by Position (averaged across sequences)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: ABSOLUTE difference by position (same method as overall MAD)
axes[1].plot(abs_diff_per_pos.numpy(), color='red', linewidth=1.5)
axes[1].axhline(y=abs_diff_per_pos.mean(), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {abs_diff_per_pos.mean():.6f}')
axes[1].set_xlabel('Position')
axes[1].set_ylabel('|LLAMA3 - BioNeMo| (absolute difference)')
axes[1].set_title('Mean Absolute Difference by Position (averaged across sequences)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, max(abs_diff_per_pos.max().item() * 1.1, 3.0)])

plt.tight_layout()
plt.savefig('comparison_by_position.png', dpi=150)
print("\n✅ Saved plots to comparison_by_position.png")

print("\n" + "=" * 80)
print("PER-POSITION STATISTICS")
print("=" * 80)
print(f"\nAbsolute diff per position:")
print(f"  Range: [{abs_diff_per_pos.min():.6f}, {abs_diff_per_pos.max():.6f}]")
print(f"  Mean: {abs_diff_per_pos.mean():.6f} (this equals overall MAD)")
print(f"\nSigned diff per position (for reference):")
print(f"  Range: [{signed_diff_per_pos.min():.6f}, {signed_diff_per_pos.max():.6f}]")
print(f"  Mean: {signed_diff_per_pos.mean():.6f} (should be ~0)")
print("\nNote: Position-averaged values are smaller than individual token diffs")
print("because differences at each position cancel across sequences")
print("=" * 80)

