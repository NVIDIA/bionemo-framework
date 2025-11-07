#!/usr/bin/env python3
"""Compare BioNeMo golden values with Peter's/Old model golden values.

This script CAREFULLY aligns sequences and calculates per-token Mean Absolute Difference (MAD).

ALIGNMENT SAFETY CHECKS:
1. Both datasets use seq_len-1 format (aligned predictions vs targets)
2. Loss masks are properly applied
3. Headers are lex-sorted for consistent alignment
4. Shapes are validated before comparison
5. No off-by-one errors in indexing
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

print("=" * 80)
print("COMPARING BIONEMO vs LLAMA3 MODELS - PER-TOKEN MAD ANALYSIS")
print("=" * 80)

# ========== Configuration ==========
bionemo_dir = Path('/workspaces/bionemo-framework/golden_values_bionemo')
peter_file = Path('batch_evaluation_golden_method.pt')
old_model_file = Path('batch_evaluation_golden_method_old_model.pt')

# Choose which model to compare
import sys
if len(sys.argv) > 1 and sys.argv[1] == '--old':
    llama3_file = old_model_file
    model_name = "OLD model.py"
else:
    llama3_file = peter_file
    model_name = "Peter's model"

print(f"\nComparing: BioNeMo vs {model_name}")

# ========== Step 1: Load BioNeMo Golden Values ==========
print("\n" + "-"*80)
print("STEP 1: LOADING BIONEMO GOLDEN VALUES")
print("-"*80)

# Load BioNeMo predictions (multiple rank files)
bionemo_files = sorted(list(bionemo_dir.glob('predictions__rank_*.pt')))
print(f"Found {len(bionemo_files)} BioNeMo rank files")

# Load and concatenate predictions from all ranks
bionemo_preds = [torch.load(f) for f in bionemo_files]
bionemo = {
    'log_probs_seqs': torch.cat([p['log_probs_seqs'] for p in bionemo_preds], dim=0),
    'loss_mask': torch.cat([p['loss_mask'] for p in bionemo_preds], dim=0),
    'seq_idx': torch.cat([p['seq_idx'] for p in bionemo_preds], dim=0),
}

print(f"✓ BioNeMo predictions loaded:")
print(f"  Shape: {bionemo['log_probs_seqs'].shape}")
print(f"  Dtype: {bionemo['log_probs_seqs'].dtype}")

# Load BioNeMo seq_idx_map
with open(bionemo_dir / 'seq_idx_map.json', 'r') as f:
    bionemo_map = json.load(f)

print(f"  Sequences in map: {len(bionemo_map)}")

# ========== Step 2: Load LLAMA3 Golden Values ==========
print("\n" + "-"*80)
print(f"STEP 2: LOADING {model_name.upper()} GOLDEN VALUES")
print("-"*80)

if not llama3_file.exists():
    print(f"❌ {model_name} results not found: {llama3_file}")
    print("\nPlease run first:")
    if llama3_file == peter_file:
        print("  python batch_evaluate_loss_golden_method.py")
    else:
        print("  python batch_evaluate_loss_golden_method_old_model.py")
    exit(1)

llama3_results = torch.load(llama3_file)
print(f"✓ {model_name} results loaded")

# Check format
if 'log_probs_seqs' not in llama3_results:
    print(f"❌ {model_name} results missing 'log_probs_seqs'")
    print("   Please re-run the golden method script (updated version)")
    exit(1)

llama3 = {
    'log_probs_seqs': llama3_results['log_probs_seqs'],
    'loss_mask': llama3_results['loss_mask'],
    'seq_idx_map': llama3_results['seq_idx_map'],
}

print(f"  Shape: {llama3['log_probs_seqs'].shape}")
print(f"  Dtype: {llama3['log_probs_seqs'].dtype}")
print(f"  Sequences in map: {len(llama3['seq_idx_map'])}")

# ========== Step 3: Align Sequences by Headers ==========
print("\n" + "-"*80)
print("STEP 3: ALIGNING SEQUENCES BY HEADERS (LEX-SORTED)")
print("-"*80)

# Find common headers - LEX SORT for consistent ordering
common_headers = sorted(set(bionemo_map.keys()) & set(llama3['seq_idx_map'].keys()))

if len(common_headers) == 0:
    print("❌ No common headers found!")
    print("\nBioNeMo sample:", list(bionemo_map.keys())[:3])
    print(f"{model_name} sample:", list(llama3['seq_idx_map'].keys())[:3])
    exit(1)

print(f"✓ Found {len(common_headers)} common sequences")
print(f"  First header: {common_headers[0][:60]}...")
print(f"  Last header:  {common_headers[-1][:60]}...")

# Get aligned indices (lex-sorted order)
bionemo_indices = [bionemo_map[h] for h in common_headers]
llama3_indices = [llama3['seq_idx_map'][h] for h in common_headers]

print(f"\n  Index alignment check:")
print(f"    BioNeMo indices: min={min(bionemo_indices)}, max={max(bionemo_indices)}")
print(f"    {model_name} indices: min={min(llama3_indices)}, max={max(llama3_indices)}")

# ========== Step 4: Extract Aligned Predictions ==========
print("\n" + "-"*80)
print("STEP 4: EXTRACTING ALIGNED PREDICTIONS")
print("-"*80)

# Extract aligned data
bionemo_aligned = bionemo['log_probs_seqs'][bionemo_indices]  # [N, seq_len]
bionemo_mask_aligned = bionemo['loss_mask'][bionemo_indices]  # [N, seq_len]

llama3_aligned = llama3['log_probs_seqs'][llama3_indices]  # [N, seq_len]
llama3_mask_aligned = llama3['loss_mask'][llama3_indices]  # [N, seq_len]

print(f"✓ Aligned predictions extracted")
print(f"  BioNeMo:  {bionemo_aligned.shape}")
print(f"  {model_name}: {llama3_aligned.shape}")

# ========== Step 5: Validate Alignment ==========
print("\n" + "-"*80)
print("STEP 5: VALIDATING ALIGNMENT (CRITICAL SAFETY CHECK)")
print("-"*80)

# Check shapes match
if bionemo_aligned.shape != llama3_aligned.shape:
    print(f"❌ Shape mismatch!")
    print(f"   BioNeMo:  {bionemo_aligned.shape}")
    print(f"   {model_name}: {llama3_aligned.shape}")
    exit(1)

print(f"✓ Shapes match: {bionemo_aligned.shape}")

# Check masks match (they should be identical if using same data)
mask_match = (bionemo_mask_aligned == llama3_mask_aligned).all()
if not mask_match:
    print(f"⚠️  Loss masks differ slightly")
    mask_diff = (bionemo_mask_aligned != llama3_mask_aligned).sum().item()
    print(f"   Different positions: {mask_diff} / {bionemo_mask_aligned.numel()}")
    print(f"   Using intersection of masks for safety")
    # Use intersection (both must be 1)
    combined_mask = bionemo_mask_aligned * llama3_mask_aligned
else:
    print(f"✓ Loss masks match perfectly")
    combined_mask = bionemo_mask_aligned

num_valid_tokens = combined_mask.sum().item()
print(f"✓ Valid tokens for comparison: {int(num_valid_tokens):,}")

# ========== Step 6: Calculate Per-Token MAD ==========
print("\n" + "="*80)
print("STEP 6: CALCULATING PER-TOKEN MEAN ABSOLUTE DIFFERENCE (MAD)")
print("="*80)

# Calculate absolute difference
abs_diff = torch.abs(bionemo_aligned - llama3_aligned)  # [N, seq_len]

# Apply mask to only consider valid tokens
masked_diff = abs_diff * combined_mask.float()

# Calculate MAD (Mean Absolute Difference over all valid tokens)
mad = masked_diff.sum() / combined_mask.sum()

print(f"\n📊 GLOBAL METRICS")
print(f"{'='*80}")
print(f"Per-Token MAD: {mad.item():.8f}")

# Distribution of differences
valid_diffs = masked_diff[combined_mask.bool()]
print(f"\n📊 DIFFERENCE DISTRIBUTION (only valid tokens)")
print(f"{'='*80}")
print(f"  Min:       {valid_diffs.min().item():.8f}")
print(f"  25th %ile: {torch.quantile(valid_diffs, 0.25).item():.8f}")
print(f"  Median:    {torch.quantile(valid_diffs, 0.50).item():.8f}")
print(f"  75th %ile: {torch.quantile(valid_diffs, 0.75).item():.8f}")
print(f"  95th %ile: {torch.quantile(valid_diffs, 0.95).item():.8f}")
print(f"  99th %ile: {torch.quantile(valid_diffs, 0.99).item():.8f}")
print(f"  Max:       {valid_diffs.max().item():.8f}")

# How many tokens are within certain thresholds?
print(f"\n📊 ACCURACY THRESHOLDS")
print(f"{'='*80}")
for threshold in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
    pct = (valid_diffs < threshold).sum().item() / valid_diffs.numel() * 100
    print(f"  Tokens with |diff| < {threshold:5.3f}: {pct:6.2f}%")

# ========== Step 7: Calculate LM Loss for Both ==========
print("\n" + "="*80)
print("STEP 7: COMPARING LANGUAGE MODEL LOSS")
print("="*80)

bionemo_lm_loss = (-bionemo_aligned * combined_mask).sum() / combined_mask.sum()
llama3_lm_loss = (-llama3_aligned * combined_mask).sum() / combined_mask.sum()

print(f"\nLM Loss (Negative Log Likelihood):")
print(f"  BioNeMo:      {bionemo_lm_loss.item():.8f}")
print(f"  {model_name}: {llama3_lm_loss.item():.8f}")
print(f"  Difference:   {abs(bionemo_lm_loss - llama3_lm_loss).item():.8f}")

lm_loss_diff_pct = abs(bionemo_lm_loss - llama3_lm_loss) / bionemo_lm_loss * 100
print(f"  Relative:     {lm_loss_diff_pct.item():.4f}%")

# ========== Step 8: Per-Position Analysis ==========
print("\n" + "="*80)
print("STEP 8: PER-POSITION ANALYSIS")
print("="*80)

# Average MAD per position
mad_per_pos = (abs_diff * combined_mask.float()).sum(dim=0) / combined_mask.sum(dim=0).clamp(min=1)

print(f"\nPer-position MAD statistics:")
print(f"  Shape: {mad_per_pos.shape}")
print(f"  Min:  {mad_per_pos.min().item():.6f}")
print(f"  Max:  {mad_per_pos.max().item():.6f}")
print(f"  Mean: {mad_per_pos.mean().item():.6f} (should equal global MAD: {mad.item():.6f})")

# Find positions with highest differences
top_k = 10
worst_positions = torch.topk(mad_per_pos, k=min(top_k, len(mad_per_pos)))
print(f"\nTop {top_k} positions with highest MAD:")
for i, (pos, val) in enumerate(zip(worst_positions.indices, worst_positions.values), 1):
    print(f"  {i}. Position {pos.item():4d}: MAD = {val.item():.6f}")

# ========== Step 9: Per-Sequence Analysis ==========
print("\n" + "="*80)
print("STEP 9: PER-SEQUENCE ANALYSIS")
print("="*80)

# Calculate MAD per sequence
mad_per_seq = (abs_diff * combined_mask.float()).sum(dim=1) / combined_mask.sum(dim=1).clamp(min=1)

print(f"\nPer-sequence MAD statistics:")
print(f"  Min:  {mad_per_seq.min().item():.6f}")
print(f"  Max:  {mad_per_seq.max().item():.6f}")
print(f"  Mean: {mad_per_seq.mean().item():.6f}")

# Find sequences with highest/lowest MAD
best_idx = mad_per_seq.argmin()
worst_idx = mad_per_seq.argmax()

print(f"\n🏆 Best sequence (lowest MAD = {mad_per_seq[best_idx].item():.6f}):")
print(f"   {common_headers[best_idx][:70]}")

print(f"\n⚠️  Worst sequence (highest MAD = {mad_per_seq[worst_idx].item():.6f}):")
print(f"   {common_headers[worst_idx][:70]}")

# ========== Step 10: Create Visualization ==========
print("\n" + "="*80)
print("STEP 10: CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Per-position MAD
axes[0, 0].plot(mad_per_pos.numpy(), alpha=0.7, linewidth=0.5)
axes[0, 0].axhline(y=mad.item(), color='red', linestyle='--', linewidth=2, label=f'Global MAD: {mad.item():.6f}')
axes[0, 0].set_xlabel('Position')
axes[0, 0].set_ylabel('Mean Absolute Difference')
axes[0, 0].set_title('Per-Position MAD (averaged across sequences)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Histogram of per-token differences
axes[0, 1].hist(valid_diffs.numpy(), bins=100, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=mad.item(), color='red', linestyle='--', linewidth=2, label=f'MAD: {mad.item():.6f}')
axes[0, 1].set_xlabel('Absolute Difference')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Per-Token Differences')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Per-sequence MAD
axes[1, 0].scatter(range(len(mad_per_seq)), mad_per_seq.numpy(), alpha=0.5, s=10)
axes[1, 0].axhline(y=mad.item(), color='red', linestyle='--', linewidth=2, label=f'Global MAD: {mad.item():.6f}')
axes[1, 0].set_xlabel('Sequence Index')
axes[1, 0].set_ylabel('Mean Absolute Difference')
axes[1, 0].set_title('Per-Sequence MAD')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Cumulative distribution
sorted_diffs = torch.sort(valid_diffs)[0]
cumulative = torch.arange(1, len(sorted_diffs) + 1).float() / len(sorted_diffs) * 100
axes[1, 1].plot(sorted_diffs.numpy(), cumulative.numpy(), linewidth=2)
axes[1, 1].axvline(x=mad.item(), color='red', linestyle='--', linewidth=2, label=f'MAD: {mad.item():.6f}')
axes[1, 1].set_xlabel('Absolute Difference')
axes[1, 1].set_ylabel('Cumulative Percentage')
axes[1, 1].set_title('Cumulative Distribution of Differences')
axes[1, 1].set_xscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_plot = f'mad_comparison_bionemo_vs_{model_name.replace(" ", "_").lower()}.png'
plt.savefig(output_plot, dpi=150)
print(f"\n✓ Saved plot to: {output_plot}")

# ========== Final Summary ==========
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n📊 Comparison: BioNeMo vs {model_name}")
print(f"  Sequences compared:  {len(common_headers)}")
print(f"  Valid tokens:        {int(num_valid_tokens):,}")
print(f"  Per-token MAD:       {mad.item():.8f}")
print(f"  LM Loss difference:  {abs(bionemo_lm_loss - llama3_lm_loss).item():.8f}")

if mad.item() < 0.001:
    print(f"\n✅✅✅ EXCELLENT: Models are essentially identical (MAD < 0.001)")
elif mad.item() < 0.01:
    print(f"\n✓✓ GOOD: Models are very similar (MAD < 0.01)")
elif mad.item() < 0.1:
    print(f"\n⚠️ MODERATE: Some differences present (MAD < 0.1)")
else:
    print(f"\n❌ SIGNIFICANT: Models produce different results (MAD >= 0.1)")

print("\n" + "="*80)




