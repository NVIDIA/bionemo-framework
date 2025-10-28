#!/usr/bin/env python3
"""
Compare layer-by-layer outputs from sequential execution.
This shows WHERE and HOW divergence grows through the network.

Unlike isolated layer comparison, this tracks error compounding.
"""

import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("=" * 80)
print("COMPARING SEQUENTIAL LAYER OUTPUTS (TRACKING DIVERGENCE)")
print("=" * 80)

nemo_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/nemo_sequential_layers")
hf_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/hf_sequential_layers")

def compare_tensors(nemo_tensor, hf_tensor):
    """Compare two tensors, handling NeMo's sequence-first format."""
    # NeMo uses [seq, batch, hidden], HF uses [batch, seq, hidden]
    if len(nemo_tensor.shape) == 3 and nemo_tensor.shape[0] > nemo_tensor.shape[1]:
        nemo_tensor = nemo_tensor.transpose(0, 1)
    
    if nemo_tensor.shape != hf_tensor.shape:
        return None  # Shape mismatch
    
    diff = (nemo_tensor - hf_tensor).abs()
    
    # Per-token analysis: mean absolute diff for each token (average over hidden dim)
    # Shape: [batch, seq, hidden] -> [batch, seq] -> [seq] (squeeze batch)
    per_token_diff = diff.mean(dim=-1).squeeze(0)  # Remove batch dimension
    
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "median_diff": diff.median().item(),
        "close_1e-3_pct": 100.0 * (diff < 1e-3).sum().item() / diff.numel(),
        "close_1e-2_pct": 100.0 * (diff < 1e-2).sum().item() / diff.numel(),
        "close_1e-1_pct": 100.0 * (diff < 1e-1).sum().item() / diff.numel(),
        # Per-token metrics
        "per_token_mean": per_token_diff.mean().item(),
        "per_token_max": per_token_diff.max().item(),
        "per_token_max_pos": per_token_diff.argmax().item(),
        "per_token_diff": per_token_diff,  # Full per-token array
    }

# ========== Check Files ==========
print("\n" + "="*80)
print("CHECKING FILES")
print("="*80)

if not nemo_dir.exists():
    print(f"❌ NeMo directory not found: {nemo_dir}")
    print("   Run save_nemo_all_layers_sequential.py first!")
    exit(1)

if not hf_dir.exists():
    print(f"❌ HF directory not found: {hf_dir}")
    print("   Run save_hf_all_layers_sequential.py first!")
    exit(1)

print(f"✓ NeMo directory: {nemo_dir}")
print(f"✓ HF directory: {hf_dir}")

# ========== Compare Layer by Layer ==========
print("\n" + "="*80)
print("LAYER-BY-LAYER COMPARISON")
print("="*80)

results = []

# Compare embeddings (layer -1)
nemo_emb = torch.load(nemo_dir / "layer_-1_output.pt")
hf_emb = torch.load(hf_dir / "layer_-1_output.pt")
emb_result = compare_tensors(nemo_emb, hf_emb)
if emb_result:
    print(f"Layer -1 (Embeddings): max_diff={emb_result['max_diff']:.6f} ({'✅ MATCH' if emb_result['max_diff'] < 1e-6 else '❌ DIFFER'})")
    results.append({"layer": -1, "name": "Embeddings", **emb_result})

# Compare each layer
for i in range(32):
    nemo_path = nemo_dir / f"layer_{i}_output.pt"
    hf_path = hf_dir / f"layer_{i}_output.pt"
    
    if not nemo_path.exists() or not hf_path.exists():
        print(f"Layer {i:2d}: ⚠️ Files missing")
        continue
    
    nemo_output = torch.load(nemo_path)
    hf_output = torch.load(hf_path)
    
    result = compare_tensors(nemo_output, hf_output)
    if result is None:
        print(f"Layer {i:2d}: ❌ Shape mismatch")
        continue
    
    status = "✅" if result["max_diff"] < 1e-3 else "⚠️" if result["max_diff"] < 1e-2 else "❌"
    print(f"Layer {i:2d}: {status} max={result['max_diff']:.6f}, mean={result['mean_diff']:.6f}, per_token_mean={result['per_token_mean']:.6f}")
    
    results.append({"layer": i, "name": f"Layer {i}", **result})

# Compare final output
nemo_final = torch.load(nemo_dir / "final_output.pt")
hf_final = torch.load(hf_dir / "final_output.pt")
final_result = compare_tensors(nemo_final, hf_final)
if final_result:
    status = "✅" if final_result["max_diff"] < 1e-3 else "⚠️" if final_result["max_diff"] < 1e-2 else "❌"
    print(f"Final (after LN): {status} max={final_result['max_diff']:.6f}, mean={final_result['mean_diff']:.6f}")
    results.append({"layer": 32, "name": "Final", **final_result})

# ========== Analysis ==========
print("\n" + "="*80)
print("DIVERGENCE ANALYSIS")
print("="*80)

# Find where divergence crosses thresholds
for threshold, name in [(1e-3, "1e-3"), (1e-2, "1e-2"), (1e-1, "1e-1")]:
    for result in results:
        if result["layer"] >= 0 and result["max_diff"] >= threshold:
            print(f"\nFirst layer exceeding {name}: Layer {result['layer']}")
            print(f"  Max diff: {result['max_diff']:.6f}")
            print(f"  Mean diff: {result['mean_diff']:.6f}")
            break

# Show trend every 4 layers
print("\n" + "-"*80)
print("DIVERGENCE TREND (every 4 layers):")
print("-"*80)
for result in results:
    if result["layer"] % 4 == 0 or result["layer"] in [-1, 31, 32]:
        print(f"{result['name']:15s}: max={result['max_diff']:.6f}, mean={result['mean_diff']:.6f}")

# ========== Create Plots ==========
print("\n" + "-"*80)
print("GENERATING PLOTS")
print("-"*80)

layer_nums = [r["layer"] for r in results if r["layer"] >= 0]
max_diffs = [r["max_diff"] for r in results if r["layer"] >= 0]
mean_diffs = [r["mean_diff"] for r in results if r["layer"] >= 0]
per_token_means = [r["per_token_mean"] for r in results if r["layer"] >= 0]

# Main divergence plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

# Plot 1: Max difference
ax1.plot(layer_nums, max_diffs, 'o-', label='Max Difference', linewidth=2, markersize=4)
ax1.axhline(y=1e-3, color='g', linestyle='--', alpha=0.5, label='Threshold: 1e-3')
ax1.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.5, label='Threshold: 1e-2')
ax1.axhline(y=1e-1, color='r', linestyle='--', alpha=0.5, label='Threshold: 1e-1')
ax1.set_xlabel('Layer Number')
ax1.set_ylabel('Max Absolute Difference')
ax1.set_title('Layer-by-Layer Divergence: Maximum Difference')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Mean difference  
ax2.plot(layer_nums, mean_diffs, 's-', label='Mean Difference (All Elements)', linewidth=2, markersize=4, color='purple')
ax2.plot(layer_nums, per_token_means, 'd-', label='Per-Token Mean', linewidth=2, markersize=4, color='darkgreen')
ax2.axhline(y=1e-3, color='g', linestyle='--', alpha=0.5, label='Threshold: 1e-3')
ax2.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.5, label='Threshold: 1e-2')
ax2.set_xlabel('Layer Number')
ax2.set_ylabel('Mean Absolute Difference')
ax2.set_title('Layer-by-Layer Divergence: Mean Difference (Global vs Per-Token)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Per-token max position (which token diverges most)
per_token_max_positions = [r["per_token_max_pos"] for r in results if r["layer"] >= 0]
ax3.plot(layer_nums, per_token_max_positions, 'v-', label='Token with Max Divergence', linewidth=2, markersize=4, color='red')
ax3.set_xlabel('Layer Number')
ax3.set_ylabel('Token Position')
ax3.set_title('Which Token Position Diverges Most (by layer)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/divergence_plot.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved main divergence plot to: {plot_path}")

# Per-token heatmap (select key layers to avoid overcrowding)
key_layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]  # Every 4th layer + first and last
per_token_data = []
per_token_labels = []

for r in results:
    if r["layer"] in key_layers and "per_token_diff" in r:
        # Convert to float32 first (BFloat16 can't be directly converted to numpy)
        per_token_data.append(r["per_token_diff"].float().numpy())
        per_token_labels.append(f"L{r['layer']}")

if per_token_data:
    fig2, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(per_token_data, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_yticks(range(len(per_token_labels)))
    ax.set_yticklabels(per_token_labels)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title('Per-Token Divergence Heatmap (Key Layers)')
    plt.colorbar(im, ax=ax, label='Mean Absolute Difference per Token')
    plt.tight_layout()
    
    heatmap_path = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/per_token_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved per-token heatmap to: {heatmap_path}")

# ========== Summary ==========
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all(r["max_diff"] < 1e-3 for r in results if r["layer"] >= 0):
    print("✅ ALL LAYERS MATCH - No significant divergence detected")
elif all(r["max_diff"] < 1e-2 for r in results if r["layer"] >= 0):
    print("⚠️ SMALL DIVERGENCE - Differences within acceptable tolerance")
else:
    print("❌ SIGNIFICANT DIVERGENCE DETECTED")
    worst = max((r for r in results if r["layer"] >= 0), key=lambda x: x["max_diff"])
    print(f"   Worst layer: {worst['name']} (max diff: {worst['max_diff']:.6f})")

print("\n" + "="*80)
print("✓ COMPARISON COMPLETE")
print("="*80)
print(f"\nPlots generated:")
print(f"  Main divergence plot: {plot_path}")
if per_token_data:
    print(f"  Per-token heatmap:    {heatmap_path}")

