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
    
    # Count positions exceeding thresholds
    total_elements = diff.numel()
    exceed_1e3 = (diff >= 1e-3).sum().item()
    exceed_1e2 = (diff >= 1e-2).sum().item()
    exceed_1e1 = (diff >= 1e-1).sum().item()
    
    # Find top 20 divergent positions (flatten and get top-k)
    diff_flat = diff.flatten()
    top20_vals, top20_indices = diff_flat.topk(20)
    
    # Convert flat indices back to (batch, seq, hidden) positions
    top20_positions = []
    for idx in top20_indices:
        pos = torch.unravel_index(idx, diff.shape)
        top20_positions.append({
            "batch": pos[0].item(),
            "token": pos[1].item(),
            "hidden": pos[2].item(),
            "diff": diff[pos].item(),
            "nemo_val": nemo_tensor[pos].item(),
            "hf_val": hf_tensor[pos].item(),
        })
    
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "median_diff": diff.median().item(),
        "close_1e-3_pct": 100.0 * (diff < 1e-3).sum().item() / total_elements,
        "close_1e-2_pct": 100.0 * (diff < 1e-2).sum().item() / total_elements,
        "close_1e-1_pct": 100.0 * (diff < 1e-1).sum().item() / total_elements,
        # Counts of positions exceeding thresholds
        "exceed_1e-3_count": exceed_1e3,
        "exceed_1e-2_count": exceed_1e2,
        "exceed_1e-1_count": exceed_1e1,
        "exceed_1e-3_pct": 100.0 * exceed_1e3 / total_elements,
        "exceed_1e-2_pct": 100.0 * exceed_1e2 / total_elements,
        "exceed_1e-1_pct": 100.0 * exceed_1e1 / total_elements,
        "total_elements": total_elements,
        # Per-token metrics
        "per_token_mean": per_token_diff.mean().item(),
        "per_token_max": per_token_diff.max().item(),
        "per_token_max_pos": per_token_diff.argmax().item(),
        "per_token_diff": per_token_diff,  # Full per-token array
        # Top 20 divergent positions
        "top20_positions": top20_positions,
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
    print(f"         Positions exceeding: >1e-3: {result['exceed_1e-3_count']:,} ({result['exceed_1e-3_pct']:.2f}%), "
          f">1e-2: {result['exceed_1e-2_count']:,} ({result['exceed_1e-2_pct']:.2f}%), "
          f">1e-1: {result['exceed_1e-1_count']:,} ({result['exceed_1e-1_pct']:.2f}%)")
    
    results.append({"layer": i, "name": f"Layer {i}", **result})

# Compare final output
nemo_final = torch.load(nemo_dir / "final_output.pt")
hf_final = torch.load(hf_dir / "final_output.pt")
final_result = compare_tensors(nemo_final, hf_final)
if final_result:
    status = "✅" if final_result["max_diff"] < 1e-3 else "⚠️" if final_result["max_diff"] < 1e-2 else "❌"
    print(f"Final (after LN): {status} max={final_result['max_diff']:.6f}, mean={final_result['mean_diff']:.6f}")
    results.append({"layer": 32, "name": "Final", **final_result})

# ========== Detailed Final Output Analysis ==========
print("\n" + "="*80)
print("DETAILED FINAL OUTPUT ANALYSIS")
print("="*80)

# Transpose NeMo if needed
if nemo_final.shape[0] > nemo_final.shape[1]:
    nemo_final_t = nemo_final.transpose(0, 1)
else:
    nemo_final_t = nemo_final

# Compute additional metrics
diff_final = (nemo_final_t - hf_final).abs()

print(f"\nFinal hidden states (after layer norm, before LM head):")
print(f"  Shape: {hf_final.shape}")
print(f"  Mean absolute difference: {diff_final.mean().item():.6f}")
print(f"  Max absolute difference:  {diff_final.max().item():.6f}")

# Cosine similarity (flatten tensors to compute overall similarity)
nemo_flat = nemo_final_t.flatten()
hf_flat = hf_final.flatten()
cosine_sim = torch.nn.functional.cosine_similarity(
    nemo_flat.unsqueeze(0), 
    hf_flat.unsqueeze(0), 
    dim=1
).item()

print(f"  Cosine similarity:        {cosine_sim:.8f}")

# Per-token cosine similarity (average over hidden dimension)
# Reshape to [batch*seq, hidden] and compute per-token
nemo_reshaped = nemo_final_t.reshape(-1, nemo_final_t.shape[-1])  # [batch*seq, hidden]
hf_reshaped = hf_final.reshape(-1, hf_final.shape[-1])

# Compute cosine similarity more carefully with explicit normalization
nemo_norm = torch.nn.functional.normalize(nemo_reshaped, p=2, dim=1)
hf_norm = torch.nn.functional.normalize(hf_reshaped, p=2, dim=1)
per_token_cosine = (nemo_norm * hf_norm).sum(dim=1)  # Dot product of normalized vectors

# Clamp to valid range [-1, 1] in case of numerical errors
per_token_cosine = torch.clamp(per_token_cosine, -1.0, 1.0)

print(f"\nPer-token cosine similarity:")
print(f"  Min:    {per_token_cosine.min().item():.8f}")
print(f"  Median: {per_token_cosine.median().item():.8f}")
print(f"  Mean:   {per_token_cosine.mean().item():.8f}")
print(f"  Max:    {per_token_cosine.max().item():.8f}")

# Check how many are very close to 1.0 (nearly identical directions)
very_similar = (per_token_cosine > 0.9999).sum().item()
print(f"  Tokens with cosine > 0.9999: {very_similar:,} / {len(per_token_cosine):,} ({100.0*very_similar/len(per_token_cosine):.2f}%)")

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
        print(f"{result['name']:15s}: max={result['max_diff']:.6f}, mean={result['mean_diff']:.6f}, "
              f"positions >1e-2: {result.get('exceed_1e-2_count', 0):,} ({result.get('exceed_1e-2_pct', 0):.2f}%)")

# ========== Create Plots ==========
print("\n" + "-"*80)
print("GENERATING PLOTS")
print("-"*80)

layer_nums = [r["layer"] for r in results if r["layer"] >= 0]
max_diffs = [r["max_diff"] for r in results if r["layer"] >= 0]
mean_diffs = [r["mean_diff"] for r in results if r["layer"] >= 0]
per_token_means = [r["per_token_mean"] for r in results if r["layer"] >= 0]
exceed_1e3_pcts = [r["exceed_1e-3_pct"] for r in results if r["layer"] >= 0]
exceed_1e2_pcts = [r["exceed_1e-2_pct"] for r in results if r["layer"] >= 0]
exceed_1e1_pcts = [r["exceed_1e-1_pct"] for r in results if r["layer"] >= 0]

# Main divergence plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

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

# Plot 4: Percentage of positions exceeding thresholds
ax4.plot(layer_nums, exceed_1e3_pcts, 'o-', label='Positions > 1e-3', linewidth=2, markersize=4, color='red')
ax4.plot(layer_nums, exceed_1e2_pcts, 's-', label='Positions > 1e-2', linewidth=2, markersize=4, color='orange')
ax4.plot(layer_nums, exceed_1e1_pcts, '^-', label='Positions > 1e-1', linewidth=2, markersize=4, color='darkred')
ax4.set_xlabel('Layer Number')
ax4.set_ylabel('Percentage of Positions Exceeding Threshold')
ax4.set_title('Growth of Divergent Positions Across Layers')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(bottom=0)

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

# Top divergent positions tracking
print("\nGenerating top divergent positions plot...")
fig3, ax = plt.subplots(figsize=(14, 10))

# Plot top 20 positions for each layer
for layer_idx, r in enumerate(results):
    if r["layer"] >= 0 and "top20_positions" in r:
        layer_num = r["layer"]
        # Plot token positions of top 20 divergent elements
        token_positions = [p["token"] for p in r["top20_positions"][:20]]
        diffs = [p["diff"] for p in r["top20_positions"][:20]]
        
        # Use color to show divergence magnitude
        if len(diffs) > 0 and max(diffs) > min(diffs):
            colors = plt.cm.hot([(d - min(diffs)) / (max(diffs) - min(diffs)) for d in diffs])
        else:
            colors = ['red'] * len(diffs)
        ax.scatter([layer_num] * len(token_positions), token_positions, 
                  c=colors, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Layer Number', fontsize=12)
ax.set_ylabel('Token Position', fontsize=12)
ax.set_title('Top 20 Divergent Token Positions Per Layer\n(Color intensity = divergence magnitude)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 33)
ax.set_ylim(-10, 1034)

plt.tight_layout()
top_pos_path = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/top_divergent_positions.png")
plt.savefig(top_pos_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved top divergent positions plot to: {top_pos_path}")

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
print(f"  Main divergence plot (4 panels):     {plot_path}")
if per_token_data:
    print(f"  Per-token heatmap:                   {heatmap_path}")
print(f"  Top divergent positions:             {top_pos_path}")

