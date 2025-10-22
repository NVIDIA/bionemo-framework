#!/usr/bin/env python3
"""
Compare isolated layer 0 outputs between NeMo and HuggingFace.
"""

import torch
from pathlib import Path

print("=" * 80)
print("COMPARING ISOLATED LAYER 0 OUTPUTS")
print("=" * 80)

nemo_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/layer0_nemo_isolated_outputs")
hf_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/layer0_hf_isolated_outputs")

# Mapping from NeMo submodule names to HF submodule names
# CORRECTED: Only compare outputs that exist at the same computation stage
nemo_to_hf = {
    # Attention path (in order):
    "self_attention_linear_qkv": "self_attention_layernorm_qkv",  # Fused LayerNorm + QKV projection
    "self_attention_core_attention_flash_attention": "self_attention_core_attention_flash_attention",  # Flash attention output
    "self_attention_linear_proj": "self_attention_proj",  # Attention output projection
    "self_attention": "self_attention",  # Final self-attention output (after proj + residual)
    
    # MLP path - ONLY final output (HF's LayerNormMLP is fully fused, no intermediates exposed):
    "mlp": "layernorm_mlp",  # Both are final MLP outputs (RMSNorm → FC1 → SwiGLU → FC2)
    
    # Final:
    "layer0_output": "layer0_output",  # Final layer 0 output (after both blocks)
}

# NOTE: The following NeMo outputs have NO equivalent in HF because HF's LayerNormMLP is fused:
# - "pre_mlp_layernorm": RMSNorm output before MLP (not exposed in HF)
# - "mlp_linear_fc1": FC1 output before SwiGLU (not exposed in HF)
# - "mlp_linear_fc2": FC2 output (same as "mlp" in NeMo, maps to "layernorm_mlp" in HF)

def compare_tensors(nemo_tensor, hf_tensor, name):
    """Compare two tensors, handling NeMo's sequence-first format."""
    result = {"name": name}
    
    # NeMo uses [seq, batch, hidden], HF uses [batch, seq, hidden]
    # Transpose NeMo tensor to match HF format ONLY if it's sequence-first
    # Sequence-first: [seq, batch, hidden] where seq >> batch (e.g. [1024, 1, 4096])
    # Batch-first: [batch, seq, hidden] where batch << seq (e.g. [1, 1024, 4096])
    if len(nemo_tensor.shape) == 3 and nemo_tensor.shape[0] > nemo_tensor.shape[1]:
        # shape[0] > shape[1] means sequence-first, needs transpose
        nemo_tensor_transposed = nemo_tensor.transpose(0, 1)
        result["shape_nemo_original"] = nemo_tensor.shape
        result["shape_nemo_transposed"] = nemo_tensor_transposed.shape
        result["shape_hf"] = hf_tensor.shape
        print(f"  📝 Transposing NeMo tensor from {nemo_tensor.shape} to {nemo_tensor_transposed.shape}")
        nemo_tensor = nemo_tensor_transposed
    else:
        result["shape_nemo"] = nemo_tensor.shape
        result["shape_hf"] = hf_tensor.shape
    
    # Check shapes match after transpose

    if nemo_tensor.shape != hf_tensor.shape:
        print(f"  📝 Shape mismatch: {nemo_tensor.shape} != {hf_tensor.shape}")
        print(f"  📝 NeMo tensor: {nemo_tensor}")
        print(f"  📝 HF tensor: {hf_tensor}")
        result["status"] = "SHAPE_MISMATCH"
        return result
    
    # Compare values
    diff = (nemo_tensor - hf_tensor).abs()
    result["max_diff"] = diff.max().item()
    result["mean_diff"] = diff.mean().item()
    result["median_diff"] = diff.median().item()
    
    # Find position of max diff
    max_diff_idx = diff.argmax()
    max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
    result["max_diff_pos"] = tuple(p.item() for p in max_diff_pos)
    
    # Get values at max diff position
    result["nemo_at_max"] = nemo_tensor[max_diff_pos].item()
    result["hf_at_max"] = hf_tensor[max_diff_pos].item()
    
    # Count matching vs non-matching elements at different tolerances
    exact_match = (diff == 0).sum().item()
    close_match = (diff < 1e-3).sum().item()
    total = diff.numel()
    
    result["exact_matches"] = exact_match
    result["close_matches"] = close_match
    result["total_elements"] = total
    result["exact_match_pct"] = 100.0 * exact_match / total
    result["close_match_pct"] = 100.0 * close_match / total
    
    # Pass/fail based on tolerances (adjusted for bfloat16 precision)
    if result["max_diff"] < 1e-3:
        result["status"] = "PASS"
    elif result["max_diff"] < 1e-2:
        result["status"] = "CLOSE"
    else:
        result["status"] = "FAIL"
    
    return result

print("\n" + "="*80)
print("COMPARING EMBEDDINGS")
print("="*80)

nemo_emb = torch.load(nemo_dir / "embeddings.pt")
hf_emb = torch.load(hf_dir / "embeddings.pt")

print(f"NeMo embeddings: {nemo_emb.shape} (sequence-first: [seq, batch, hidden])")
print(f"HF embeddings:   {hf_emb.shape} (batch-first: [batch, seq, hidden])")

emb_result = compare_tensors(nemo_emb, hf_emb, "embeddings")
if emb_result["status"] == "SHAPE_MISMATCH":
    print(f"❌ Shape mismatch!")
else:
    print(f"{'✅' if emb_result['status'] == 'PASS' else '⚠️'} Max diff: {emb_result['max_diff']:.6f}, Mean diff: {emb_result['mean_diff']:.6f}")
    if emb_result['max_diff'] > 0:
        print(f"  Max diff at position {emb_result['max_diff_pos']}: NeMo={emb_result['nemo_at_max']:.6f}, HF={emb_result['hf_at_max']:.6f}")
    print(f"  Exact matches: {emb_result['exact_matches']:,}/{emb_result['total_elements']:,} ({emb_result['exact_match_pct']:.2f}%)")

print("\n" + "="*80)
print("COMPARING LAYER 0 SUBMODULE OUTPUTS")
print("="*80)

results = {}

for nemo_name, hf_name in nemo_to_hf.items():
    # Find NeMo files
    nemo_pattern = f"layer0_{nemo_name.replace('.', '_')}_*.pt"
    nemo_files = sorted(nemo_dir.glob(nemo_pattern))
    
    # Find HF files
    hf_pattern = f"layer0_{hf_name.replace('.', '_')}_*.pt"
    hf_files = sorted(hf_dir.glob(hf_pattern))
    
    if not nemo_files:
        print(f"⚠️  {nemo_name:<50} - No NeMo file found")
        continue
    
    if not hf_files:
        print(f"⚠️  {nemo_name:<50} - No HF file found")
        continue
    
    # Compare first occurrence (usually there's only one per submodule)
    print(nemo_files[0], hf_files[0])
    nemo_tensor = torch.load(nemo_files[0])
    hf_tensor = torch.load(hf_files[0])
    
    result = compare_tensors(nemo_tensor, hf_tensor, nemo_name)
    results[nemo_name] = result
    
    status_symbol = "✅" if result["status"] == "PASS" else "⚠️" if result["status"] == "CLOSE" else "❌"
    if result["status"] == "SHAPE_MISMATCH":
        print(f"{status_symbol} {nemo_name:<50} - SHAPE MISMATCH: {result.get('shape_nemo_transposed', result.get('shape_nemo'))} vs {result.get('shape_hf')}")
    else:
        print(f"{status_symbol} {nemo_name:<50} - Max diff: {result['max_diff']:.6f}, Mean diff: {result['mean_diff']:.6f}")
        print(f"     Max diff at position {result['max_diff_pos']}: NeMo={result['nemo_at_max']:.6f}, HF={result['hf_at_max']:.6f}")
        print(f"     Exact matches: {result['exact_matches']:,}/{result['total_elements']:,} ({result['exact_match_pct']:.2f}%)")
        print(f"     Close matches (< 1e-3): {result['close_matches']:,}/{result['total_elements']:,} ({result['close_match_pct']:.2f}%)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

pass_count = sum(1 for r in results.values() if r["status"] == "PASS")
close_count = sum(1 for r in results.values() if r["status"] == "CLOSE")
fail_count = sum(1 for r in results.values() if r["status"] == "FAIL")
shape_mismatch_count = sum(1 for r in results.values() if r["status"] == "SHAPE_MISMATCH")

print(f"✅ PASS:  {pass_count}")
print(f"⚠️  CLOSE: {close_count}")
print(f"❌ FAIL:  {fail_count}")
print(f"❌ SHAPE MISMATCH: {shape_mismatch_count}")

if fail_count > 0 or shape_mismatch_count > 0:
    print("\n🔍 First divergence point:")
    for name, result in results.items():
        if result["status"] in ["FAIL", "SHAPE_MISMATCH"]:
            print(f"  → {name}")
            break

print("\n" + "=" * 80)
print("✓ COMPARISON COMPLETE")
print("=" * 80)
print("\nNOTE: HF's LayerNormMLP is fully fused, so the following NeMo outputs have no HF equivalent:")
print("  - pre_mlp_layernorm (RMSNorm output before MLP)")
print("  - mlp.linear_fc1 (FC1 output before SwiGLU)")
print("  - mlp.linear_fc2 (same as 'mlp', both map to HF's layernorm_mlp)")
print("\nOnly final outputs at equivalent computation stages can be meaningfully compared.")
print("=" * 80)
