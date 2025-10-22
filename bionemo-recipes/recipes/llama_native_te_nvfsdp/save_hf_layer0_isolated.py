#!/usr/bin/env python3
"""
Save HuggingFace Layer 0 outputs in ISOLATION (not full model forward pass).
This runs: tokens -> embeddings -> layer 0 only.
Designed for slurm execution and direct comparison with NeMo isolated layer 0.
"""

import os
from pathlib import Path
import torch
import re

# Memory hygiene
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.cuda.empty_cache()

# Paths
hf_ckpt = "/workspaces/bionemo-framework/checkpoints/bcr_eden_checkpoint_hf"
hardcoded_input = "/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/hardcoded_input.pt"
out_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/layer0_hf_isolated_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HF LAYER 0 ISOLATED EXECUTION")
print("=" * 80)

# ========== Load Input ==========
print("\n" + "-"*80)
print("LOADING INPUT")
print("-"*80)

blob = torch.load(hardcoded_input)
tokens = blob["tokens_tensor"]  # shape [B, T]

# Use full 1024 tokens (no memory issues with isolated layer 0)
MAX_SEQ_LEN = 1024
tokens = tokens[:, :MAX_SEQ_LEN]

print(f"Using tokens shape: {tokens.shape}, dtype: {tokens.dtype}")

# ========== Load HF Model ==========
print("\n" + "-"*80)
print("LOADING HF MODEL")
print("-"*80)

from model import NVLlamaForCausalLM

model = NVLlamaForCausalLM.from_pretrained(
    hf_ckpt,
    torch_dtype=torch.bfloat16,
)
model.eval()
model.config.use_cache = False
model = model.to("cuda")

print(f"Model type: {type(model).__name__}")

# Get layer 0 (NVLlamaModel is at model.model)
layer0 = model.model.layers[0]
print(f"Layer 0 type: {type(layer0).__name__}")

# CRITICAL: Update RoPE embeddings for the sequence length we're using
# NVLlamaDecoderLayer uses self.te_rope_emb which must match our sequence length
print(f"Layer 0 te_rope_emb shape: {layer0.te_rope_emb.shape if hasattr(layer0, 'te_rope_emb') else 'Not found'}")

# ========== Print Layer 0 Structure ==========
print("\n" + "-"*80)
print("LAYER 0 STRUCTURE")
print("-"*80)
print(f"\nLayer 0 named modules:")
for name, module in layer0.named_modules():
    if name == "":  # Skip the root module itself
        continue
    module_type = type(module).__name__
    # Count parameters
    num_params = sum(p.numel() for p in module.parameters())
    has_params = "✓" if num_params > 0 else " "
    print(f"  {has_params} {name:<50} {module_type:<30} ({num_params:>12,} params)")

print(f"\nLayer 0 direct children:")
for name, module in layer0.named_children():
    print(f"  - {name}: {type(module).__name__}")

print(f"\nLayer 0 parameters:")
for name, param in layer0.named_parameters():
    print(f"  - {name}: {param.shape}")

# ========== Setup Hooks ==========
print("\n" + "-"*80)
print("SETTING UP HOOKS")
print("-"*80)

handles = []
call_index = {}

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)

def make_hook(name):
    def hook(_mod, _inp, out):
        # Normalize output to a tensor (handle tuples/lists)
        x = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(x):
            return  # skip non-tensor outputs
        
        # Save full tensor
        idx = call_index.get(name, 0)
        call_index[name] = idx + 1
        fname = out_dir / f"layer0_{sanitize(name)}_{idx:02d}.pt"
        # IMPORTANT: Use .clone() to avoid capturing a view that might be modified later
        x_cpu = x.detach().clone().to("cpu")
        torch.save(x_cpu, fname)
        # Explicit cleanup
        del x_cpu
    return hook

# Define key modules to hook (but don't register yet - see workaround below)
key_modules = [
    ("self_attention.layernorm_qkv", layer0.self_attention.layernorm_qkv),
    ("self_attention.core_attention.flash_attention", layer0.self_attention.core_attention.flash_attention),
    ("self_attention.proj", layer0.self_attention.proj),
    ("self_attention", layer0.self_attention),
    ("layernorm_mlp", layer0.layernorm_mlp),
    ("layer0_output", layer0),
]

print(f"Identified {len(key_modules)} key layer 0 modules to hook")

# ========== Run Layer 0 in Isolation ==========
print("\n" + "-"*80)
print("RUNNING LAYER 0 IN ISOLATION")
print("-"*80)

# Move tokens to GPU
tokens_gpu = tokens.to('cuda')
batch_size, seq_len = tokens_gpu.shape

print(f"Input shape: {tokens_gpu.shape}")

with torch.no_grad():
    # Step 1: Generate embeddings
    # HF format: model.model.embed_tokens(tokens) -> [batch, seq, hidden]
    embeddings = model.model.embed_tokens(tokens_gpu)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    
    # Save embeddings for comparison
    torch.save(embeddings.detach().cpu(), out_dir / "embeddings.pt")
    print(f"Saved embeddings")
    
    # WORKAROUND: TE bug - flash attention outputs zeros if hooks registered before first run
    # This only affects HF/NVLlama model in script execution (NeMo doesn't have this issue)
    # Solution: Run layer0 once to initialize TE, then register hooks for actual capture
    print("\n[Workaround] Initializing TE by running layer0 once...")
    _ = layer0(embeddings, attention_mask=None)
    
    # NOW register hooks for the actual capture run
    print("[Workaround] Registering hooks for capture...")
    for name, module in key_modules:
        handles.append(module.register_forward_hook(make_hook(name)))
    print(f"[Workaround] Registered {len(handles)} hooks\n")
    
    # Step 2: Let TE create the causal mask internally
    # According to TE docs: attention_mask should be None for causal masks
    # self_attn_mask_type='causal' is the default, which creates the mask internally
    print(f"Using TE internal causal masking (attention_mask=None)")
    
    # Step 3: Run through layer 0 ONLY
    # NVLlamaDecoderLayer signature: forward(hidden_states, attention_mask=None)
    # TE will create causal mask internally with self_attn_mask_type='causal' (default)
    layer0_output = layer0(embeddings, attention_mask=None)
    
    # Handle tuple output
    if isinstance(layer0_output, tuple):
        layer0_output = layer0_output[0]
    
    print(f"Layer 0 output shape: {layer0_output.shape}, dtype: {layer0_output.dtype}")
    
    # Note: The hook on layer0_output will have saved this already

print("\n✓ Layer 0 execution complete")

# ========== Cleanup ==========
print("\n" + "-"*80)
print("CLEANUP")
print("-"*80)

for handle in handles:
    handle.remove()
print(f"Removed {len(handles)} hooks")

# ========== List Saved Files ==========
print("\n" + "-"*80)
print("SAVED FILES")
print("-"*80)

saved_files = sorted(out_dir.glob("*.pt"))
print(f"Total files saved: {len(saved_files)}")
for f in saved_files:
    tensor = torch.load(f)
    print(f"  {f.name}: {tensor.shape}")

print("\n" + "=" * 80)
print("✓ HF LAYER 0 ISOLATED EXECUTION COMPLETE")
print("=" * 80)

