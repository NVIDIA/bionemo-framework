#!/usr/bin/env python3
"""
Save NeMo Layer 0 outputs in ISOLATION (not full model forward pass).
This runs: tokens -> embeddings -> layer 0 only.

IMPORTANT: Loads model WITHOUT inference wrapper (per John's recommendation).
The inference wrapper may change the attention mechanism, so we use the raw model.

Designed for slurm execution and direct comparison with HF isolated layer 0.
"""

import os
from pathlib import Path
import torch
import tempfile
import re

# Memory hygiene
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.cuda.empty_cache()

# Paths
nemo_ckpt = "/workspaces/bionemo-framework/checkpoints/bcr_eden_checkpoint"
hardcoded_input = "/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/hardcoded_input.pt"
out_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/layer0_nemo_isolated_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("NEMO LAYER 0 ISOLATED EXECUTION")
print("=" * 80)

# ========== Load Input ==========
print("\n" + "-"*80)
print("LOADING INPUT")
print("-"*80)

blob = torch.load(hardcoded_input)
tokens = blob["tokens_tensor"]  # shape [B, T]

# Use full 1024 tokens (isolated layer 0 is memory efficient)
MAX_SEQ_LEN = 1024
tokens = tokens[:, :MAX_SEQ_LEN]

print(f"Using tokens shape: {tokens.shape}, dtype: {tokens.dtype}")

# ========== Load NeMo Model ==========
print("\n" + "-"*80)
print("LOADING NEMO MODEL")
print("-"*80)

import nemo.lightning as nl
from nemo.lightning import NeMoLogger, io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model

# Setup trainer with MegatronStrategy
work_dir = Path(tempfile.mkdtemp())
nemo_logger = NeMoLogger(log_dir=work_dir)

trainer = nl.Trainer(
    devices=1,
    accelerator="gpu",
    strategy=nl.MegatronStrategy(),
    enable_checkpointing=False,
    logger=nemo_logger,
)

# Load model WITHOUT inference wrapper (John's recommendation)
# This avoids potential attention mechanism changes introduced by the wrapper
print("Loading model config from checkpoint...")
model = io.load_context(path=ckpt_to_context_subdir(Path(nemo_ckpt)), subpath="model")
print(f"✓ Loaded config, model type: {type(model).__name__}")

print("Setting up trainer and restoring weights...")
_setup_trainer_and_restore_model(path=Path(nemo_ckpt), trainer=trainer, model=model)
print(f"✓ Weights restored")

# No wrapper! Model is accessed directly
print(f"Model loaded WITHOUT inference wrapper: {type(model).__name__}")
tokenizer = model.tokenizer
print(f"Tokenizer: {type(tokenizer).__name__}")

# Disable KV cache
if hasattr(model, 'config'):
    model.config.use_cache = False
    print(f"  Disabled KV cache on {type(model).__name__}.config")

print("✓ Model loaded (no inference wrapper)")

# Clear GPU cache
torch.cuda.empty_cache()
print("✓ GPU cache cleared")

# ========== Get Layer 0 ==========
print("\n" + "-"*80)
print("ACCESSING LAYER 0")
print("-"*80)

# Access the Megatron module (actual transformer)
# Without wrapper: model.module contains the actual GPTModel from Megatron
if hasattr(model, 'module'):
    megatron_model = model.module
    print(f"Found Megatron model: {type(megatron_model).__name__}")
else:
    raise AttributeError("Model not configured - no 'module' attribute found")

# Access decoder and layer 0
decoder = megatron_model.decoder
layer0 = decoder.layers[0]

print(f"Decoder type: {type(decoder).__name__}")
print(f"Layer 0 type: {type(layer0).__name__}")

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
        # Note: NeMo uses sequence-first format [seq_len, batch, hidden]
        idx = call_index.get(name, 0)
        call_index[name] = idx + 1
        fname = out_dir / f"layer0_{sanitize(name)}_{idx:02d}.pt"
        # Move to CPU immediately and save
        x_cpu = x.detach().to("cpu")
        torch.save(x_cpu, fname)
        # Explicit cleanup
        del x_cpu
    return hook

# Hook every submodule of layer 0
for n, m in layer0.named_modules():
    if n == "":
        continue
    handles.append(m.register_forward_hook(make_hook(n)))

# Also hook the layer0 output
handles.append(layer0.register_forward_hook(make_hook("layer0_output")))

print(f"Registered {len(handles)} hooks on layer 0 submodules")

# ========== Run Layer 0 in Isolation ==========
print("\n" + "-"*80)
print("RUNNING LAYER 0 IN ISOLATION")
print("-"*80)

# Move tokens to GPU
tokens_gpu = tokens.to('cuda')
batch_size, seq_len = tokens_gpu.shape

# Create position_ids (NeMo needs this for RoPE)
position_ids = torch.arange(seq_len, device=tokens_gpu.device).unsqueeze(0).expand(batch_size, -1)

print(f"Input shape: {tokens_gpu.shape}")
print(f"Position IDs shape: {position_ids.shape}")

with torch.no_grad():
    # Step 1: Generate embeddings
    # NeMo format: megatron_model.embedding(tokens, position_ids) -> [seq, batch, hidden]
    embeddings = megatron_model.embedding(tokens_gpu, position_ids)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"  Note: NeMo uses sequence-first [seq, batch, hidden] format")
    
    # Save embeddings for comparison
    torch.save(embeddings.detach().cpu(), out_dir / "embeddings.pt")
    print(f"Saved embeddings")
    
    # Step 2: Let TE create the causal mask internally
    # According to TE docs: attention_mask should be None for causal masks
    # self_attn_mask_type='causal' is the default, which creates the mask internally
    print(f"Using TE internal causal masking (attention_mask=None)")
    
    # Step 3: Run through layer 0 ONLY
    # NeMo TransformerLayer signature: forward(hidden_states, attention_mask=None, rotary_pos_emb, ...)
    # TE will create causal mask internally with self_attn_mask_type='causal' (default)
    # Note: position_ids are NOT passed to individual layers, RoPE is computed internally
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
print("✓ NEMO LAYER 0 ISOLATED EXECUTION COMPLETE")
print("=" * 80)

