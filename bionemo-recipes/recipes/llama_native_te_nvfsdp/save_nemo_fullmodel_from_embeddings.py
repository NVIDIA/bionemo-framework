#!/usr/bin/env python3
"""
Run full NeMo model (all 32 layers) starting from embeddings.
This runs: embeddings -> all layers -> final output.

IMPORTANT: 
1. Uses pre-saved embeddings from layer 0 outputs.
2. Loads model WITHOUT inference wrapper (per John's recommendation).
"""

import os
from pathlib import Path
import torch
import tempfile

# Memory hygiene
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.cuda.empty_cache()

# Paths
nemo_ckpt = "/workspaces/bionemo-framework/checkpoints/bcr_eden_checkpoint"
embeddings_path = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/layer0_nemo_isolated_outputs_nemo_checkpoint/embeddings.pt")
out_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/fullmodel_nemo_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("NEMO FULL MODEL EXECUTION")
print("=" * 80)

# ========== Load Embeddings ==========
print("\n" + "-"*80)
print("LOADING EMBEDDINGS")
print("-"*80)

if not embeddings_path.exists():
    raise FileNotFoundError(
        f"Embeddings not found at {embeddings_path}\n"
        "You must run save_nemo_layer0_isolated.py first to generate embeddings!"
    )

embeddings = torch.load(embeddings_path).to('cuda')
print(f"Loaded embeddings from: {embeddings_path}")
print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
print(f"  Note: NeMo uses sequence-first [seq, batch, hidden] format")

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

# Load model WITHOUT inference wrapper
print("Loading model config from checkpoint...")
model = io.load_context(path=ckpt_to_context_subdir(Path(nemo_ckpt)), subpath="model")
print(f"✓ Loaded config, model type: {type(model).__name__}")

print("Setting up trainer and restoring weights...")
_setup_trainer_and_restore_model(path=Path(nemo_ckpt), trainer=trainer, model=model)
print(f"✓ Weights restored")

print(f"Model loaded WITHOUT inference wrapper: {type(model).__name__}")

# Disable KV cache
if hasattr(model, 'config'):
    model.config.use_cache = False

# Clear GPU cache
torch.cuda.empty_cache()

# ========== Access Model Components ==========
print("\n" + "-"*80)
print("ACCESSING MODEL COMPONENTS")
print("-"*80)

# Access the Megatron module
if hasattr(model, 'module'):
    megatron_model = model.module
    print(f"Found Megatron model: {type(megatron_model).__name__}")
else:
    raise AttributeError("Model not configured - no 'module' attribute found")

decoder = megatron_model.decoder
print(f"Decoder type: {type(decoder).__name__}")
print(f"Number of layers: {len(decoder.layers)}")

# ========== Compute RoPE Embeddings ==========
print("\n" + "-"*80)
print("COMPUTING ROPE EMBEDDINGS")
print("-"*80)

if hasattr(megatron_model, 'rotary_pos_emb'):
    rotary_pos_emb_module = megatron_model.rotary_pos_emb
    print(f"RoPE module type: {type(rotary_pos_emb_module).__name__}")
    
    # Compute rotary embeddings for the sequence length
    # For sequence-first format [seq, batch, hidden], seq_len is dimension 0
    rotary_seq_len = embeddings.shape[0]
    rotary_pos_emb = rotary_pos_emb_module(rotary_seq_len)
    
    print(f"rotary_pos_emb computed: {type(rotary_pos_emb)}")
    if isinstance(rotary_pos_emb, tuple):
        print(f"  rotary_pos_emb[0] shape: {rotary_pos_emb[0].shape}")
        print(f"  rotary_pos_emb[1] shape: {rotary_pos_emb[1].shape}")
    else:
        print(f"  rotary_pos_emb shape: {rotary_pos_emb.shape}")
else:
    print("WARNING: No rotary_pos_emb module found!")
    rotary_pos_emb = None

# ========== Run Full Model ==========
print("\n" + "-"*80)
print("RUNNING FULL MODEL (ALL 32 LAYERS)")
print("-"*80)

with torch.no_grad():
    # Run through all decoder layers
    hidden_states = embeddings
    
    for i, layer in enumerate(decoder.layers):
        print(f"Processing layer {i}...", end='\r')
        layer_output = layer(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb
        )
        
        # Handle tuple output
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output
    
    print(f"\n✓ Processed all {len(decoder.layers)} layers")
    
    # Apply final layer norm
    final_output = decoder.final_layernorm(hidden_states)
    print(f"Final output shape: {final_output.shape}, dtype: {final_output.dtype}")
    
    # Save final output
    output_path = out_dir / "final_output.pt"
    torch.save(final_output.detach().cpu(), output_path)
    print(f"✓ Saved final output to: {output_path}")

print("\n" + "=" * 80)
print("✓ NEMO FULL MODEL EXECUTION COMPLETE")
print("=" * 80)

