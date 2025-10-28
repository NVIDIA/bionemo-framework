#!/usr/bin/env python3
"""
Run NeMo model through ALL 32 layers sequentially, saving output after EACH layer.
This allows us to track where divergence occurs without loading both models together.

Unlike isolated layer scripts, this runs layers in sequence (letting errors compound).
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
out_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/nemo_sequential_layers")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("NEMO SEQUENTIAL LAYER EXECUTION (WITH ERROR COMPOUNDING)")
print("=" * 80)

# ========== Load Embeddings ==========
print("\n" + "-"*80)
print("LOADING EMBEDDINGS")
print("-"*80)

if not embeddings_path.exists():
    raise FileNotFoundError(
        f"Embeddings not found at {embeddings_path}\n"
        "You must run save_nemo_layer0_isolated.py first!"
    )

embeddings = torch.load(embeddings_path).to('cuda')
print(f"Loaded embeddings from: {embeddings_path}")
print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
print(f"  Note: NeMo uses sequence-first [seq, batch, hidden] format")

# Save embeddings as layer -1 (before any layers)
torch.save(embeddings.detach().cpu(), out_dir / "layer_-1_output.pt")
print(f"✓ Saved embeddings as layer -1")

# ========== Load NeMo Model ==========
print("\n" + "-"*80)
print("LOADING NEMO MODEL")
print("-"*80)

import nemo.lightning as nl
from nemo.lightning import NeMoLogger, io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model

# Setup trainer
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

# Disable KV cache
if hasattr(model, 'config'):
    model.config.use_cache = False

torch.cuda.empty_cache()

# ========== Access Model Components ==========
print("\n" + "-"*80)
print("ACCESSING MODEL COMPONENTS")
print("-"*80)

if hasattr(model, 'module'):
    megatron_model = model.module
    print(f"Found Megatron model: {type(megatron_model).__name__}")
else:
    raise AttributeError("Model not configured - no 'module' attribute found")

decoder = megatron_model.decoder
print(f"Decoder type: {type(decoder).__name__}")
print(f"Number of layers: {len(decoder.layers)}")

# ========== Compute RoPE ==========
print("\n" + "-"*80)
print("COMPUTING ROPE EMBEDDINGS")
print("-"*80)

if hasattr(megatron_model, 'rotary_pos_emb'):
    rotary_pos_emb_module = megatron_model.rotary_pos_emb
    print(f"RoPE module type: {type(rotary_pos_emb_module).__name__}")
    
    rotary_seq_len = embeddings.shape[0]  # sequence dimension
    rotary_pos_emb = rotary_pos_emb_module(rotary_seq_len)
    
    print(f"✓ RoPE computed for sequence length {rotary_seq_len}")
else:
    print("WARNING: No rotary_pos_emb module found!")
    rotary_pos_emb = None

# ========== Run Sequentially Through All Layers ==========
print("\n" + "-"*80)
print("RUNNING ALL LAYERS SEQUENTIALLY (SAVING AFTER EACH)")
print("-"*80)
print("This lets errors compound layer-to-layer (unlike isolated testing)")
print("-"*80)

with torch.no_grad():
    hidden_states = embeddings
    
    for i, layer in enumerate(decoder.layers):
        # Run layer
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
        
        # Save output after this layer
        output_path = out_dir / f"layer_{i}_output.pt"
        torch.save(hidden_states.detach().cpu(), output_path)
        
        print(f"✓ Layer {i:2d} complete - saved to {output_path.name}")
    
    print(f"\n✓ Processed all {len(decoder.layers)} layers")
    
    # Apply final layer norm and save
    final_output = decoder.final_layernorm(hidden_states)
    final_path = out_dir / "final_output.pt"
    torch.save(final_output.detach().cpu(), final_path)
    print(f"✓ Saved final output (after layer norm) to {final_path.name}")

# ========== Summary ==========
print("\n" + "-"*80)
print("SAVED FILES")
print("-"*80)

saved_files = sorted(out_dir.glob("*.pt"))
print(f"Total files: {len(saved_files)}")
print(f"  layer_-1_output.pt (embeddings)")
print(f"  layer_0_output.pt through layer_31_output.pt (after each layer)")
print(f"  final_output.pt (after final layer norm)")

print("\n" + "=" * 80)
print("✓ NEMO SEQUENTIAL EXECUTION COMPLETE")
print("=" * 80)
print("\nNext: Run compare_sequential_layers.py to see where divergence grows")

