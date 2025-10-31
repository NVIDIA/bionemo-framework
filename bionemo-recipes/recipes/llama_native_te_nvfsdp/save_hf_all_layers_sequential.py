#!/usr/bin/env python3
"""
Run HF model through ALL 32 layers sequentially, saving output after EACH layer.
This allows us to track where divergence occurs without loading both models together.

Unlike isolated layer scripts, this runs layers in sequence (letting errors compound).
"""

import os
from pathlib import Path
import torch

# Memory hygiene
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.cuda.empty_cache()

# Paths
hf_ckpt = "/workspaces/bionemo-framework/checkpoints/bcr_eden_checkpoint_hf"
embeddings_path = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/layer0_hf_isolated_outputs/embeddings.pt")
out_dir = Path("/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/hf_sequential_layers")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HF SEQUENTIAL LAYER EXECUTION (WITH ERROR COMPOUNDING)")
print("=" * 80)

# ========== Load Embeddings ==========
print("\n" + "-"*80)
print("LOADING EMBEDDINGS")
print("-"*80)

if not embeddings_path.exists():
    raise FileNotFoundError(
        f"Embeddings not found at {embeddings_path}\n"
        "You must run save_hf_layer0_isolated.py first!"
    )

embeddings = torch.load(embeddings_path).to('cuda')
print(f"Loaded embeddings from: {embeddings_path}")
print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

# Save embeddings as layer -1 (before any layers)
torch.save(embeddings.detach().cpu(), out_dir / "layer_-1_output.pt")
print(f"✓ Saved embeddings as layer -1")

# ========== Load HF Model ==========
print("\n" + "-"*80)
print("LOADING HF MODEL WITH NOOP_CAT PATCH")
print("-"*80)

from model import NVLlamaForCausalLM
import transformer_engine.pytorch.module._common as te_common
te_common.noop_cat = lambda tensors, dim=0: torch.cat(tensors, dim=dim).contiguous()
from transformer_engine.pytorch.module import layernorm_linear
layernorm_linear.noop_cat = lambda tensors, dim=0: torch.cat(tensors, dim=dim).contiguous()
print("✓ Applied noop_cat patch")

model = NVLlamaForCausalLM.from_pretrained(
    hf_ckpt,
    torch_dtype=torch.bfloat16,
)
model.eval()
model.config.use_cache = False
model = model.to("cuda")

print(f"Model type: {type(model).__name__}")
print(f"Number of layers: {len(model.model.layers)}")

# ========== Run Sequentially Through All Layers ==========
print("\n" + "-"*80)
print("RUNNING ALL LAYERS SEQUENTIALLY (SAVING AFTER EACH)")
print("-"*80)
print("This lets errors compound layer-to-layer (unlike isolated testing)")
print("-"*80)

with torch.no_grad():
    hidden_states = embeddings
    
    for i, layer in enumerate(model.model.layers):
        # Run layer
        layer_output = layer(hidden_states, attention_mask=None)
        
        # Handle tuple output
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output
        
        # Save output after this layer
        output_path = out_dir / f"layer_{i}_output.pt"
        torch.save(hidden_states.detach().cpu(), output_path)
        
        print(f"✓ Layer {i:2d} complete - saved to {output_path.name}")
    
    print(f"\n✓ Processed all {len(model.model.layers)} layers")
    
    # Apply final layer norm and save
    final_output = model.model.norm(hidden_states)
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
print("✓ HF SEQUENTIAL EXECUTION COMPLETE")
print("=" * 80)
print("\nNext: Run save_nemo_all_layers_sequential.py")
print("Then: Run compare_sequential_layers.py to see where divergence grows")

