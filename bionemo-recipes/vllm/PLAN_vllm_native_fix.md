# Plan: Long-term vLLM compatibility fix for NVEsm

## Problem

The `nvidia/esm2_*` checkpoints are saved from `NVEsmForMaskedLM`, which wraps
`NVEsmModel` inside `self.esm`. This means checkpoint weight keys have an `esm.`
prefix (e.g. `esm.embeddings.*`, `esm.encoder.*`).

When vLLM loads the model, it calls `AutoModel.from_config()` which returns
`NVEsmModel` directly (via `auto_map`). `NVEsmModel` has no `esm.` prefix in its
module tree. vLLM's generic `TransformersForEmbedding` wrapper then adds a
`model.` prefix, expecting weights like `model.embeddings.*`.

The default `hf_to_vllm_mapper` prepends `model.` to all checkpoint keys, turning
`esm.embeddings.*` into `model.esm.embeddings.*`, which doesn't exist.

Transformers v5 handles this via `base_model_prefix = "esm"` (line 234 of
`esm_nv.py`), but vLLM's weight loader doesn't respect `base_model_prefix`.

## Fix: Two changes needed in the HuggingFace model repository

### Change 1: Re-save the checkpoint without the `esm.` prefix

Re-save the safetensors files so weight keys match `NVEsmModel.state_dict()`:

```
esm.embeddings.word_embeddings.weight  ->  embeddings.word_embeddings.weight
esm.encoder.layers.0.*                 ->  encoder.layers.0.*
esm.encoder.emb_layer_norm_after.*     ->  encoder.emb_layer_norm_after.*
lm_head.*                              ->  (drop, or keep for MaskedLM users)
```

This makes the checkpoint directly compatible with both:

- Transformers v5 `AutoModel` (keys match `NVEsmModel.state_dict()` exactly)
- vLLM (mapper adds `model.` -> `model.embeddings.*` which matches the wrapper)

**Script to generate new checkpoint:**

```python
from safetensors.torch import load_file, save_file

weights = load_file("model.safetensors")
new_weights = {}
for name, tensor in weights.items():
    if name.startswith("esm."):
        new_weights[name[4:]] = tensor  # strip "esm." prefix
    # optionally keep lm_head.* for MaskedLM users
    # else:
    #     new_weights[name] = tensor
save_file(new_weights, "model.safetensors")
```

### Change 2: Update `NVEsmForMaskedLM` to load the new key format

Since `NVEsmForMaskedLM` has `self.esm = NVEsmModel(...)`, its state_dict keys
are `esm.embeddings.*`. After re-saving without the `esm.` prefix, loading into
`NVEsmForMaskedLM` would fail because the keys no longer match.

**Option A** (recommended): Override `_load_pretrained_model` or add custom
`state_dict`/`load_state_dict` logic in `NVEsmForMaskedLM` that re-adds `esm.`
when loading from the new format.

**Option B**: Ship two safetensors files (one for base model, one for MaskedLM)
using a safetensors index file.

**Option C**: Rename `self.esm` to `self.model` in `NVEsmForMaskedLM` to match
the HuggingFace convention (most models use `self.model`). Then both formats
align:

- Checkpoint keys: `model.embeddings.*` (or `embeddings.*` for base model)
- `NVEsmForMaskedLM.state_dict()`: `model.embeddings.*`
- `NVEsmModel.state_dict()`: `embeddings.*`
- vLLM wrapper: `model.embeddings.*`

This is the standard pattern used by most HF models (LLaMA, Mistral, etc.) and
would make vLLM work out of the box without any custom mapper.

### Change 3 (optional): Add vLLM weight mapper as safety net

Even with the checkpoint fix, adding a mapper to `esm_nv.py` provides a safety
net for older checkpoints. vLLM's transformers backend merges
`hf_to_vllm_mapper` attributes from model classes during `__init_subclass__`.

For trust_remote_code models wrapped in `TransformersForEmbedding`, this
requires vLLM to check the inner model for a mapper attribute. This may need a
vLLM-side change (check if the inner HF model has `hf_to_vllm_mapper` and merge
it). File a feature request or PR to vLLM if pursuing this route.

## Recommended approach

**Option C from Change 2** is the cleanest long-term fix:

1. Rename `self.esm` -> `self.model` in `NVEsmForMaskedLM` and
   `NVEsmForTokenClassification`
2. Update `base_model_prefix` from `"esm"` to `"model"`
3. Update `_tied_weights_keys` from `esm.embeddings...` to `model.embeddings...`
4. Re-save checkpoints with `model.` prefix (matching `NVEsmForMaskedLM`) or
   bare keys (matching `NVEsmModel`)
5. Both Transformers v5 and vLLM then work without any conversion scripts

This matches the pattern every modern HF model uses and requires zero special
handling in vLLM.
