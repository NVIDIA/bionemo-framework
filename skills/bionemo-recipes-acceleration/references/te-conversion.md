# Transformer Engine conversion

Phase 4. Pick one depth based on what the target code already has, then follow the matching
reference implementation exactly. Do not blend depths.

## Choosing the depth

| Condition                                                                                                                                                                                   | Depth                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Code already constructs `te.TransformerLayer`                                                                                                                                               | **A** — config layer only         |
| HF-style transformer blocks, pre-norm, no TE                                                                                                                                                | **B** — full port + converter     |
| HF-style transformer blocks, post-norm, no TE                                                                                                                                               | **B-postnorm** — hand-built block |
| Two stacks with cross-attention (T5-style)                                                                                                                                                  | **B-encdec** — two TE stacks      |
| Block cannot become a TE layer (structural quirk, non-dot-product attention mechanism, or no TE analogue for the sequence-mixing core) but has clean `nn.Linear` / norm / attention sub-ops | **C** — kernel swaps only         |

Depth C is the landing spot for a target whose block has a structural quirk (an extra gate, an
unusual residual, exotic MoE routing) or an attention mechanism that isn't maskable scaled-dot-product
attention at all (linear/kernelized attention, retrieval-augmented attention, block-sparse routing).
Causal vs bidirectional, sliding window, and custom masks are not Depth C triggers — those are
`self_attn_mask_type` / `window_size` config on `te.TransformerLayer`, see
`references/architecture-matching.md`. Depth C is also the landing spot for diffusion,
GNN/equivariant, and state-space architectures:
the sequence-mixing core (a scan, message passing, denoiser conditioning) stays custom, but the
projections, norms, MLPs, and — where a bare attention core fits — the attention computation itself
still swap. It is a real outcome, not a consolation prize: the report says plainly that no TE
transformer block was substituted, and lists exactly which wins were applied. Only a target with no
runnable forward pass at all — so neither static source nor a probing run over the live module tree
can recover the architecture — produces no port at all; see "No static source: probe the live model
before giving up" in `references/architecture-matching.md`.

### Depth B-postnorm

`te.TransformerLayer` is pre-norm. When the target applies LayerNorm *after* the residual add, do
not try to configure your way out of it — assemble the block by hand, exactly as
`$BIONEMO_RECIPES/models/geneformer/src/geneformer/modeling_bert_te.py::TEBertLayer` does:
`te.MultiheadAttention` with `input_layernorm=False`, a `te.LayerNorm` after the attention residual,
`te.Linear` → activation → `te.Linear` for the MLP, and a second `te.LayerNorm` after the MLP
residual.

Two traps:

- **`input_layernorm=False` is load-bearing.** Leave it at the default and you get a pre-norm block
  wearing post-norm parameter names, which passes a smoke test and fails golden values.
- **Parameter naming must match the converter.** Geneformer names them `layernorm`,
  `layernorm_mlp.fc1`, `layernorm_mlp.activation`, `layernorm_mlp.fc2`, `layernorm_mlp.layer_norm`
  precisely so `$BIONEMO_RECIPES/models/geneformer/src/geneformer/convert.py` can map HF BERT
  checkpoints onto it. Pick names before you write the converter, not after.

### Depth B-encdec

Build the encoder stack as in Depth B, then the decoder stack with `layer_type="decoder"` — TE adds
an `inter_attention` cross-attention block after self-attn and takes `encoder_output` and
`enc_dec_attn_mask` in the forward. See
`$BIONEMO_RECIPES/recipes/codonfm_ptl_te/src/models/components/encodon_te_layer.py` for the
construction and the forward signature.

Set `self_attn_mask_type` causal on the decoder and `enc_dec_attn_mask_type` for the cross path. The
converter needs a hand-written transform for `inter_attention` — Q comes from the decoder hidden
states and KV from the encoder output, so `$BIONEMO_RECIPES/models/esm2/convert.py::_pack_qkv_weight`
does not apply unchanged. No in-repo model exercises this path, so Tier 1 parity is the primary
proof and every gap goes in the report.

Depth C is specified in full under "Depth C — targeted kernel swaps" below.

______________________________________________________________________

## Depth A — precision configuration layer

The code already runs TE layers. Add three things.

### 1. Dotted-path recipe config

The reason recipes are configured as dotted class paths is that switching FP8 → MXFP8 → NVFP4
becomes a one-line change. Copy the block shape from
`$BIONEMO_RECIPES/recipes/esm2_native_te/hydra_config/defaults.yaml`:

```yaml
fp8_config:
  enabled: false
  fp8_recipe: transformer_engine.common.recipe.DelayedScaling
  fp8_format: "HYBRID"
  fp8_recipe_kwargs: {}

fp4_config:
  enabled: false
  fp4_recipe: transformer_engine.common.recipe.NVFP4BlockScaling
  fp4_format: "E2M1"
  fp4_recipe_kwargs: {}

# 1-indexed layer lists, converted to 0-indexed at runtime; null means "all layers"
fp8_layers: null
fp4_layers: null
```

Resolved in the training script exactly as `$BIONEMO_RECIPES/recipes/esm2_native_te/train_ddp.py` does:

```python
from transformer_engine.common.recipe import Format

fp8_recipe = None
if args.fp8_config.enabled:
    fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
        fp8_format=Format[args.fp8_config.fp8_format],
        **args.fp8_config.fp8_recipe_kwargs
    )
```

If the target does not use Hydra, replace `hydra.utils.get_class` with `importlib`-based resolution
of the same dotted string — keep the string form, it is the whole point.

### 2. Layer-precision resolution

Port `$BIONEMO_RECIPES/recipes/esm2_native_te/quantization.py::resolve_layer_precision`. It maps
`(num_layers, fp8_enabled, fp4_enabled, fp8_layers, fp4_layers)` to a `list[str | None]` of
`"fp8" | "fp4" | None` per layer, stored as `config.layer_precision`. Its rules, which you must
preserve: an enabled precision with no layer list claims all remaining layers; overlapping lists
raise `ValueError`; both enabled with neither list raises `ValueError`.

### 3. Autocast contexts

Follow `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py::NVEsmModel.get_autocast_context`. The critical detail:

> **`DelayedScaling` needs one *outer* autocast wrapping the entire stack, once per forward pass**,
> so its amax-history post-processing runs exactly once. Per-layer contexts alone are not enough.

The ESM2 shape — outer context plus per-layer contexts, both owned by the model:

```python
def get_autocast_context(self, layer_number, init=False, outer=False) -> ContextManager:
    if self.config.layer_precision is None:
        return nullcontext()
    if outer:
        if "fp8" not in self.config.layer_precision:
            return nullcontext()
        return transformer_engine.pytorch.autocast(
            enabled=True, recipe=self._fp8_recipe
        )
    precision = self.config.layer_precision[layer_number]
    recipe = {"fp8": self._fp8_recipe, "fp4": self._fp4_recipe}.get(precision)
    if init and self.config.use_quantized_model_init:
        if precision in ("fp8", "fp4"):
            return transformer_engine.pytorch.quantized_model_init(recipe=recipe)
        return nullcontext()
    if precision in ("fp8", "fp4"):
        return transformer_engine.pytorch.autocast(enabled=True, recipe=recipe)
    return transformer_engine.pytorch.autocast(enabled=False)
```

The Llama3 variant (`$BIONEMO_RECIPES/models/llama3/modeling_llama_te.py::NVLlamaModel.get_layer_autocast`) inverts
the ownership: FP8 layers return `nullcontext()` so an outer autocast *in the training script*
applies, and FP4 layers override. Use this shape when the target shards with FSDP, because recipe
objects are not serializable and must be attached **after** sharding — hence
`NVLlamaModel.set_recipes(fp8_recipe, fp4_recipe)`, called from
`$BIONEMO_RECIPES/recipes/llama3_native_te/train_fsdp2.py` after the model is wrapped.

Pick one ownership model and be consistent. Mixing them double-applies autocast.

Use `transformer_engine.pytorch.autocast(enabled=..., recipe=...)`, not the legacy
`te.fp8_autocast(fp8_recipe=...)`.

______________________________________________________________________

## Depth B — full port to `te.TransformerLayer`

### Build the layer

Mirror the family's reference. For causal LMs, `$BIONEMO_RECIPES/models/llama3/modeling_llama_te.py`:

```python
transformer_engine.pytorch.TransformerLayer(
    hidden_size=config.hidden_size,
    ffn_hidden_size=config.intermediate_size,
    num_attention_heads=config.num_attention_heads,
    layernorm_epsilon=config.rms_norm_eps,
    hidden_dropout=0,
    attention_dropout=0,
    fuse_qkv_params=True,
    qkv_weight_interleaved=True,
    normalization="RMSNorm",
    activation="swiglu",
    attn_input_format=config.attn_input_format,  # "bshd" or "thd"
    self_attn_mask_type=config.self_attn_mask_type,  # "causal" for decoder, "padding" for encoder
    num_gqa_groups=config.num_key_value_heads,
    layer_number=layer_idx + 1,  # 1-indexed, TE convention
    params_dtype=config.dtype,
    device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
    init_method=_init_method,
    output_layer_init_method=_init_method,
)
```

For encoders, `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py` uses `normalization="LayerNorm"`,
`activation="gelu"`, and no `num_gqa_groups`.

Rotary embeddings: use TE's `RotaryPositionEmbedding`, but **seed `inv_freq` from the HF module** so
numerics match the original —

```python
self.rotary_emb = RotaryPositionEmbedding(
    config.hidden_size // config.num_attention_heads
)
self.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=config).inv_freq
```

Skipping this is the single most common cause of a port that trains but does not match golden
values.

### TE-specific config knobs

From `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py::NVEsmConfig`:

- `fuse_qkv_params` / `qkv_weight_interleaved` — fused QKV. Changes the weight layout, so the
  converter must pack accordingly.
- `attn_input_format` — `"bshd"` or `"thd"`. See `references/sequence-packing.md`.
- `padded_vocab_size` — vocab padded to a multiple of 64 so FP8 GEMMs align. Load-bearing for FP8
  performance; the converter must pad and unpad.
- `micro_batch_size` / `max_seq_length` — used for JIT warmup.
- `layer_precision`, `use_quantized_model_init` — from Depth A.

### Generate the converter

Model it on `$BIONEMO_RECIPES/models/esm2/convert.py`. Three parts:

1. **A flat wildcard mapping dict**, HF key → TE key, e.g.
   `esm.encoder.layer.*.attention.LayerNorm.weight` →
   `model.encoder.layers.*.self_attention.layernorm_qkv.layer_norm_weight`, and
   `...intermediate.dense.weight` → `...layernorm_mlp.fc1_weight`. Build the reverse with
   `{v: k for k, v in mapping.items()}`.
2. **`@state.state_transform`-decorated functions** for anything not a rename:
   `_pack_qkv_weight` / `_pack_qkv_bias` (interleaved head-major QKV fusion) and their
   `_unpack_*` inverses; `_pad_weights` / `_pad_bias` (vocab padding — bias padded with
   `torch.finfo(dtype).min`, not zero) and their `_unpad_*` inverses.
3. **Two entry points**: `convert_<model>_hf_to_te(model_hf, **config_kwargs)` and
   `convert_<model>_te_to_hf(model_te, **config_kwargs)`.

Vendor `$BIONEMO_RECIPES/models/esm2/state.py` into the target — it is the self-contained transform engine
(`apply_transforms`, `TransformCTX`, `state_transform`, `TransformFns`) that the mapping relies on,
adapted from `nemo.lightning.io.state`.

Family-specific converter references: `$BIONEMO_RECIPES/models/llama3/convert.py`, `$BIONEMO_RECIPES/models/qwen/convert_qwen2.py`,
`$BIONEMO_RECIPES/models/qwen/convert_qwen3.py`, `$BIONEMO_RECIPES/models/mixtral/convert.py`.

### Preserve HuggingFace compatibility

From `$BIONEMO_RECIPES/models/esm2/modeling_esm_te.py`:

- Set `auto_map` on the config so `trust_remote_code=True` loading works.
- `init_empty_weights()` must call `reset_parameters()` on every TE module and `_init_weights` only
  on non-TE modules — TE modules do not initialize through the HF path.
- `state_dict()` must strip `_extra_state` keys (TE quantization state) and `.inv_freq` (a derived
  buffer). Leaving them in breaks checkpoint round-trips and `test_convert_te_to_hf_roundtrip`.

### Memory extras — generate, default off

Build the model inside `quantized_model_init` so only quantized weights are materialized, with no
BF16 master copy, and pair it with TE's `FusedAdam(master_weights=True)`. From
`$BIONEMO_RECIPES/recipes/llama3_native_te/train_fsdp2.py`:

```python
with (
    torch.device("meta") if args.use_meta_device else nullcontext(),
    transformer_engine.pytorch.quantized_model_init(
        recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
    ),
):
    model = model_class(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
```

Guard with a `ValueError` if `quantized_model_init_kwargs.enabled` is set without
`fp8_config.enabled or fp4_config.enabled`. Config shape is in
`$BIONEMO_RECIPES/recipes/llama3_native_te/hydra_config/L2_lingua_8b_mxfp8_qinit.yaml`.

**Known incompatibility:** mFSDP + `quantized_model_init` is xfail'd in
`$BIONEMO_RECIPES/recipes/esm2_native_te/tests/test_train.py` (BIONEMO-3012). If the target uses `megatron-fsdp`, do
not enable it; say so in the report.

______________________________________________________________________

## Depth C — targeted kernel swaps

The fallback when the block has a quirk no TE layer can express, including a non-dot-product
attention mechanism or a sequence-mixing core (scan, message passing, denoiser conditioning) with no
TE analogue at all. Apply only these, and only where the shapes are compatible:

- `nn.Linear` → `transformer_engine.pytorch.Linear`
- `LayerNorm` immediately followed by `Linear` → `transformer_engine.pytorch.LayerNormLinear`
- `LayerNorm` + two-layer MLP → `transformer_engine.pytorch.LayerNormMLP`
- `torch.optim.Adam*` → `transformer_engine.pytorch.optimizers.FusedAdam`
- A standalone attention core with custom QKV/mask shaping around it (self- or cross-attention) →
  `transformer_engine.pytorch.attention.dot_product_attention.DotProductAttention`, keeping the
  custom reshape/GQA/masking logic in place. This does not require a full `te.TransformerLayer`; see
  `$BIONEMO_RECIPES/recipes/codonfm_ptl_te/src/models/components/encodon_te_mha.py` for a worked
  example of both self- and T5-style cross-attention built this way.

THD packing and `quantized_model_init` are **not** blocked by the missing `TransformerLayer` — both
are properties of the individual TE modules above, not the layer wrapper:

- **THD packing works on a standalone `DotProductAttention`.** `forward()` takes `qkv_format="thd"`,
  `cu_seqlens_q` / `cu_seqlens_kv`, and `max_seqlen_q` / `max_seqlen_kv` directly — the varlen kernel
  dispatch this buys is inside the attention core being swapped, not the surrounding layer. The
  target still has to thread `cu_seqlens` from its own collator into that call by hand, same as the
  custom QKV/mask shaping already required above.
- **`quantized_model_init` wraps module construction, not `TransformerLayer` construction
  specifically.** Any `transformer_engine.pytorch.Linear` / `LayerNormLinear` / `LayerNormMLP` built
  inside the context manager gets quantized-only parameters, per its own docstring example
  (`with quantized_model_init(enabled=True): model = transformer_engine.pytorch.Linear(768, 768)`).
  Wrap the Depth C module construction in it the same way Depth B wraps the full model.

What genuinely has no Depth C equivalent is **fused QKV** (`fuse_qkv_params` /
`qkv_weight_interleaved`): that packed weight layout is a `te.TransformerLayer` /
`te.MultiheadAttention` construction detail, not a property of bare `DotProductAttention`, which
takes already-separated Q/K/V tensors.

Also available, and architecture-agnostic: `torch.compile` on the model or step function, and
AMP/BF16 autocast if the loop is still FP32.

Wrap the forward in a single outer `transformer_engine.pytorch.autocast(enabled=True, recipe=...)`.

If the target already builds its block from named, independently swappable sub-modules (a
submodules dataclass, a `build_module`-style factory), mirror that shape rather than patching calls
inline — see
`$BIONEMO_RECIPES/recipes/evo2_megatron/src/bionemo/evo2/models/megatron/hyena/hyena_layer_specs.py`
and `hyena_layer.py`, where the Hyena (state-space) block's projections, norm, and MLP are each an
independently swappable slot and only the scan/conv mixer itself is unconditionally custom. This is
the ideal structure when the target already supports it, not a requirement — an inline pass over
`nn.Linear`/`LayerNorm`/attention calls is an acceptable outcome too.

Validation is Tier 1 only. State the explicitly-not-applied list in the report: fused QKV requires a
`TransformerLayer` and is unavailable at this depth; THD packing and `quantized_model_init` are
available (see above) but hand-wired rather than `TransformerLayer`-managed, so say so explicitly if
either was skipped anyway. Name the structural quirk or sequence-mixing core that blocked a deeper
port, and do not present the result as a full FP8 port. Expect a materially smaller speedup than the
GEMM benchmark
suggests, because only the swapped sub-layers are affected.

When no BioNeMo Recipes reference or Depth C pattern above covers the target's sequence-mixing core
at all, `github.com/NVIDIA/Megatron-LM` and `github.com/NVIDIA/Megatron-Bridge` are read-only
inspiration for how that layer type has been wired to TE elsewhere — same status as TE's own docs,
never a repo to edit or copy wholesale. Treat anything sourced from them as **unvalidated by
BioNeMo's `BaseModelTest` suite**: say so explicitly in the report, and hold it to Tier 1 parity at
minimum, not a lighter bar than the rest of Depth C.

______________________________________________________________________

## Throughput instrumentation (all depths)

Vendor a `PerfLogger` based on `$BIONEMO_RECIPES/recipes/esm2_native_te/perf_logger.py`. It emits
`train/tokens_per_second_per_gpu`, `train/unpadded_tokens_per_second_per_gpu`,
`train/step_time`, and `train/gpu_memory_allocated_max_gb`. This is what converts the GEMM
benchmark's upper bound into a real before/after number for the report. The llama3 variant
(`$BIONEMO_RECIPES/recipes/llama3_native_te/perf_logger.py`) adds gradient-accumulation-aware
`log_micro_step` and an `NsightProfiler` — use it when the target accumulates gradients.

## Branch and file hygiene

- Create `bionemo-accel/<family>` in the target repo before the first edit.
- Never overwrite the original model file in place — the parity check needs it. Add the TE model
  alongside as `modeling_<name>_te.py`.
- One commit per phase, so the user can bisect the port.
