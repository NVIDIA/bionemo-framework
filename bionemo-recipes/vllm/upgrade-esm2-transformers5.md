# ESM2 + vLLM Integration: Technical Analysis and Recommendations

## Executive Summary

This document analyzes the challenges of running ESM2 with vLLM and provides clear
recommendations based on use case. There are **two separate issues** that need to be
understood:

1. **Transformers 5.0 compatibility** - Required for vLLM encoder-only support
2. **TransformerEngine vs vLLM architecture conflict** - Fundamental design tension

## The Two Problems

### Problem 1: Transformers 5.0 API Change (Fixable)

vLLM requires `transformers >= 5.0.0` for encoder-only models. Our ESM2 model fails with:

```
AttributeError: 'NVEsmForMaskedLM' object has no attribute 'all_tied_weights_keys'.
Did you mean: '_tied_weights_keys'?
```

**Root cause**: Transformers 5.0 changed the tied weights API:

- **v4.x**: `_tied_weights_keys = ("lm_head.decoder.weight",)` (class attribute, tuple)
- **v5.0**: `all_tied_weights_keys` (instance attribute, dict, set in `post_init()`)

**Fix**: Add `get_expanded_tied_weights_keys()` method and update `post_init()`.

### Problem 2: TransformerEngine vs vLLM Attention (Architectural Conflict)

Even after fixing Problem 1, there's a deeper issue. vLLM's Transformers backend requires:

```python
class MyAttention(nn.Module):
    is_causal = False  # Required for encoder-only

    def forward(self, hidden_states, **kwargs):
        # MUST use vLLM's attention interface
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        ...


class MyModel(PreTrainedModel):
    _supports_attention_backend = True  # Required
```

**Our ESM2 model FAILS these requirements because:**

| Requirement                          | Our Status | Why                    |
| ------------------------------------ | ---------- | ---------------------- |
| `_supports_attention_backend = True` | ❌ Missing | Not declared           |
| Uses `ALL_ATTENTION_FUNCTIONS`       | ❌ No      | Uses TE's attention    |
| `is_causal = False` on attention     | ❌ No      | TE doesn't expose this |

**Why this matters**: vLLM wants to REPLACE your attention with its own optimized kernels.
TransformerEngine's `TransformerLayer` is a monolithic fused module (LayerNorm + QKV +
Attention + MLP), and you can't easily swap just the attention.

When these requirements aren't met, vLLM falls back to a different loading path that
causes the `model.` prefix weight naming issue we observed.

## What Does vLLM Actually Provide?

| vLLM Feature              | For Decoder (LLM)      | For Encoder (ESM2)            |
| ------------------------- | ---------------------- | ----------------------------- |
| **PagedAttention**        | ✅ Huge win (KV cache) | ❌ N/A (no autoregressive)    |
| **Continuous batching**   | ✅ High throughput     | ✅ Useful for serving         |
| **Optimized attention**   | ✅ Fast inference      | ❌ Would replace TE attention |
| **Tensor parallelism**    | ✅ Multi-GPU           | ✅ Useful                     |
| **OpenAI-compatible API** | ✅ Easy integration    | ✅ Useful                     |

**Key insight**: vLLM's main optimization (PagedAttention) is designed for autoregressive
decoder models. For encoder-only models like ESM2, the primary benefit is serving
infrastructure, not raw compute performance.

## Recommendations by Use Case

### Use Case A: Maximum Inference Performance

**Recommendation: Use TransformerEngine model directly (no vLLM)**

```python
# Simple, fast, uses all TE optimizations (FP8, fused kernels)
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("nvidia/esm2_t6_8M_UR50D", trust_remote_code=True)
model = model.cuda().eval()

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(**inputs)
```

For serving, wrap with FastAPI or NVIDIA Triton Inference Server.

**Pros**: Full TE performance (FP8, fused ops)
**Cons**: DIY batching, no vLLM API

### Use Case B: vLLM Serving Infrastructure Needed

**Recommendation: Use HuggingFace ESM2 (non-TE) with vLLM**

```python
from vllm import LLM

# Uses facebook's original ESM2, not NVIDIA's TE version
model = LLM(model="facebook/esm2_t12_35M_UR50D", runner="pooling")
outputs = model.embed(sequences)
```

**Pros**: vLLM batching, OpenAI-compatible API, easy deployment
**Cons**: Loses TE optimizations (FP8, fused kernels)

### Use Case C: Both TE Performance AND vLLM Serving (Long-term)

**Recommendation: Write a native vLLM model implementation**

This means creating a custom vLLM model class (like they have for Llama, Qwen) that:

1. Uses TE layers directly
2. Registers in vLLM's model registry
3. Bypasses the Transformers backend entirely

**Effort**: High (significant development)
**Pros**: Best of both worlds
**Cons**: Maintenance burden, needs vLLM expertise

## Implementation Plan

### Phase 1: Fix Transformers 5.0 Compatibility (Required for any vLLM path)

#### 1.1 Add `get_expanded_tied_weights_keys` Method

```python
class NVEsmPreTrainedModel(EsmPreTrainedModel):
    def get_expanded_tied_weights_keys(
        self, all_submodels: bool = True
    ) -> dict[str, str]:
        """Return tied weight keys in transformers 5.0 format."""
        if self.config.tie_word_embeddings:
            return {"lm_head.decoder.weight": "esm.embeddings.word_embeddings.weight"}
        return {}
```

#### 1.2 Update `post_init()`

```python
def post_init(self):
    """Post-initialization hook for transformers 5.0 compatibility."""
    self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
        all_submodels=False
    )
    super().post_init()
```

#### 1.3 Version-Conditional Logic

```python
from packaging import version
from transformers import __version__ as transformers_version

IS_TRANSFORMERS_5 = version.parse(transformers_version) >= version.parse("5.0.0")


def post_init(self):
    if IS_TRANSFORMERS_5:
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
            all_submodels=False
        )
    super().post_init()
```

### Phase 2: vLLM Integration (Choose based on Use Case)

#### Option A: Accept vLLM Fallback Path + Weight Mapping

If you choose to use the TE model with vLLM despite not meeting all requirements:

```python
@property
def hf_to_vllm_mapper(self):
    """Map HuggingFace weight names to vLLM expected names."""

    def mapper(name: str) -> str:
        if name.startswith("esm.") or name.startswith("lm_head."):
            return f"model.{name}"
        return name

    return mapper
```

**Note**: This uses vLLM's fallback loading, not the optimized Transformers backend.
You keep TE layers but may not get all vLLM benefits.

#### Option B: Use Non-TE ESM2 with vLLM

No model changes needed. Just use the original Facebook ESM2:

```python
model = LLM(model="facebook/esm2_t12_35M_UR50D", runner="pooling")
```

## Files to Modify

1. **`bionemo-recipes/models/esm2/src/esm/modeling_esm_te.py`**

   - Add `get_expanded_tied_weights_keys()` method
   - Update `post_init()` for transformers 5.0
   - (Optional) Add `hf_to_vllm_mapper` property

2. **`bionemo-recipes/models/esm2/tests/test_transformers5_compat.py`** (new file)

   - Unit tests for transformers 5.0 compatibility

3. **`bionemo-recipes/models/esm2/README.md`**

   - Document version requirements and trade-offs

## Summary: Decision Matrix

| Priority                | Recommended Approach                |
| ----------------------- | ----------------------------------- |
| **Max inference speed** | TE model + FastAPI/Triton (no vLLM) |
| **Easy deployment**     | HuggingFace ESM2 + vLLM (no TE)     |
| **Both (long-term)**    | Native vLLM implementation with TE  |

## Appendix: Why vLLM's Architecture Conflicts with TransformerEngine

vLLM's Transformers backend is designed to:

1. Load any HuggingFace model
2. **Replace the attention with vLLM's optimized attention** (FlashAttention, PagedAttention)
3. Use vLLM's KV cache management

TransformerEngine's `TransformerLayer` is designed to:

1. Fuse LayerNorm + QKV projection + Attention + MLP into one optimized kernel
2. Support FP8 quantization throughout
3. Use TE's own attention implementation

These are fundamentally different approaches to optimization. You can't easily take
"just the attention" from either one - they're designed as integrated systems.

## References

- [vLLM Supported Models - Transformers Backend](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers)
- [vLLM Writing Custom Models](https://docs.vllm.ai/en/latest/models/supported_models.html#writing-custom-models)
- [Transformers 5.0 Migration](https://github.com/huggingface/transformers/issues/42832)
- [TransformerEngine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/)
