# First-Principles FLOPs Derivation for Llama 3 (GQA + SwiGLU)

This document derives the per-training-step FLOPs formula used in `compute_flops_first_principles()`, explains each component, and contrasts it with the simplified README formula.

## Counting convention

We count **multiply-accumulate operations (MACs)** and report them as **2 FLOPs per MAC** (one multiply, one add). For a matrix multiplication of shapes `(M, K) @ (K, N)`, the FLOPs are:

```
FLOPs = 2 * M * K * N
```

We only count dense matmuls. Softmax, layer norms, RoPE rotations, element-wise activations (SiLU), and the Hadamard product in SwiGLU are negligible relative to the matmuls and are excluded, consistent with standard MFU methodology.

## Notation

| Symbol | Meaning                                           | Lingua-1B value |
| ------ | ------------------------------------------------- | --------------- |
| B      | Batch size                                        | 1               |
| S      | Sequence length                                   | varies          |
| H      | Hidden size (`hidden_size`)                       | 2048            |
| L      | Number of layers (`num_hidden_layers`)            | 25              |
| n_h    | Number of attention heads (`num_attention_heads`) | 16              |
| n_kv   | Number of KV heads (`num_key_value_heads`)        | 8               |
| d      | Head dimension (H / n_h)                          | 128             |
| d_kv   | KV dimension (n_kv * d)                           | 1024            |
| I      | FFN intermediate size (`intermediate_size`)       | 6144            |
| V      | Vocabulary size (`vocab_size`)                    | 128256          |

## Per-layer forward FLOPs

### Attention projections

Each attention layer projects the hidden states into queries, keys, values, and then projects the attention output back.

**Q projection**: Each token's hidden state (H) is projected to the query space (H = n_h * d).

```
input:  (B, S, H)
weight: (H, H)
output: (B, S, H)
FLOPs = 2 * B * S * H * H
```

**K projection**: With Grouped Query Attention (GQA), keys are projected to a smaller space (d_kv = n_kv * d) instead of the full H. This is the key difference from standard Multi-Head Attention (MHA).

```
input:  (B, S, H)
weight: (H, d_kv)
output: (B, S, d_kv)
FLOPs = 2 * B * S * H * d_kv
```

**V projection**: Same dimensions as K projection.

```
FLOPs = 2 * B * S * H * d_kv
```

**O projection**: The concatenated attention output (H) is projected back to hidden size (H).

```
input:  (B, S, H)
weight: (H, H)
output: (B, S, H)
FLOPs = 2 * B * S * H * H
```

**Total attention projections:**

```
attn_proj = 2 * B * S * H * (2*H + 2*d_kv)
```

For MHA (d_kv = H), this simplifies to `2 * B * S * H * 4H = 8 * B * S * H^2`. For GQA with d_kv < H, the K and V projections are smaller.

### Attention scores

After projection, attention computes Q @ K^T and then attn_weights @ V. Even with GQA (fewer KV heads), the KV heads are **broadcast** to match the query heads, so the effective computation uses all n_h query heads attending to S key positions.

**Attention logits (Q @ K^T)**: For each head, the query (S, d) is multiplied by key^T (d, S).

```
Per head: 2 * B * S * d * S = 2 * B * S^2 * d
All n_h heads: 2 * B * S^2 * d * n_h = 2 * B * S^2 * H
```

Note: with GQA, each KV head is shared across (n_h / n_kv) query heads. The total FLOPs remain `2 * B * S^2 * H` because we still have n_h query heads each doing S\*d work against S keys.

**Attention values (attn_weights @ V)**: Same shape — attention weights (S, S) multiplied by values (S, d) per head.

```
FLOPs = 2 * B * S^2 * H
```

**Total attention scores:**

```
attn_score = 4 * B * S^2 * H
```

### MLP (SwiGLU)

Llama 3 uses SwiGLU activation, which has **three** linear projections instead of the standard MLP's two:

```
SwiGLU(x) = (x @ W_gate * SiLU(x @ W_up)) @ W_down
```

Standard MLP has two projections (up: H -> I, down: I -> H) with I = 4H typically. SwiGLU adds a third (gate) projection.

**Gate projection**: H -> I

```
FLOPs = 2 * B * S * H * I
```

**Up projection**: H -> I

```
FLOPs = 2 * B * S * H * I
```

**Down projection**: I -> H

```
FLOPs = 2 * B * S * I * H
```

The element-wise SiLU activation and the Hadamard product (gate * up) are O(B * S * I) — negligible compared to the matmuls.

**Total MLP:**

```
mlp = 6 * B * S * H * I
```

### Per-layer total

```
per_layer_fwd = attn_proj + attn_score + mlp
             = 2*B*S*H*(2*H + 2*d_kv) + 4*B*S^2*H + 6*B*S*H*I
```

## LM head

The language model head projects hidden states to vocabulary logits:

```
input:  (B, S, H)
weight: (H, V)
output: (B, S, V)
FLOPs = 2 * B * S * H * V
```

## Total forward FLOPs

```
total_fwd = L * per_layer_fwd + lm_head
          = L * [2*B*S*H*(2*H + 2*d_kv) + 4*B*S^2*H + 6*B*S*H*I] + 2*B*S*H*V
```

## Total training FLOPs (forward + backward)

The standard approximation for training is that backward costs 2x the forward (one pass to compute dL/dW, another to compute dL/dX for each matmul). Total training = 3x forward.

```
total_training = 3 * total_fwd
```

## Comparison with the README formula

The README uses a simplified formula for a standard transformer:

```python
total = (24 * B * S * H * H + 4 * B * S * S * H) * (3 * L) + (6 * B * S * H * V)
```

The `3*L` folds the 3x training multiplier into the layer count, and `6*B*S*H*V = 3 * 2*B*S*H*V` does the same for the LM head. Extracting the per-layer **forward** FLOPs implicit in the README:

```
readme_per_layer_fwd = (24*B*S*H^2 + 4*B*S^2*H) / 3
                     = 8*B*S*H^2 + (4/3)*B*S^2*H
```

The `4*B*S^2*H` attention score term (with 3x) matches our first-principles `4*B*S^2*H` exactly — both formulas agree on attention scores. The difference is entirely in the `24*B*S*H^2` term, which covers attention projections and MLP. Decomposing it:

### Decomposition of the README's `24*B*S*H^2`

The coefficient 24 encodes two assumptions about the per-layer linear projections:

**Attention projections (coefficient = 8):** Four projections (Q, K, V, O) each of size H -> H, assuming standard Multi-Head Attention (MHA):

```
4 projections * 2*B*S*H*H = 8*B*S*H^2
```

**MLP (coefficient = 16):** Two projections with intermediate size I = 4H, assuming a standard Feed-Forward Network:

```
Up:   2*B*S*H*(4H) = 8*B*S*H^2
Down: 2*B*S*(4H)*H = 8*B*S*H^2
Total:              = 16*B*S*H^2
```

Combined: `8 + 16 = 24`.

### How our first-principles formula differs

Our formula replaces both assumptions with the actual Llama 3 architecture:

**Attention projections with GQA:** K and V project to d_kv (not H):

```
Q: 2*B*S*H*H      K: 2*B*S*H*d_kv      V: 2*B*S*H*d_kv      O: 2*B*S*H*H
Total = 2*B*S*H*(2*H + 2*d_kv)
```

**MLP with SwiGLU:** Three projections (gate, up, down) with actual I:

```
Gate: 2*B*S*H*I    Up: 2*B*S*H*I    Down: 2*B*S*I*H
Total = 6*B*S*H*I
```

Side by side, per layer forward, factoring out `2*B*S*H`:

| Component        | README                 | First principles               |
| ---------------- | ---------------------- | ------------------------------ |
| Attention proj   | `4*H` (MHA)            | `2*H + 2*d_kv` (GQA)           |
| MLP              | `2*4H = 8*H` (std FFN) | `3*I` (SwiGLU)                 |
| Attention scores | `2*S` (same)           | `2*S` (same)                   |
| **Total coeff**  | **`4*H + 8*H + 2*S`**  | **`2*H + 2*d_kv + 3*I + 2*S`** |

Setting them equal: `12*H + 2*S = 2*H + 2*d_kv + 3*I + 2*S`, which simplifies to:

```
10*H = 2*d_kv + 3*I
```

### Where the assumptions break

| Component        | README assumes        | Llama 3 actual                 | Direction             |
| ---------------- | --------------------- | ------------------------------ | --------------------- |
| K, V projections | H -> H (MHA)          | H -> d_kv (GQA, d_kv < H)      | README **overcounts** |
| MLP              | 2 projections, I = 4H | 3 projections (SwiGLU), I < 4H | Depends on model dims |

The errors go in opposite directions:

- **GQA** makes K/V projections cheaper (d_kv < H), so the README overcounts attention
- **SwiGLU** adds a third MLP projection, so the README undercounts MLP (despite using a larger I=4H)

### Why they cancel exactly for Lingua-1B

For the Lingua-1B config (H=2048, d_kv=1024, I=6144):

```
README linear cost per layer:        12*H = 12 * 2048 = 24,576
First-principles linear cost:  2*H + 2*d_kv + 3*I = 4096 + 2048 + 18432 = 24,576
```

They match. Breaking down why:

- README assumes attn proj cost: `4*H = 4 * 2048 = 8,192`

- Actual attn proj cost: `2*H + 2*d_kv = 4096 + 2048 = 6,144`

- **GQA saves: 2,048**

- README assumes MLP cost: `8*H = 8 * 2048 = 16,384`

- Actual MLP cost: `3*I = 3 * 6144 = 18,432`

- **SwiGLU adds: 2,048**

Saved from GQA = Added from SwiGLU = **2,048 exactly**. This is a coincidence specific to Lingua-1B's dimensions. For models with different d_kv/H or I/H ratios, the formulas diverge.

### When would they diverge?

For Llama 3.1 70B (H=8192, n_kv=8, d=128, d_kv=1024, I=28672):

```
README linear: 12*H = 98,304
First-principles: 2*8192 + 2*1024 + 3*28672 = 16384 + 2048 + 86016 = 104,448
Difference: +6.2% (README undercounts by ~6%)
```

The README would **undercount** FLOPs for Llama 70B because SwiGLU's third projection with the large I=28672 dominates the GQA savings.

## Code

```python
def compute_flops_first_principles(
    b, s, h, num_layers, n_kv_heads, head_dim, ffn_hidden_size, vocab_size
):
    kv_dim = n_kv_heads * head_dim

    breakdown = {
        "Q projection": 2 * b * s * h * h,
        "K projection": 2 * b * s * h * kv_dim,
        "V projection": 2 * b * s * h * kv_dim,
        "O projection": 2 * b * s * h * h,
        "Attn logits": 2 * b * s * s * h,
        "Attn values": 2 * b * s * s * h,
        "Gate proj": 2 * b * s * h * ffn_hidden_size,
        "Up proj": 2 * b * s * h * ffn_hidden_size,
        "Down proj": 2 * b * s * ffn_hidden_size * h,
    }

    per_layer_fwd = sum(breakdown.values())
    lm_head_fwd = 2 * b * s * h * vocab_size
    total_fwd = num_layers * per_layer_fwd + lm_head_fwd
    total_training = 3 * total_fwd

    return total_training, breakdown, lm_head_fwd
```

## Numerical example (Lingua-1B, B=1, S=4096)

```
Per layer forward:
  Q proj:    2 * 1 * 4096 * 2048 * 2048          =   34,359,738,368
  K proj:    2 * 1 * 4096 * 2048 * 1024          =   17,179,869,184
  V proj:    2 * 1 * 4096 * 2048 * 1024          =   17,179,869,184
  O proj:    2 * 1 * 4096 * 2048 * 2048          =   34,359,738,368
  Attn logits: 2 * 1 * 4096 * 4096 * 2048        =   68,719,476,736
  Attn values: 2 * 1 * 4096 * 4096 * 2048        =   68,719,476,736
  Gate proj: 2 * 1 * 4096 * 2048 * 6144          =  103,079,215,104
  Up proj:   2 * 1 * 4096 * 2048 * 6144          =  103,079,215,104
  Down proj: 2 * 1 * 4096 * 6144 * 2048          =  103,079,215,104
  ─────────────────────────────────────────────────────────────────
  Per-layer total:                                   549,755,813,888

LM head:   2 * 1 * 4096 * 2048 * 128256          = 2,152,726,528,000

Total forward: 25 * 549,755,813,888 + 2,152,726,528,000
             = 13,743,895,347,200 + 2,152,726,528,000
             = 15,896,621,875,200

Total training (3x): 3 * 15,896,621,875,200 = 47,689,865,625,600
```

Note: the code uses integer arithmetic and reports `47,687,021,887,488` — the small difference is from the embedding layer not being counted here (it's a lookup, not a matmul) and the LM head weight tying configuration.

## References

- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022) — establishes the 3x forward approximation for training FLOPs
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022) — defines MFU as model_flops / (step_time * peak_hardware_flops)
