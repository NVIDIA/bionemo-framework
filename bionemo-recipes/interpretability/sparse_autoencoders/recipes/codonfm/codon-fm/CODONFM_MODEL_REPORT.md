# CodonFM Model Report

## Overview

CodonFM is a suite of foundation models developed by NVIDIA that operates directly on **codon sequences** (3-nucleotide units of DNA/RNA) rather than amino acid sequences. It learns contextual codon representations to enable downstream codon-aware tasks such as variant effect prediction, codon optimization, and translation efficiency estimation.

The suite contains two model families:

- **Encodon** (bidirectional encoder): Trained with masked language modeling (MLM). Sizes range from 80M to 10B parameters.
- **Decodon** (autoregressive decoder): Trained with causal language modeling (CLM) and organism conditioning. Sizes range from 200M to 1B parameters.

---

## Encodon 5B Architecture

The Encodon 5B is the second-largest encoder variant. Its architecture parameters:

| Parameter | Value |
|---|---|
| Hidden size | 4096 |
| Intermediate (FFN) size | 16384 |
| Attention heads | 32 |
| Transformer layers | 24 |
| Head dimension | 128 (4096 / 32) |
| Vocabulary size | 69 |
| Max sequence length | 2048 codons |
| Position encoding | Rotary (RoPE), theta = 10,000 |
| Activation function | GELU |
| Dropout | 0.1 (hidden & attention) |
| Layer norm epsilon | 1e-12 |
| Post-embedding LayerNorm | Yes |

### Transformer Block

Each of the 24 layers follows a **Pre-LayerNorm** design:

```
Attention sub-block:
  x' = x + Dropout(Dense(PostLN(MHA(PreLN(x)))))

FFN sub-block:
  x' = x + Dropout(Dense_down(PostLN(GELU(Dense_up(PreLN(x))))))
```

Key components:
- **Multi-Head Attention** uses xformers memory-efficient attention with Rotary Positional Embeddings (RoPE).
- **Feed-forward network** expands from 4096 to 16384 dimensions (4x expansion) and contracts back.
- Weight initialization follows the **MAGNETO** scheme (Xavier normal with scaled gain).

### MLM Prediction Head

A 3-layer MLP sits on top of the encoder for pretraining:

```
Linear(4096 → 4096) → GELU → LayerNorm → Linear(4096 → 69)
```

### Training Configuration (5B)

From the pretraining script (`experiment_scripts/pretraining/encodon_filtered/mlm/encodon_5b.sh`):

| Setting | Value |
|---|---|
| Learning rate | 1e-5 |
| Nodes | 64 |
| GPUs per node | 8 |
| Batch size per GPU | 2 |
| Effective batch size | 1024 |
| Precision | bf16-mixed |
| Distributed strategy | FSDP (sharded state dict) |
| Masking probability | 0.15 |
| Mask strategy | 80% `<MASK>`, 10% random, 10% unchanged |

A **codon-frequency-weighted** (CDSWT) variant also exists, which weights masking probability by codon usage frequency to address codon usage bias across organisms.

---

## Inputs and Outputs

### Input Format

**Raw input:** DNA or RNA coding sequences composed of 3-letter codons (e.g., `ATGAAAGCCTTT...`).

**After tokenization:** Integer token IDs in range 0-68.

| Token ID | Token |
|---|---|
| 0 | `<CLS>` |
| 1 | `<SEP>` |
| 2 | `<UNK>` |
| 3 | `<PAD>` |
| 4 | `<MASK>` |
| 5-68 | 64 DNA codons (AAA, AAC, AAG, ..., TTT) |

**Input tensor shape:** `(batch_size, sequence_length)` where `sequence_length <= 2048`.

A sequence is preprocessed as: `<CLS> codon_1 codon_2 ... codon_N <SEP> <PAD> ... <PAD>`

### Output Format

The model returns an `EnCodonOutput` dataclass:

| Field | Shape | Description |
|---|---|---|
| `logits` | `(B, L, 69)` | Per-position probability distribution over the 69-token vocabulary |
| `last_hidden_state` | `(B, L, 4096)` | Final-layer hidden representations |
| `all_hidden_states` | List of `(B, L, 4096)` | Optional: hidden states from all 24 layers |

### Inference Tasks

| Task | Output | Description |
|---|---|---|
| `masked_language_modeling` | Predicted codon tokens at masked positions | Fill-in-the-blank codon prediction |
| `mutation_prediction` | Log-likelihood ratio (ref vs. alt) | Zero-shot variant effect scoring |
| `fitness_prediction` | Scalar per sequence | Mean log-likelihood as sequence fitness proxy |
| `embedding_prediction` | `(B, 4096)` vector | CLS token embedding for downstream use |
| `downstream_prediction` | Task-dependent | Cross-attention head for classification/regression |

---

## How CodonFM Differs from ESM2

ESM2 (Evolutionary Scale Modeling 2) is a protein language model that operates on amino acid sequences. CodonFM operates at a fundamentally different biological level. The key differences are:

### 1. Input Granularity

| | CodonFM | ESM2 |
|---|---|---|
| **Input unit** | Codons (3-nucleotide DNA/RNA) | Single amino acids |
| **Vocabulary** | 69 tokens (5 special + 64 codons) | ~33 tokens (20 AAs + special) |
| **Biological level** | Nucleotide / mRNA | Protein |

Because the genetic code is degenerate (multiple codons encode the same amino acid), CodonFM preserves information that ESM2 discards. For example, the amino acid Leucine (L) can be encoded by 6 different codons (TTA, TTG, CTT, CTC, CTA, CTG). ESM2 sees a single "L" token; CodonFM sees the specific codon used.

### 2. Synonymous Variation

CodonFM can distinguish between **synonymous variants** — mutations that change the codon but not the amino acid. These variants affect:
- mRNA stability and secondary structure
- Translation speed and efficiency (codon usage bias)
- Co-translational protein folding

ESM2 is blind to synonymous variation since the amino acid sequence is unchanged.

### 3. Position Encoding

| | CodonFM | ESM2 |
|---|---|---|
| **Method** | Rotary Position Embeddings (RoPE) | Learned absolute position embeddings |
| **Benefit** | Better length extrapolation, relative position awareness | Fixed max length |

### 4. Architecture Variants

CodonFM includes both an encoder (Encodon) and a decoder (Decodon), whereas ESM2 is encoder-only. The Decodon variant supports:
- **Organism conditioning**: species-specific codon optimization by prepending an organism token (e.g., `<9606>` for human)
- **Autoregressive generation**: native sequence generation with temperature sampling and beam search

### 5. Downstream Applications

| Capability | CodonFM | ESM2 |
|---|---|---|
| Synonymous variant scoring | Yes | No |
| Codon optimization | Yes | No |
| Translation efficiency prediction | Yes | No |
| mRNA stability prediction | Yes | Indirect |
| Missense variant scoring | Yes | Yes |
| Protein structure prediction | No (different level) | Yes |
| Protein function prediction | Indirect | Yes |

### 6. Training Data

| | CodonFM | ESM2 |
|---|---|---|
| **Source** | NCBI coding DNA sequences (CDS) | UniRef protein databases |
| **Data type** | Nucleotide sequences | Amino acid sequences |
| **Filtering** | Pathogen exclusion, taxonomic filtering | Sequence clustering |

---

## Model Family Summary

### Encodon (Encoder) Models

| Model | Params | Hidden | Layers | Heads | FFN | Checkpoint |
|---|---|---|---|---|---|---|
| Encodon 80M | 80M | 1024 | 6 | 8 | 4096 | [HuggingFace](https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1) |
| Encodon 600M | 600M | 2048 | 12 | 16 | 8192 | [HuggingFace](https://huggingface.co/nvidia/NV-CodonFM-Encodon-600M-v1) |
| Encodon 1B | 1B | 2048 | 18 | 16 | 8192 | [HuggingFace](https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1) |
| Encodon 1B CDSWT | 1B | 2048 | 18 | 16 | 8192 | [HuggingFace](https://huggingface.co/nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1) |
| **Encodon 5B** | **~5B** | **4096** | **24** | **32** | **16384** | Internal |
| Encodon 10B | ~10B | 5120 | 34 | 40 | 20480 | Internal |

### Decodon (Decoder) Models

| Model | Params | Hidden | Layers | Heads | FFN |
|---|---|---|---|---|---|
| Decodon 200M | 200M | 1024 | 16 | 16 | 4096 |
| Decodon 1B | 1B | 2048 | 18 | 16 | 8192 |

---

## Fine-tuning Options

| Strategy | Description |
|---|---|
| `full` | All parameters trained end-to-end |
| `lora` | Low-rank adapters (r=16, alpha=32, dropout=0.1) on Q/V/FFN projections |
| `head_only_pretrained` | Frozen backbone, train pretrained prediction head |
| `head_only_random` | Frozen backbone, train randomly initialized head |

---

## Codon Optimization Pipeline

CodonFM includes a codon optimization pipeline that combines model-guided sequence generation with a **Genetic Algorithm (GA)** for multi-objective optimization. Given an amino acid sequence, it generates codon-optimized DNA sequences that maximize model-predicted fitness while respecting biological constraints.

**Generation modes:**
- **Bidirectional (Encodon):** Iterative mask-and-predict with temperature annealing (1.2 → 0.5)
- **Autoregressive (Decodon):** Left-to-right generation with organism conditioning

**GA parameters:** population size of 100, crossover rate 0.7, mutation rate 0.02, configurable fitness weights.

---

## Key Takeaway

CodonFM fills a gap that protein-level models like ESM2 cannot address: understanding the **codon-level language** of translation. While ESM2 excels at protein structure and function tasks, CodonFM captures codon usage bias, synonymous variation effects, translational dynamics, and mRNA-level properties — making the two model families complementary for different biological questions.
