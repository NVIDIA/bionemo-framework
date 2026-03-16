# CodonFM SAE Experiments

## Setup

All experiments use the Primates 10k activation cache (~6M tokens, 2048 hidden dim, layer 16/18):

```
Cache: .cache/activations/primates_10k_1b_layer-2
Model: Encodon 1B
Layer: -2 (penultimate, layer 16 of 18)
Tokens: ~5.9M
```

### Base command

```bash
cd /data/jwilber/biosae_refactor/recipes/codonfm

python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 \
    --wandb --wandb-project sae_codonfm_recipe \
    <experiment-specific flags>
```

### Evaluation (after each training run)

```bash
# Dashboard data
python scripts/dashboard.py \
    --checkpoint ./outputs/<run>/checkpoints/checkpoint_final.pt \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 --top-k <K> \
    --csv-path ../../../codonfm/data/merged_seqs_for_validation.csv \
    --num-sequences 431 \
    --output-dir ./outputs/<run>/dashboard

# Analysis (vocab logits, correlations, annotations)
python scripts/analyze.py \
    --checkpoint ./outputs/<run>/checkpoints/checkpoint_final.pt \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 --top-k <K> \
    --csv-path ../../../codonfm/data/merged_seqs_for_validation.csv \
    --dashboard-dir ./outputs/<run>/dashboard \
    --output-dir ./outputs/<run>/analysis
```

---

## Experiment 1: Expansion Factor Sweep

**Goal:** Find the right number of features for 6M tokens.
**Vary:** expansion factor (2, 4, 8, 16)
**Fixed:** top_k=32, lr=3e-4, epochs=10, batch_size=2048, auxk=256

| Run | Features | Expansion | Command |
|-----|----------|-----------|---------|
| ef2 | 4096 | 2 | below |
| ef4 | 8192 | 4 | below |
| ef8 | 16384 | 8 | below |
| ef16 | 32768 | 16 | below |

```bash
# ef2 — 4096 features
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 2 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/ef2/checkpoints --output-dir ./outputs/ef2 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ef2_k32"

# ef4 — 8192 features
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 4 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/ef4/checkpoints --output-dir ./outputs/ef4 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ef4_k32"

# ef8 — 16384 features
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/ef8/checkpoints --output-dir ./outputs/ef8 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ef8_k32"

# ef16 — 32768 features
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 16 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/ef16/checkpoints --output-dir ./outputs/ef16 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ef16_k32"
```

**What to compare:** final loss, variance explained, % dead latents, interpretability in dashboard

**Expected:** ef2/ef4 will have fewer dead latents but less feature specificity. ef8/ef16 will have more dead but sharper features. Sweet spot is likely ef4 or ef8 for 6M tokens.

---

## Experiment 2: Top-K Sweep

**Goal:** Find optimal sparsity level.
**Vary:** top_k (16, 32, 64, 128)
**Fixed:** expansion_factor=8, lr=3e-4, epochs=10, batch_size=2048, auxk=256

```bash
# k16
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 16 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/k16/checkpoints --output-dir ./outputs/k16 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ef8_k16"

# k32
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/k32/checkpoints --output-dir ./outputs/k32 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ef8_k32"

# k64
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 64 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/k64/checkpoints --output-dir ./outputs/k64 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ef8_k64"

# k128
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 128 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/k128/checkpoints --output-dir ./outputs/k128 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ef8_k128"
```

**What to compare:** reconstruction quality (var_exp) vs sparsity (avg_nonzero_act) vs dead latents. Lower k = sparser but harder to reconstruct. Higher k = better reconstruction but less interpretable.

**Expected:** k32 is the standard default. k64 may give better reconstruction with acceptable sparsity. k16 will be very sparse with higher loss.

---

## Experiment 3: AuxK Sweep

**Goal:** Find optimal dead latent recovery.
**Vary:** auxk (0, 64, 256, 512)
**Fixed:** expansion_factor=8, top_k=32, lr=3e-4, epochs=10, batch_size=2048

```bash
# No auxk
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/auxk0/checkpoints --output-dir ./outputs/auxk0 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_auxk0"

# auxk=64
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 64 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/auxk64/checkpoints --output-dir ./outputs/auxk64 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_auxk64"

# auxk=256
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/auxk256/checkpoints --output-dir ./outputs/auxk256 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_auxk256"

# auxk=512
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 512 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/auxk512/checkpoints --output-dir ./outputs/auxk512 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_auxk512"
```

**What to compare:** dead latent % at end of training. auxk=0 will have highest dead %, auxk=512 lowest.

**Expected:** auxk=256 or 512 should keep dead latents under 30% even with 6M tokens.

---

## Experiment 4: Learning Rate Sweep

**Goal:** Find optimal learning rate.
**Vary:** lr (1e-4, 3e-4, 1e-3, 3e-3)
**Fixed:** expansion_factor=8, top_k=32, auxk=256, epochs=10, batch_size=2048

```bash
# lr=1e-4
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 1e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/lr1e4/checkpoints --output-dir ./outputs/lr1e4 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_lr1e-4"

# lr=3e-4 (baseline)
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/lr3e4/checkpoints --output-dir ./outputs/lr3e4 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_lr3e-4"

# lr=1e-3
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 1e-3 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/lr1e3/checkpoints --output-dir ./outputs/lr1e3 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_lr1e-3"

# lr=3e-3
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-3 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/lr3e3/checkpoints --output-dir ./outputs/lr3e3 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_lr3e-3"
```

**What to compare:** loss curves, final var_exp, training stability.

**Expected:** 3e-4 is the standard default. 1e-3 may converge faster. 3e-3 may be unstable.

---

## Experiment 5: Epoch Sweep (Overfitting Check)

**Goal:** With 6M tokens, how many epochs before overfitting?
**Vary:** n_epochs (3, 10, 30, 100)
**Fixed:** expansion_factor=8, top_k=32, lr=3e-4, auxk=256, batch_size=2048

```bash
# 3 epochs
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 3 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/ep3/checkpoints --output-dir ./outputs/ep3 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ep3"

# 10 epochs
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 10 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/ep10/checkpoints --output-dir ./outputs/ep10 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ep10"

# 30 epochs
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 30 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/ep30/checkpoints --output-dir ./outputs/ep30 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ep30"

# 100 epochs
python scripts/train.py \
    --cache-dir .cache/activations/primates_10k_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 2048 --n-epochs 100 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 500000 \
    --checkpoint-dir ./outputs/ep100/checkpoints --output-dir ./outputs/ep100 \
    --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_ep100"
```

**What to compare:** loss curves — when does the loss plateau? Does var_exp keep improving or stall?

**Expected:** With 6M tokens and 16k features, 10-30 epochs should be sufficient. 100 will likely overfit (loss stops decreasing, features memorize specific sequences).

---

## Experiment 6: Layer Sweep

**Goal:** Which layer produces the most interpretable features?
**Vary:** layer (-1, -2, -4, -9 which maps to last, penultimate, layer 14, layer 9)
**Fixed:** expansion_factor=8, top_k=32, lr=3e-4, auxk=256, epochs=10

Note: requires re-extracting activations for each layer.

```bash
# Extract layer -1 (last, layer 17)
python scripts/extract.py \
    --csv-path ../../../codonfm/data/Primates.csv \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -1 --batch-size 8 --num-sequences 10000 \
    --output .cache/activations/primates_10k_1b_layer-1

# Extract layer -4 (layer 14)
python scripts/extract.py \
    --csv-path ../../../codonfm/data/Primates.csv \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -4 --batch-size 8 --num-sequences 10000 \
    --output .cache/activations/primates_10k_1b_layer-4

# Extract layer -9 (layer 9)
python scripts/extract.py \
    --csv-path ../../../codonfm/data/Primates.csv \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -9 --batch-size 8 --num-sequences 10000 \
    --output .cache/activations/primates_10k_1b_layer-9

# Train on each (same hyperparams)
for LAYER in -1 -2 -4 -9; do
  python scripts/train.py \
      --cache-dir .cache/activations/primates_10k_1b_layer${LAYER} \
      --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer ${LAYER} \
      --model-type topk --expansion-factor 8 --top-k 32 \
      --batch-size 2048 --n-epochs 10 --lr 3e-4 \
      --auxk 256 --dead-tokens-threshold 500000 \
      --checkpoint-dir ./outputs/layer${LAYER}/checkpoints --output-dir ./outputs/layer${LAYER} \
      --wandb --wandb-project sae_codonfm_recipe --wandb-run-name "p10k_layer${LAYER}"
done
```

**What to compare:** var_exp, dead latents, and feature interpretability in dashboard. Earlier layers capture lower-level patterns (codon identity), later layers capture higher-level context.

**Expected:** Layer -2 (penultimate) is the standard choice. Layer -1 (last) may be too specialized for MLM. Middle layers may capture more syntactic/positional patterns.

---

## Suggested Run Order

1. **Experiment 3 (AuxK)** first — establishes whether auxk helps with dead latents on this data
2. **Experiment 1 (Expansion Factor)** — most impactful for feature quality
3. **Experiment 2 (Top-K)** — fine-tune sparsity
4. **Experiment 5 (Epochs)** — know when to stop
5. **Experiment 4 (LR)** — minor tuning
6. **Experiment 6 (Layer)** — most expensive, do last with best hyperparams

## Key Metrics to Track (W&B)

- `loss` — reconstruction loss (lower = better)
- `var_exp` — variance explained (higher = better, target > 0.8)
- `dead_latents (%)` — dead features (lower = better, target < 30%)
- `fvu` — fraction of variance unexplained (1 - var_exp)
- `mse` — mean squared error

## Scaling Up

Once best configs are found on 10k sequences (~6M tokens), scale to full Primates:

```bash
# Extract all 1.68M sequences (~1B tokens)
python scripts/extract.py \
    --csv-path ../../../codonfm/data/Primates.csv \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 --batch-size 8 \
    --output .cache/activations/primates_full_1b_layer-2

# Train with best config (example: ef8, k32, auxk256, lr3e-4, 3 epochs)
python scripts/train.py \
    --cache-dir .cache/activations/primates_full_1b_layer-2 \
    --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors --layer -2 \
    --model-type topk --expansion-factor 8 --top-k 32 \
    --batch-size 4096 --n-epochs 3 --lr 3e-4 \
    --auxk 256 --dead-tokens-threshold 10000000 \
    --init-pre-bias \
    --checkpoint-dir ./outputs/primates_full/checkpoints \
    --output-dir ./outputs/primates_full \
    --wandb --wandb-project sae_codonfm_recipe \
    --wandb-run-name "primates_full_ef8_k32_auxk256"
```

Note: at 1B tokens, increase `--dead-tokens-threshold` back to 10M and `--batch-size` to 4096. Consider `--init-pre-bias` for better convergence. Training will take ~8-9 hours single GPU.
