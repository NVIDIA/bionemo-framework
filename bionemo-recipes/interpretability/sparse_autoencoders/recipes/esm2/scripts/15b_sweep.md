# ESM2-15B SAE Sweep Plan

## Goal

Find the best SAE hyperparameters for ESM2-15B before a full InterPLM-scale run.
We sweep over expansion factor, top-k, and learning rate using the pre-extracted
activation caches. Each sweep run trains for 1 epoch (fast iteration), then the
best config gets a full multi-epoch run.

## Datasets

Two pre-extracted activation caches are available:

| Cache | Layer | Proteins | Size | Path |
|-------|-------|----------|------|------|
| 3M filtered | 36 | ~3M | 245G | `.cache/activations/esm2_15B_layer36_uniref50_3m_filtered` |
| 50k | 24 | 50k | 284G | `.cache/activations/15b_50k_layer24` |

**Use the 3M dataset (layer 36)** for sweeps. Layer 36 is deeper in the 48-layer
model (75% depth), which is where InterPLM found the most interpretable features.
The 3M protein count also gives much better coverage — dead latent rates will be
lower and features more meaningful.

> The 50k/layer24 cache is useful for quick smoke tests but too small for
> production-quality SAEs.

## Sweep Axes

### Phase 1: Expansion Factor + Top-K (4 runs)

These control the capacity vs sparsity tradeoff. Run all with lr=3e-4, 1 epoch.

```bash
#!/bin/bash
set -e

CACHE=.cache/activations/esm2_15B_layer36_uniref50_3m_filtered
MODEL=nvidia/esm2_t48_15B_UR50D
LAYER=36
WANDB_PROJECT=sae_esm2_15b_sweep

for EF in 8 16; do
  for K in 32 64; do
    AUXK=$((K * 2))
    RUN_NAME="ef${EF}_k${K}"
    echo ""
    echo "============================================================"
    echo "  Phase 1: ${RUN_NAME}"
    echo "============================================================"

    torchrun --nproc_per_node=8 scripts/train.py \
        --cache-dir $CACHE \
        --model-name $MODEL \
        --layer $LAYER \
        --expansion-factor $EF --top-k $K \
        --auxk $AUXK --auxk-coef 0.03125 \
        --init-pre-bias \
        --n-epochs 1 --batch-size 4096 --lr 3e-4 \
        --log-interval 100 \
        --dp-size 8 --seed 42 \
        --wandb --wandb-project $WANDB_PROJECT \
        --wandb-group esm2_15b_sweep \
        --wandb-run-name "$RUN_NAME" \
        --output-dir outputs/sweep_15b/$RUN_NAME \
        --checkpoint-dir outputs/sweep_15b/$RUN_NAME/checkpoints
  done
done
```

**What to compare**: FVU (fraction of variance unexplained — lower is better),
dead latent %, and loss. These are all logged to W&B.

### Phase 2: Learning Rate (3 runs)

Take the best (ef, k) from Phase 1 and plug in below.
Edit `BEST_EF`, `BEST_K`, `BEST_AUXK` before running.

```bash
#!/bin/bash
set -e

CACHE=.cache/activations/esm2_15B_layer36_uniref50_3m_filtered
MODEL=nvidia/esm2_t48_15B_UR50D
LAYER=36
WANDB_PROJECT=sae_esm2_15b_sweep

# Fill in from Phase 1 results
BEST_EF=16
BEST_K=64
BEST_AUXK=128

for LR in 1e-4 3e-4 1e-3; do
  RUN_NAME="ef${BEST_EF}_k${BEST_K}_lr${LR}"
  echo ""
  echo "============================================================"
  echo "  Phase 2: ${RUN_NAME}"
  echo "============================================================"

  torchrun --nproc_per_node=8 scripts/train.py \
      --cache-dir $CACHE \
      --model-name $MODEL \
      --layer $LAYER \
      --expansion-factor $BEST_EF --top-k $BEST_K \
      --auxk $BEST_AUXK --auxk-coef 0.03125 \
      --init-pre-bias \
      --n-epochs 1 --batch-size 4096 --lr $LR \
      --log-interval 100 \
      --dp-size 8 --seed 42 \
      --wandb --wandb-project $WANDB_PROJECT \
      --wandb-group esm2_15b_sweep \
      --wandb-run-name "$RUN_NAME" \
      --output-dir outputs/sweep_15b/$RUN_NAME \
      --checkpoint-dir outputs/sweep_15b/$RUN_NAME/checkpoints
done
```

### Phase 3: Full Training Run

Take the best config from Phases 1+2 and train for multiple epochs.
Edit all `BEST_*` variables before running.

```bash
#!/bin/bash
set -e

CACHE=.cache/activations/esm2_15B_layer36_uniref50_3m_filtered
MODEL=nvidia/esm2_t48_15B_UR50D
LAYER=36
WANDB_PROJECT=sae_esm2_15b_sweep

# Fill in from Phase 1+2 results
BEST_EF=16
BEST_K=64
BEST_AUXK=128
BEST_LR=3e-4

RUN_NAME="final_ef${BEST_EF}_k${BEST_K}_lr${BEST_LR}"

torchrun --nproc_per_node=8 scripts/train.py \
    --cache-dir $CACHE \
    --model-name $MODEL \
    --layer $LAYER \
    --expansion-factor $BEST_EF --top-k $BEST_K \
    --auxk $BEST_AUXK --auxk-coef 0.03125 \
    --init-pre-bias \
    --n-epochs 5 --batch-size 4096 --lr $BEST_LR \
    --log-interval 100 \
    --dp-size 8 --seed 42 \
    --wandb --wandb-project $WANDB_PROJECT \
    --wandb-group esm2_15b_sweep \
    --wandb-run-name "$RUN_NAME" \
    --output-dir outputs/interplm_15b \
    --checkpoint-dir outputs/interplm_15b/checkpoints \
    --checkpoint-steps 5000
```

### Phase 4: Eval + Dashboard

```bash
#!/bin/bash
set -e

# Force re-download annotations with no score filter
rm -f data/swissprot_annotations.tsv.gz

# Fill in from best run
BEST_K=64

python scripts/eval.py \
    --checkpoint outputs/interplm_15b/checkpoints/checkpoint_final.pt \
    --top-k $BEST_K \
    --model-name nvidia/esm2_t48_15B_UR50D \
    --layer 36 \
    --batch-size 1 \
    --dtype bf16 \
    --num-proteins 5000 \
    --f1-max-proteins 50000 \
    --f1-min-positives 5 \
    --f1-threshold 0.2 \
    --normalization-n-proteins 5000 \
    --umap-n-neighbors 50 \
    --umap-min-dist 0.0 \
    --hdbscan-min-cluster-size 20 \
    --output-dir outputs/interplm_15b/eval
```

## Notes

- **Layer choice**: The 3M cache uses layer 36 (75% depth). InterPLM found
  middle-to-late layers most interpretable. Layer 24 (50% depth) is also viable
  but layer 36 should give richer features for the 15B model.
- **auxk**: Set to 2x top-k as a starting point. This controls the auxiliary
  dead-latent loss. Higher auxk = more aggressive dead latent revival.
- **Epochs**: 1 epoch for sweeps is sufficient since the 3M dataset is large.
  The full run uses 3-5 epochs for convergence.
- **Memory**: ef=16 with 15B (hidden_dim=5120) gives 81,920 latents.
  At batch_size=4096 this needs ~2-3GB per GPU. Should fit on 80GB A100s.
- **Streaming**: The 245G dataset won't fit in RAM. The trainer auto-detects
  this and streams from disk (>50GB threshold).
