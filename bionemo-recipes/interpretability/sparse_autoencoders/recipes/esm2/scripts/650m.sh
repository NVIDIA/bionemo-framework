#!/bin/bash
set -e

echo "============================================================"
echo "STEP 1: Extract activations from ESM2-650M"
echo "============================================================"

torchrun --nproc_per_node=4 scripts/extract.py \
    --source uniref50 \
    --num-proteins 5000 \
    --data-dir ./data \
    --layer 24 \
    --model-name nvidia/esm2_t33_650M_UR50D \
    --batch-size 16 \
    --max-length 1024 \
    --filter-length \
    --output .cache/activations/650m_5k_layer24

echo ""
echo "============================================================"
echo "STEP 2: Train SAE on cached activations"
echo "============================================================"

torchrun --nproc_per_node=4 scripts/train.py \
    --cache-dir .cache/activations/650m_5k_layer24 \
    --model-name nvidia/esm2_t33_650M_UR50D \
    --layer 24 \
    --model-type topk \
    --expansion-factor 8 \
    --top-k 32 \
    --auxk 64 \
    --auxk-coef 0.03125 \
    --init-pre-bias \
    --n-epochs 3 \
    --batch-size 4096 \
    --lr 3e-4 \
    --log-interval 50 \
    --no-wandb \
    --dp-size 4 \
    --seed 42 \
    --num-proteins 5000 \
    --output-dir "$(pwd)/outputs/650m_5k" \
    --checkpoint-dir "$(pwd)/outputs/650m_5k/checkpoints" \
    --checkpoint-steps 999999

echo ""
echo "============================================================"
echo "STEP 3: Evaluate SAE + build dashboard"
echo "============================================================"

python scripts/eval.py \
    --checkpoint ./outputs/650m_5k/checkpoints/checkpoint_final.pt \
    --top-k 32 \
    --model-name nvidia/esm2_t33_650M_UR50D \
    --layer 24 \
    --batch-size 16 \
    --dtype bf16 \
    --num-proteins 2000 \
    --f1-max-proteins 8000 \
    --f1-min-positives 10 \
    --f1-threshold 0.3 \
    --normalization-n-proteins 3000 \
    --umap-n-neighbors 50 \
    --umap-min-dist 0.0 \
    --hdbscan-min-cluster-size 20 \
    --output-dir ./outputs/650m_5k/eval

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
