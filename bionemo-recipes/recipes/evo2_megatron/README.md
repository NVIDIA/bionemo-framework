# Evo2 Recipe

This recipe is work-in-progress rewrite of the nemo2 based bionemo/evo2 package into a self-contained
training repository that makes use of megatron bridge.

## Installation

```
# 1. Create venv (CRITICAL: include system packages so it sees the container's PyTorch)
export UV_LINK_MODE=copy
uv venv --system-site-packages --seed /workspace/.venv

# 2. Activate the environment
source /workspace/.venv/bin/activate
pip freeze | grep transformer_engine > pip-constraints.txt
uv pip install -r build_requirements.txt --no-build-isolation  # some extra requirements are needed for building
uv pip install -c pip-constraints.txt -e . --no-build-isolation
```

## Usage

```
# 3. Run an example job
## 2. if on a6000s, you may need to disable p2p to avoid crashing
export NCCL_P2P_DISABLE=1
## 3. Run the job:
torchrun --nproc-per-node 8 --no-python \
  train_evo2 \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_256 \
  --model-size striped_hyena_1b_nv_parallel --max-steps 12 --eval-interval 10 \
  --eval-iters 3 --mock-data \
  --micro-batch-size 32 --global-batch-size 256 --seq-length 1024 \
  --tensor-model-parallel 1 \
  --use-precision-aware-optimizer --dataset-seed 33 \
  --seed 41 --ckpt-async-save  --spike-no-more-embedding-init \
  --no-weight-decay-embeddings --cross-entropy-loss-fusion \
  --align-param-gather --overlap-param-gather  --grad-reduce-in-fp32 \
  --decay-steps 100 --warmup-steps 10 \
  --mixed-precision-recipe bf16-mixed \
  --no-fp32-residual-connection --activation-checkpoint-recompute-num-layers 1 \
  --attention-dropout 0.001 --hidden-dropout 0.001 \
  --eod-pad-in-loss-mask --enable-preemption \
  --log-interval 5 --debug-ddp-parity-freq 10 \
  --wandb-project evo2-recipes-verification-tmp \
  --wandb-run-name tmp_workstation_run_mock_data \
  --result-dir tmpbf16 --no-renormalize-loss
```

## Docker build

```
docker build -t evo2_megatron_recipe-$(git rev-parse --short HEAD) .
```
