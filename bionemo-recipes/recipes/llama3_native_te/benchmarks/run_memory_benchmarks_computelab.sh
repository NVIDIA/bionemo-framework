#!/bin/bash
# ============================================================================
# ComputeLab: Run all 6 memory profiler benchmarks (8B + 70B x BF16/MXFP8/qinit)
#
# Usage (on a ComputeLab node with 8 GPUs):
#   bash run_memory_benchmarks_computelab.sh
#
# Prerequisites:
#   - 8 GPUs visible (nvidia-smi should show 8)
#   - Docker available
#   - TE repo cloned and built for B300 (103a)
#   - DCLM data available
#   - WANDB_API_KEY and HUGGING_FACE_HUB_TOKEN set
# ============================================================================
set -euxo pipefail

# Paths on ComputeLab (code/outputs in scratch_other, large assets in scratch_other_1)
SCRATCH="/home/scratch.savithas_other"
SCRATCH_1="/home/scratch.savithas_other_1"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH_1}/data"
TE_DIR="${SCRATCH_1}/TransformerEngine"
RESULTS_BASE="${SCRATCH}/results"
SNAPSHOT_DIR="${SCRATCH}/memory_snapshots"
IMAGE="nvcr.io/nvidia/pytorch:26.03-py3"

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN}"

mkdir -p "${RESULTS_BASE}" "${SNAPSHOT_DIR}"

# Common docker args
DOCKER_ARGS=(
  --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
  -v "${CODE_DIR}:/workspace/bionemo"
  -v "${DATA_DIR}:/workspace/data"
  -v "${TE_DIR}:/workspace/transformer_engine"
  -v "${SNAPSHOT_DIR}:/workspace/memory_snapshots"
  -e WANDB_API_KEY="${WANDB_API_KEY}"
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  -e HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  -e HYDRA_FULL_ERROR=1
  -e WANDB_CONSOLE_SUMMARY_MAX_ROWS=20
)

# --------------------------------------------------------------------------
# 8B benchmarks: train_fsdp2.py, mbs=4
# --------------------------------------------------------------------------
COMMON_8B_ARGS=(
  "dataset.micro_batch_size=4"
  "dataset.use_stateful_dataloader=true"
  "grad_acc_steps=1"
  "num_train_steps=100"
  "checkpoint.save_every_n_steps=999999"
  "checkpoint.resume_from_checkpoint=false"
  "logger.frequency=10"
  "memory_profiler.enabled=true"
  "memory_profiler.snapshot_after_first_step=true"
  "wandb.project=lingua-memory"
  "hydra.run.dir=/tmp/hydra_outputs"
)

RESULTS_CAPTURE="${SNAPSHOT_DIR}/results"
mkdir -p "${RESULTS_CAPTURE}"

run_8b() {
  local name="$1"
  local config="$2"
  shift 2
  local extra_args=("$@")

  echo ""
  echo "========================================="
  echo "Running 8B: ${name}"
  echo "========================================="

  local snap_dir="/workspace/memory_snapshots/${name}"
  mkdir -p "${SNAPSHOT_DIR}/${name}" "${RESULTS_BASE}/${name}"

  docker run "${DOCKER_ARGS[@]}" \
    -v "${RESULTS_BASE}/${name}:/workspace/bionemo/results" \
    "${IMAGE}" \
    bash -c "
set -euxo pipefail
export PYTHONPATH=/workspace/transformer_engine:\${PYTHONPATH:-}
python -c \"import transformer_engine; print(f'TE version: {transformer_engine.__version__}')\"
cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te
torchrun --nproc_per_node=8 train_fsdp2.py --config-name ${config} \
  ${COMMON_8B_ARGS[*]} \
  memory_profiler.snapshot_dir=${snap_dir} \
  wandb.name=${name} \
  wandb.id=${name} \
  ${extra_args[*]:-}
echo 'Done: ${name}'
" 2>&1 | tee "${RESULTS_CAPTURE}/${name}.log"
  echo "Snapshot: ${SNAPSHOT_DIR}/${name}/"
}

# --------------------------------------------------------------------------
# 70B benchmarks: train_fsdp2_cp.py, CP=4, mbs=1
# --------------------------------------------------------------------------
COMMON_70B_ARGS=(
  "num_train_steps=100"
  "checkpoint.save_every_n_steps=999999"
  "checkpoint.resume_from_checkpoint=false"
  "logger.frequency=10"
  "memory_profiler.enabled=true"
  "memory_profiler.snapshot_after_first_step=true"
  "wandb.project=lingua-memory"
  "hydra.run.dir=/tmp/hydra_outputs"
)

run_70b() {
  local name="$1"
  shift
  local extra_args=("$@")

  echo ""
  echo "========================================="
  echo "Running 70B: ${name}"
  echo "========================================="

  local snap_dir="/workspace/memory_snapshots/${name}"
  mkdir -p "${SNAPSHOT_DIR}/${name}" "${RESULTS_BASE}/${name}"

  docker run "${DOCKER_ARGS[@]}" \
    -v "${RESULTS_BASE}/${name}:/workspace/bionemo/results" \
    "${IMAGE}" \
    bash -c "
set -euxo pipefail
export PYTHONPATH=/workspace/transformer_engine:\${PYTHONPATH:-}
python -c \"import transformer_engine; print(f'TE version: {transformer_engine.__version__}')\"
cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name L2_lingua_70b_computelab \
  ${COMMON_70B_ARGS[*]} \
  memory_profiler.snapshot_dir=${snap_dir} \
  wandb.name=${name} \
  wandb.id=${name} \
  ${extra_args[*]:-}
echo 'Done: ${name}'
" 2>&1 | tee "${RESULTS_CAPTURE}/${name}.log"
  echo "Snapshot: ${SNAPSHOT_DIR}/${name}/"
}

# ========================== 8B BENCHMARKS ==========================

# 1. 8B BF16
run_8b "8b_bf16_mbs4" "L2_lingua_7b"

# 2. 8B MXFP8 no qinit (all layers FP8)
run_8b "8b_mxfp8_no_qinit_mbs4" "L2_lingua_7b_mxfp8" "fp8_layers=null"

# 3. 8B MXFP8 qinit (preserve_high_precision_init_val=false)
run_8b "8b_mxfp8_qinit_mbs4" "L2_lingua_7b_mxfp8_qinit" \
  "fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=false"

# ========================== 70B BENCHMARKS ==========================

# 4. 70B BF16 (CP=4 from L2_lingua_70b_computelab)
run_70b "70b_bf16_mbs1"

# 5. 70B MXFP8 no qinit (all layers FP8)
run_70b "70b_mxfp8_no_qinit_mbs1" \
  "fp8_config.enabled=true" \
  "fp8_config.fp8_recipe=transformer_engine.common.recipe.MXFP8BlockScaling" \
  "fp8_config.fp8_format=E4M3" \
  "fp8_layers=null"

# 6. 70B MXFP8 qinit (preserve_high_precision_init_val=false)
run_70b "70b_mxfp8_qinit_mbs1" \
  "fp8_config.enabled=true" \
  "fp8_config.fp8_recipe=transformer_engine.common.recipe.MXFP8BlockScaling" \
  "fp8_config.fp8_format=E4M3" \
  "fp8_config.quantized_model_init_kwargs.enabled=true" \
  "fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=false"

echo ""
echo "========================================="
echo "All 6 benchmarks complete!"
echo "Memory snapshots in: ${SNAPSHOT_DIR}/"
echo "  8b_bf16_mbs4/"
echo "  8b_mxfp8_no_qinit_mbs4/"
echo "  8b_mxfp8_qinit_mbs4/"
echo "  70b_bf16_mbs1/"
echo "  70b_mxfp8_no_qinit_mbs1/"
echo "  70b_mxfp8_qinit_mbs1/"
echo "View .pickle files at: https://pytorch.org/memory_viz"
echo "========================================="
