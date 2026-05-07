#!/bin/bash
# ============================================================================
# ComputeLab: Run 8B memory profiler benchmarks (BF16, MXFP8, MXFP8+qinit)
#
# Usage (on a ComputeLab node with 8 GPUs):
#   bash run_8b_memory_benchmarks_computelab.sh
#
# Prerequisites:
#   - 8 GPUs visible (nvidia-smi should show 8)
#   - Docker available
#   - TE repo cloned and built for B300 (103a)
#   - DCLM data available
#   - WANDB_API_KEY and HUGGING_FACE_HUB_TOKEN set
# ============================================================================
set -euxo pipefail

# Paths on ComputeLab
SCRATCH="/home/scratch.savithas_other_1"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH}/data"
TE_DIR="${SCRATCH}/TransformerEngine"
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
)

# Common training args
COMMON_ARGS=(
  "dataset.micro_batch_size=4"
  "dataset.use_stateful_dataloader=true"
  "grad_acc_steps=1"
  "num_train_steps=100"
  "checkpoint.save_every_n_steps=999999"
  "checkpoint.resume_from_checkpoint=false"
  "logger.frequency=10"
  "memory_profiler.enabled=true"
  "memory_profiler.snapshot_dir=/workspace/memory_snapshots"
  "memory_profiler.snapshot_after_first_step=true"
  "wandb.project=lingua-7b-memory"
)

run_benchmark() {
  local name="$1"
  local config="$2"
  shift 2
  local extra_args=("$@")

  echo ""
  echo "========================================="
  echo "Running: ${name}"
  echo "========================================="

  # Create per-run snapshot subdir
  local snap_dir="/workspace/memory_snapshots/${name}"

  docker run "${DOCKER_ARGS[@]}" \
    -v "${RESULTS_BASE}/${name}:/workspace/bionemo/results" \
    "${IMAGE}" \
    bash -c "
set -euxo pipefail
export PYTHONPATH=/workspace/transformer_engine:\${PYTHONPATH:-}
python -c \"import transformer_engine; print(f'TE version: {transformer_engine.__version__}')\"
cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te
mkdir -p ${snap_dir}
torchrun --nproc_per_node=8 train_fsdp2.py --config-name ${config} \
  ${COMMON_ARGS[*]} \
  memory_profiler.snapshot_dir=${snap_dir} \
  wandb.name=${name} \
  wandb.id=${name} \
  ${extra_args[*]:-}
echo 'Done: ${name}'
"

  echo "Snapshot saved to: ${SNAPSHOT_DIR}/${name}/"
}

# ---------- 1. BF16 baseline ----------
run_benchmark \
  "8b_bf16_mbs4_computelab" \
  "L2_lingua_7b"

# ---------- 2. MXFP8 no quantized init (all layers FP8) ----------
run_benchmark \
  "8b_mxfp8_no_qinit_mbs4_computelab" \
  "L2_lingua_7b_mxfp8" \
  "fp8_layers=null"

# ---------- 3. MXFP8 quantized init (preserve_high_precision_init_val=false) ----------
run_benchmark \
  "8b_mxfp8_qinit_mbs4_computelab" \
  "L2_lingua_7b_mxfp8_qinit" \
  "fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=false"

echo ""
echo "========================================="
echo "All 3 benchmarks complete!"
echo "Memory snapshots in: ${SNAPSHOT_DIR}/"
echo "View at: https://pytorch.org/memory_viz"
echo "========================================="
