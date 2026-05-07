#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:59:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua70b.mxfp8-qi-hpiv-mbs1
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Lingua 70B MXFP8 THD WITH quantized init (preserve_high_precision_init_val=true)
# MBS=1, 1000 steps (1 node, bia B300)
# CP=2, dp_size=4. THD + sequence packing.
# Tests whether HPIV=true causes OOM on longer runs.
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH}/data"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-lingua_70b_mxfp8_qinit_hpiv_mbs1_1n_1k_bia}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}"
CKPT_ROOT="${SCRATCH}/checkpoints/${EXP_NAME}"

mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${DATA_DIR}:/workspace/data,${RESULTS_DIR}:${CODE_MOUNT}/results,${CKPT_ROOT}:${CODE_MOUNT}/checkpoints,${TE_DIR}:${TE_MOUNT}"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"

echo "========================================="
echo "Lingua 70B MXFP8 THD Quantized Init (HPIV=true) — MBS=1 Benchmark (1 node, bia B300)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "========================================="

export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
python -c "
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
import inspect
src = inspect.getsource(FusedAdam.step)
assert 'QuantizedTensor' in src, 'PR #2753 not found'
print('FusedAdam has QuantizedTensor support')
"

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

echo "Starting training..."
python train_fsdp2_cp.py --config-name L2_lingua_70b_mxfp8_qinit_thd \
  fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=true \
  dataset.micro_batch_size=1 \
  dataset.use_stateful_dataloader=true \
  dataset.pad_sequences_to_be_divisible_by=64 \
  grad_acc_steps=1 \
  num_train_steps=1000 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  logger.frequency=10 \
  checkpoint.save_every_n_steps=999999 \
  checkpoint.resume_from_checkpoint=false \
  wandb.name=${EXP_NAME} \
  wandb.id=${EXP_NAME} \
  wandb.project=lingua-70b

echo "Training complete!"
OUTER_EOF

COMMAND="export EXP_NAME=\"${EXP_NAME}\"; export WANDB_API_KEY=\"${WANDB_API_KEY}\"; export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"

echo "Launching: ${EXP_NAME}"

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
