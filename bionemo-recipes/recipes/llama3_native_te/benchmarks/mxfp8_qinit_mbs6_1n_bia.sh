#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:30:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua7b.mxfp8-mbs6
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Lingua 7B MXFP8 quantized init — MBS=6 benchmark (1 node, bia B300)
# Single node perf benchmark: max MBS sweep with grad_acc=1.
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH}/data"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-lingua_7b_mxfp8_qinit_mbs6_1n_bia}"
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
echo "Lingua 7B MXFP8 Quantized Init — MBS=6 Benchmark (1 node, bia B300)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "========================================="

export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"

python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

echo "Starting training..."
python train_fsdp2.py --config-name L2_lingua_7b_mxfp8_qinit \
  dataset.micro_batch_size=6 \
  dataset.use_stateful_dataloader=true \
  grad_acc_steps=1 \
  num_train_steps=1000 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  logger.frequency=10 \
  checkpoint.save_every_n_steps=999999 \
  checkpoint.resume_from_checkpoint=false \
  wandb.name=${EXP_NAME} \
  wandb.id=${EXP_NAME} \
  wandb.project=lingua-7b

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
