#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=4
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:55:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua7b.stopgo-continuous
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Stop-and-Go Test: CONTINUOUS baseline (1000 steps, no interruption)
# Compare loss curve with the stop-and-go run to verify checkpoint resume.
# ============================================================================

CONTAINER="/lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te.sqsh"
CODE_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework"
DATA_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/data"

export EXP_NAME="${EXP_NAME:-lingua_7b_bf16_stopgo_continuous_bia}"
RESULTS_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/results/${EXP_NAME}"
CKPT_ROOT="/lustre/fsw/healthcareeng_bionemo/savithas/checkpoints/${EXP_NAME}"

mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

CONTAINER_WORKDIR="/workspace/bionemo"
MOUNTS="${CODE_DIR}:${CONTAINER_WORKDIR},${DATA_DIR}:/workspace/data,${RESULTS_DIR}:${CONTAINER_WORKDIR}/results,${CKPT_ROOT}:${CONTAINER_WORKDIR}/checkpoints"

read -r -d '' COMMAND <<EOF || true
export EXP_NAME="${EXP_NAME}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"

set -euxo pipefail

echo "========================================="
echo "Stop-and-Go Test: CONTINUOUS (1000 steps)"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_JOB_NUM_NODES}"
echo "========================================="

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

python train_fsdp2.py --config-name L2_lingua_7b \
  dataset.micro_batch_size=2 \
  dataset.use_stateful_dataloader=false \
  grad_acc_steps=4 \
  num_train_steps=1001 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  checkpoint.save_every_n_steps=999999 \
  checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  wandb.name=\${EXP_NAME} \
  wandb.id=\${EXP_NAME} \
  wandb.project=lingua-7b

echo "========================================="
echo "Continuous run complete!"
echo "========================================="
EOF

echo "Launching: ${EXP_NAME}"
srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
