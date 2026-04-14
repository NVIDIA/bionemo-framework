#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=4
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:55:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua7b.stopgo-v4-resume
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Stop-and-Go Diagnostic v4: STOP-AND-GO with StatefulDataLoader
# Resume from step 500 checkpoint, stop/restart every 500 steps to step 3500.
# Compare with the continuous run (train_lingua_7b_stopgo_v4_continuous_bia.sh).
#
# Same as v3 but with use_stateful_dataloader=true to preserve data ordering
# across checkpoint resumes. The pin_memory bug is fixed in 26.03 container.
#
# USAGE: Submit 6 times sequentially (wait for each to finish before next):
#   STOP_AT_STEP=1001 sbatch train_lingua_7b_stopgo_v4_resume_bia.sh   # 500 -> 1000
#   STOP_AT_STEP=1501 sbatch train_lingua_7b_stopgo_v4_resume_bia.sh   # 1000 -> 1500
#   STOP_AT_STEP=2001 sbatch train_lingua_7b_stopgo_v4_resume_bia.sh   # 1500 -> 2000
#   STOP_AT_STEP=2501 sbatch train_lingua_7b_stopgo_v4_resume_bia.sh   # 2000 -> 2500
#   STOP_AT_STEP=3001 sbatch train_lingua_7b_stopgo_v4_resume_bia.sh   # 2500 -> 3000
#   STOP_AT_STEP=3501 sbatch train_lingua_7b_stopgo_v4_resume_bia.sh   # 3000 -> 3500
# ============================================================================

CONTAINER="/lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te.sqsh"
CODE_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework"
DATA_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/data"

# Source checkpoint from the v2 stop-and-go experiment
CKPT_SRC="/lustre/fsw/healthcareeng_bionemo/savithas/checkpoints/lingua_7b_bf16_stopgo_resume_v2_bia"

STOP_AT_STEP="${STOP_AT_STEP:-1001}"

export EXP_NAME="${EXP_NAME:-stopgo_v4_resume}"
RESULTS_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/results/${EXP_NAME}"
CKPT_ROOT="/lustre/fsw/healthcareeng_bionemo/savithas/checkpoints/${EXP_NAME}"

mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}/train_fsdp2"

# Auto-copy step_500 checkpoint if not already present
if [ ! -d "${CKPT_ROOT}/train_fsdp2/step_500" ]; then
  echo "Copying step_500 checkpoint from ${CKPT_SRC}..."
  cp -r "${CKPT_SRC}/train_fsdp2/step_500" "${CKPT_ROOT}/train_fsdp2/"
fi

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
echo "Stop-and-Go v4: RESUME + StatefulDataLoader (-> step ${STOP_AT_STEP})"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_JOB_NUM_NODES}"
echo "========================================="

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

echo "Checkpoint dir contents:"
ls -la /workspace/bionemo/checkpoints/train_fsdp2/ 2>/dev/null || echo "(no checkpoints yet)"

python train_fsdp2.py --config-name L2_lingua_7b \
  dataset.micro_batch_size=2 \
  dataset.use_stateful_dataloader=true \
  grad_acc_steps=4 \
  num_train_steps=${STOP_AT_STEP} \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  checkpoint.save_every_n_steps=500 \
  checkpoint.resume_from_checkpoint=true \
  logger.frequency=10 \
  wandb.name=\${EXP_NAME} \
  wandb.id=\${EXP_NAME} \
  wandb.project=lingua-7b

echo "========================================="
echo "Stop-and-go segment complete (-> step ${STOP_AT_STEP})!"
echo "========================================="
EOF

echo "Launching: ${EXP_NAME} (stop at step ${STOP_AT_STEP})"
srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
