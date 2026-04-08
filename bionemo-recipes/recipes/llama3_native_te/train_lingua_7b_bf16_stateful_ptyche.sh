#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=8
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=4
#SBATCH --time=03:55:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua7b.bf16.stateful
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# PATHS (ptyche uses ARM Grace Hopper - pull container image directly)
# ============================================================================
CONTAINER="nvcr.io/nvidia/pytorch:26.02-py3"
CODE_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework"
DATA_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/data"

# Fresh experiment with stateful dataloader
export EXP_NAME="${EXP_NAME:-lingua_7b_bf16_stateful_8n_ptyche}"
RESULTS_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/results/${EXP_NAME}"
CKPT_ROOT="/lustre/fsw/healthcareeng_bionemo/savithas/checkpoints/${EXP_NAME}"

mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}"

# ============================================================================
# SECRETS - read from environment (set in ~/.bashrc, never hardcode here)
# ============================================================================
: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

# ============================================================================
# CONTAINER MOUNTS
# ============================================================================
CONTAINER_WORKDIR="/workspace/bionemo"
MOUNTS="${CODE_DIR}:${CONTAINER_WORKDIR},${DATA_DIR}:/workspace/data,${RESULTS_DIR}:${CONTAINER_WORKDIR}/results,${CKPT_ROOT}:${CONTAINER_WORKDIR}/checkpoints"

# ============================================================================
# TRAINING COMMAND
# ============================================================================
read -r -d '' COMMAND <<EOF || true
# Export secrets without tracing (no set -x yet)
export EXP_NAME="${EXP_NAME}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"

set -euxo pipefail

echo "========================================="
echo "Starting Lingua 7B BF16 Training w/ Stateful DL (8 nodes, ptyche ARM)"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_JOB_NUM_NODES}"
echo "Tasks per node: \${SLURM_NTASKS_PER_NODE}"
echo "========================================="

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

# Only local rank 0 installs packages to avoid concurrent pip conflicts
if [ "\${SLURM_LOCALID}" = "0" ]; then
  echo "Local rank 0: installing requirements..."
  pip install -r requirements.txt
  hash -r
  touch /tmp/pip_install_done
else
  echo "Local rank \${SLURM_LOCALID}: waiting for pip install..."
  while [ ! -f /tmp/pip_install_done ]; do sleep 2; done
  echo "pip install done, proceeding."
fi

echo "Verifying mounts..."
ls -la /workspace/data/dclm-baseline/global-shard_01_of_10/ | head -5
echo "Checkpoints:" && ls -la /workspace/bionemo/checkpoints/
echo "Results:" && ls -la /workspace/bionemo/results/

echo "Starting training..."
python train_fsdp2.py --config-name L2_lingua_7b \
  dataset.micro_batch_size=2 \
  dataset.use_stateful_dataloader=true \
  grad_acc_steps=4 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  checkpoint.save_every_n_steps=1000 \
  checkpoint.resume_from_checkpoint=true \
  wandb.name=\${EXP_NAME} \
  wandb.id=\${EXP_NAME} \
  wandb.project=lingua-7b

echo "========================================="
echo "Training complete!"
echo "========================================="
EOF

# ============================================================================
# LAUNCH JOB
# ============================================================================
echo "Launching training: ${EXP_NAME}"
echo "Results: ${RESULTS_DIR}"
echo "Checkpoints: ${CKPT_ROOT}"

# AUTO-CHAIN: always resubmit on exit (success, failure, or timeout).
# Uses --dependency=singleton so only one job with this name runs at a time.
# To stop chaining: scancel the queued job.
trap 'echo "Resubmitting for next chain..."; sbatch --dependency=singleton "${BASH_SOURCE[0]}"; echo "Job finished! Check: ${RESULTS_DIR}"' EXIT

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
