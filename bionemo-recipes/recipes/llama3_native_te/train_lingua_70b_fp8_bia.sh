#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=4
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:55:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua70b.fp8-bench
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# PATHS
# ============================================================================
CONTAINER="/lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te.sqsh"
CODE_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework"
DATA_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/data"

export EXP_NAME="${EXP_NAME:-lingua_70b_mxfp8_bench_4n_bia}"
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
export EXP_NAME="${EXP_NAME}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"

set -euxo pipefail

echo "========================================="
echo "Starting Lingua 70B MXFP8 Benchmark (4 nodes, bia, GBS=32)"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_JOB_NUM_NODES}"
echo "Tasks per node: \${SLURM_NTASKS_PER_NODE}"
echo "========================================="

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

echo "Verifying mounts..."
ls -la /workspace/data/dclm-baseline/global-shard_01_of_10/ | head -5
echo "Checkpoints:" && ls -la /workspace/bionemo/checkpoints/
echo "Results:" && ls -la /workspace/bionemo/results/

echo "Starting training..."
python train_fsdp2_cp.py --config-name L2_lingua_70b_mxfp8 \
  dataset.micro_batch_size=1 \
  grad_acc_steps=1 \
  num_train_steps=300 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  checkpoint.save_every_n_steps=999999 \
  checkpoint.resume_from_checkpoint=false \
  wandb.name=\${EXP_NAME} \
  wandb.id=\${EXP_NAME} \
  wandb.project=lingua-70b

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

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
