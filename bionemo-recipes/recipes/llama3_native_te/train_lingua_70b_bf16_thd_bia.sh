#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=4
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:55:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua70b.bf16-thd-bia
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Lingua 70B BF16 THD + CP=2 benchmark (4 nodes, bia B300)
# BF16 baseline for comparison against MXFP8 quantized init.
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH}/data"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-lingua_70b_bf16_thd_bench_4n_cp2_bia}"
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
echo "Lingua 70B BF16 THD Baseline (4 nodes, CP=2, bia B300)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "========================================="

# Use same TE main build as MXFP8 run for apples-to-apples comparison
TE_SITE="/usr/local/lib/python3.12/dist-packages/transformer_engine"
cp "$TE_MOUNT"/transformer_engine_torch*.so "$TE_SITE"/ 2>/dev/null || true
cp "$TE_MOUNT"/transformer_engine_cu12*.so "$TE_SITE"/ 2>/dev/null || true
export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"

python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

echo "Starting training..."
python train_fsdp2_cp.py --config-name L2_lingua_70b_thd \
  dataset.micro_batch_size=1 \
  dataset.pad_sequences_to_be_divisible_by=64 \
  grad_acc_steps=1 \
  num_train_steps=500 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  logger.frequency=10 \
  checkpoint.save_every_n_steps=999999 \
  checkpoint.resume_from_checkpoint=false \
  wandb.name=${EXP_NAME} \
  wandb.id=${EXP_NAME} \
  wandb.project=lingua-70b

echo "========================================="
echo "Training complete!"
echo "========================================="
OUTER_EOF

# Inject credentials into the command.
COMMAND="export EXP_NAME=\"${EXP_NAME}\"; export WANDB_API_KEY=\"${WANDB_API_KEY}\"; export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"

echo "Launching: ${EXP_NAME}"

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
