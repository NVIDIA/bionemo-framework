#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=8
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:55:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua7b.mxfp8-fl1-qinit-s2
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Resume Lingua 7B MXFP8 FL1 + qinit from step 54000 with shard_02 data.
# Dataset shard_01 exhausted at ~54k steps. This resumes on fresh data
# under the SAME wandb run ID so the loss curve continues seamlessly.
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH}/data"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

# Same wandb name/id as the original run so the curve continues.
export EXP_NAME="${EXP_NAME:-lingua_7b_mxfp8_fl1_qinit_8n_prenyx}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}_shard2"
CKPT_ROOT="${SCRATCH}/checkpoints/${EXP_NAME}_shard2"

# Copy step_54000 from original run to isolated checkpoint dir (one-time).
ORIG_CKPT="${SCRATCH}/checkpoints/${EXP_NAME}/train_fsdp2/step_54000"
mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}/train_fsdp2"
if [ ! -d "${CKPT_ROOT}/train_fsdp2/step_54000" ]; then
    echo "Copying step_54000 checkpoint to isolated dir..."
    cp -a "${ORIG_CKPT}" "${CKPT_ROOT}/train_fsdp2/step_54000"
fi

: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${DATA_DIR}:/workspace/data,${RESULTS_DIR}:${CODE_MOUNT}/results,${CKPT_ROOT}:${CODE_MOUNT}/checkpoints,${TE_DIR}:${TE_MOUNT}"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"

echo "========================================="
echo "Lingua 7B MXFP8 FL1 + Quantized Init — Resume from 54000 with shard_02"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "========================================="

# TE setup
TE_SITE="/usr/local/lib/python3.12/dist-packages/transformer_engine"
cp "$TE_MOUNT"/transformer_engine_torch*.so "$TE_SITE"/ 2>/dev/null || true
cp "$TE_MOUNT"/transformer_engine_cu12*.so "$TE_SITE"/ 2>/dev/null || true
export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

python train_fsdp2.py --config-name L2_lingua_7b_mxfp8_fl1_qinit \
  dataset.micro_batch_size=2 \
  dataset.use_stateful_dataloader=true \
  'dataset.load_dataset_kwargs.data_files=/workspace/data/dclm-baseline/global-shard_02_of_10/**/*.parquet' \
  grad_acc_steps=2 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  checkpoint.save_every_n_steps=1500 \
  checkpoint.resume_from_checkpoint=true \
  logger.frequency=100 \
  wandb.name=${EXP_NAME} \
  wandb.id=${EXP_NAME} \
  wandb.project=lingua-7b

echo "========================================="
echo "Training complete!"
echo "========================================="
OUTER_EOF

# Inject credentials into the command.
COMMAND="export EXP_NAME=\"${EXP_NAME}\"; export WANDB_API_KEY=\"${WANDB_API_KEY}\"; export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"

echo "Launching: ${EXP_NAME} (shard_02 resume)"

# AUTO-CHAIN: always resubmit on exit (success, failure, or timeout).
trap 'echo "Resubmitting for next chain..."; sbatch --dependency=singleton "${BASH_SOURCE[0]}"; echo "Job finished! Check: ${RESULTS_DIR}"' EXIT

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
