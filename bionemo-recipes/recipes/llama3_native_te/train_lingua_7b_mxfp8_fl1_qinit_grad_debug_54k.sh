#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=8
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua7b.fl1-qinit-graddebug
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# FP8 gradient underflow analysis: resume FL1 qinit from step 54000.
# Uses TE's nvdlfw_inspect (quant_stats_config) to log per-layer:
#   - FP8 tensor stats: underflows%, scale_inv_min/max, MSE (activation/gradient/weight)
#   - Tensor stats: max, min, mean, std, l1_norm for dgrad/wgrad
# Runs 1000 steps (54000 -> 55000) with logging every step.
# Stats are written to log_quant_stats/ directory on each rank.
# Uses same checkpoint dir as the main convergence run.
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH}/data"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

# Use the SAME checkpoint dir as the main convergence run so we resume from step 54000
MAIN_EXP="lingua_7b_mxfp8_fl1_qinit_8n_prenyx"
export EXP_NAME="${EXP_NAME:-lingua_7b_mxfp8_fl1_qinit_grad_debug_54k}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}"
CKPT_ROOT="${SCRATCH}/checkpoints/${MAIN_EXP}"
QUANT_LOG_DIR="${RESULTS_DIR}/log_quant_stats"

mkdir -p "${RESULTS_DIR}" "${QUANT_LOG_DIR}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${DATA_DIR}:/workspace/data,${RESULTS_DIR}:${CODE_MOUNT}/results,${CKPT_ROOT}:${CODE_MOUNT}/checkpoints,${TE_DIR}:${TE_MOUNT}"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"

echo "========================================="
echo "FL1 Qinit FP8 Gradient Debug — Resume from 54000"
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

echo "Starting FP8 gradient debug run — resuming from 54000, target 55000..."
python train_fsdp2.py --config-name L2_lingua_7b_mxfp8_fl1_qinit \
  dataset.micro_batch_size=2 \
  dataset.use_stateful_dataloader=true \
  grad_acc_steps=2 \
  num_train_steps=55000 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  checkpoint.save_every_n_steps=999999 \
  checkpoint.resume_from_checkpoint=true \
  logger.frequency=10 \
  quant_stats_config.enabled=true \
  quant_stats_config.quant_stats_file=./fp8_debugging_stats.yaml \
  quant_stats_config.quant_log_dir=/workspace/bionemo/results/log_quant_stats \
  wandb.name=${EXP_NAME} \
  wandb.id=${EXP_NAME} \
  wandb.project=lingua-7b

echo "========================================="
echo "Gradient debug run complete!"
echo "Check FP8 stats in: /workspace/bionemo/results/log_quant_stats/"
echo "========================================="
OUTER_EOF

# Inject credentials into the command.
COMMAND="export EXP_NAME=\"${EXP_NAME}\"; export WANDB_API_KEY=\"${WANDB_API_KEY}\"; export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"

echo "Launching: ${EXP_NAME}"

# NO auto-chain — single run only
srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
