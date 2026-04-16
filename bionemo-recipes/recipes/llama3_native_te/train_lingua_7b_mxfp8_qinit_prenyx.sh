#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=8
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:55:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-lingua7b.mxfp8-qinit
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Lingua 7B MXFP8 with quantized model init experiment
# Tests convergence with FP8-only weights + FP32 master weights in FusedAdam
#
# PREREQUISITE: Build TE once using setup_te_prenyx.sh:
#   salloc --account=healthcareeng_bionemo --nodes=1 --ntasks-per-node=1 \
#     --time=01:00:00 --partition=batch
#   bash setup_te_prenyx.sh --build-te
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH}/data"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-lingua_7b_mxfp8_qinit_v7_no_stateful_dl_prenyx}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}"
# Resume from v6 checkpoints (step 9000) to test without stateful dataloader
CKPT_ROOT="${SCRATCH}/checkpoints/lingua_7b_mxfp8_qinit_v6_te_main_8n_prenyx"

mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${DATA_DIR}:/workspace/data,${RESULTS_DIR}:${CODE_MOUNT}/results,${CKPT_ROOT}:${CODE_MOUNT}/checkpoints,${TE_DIR}:${TE_MOUNT}"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"

echo "========================================="
echo "Lingua 7B MXFP8 Quantized Model Init (8 nodes, prenyx)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "========================================="

# TE setup: overwrite container's native libs with our Blackwell-compiled build,
# then use PYTHONPATH for TE main Python code. No pip uninstall needed (avoids
# sanity check failures), no copy to TE subdir (avoids duplicate file errors).
TE_SITE="/usr/local/lib/python3.12/dist-packages/transformer_engine"
cp "$TE_MOUNT"/transformer_engine_torch*.so "$TE_SITE"/ 2>/dev/null || true
cp "$TE_MOUNT"/transformer_engine_cu12*.so "$TE_SITE"/ 2>/dev/null || true
export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"

# Verify TE has QuantizedTensor support in FusedAdam (PR #2753)
python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
python -c "
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
import inspect
src = inspect.getsource(FusedAdam.step)
assert 'QuantizedTensor' in src, 'PR #2753 not found'
print('FusedAdam has QuantizedTensor support')
"

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

python train_fsdp2.py --config-name L2_lingua_7b_mxfp8_qinit \
  dataset.micro_batch_size=2 \
  dataset.use_stateful_dataloader=false \
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

# Inject credentials into the command
COMMAND="export EXP_NAME=\"${EXP_NAME}\"; export WANDB_API_KEY=\"${WANDB_API_KEY}\"; export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"

echo "Launching: ${EXP_NAME}"

# AUTO-CHAIN disabled for this diagnostic run.
# trap 'echo "Resubmitting for next chain..."; sbatch --dependency=singleton "${BASH_SOURCE[0]}"; echo "Job finished! Check: ${RESULTS_DIR}"' EXIT

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
