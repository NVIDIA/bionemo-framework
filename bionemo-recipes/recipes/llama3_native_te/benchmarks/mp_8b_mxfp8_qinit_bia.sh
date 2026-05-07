#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-mp.8b-mxfp8-qi
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Memory Profiler: Lingua 8B MXFP8 qinit (preserve_high_precision_init_val=false)
# MBS=4 (1 node, bia B300)
# Runs 10 steps, dumps torch memory snapshot after step 0.
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
DATA_DIR="${SCRATCH}/data"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-mp_8b_mxfp8_qinit}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}"
CKPT_ROOT="${SCRATCH}/checkpoints/${EXP_NAME}"
SNAP_DIR="${SCRATCH}/memory_snapshots/8b_mxfp8_qinit"

mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}" "${SNAP_DIR}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${DATA_DIR}:/workspace/data,${RESULTS_DIR}:${CODE_MOUNT}/results,${CKPT_ROOT}:${CODE_MOUNT}/checkpoints,${TE_DIR}:${TE_MOUNT},${SNAP_DIR}:/workspace/snapshots"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"

echo "========================================="
echo "Memory Profiler: 8B MXFP8 qinit (no HPIV) — MBS=4 (1 node, bia B300)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "========================================="

export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"

python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
python -c "
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
import inspect
src = inspect.getsource(FusedAdam.step)
assert 'QuantizedTensor' in src, 'PR #2753 not found'
print('FusedAdam has QuantizedTensor support')
"

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

echo "Starting memory profiler run..."
python train_fsdp2.py --config-name L2_lingua_7b_mxfp8_qinit \
  fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=false \
  dataset.micro_batch_size=4 \
  dataset.use_stateful_dataloader=true \
  grad_acc_steps=1 \
  num_train_steps=10 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  logger.frequency=1 \
  checkpoint.save_every_n_steps=999999 \
  checkpoint.resume_from_checkpoint=false \
  memory_profiler.enabled=true \
  memory_profiler.snapshot_dir=/workspace/snapshots \
  wandb.name=${EXP_NAME} \
  wandb.id=${EXP_NAME} \
  wandb.project=lingua-7b

echo "========================================="
echo "Memory profiler complete!"
echo "Snapshot: /workspace/snapshots/memory_snapshot.pickle"
echo "========================================="
OUTER_EOF

COMMAND="export EXP_NAME=\"${EXP_NAME}\"; export WANDB_API_KEY=\"${WANDB_API_KEY}\"; export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"

echo "Launching: ${EXP_NAME}"

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
