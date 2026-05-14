#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-mp.single-block
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Memory Profiler — NEW experiments only (original 4 modes already collected)
#
# Tests two TE FSDP2 memory issues (Varun Thumbe):
#
#   Issue 2 (quantized weights not freed without qinit):
#     fp8-no-qinit           BF16 weights + FP8 autocast, no FSDP2
#     fp8-no-qinit-fsdp2     BF16 weights + FP8 autocast + FSDP2
#
#   Issue 1 (transpose accumulation with qinit + FSDP2):
#     bare-fsdp2-4L          BF16 + FSDP2, 4 layers (baseline)
#     mxfp8-fsdp2-4L         MXFP8 + qinit + FSDP2, 4 layers
#     fp8-no-qinit-fsdp2-4L  FP8 autocast + FSDP2, 4 layers
#
# Uses 8B Llama dimensions (~490M params/layer) for manageable snapshot sizes.
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-mp_single_block_v2}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}"
SNAP_DIR="${SCRATCH}/memory_snapshots/single_block"

mkdir -p "${RESULTS_DIR}" "${SNAP_DIR}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${TE_DIR}:${TE_MOUNT},${SNAP_DIR}:/workspace/snapshots"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"
SCRIPT="/workspace/bionemo/bionemo-recipes/recipes/llama3_native_te/benchmarks/single_block_memory_profile.py"

echo "========================================="
echo "Memory Profiler — new experiments"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"
python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# --- Issue 2: FP8 autocast without qinit (quantized weights not freed) ---

echo ""
echo "=== 1/5: fp8-no-qinit (FP8 autocast, BF16 weights, no FSDP2) ==="
python $SCRIPT --mode fp8-no-qinit --snapshot-dir /workspace/snapshots

echo ""
echo "=== 2/5: fp8-no-qinit-fsdp2 (FP8 autocast, BF16 weights + FSDP2, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode fp8-no-qinit-fsdp2 --snapshot-dir /workspace/snapshots

# --- Issue 1: 4-layer FSDP2 for cross-layer transpose accumulation ---

echo ""
echo "=== 3/5: bare-fsdp2 4-layer (BF16 baseline + FSDP2, 4 layers, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode bare-fsdp2 --num-layers 4 --snapshot-dir /workspace/snapshots

echo ""
echo "=== 4/5: mxfp8-fsdp2 4-layer (MXFP8 + qinit + FSDP2, 4 layers, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode mxfp8-fsdp2 --num-layers 4 --snapshot-dir /workspace/snapshots

echo ""
echo "=== 5/5: fp8-no-qinit-fsdp2 4-layer (FP8 autocast + FSDP2, 4 layers, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode fp8-no-qinit-fsdp2 --num-layers 4 --snapshot-dir /workspace/snapshots

echo ""
echo "========================================="
echo "Done! Snapshots in /workspace/snapshots/"
find /workspace/snapshots -name "memory_snapshot.pickle" | sort
echo "========================================="
OUTER_EOF

COMMAND="export EXP_NAME=\"${EXP_NAME}\"; ${COMMAND}"

echo "Launching: ${EXP_NAME}"

srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
