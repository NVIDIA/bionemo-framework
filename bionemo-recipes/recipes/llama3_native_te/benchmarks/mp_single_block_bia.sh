#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-mp.single-block
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Single-Block Memory Profiler — runs all 4 modes sequentially
#
# Produces snapshots for:
#   bare          BF16 baseline, no FSDP2
#   mxfp8         MXFP8 + qinit + HPIV, no FSDP2
#   bare-fsdp2    BF16 + FSDP2 (2 GPUs)
#   mxfp8-fsdp2   MXFP8 + qinit + HPIV + FSDP2 (2 GPUs)
#
# Uses 70B Llama single-layer dimensions (~973M params, ~1.95 GB BF16).
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-mp_single_block}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}"
SNAP_DIR="${SCRATCH}/memory_snapshots/single_block"

mkdir -p "${RESULTS_DIR}" "${SNAP_DIR}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${TE_DIR}:${TE_MOUNT},${SNAP_DIR}:/workspace/snapshots"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"
SCRIPT="/workspace/bionemo/bionemo-recipes/recipes/llama3_native_te/benchmarks/single_block_memory_profile.py"

echo "========================================="
echo "Single-Block Memory Profiler (all 4 modes)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"
python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo ""
echo "=== Mode 1/4: bare (BF16, no FSDP2) ==="
python $SCRIPT --mode bare --snapshot-dir /workspace/snapshots

echo ""
echo "=== Mode 2/4: mxfp8 (MXFP8 + qinit, no FSDP2) ==="
python $SCRIPT --mode mxfp8 --snapshot-dir /workspace/snapshots

echo ""
echo "=== Mode 3/4: bare-fsdp2 (BF16 + FSDP2, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode bare-fsdp2 --snapshot-dir /workspace/snapshots

echo ""
echo "=== Mode 4/4: mxfp8-fsdp2 (MXFP8 + qinit + FSDP2, 2 GPUs) ==="
torchrun --nproc-per-node 2 $SCRIPT --mode mxfp8-fsdp2 --snapshot-dir /workspace/snapshots

echo ""
echo "========================================="
echo "Done! Snapshots in /workspace/snapshots/"
ls -la /workspace/snapshots/*/memory_snapshot.pickle
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
