#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-mp.fs-mxfp8
#SBATCH --mail-type=FAIL
#SBATCH --overcommit
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Memory Profiler: TE fully_shard example — MXFP8 no qinit
# Small model (256 hidden, 3 layers, 2 GPUs) for clean qinit analysis.
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
TE_DIR="${SCRATCH}/TransformerEngine"

CODE_MOUNT="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"

export EXP_NAME="${EXP_NAME:-mp_fully_shard_mxfp8}"
RESULTS_DIR="${SCRATCH}/results/${EXP_NAME}"
SNAP_DIR="${SCRATCH}/memory_snapshots/fully_shard_mxfp8"

mkdir -p "${RESULTS_DIR}" "${SNAP_DIR}"

MOUNTS="${CODE_DIR}:${CODE_MOUNT},${TE_DIR}:${TE_MOUNT},${SNAP_DIR}:/workspace/snapshots"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

TE_MOUNT="/workspace/transformer_engine"
SCRIPT="/workspace/bionemo/bionemo-recipes/recipes/llama3_native_te/benchmarks/fully_shard_memory_profile.py"

echo "========================================="
echo "Memory Profiler: fully_shard — MXFP8 no qinit"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

export PYTHONPATH="$TE_MOUNT:${PYTHONPATH:-}"

python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"

torchrun --nproc-per-node 2 $SCRIPT --mxfp8-no-qinit --snapshot-dir /workspace/snapshots

echo "========================================="
echo "Done! Snapshot: /workspace/snapshots/mxfp8_no_qinit/memory_snapshot.pickle"
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
