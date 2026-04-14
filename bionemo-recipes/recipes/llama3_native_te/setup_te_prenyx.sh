#!/bin/bash
# ============================================================================
# TE Development Environment (following Jonathan Mitchell's approach)
#
# Uses enroot to create a persistent writable container, then builds or loads
# TransformerEngine from source with the correct GPU architecture.
#
# USAGE:
#   # One-time: build TE from source (takes ~30 min)
#   salloc --account=healthcareeng_bionemo --nodes=1 --ntasks-per-node=1 \
#     --time=01:00:00 --partition=batch
#   srun bash setup_te_prenyx.sh --build-te
#
#   # Quick start: load pre-built TE (takes seconds)
#   srun bash setup_te_prenyx.sh --copy-so
#
#   # Override arch (default: 103a for B300):
#   srun bash setup_te_prenyx.sh --build-te 100a
# ============================================================================
set -euo pipefail

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CODE_SRC="${SCRATCH}/bionemo-framework"
CODE_MOUNT="/workspace/bionemo"
TE_SRC="${SCRATCH}/TransformerEngine"
TE_MOUNT="/workspace/transformer_engine"
DATA_DIR="${SCRATCH}/data"
DATA_MOUNT="/workspace/data"

CONTAINER_IMAGE="${SCRATCH}/enroot/llama3_native_te.sqsh"
CONTAINER_NAME="bionemo-te-dev"

# ── Credentials ───────────────────────────────────────
: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

# ── Container creation (idempotent) ───────────────────
if ! enroot list | grep -q "^${CONTAINER_NAME}$"; then
    echo "Creating enroot container '${CONTAINER_NAME}'..."
    enroot create --name "${CONTAINER_NAME}" "${CONTAINER_IMAGE}"
fi

MODE="${1:-}"
NVTE_ARCH="${2:-103a}"

# ── Launch ─────────────────────────────────────────────
enroot start --rw \
    --mount "${CODE_SRC}:${CODE_MOUNT}" \
    --mount "${TE_SRC}:${TE_MOUNT}" \
    --mount "${DATA_DIR}:${DATA_MOUNT}" \
    --env HOME=/workspace \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
    --env MODE="${MODE}" \
    --env NVTE_ARCH="${NVTE_ARCH}" \
    --env TE_MOUNT="${TE_MOUNT}" \
    "${CONTAINER_NAME}" \
    bash -lc '
set -euo pipefail

echo "MODE=$MODE"
echo "NVTE_ARCH=$NVTE_ARCH"

case "$MODE" in
  --build-te)
    echo "Building TE for arch: $NVTE_ARCH"
    pip uninstall transformer-engine transformer-engine-torch -y 2>/dev/null || true
    cd "$TE_MOUNT"
    rm -rf build/cmake
    NVTE_USE_CCACHE=0 NVTE_CCACHE_BIN="" \
    NVTE_CUDA_ARCHS="$NVTE_ARCH" \
    NVTE_BUILD_THREADS_PER_JOB=4 \
    CUDA_CACHE_PATH=/.cache/ComputeCache \
    NVTE_FRAMEWORK=pytorch \
    pip install -v --no-build-isolation -e .
    exec bash
    ;;

  --copy-so)
    pip uninstall transformer-engine transformer-engine-torch -y 2>/dev/null || true
    cp "$TE_MOUNT"/transformer_engine_torch*.so "$TE_MOUNT"/transformer_engine/ 2>/dev/null || true
    cp "$TE_MOUNT"/transformer_engine_cu12*.so "$TE_MOUNT"/transformer_engine/ 2>/dev/null || true
    export PYTHONPATH="$TE_MOUNT:$PYTHONPATH"
    exec bash
    ;;

  *)
    exec bash
    ;;
esac
'
