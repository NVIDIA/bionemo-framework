#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch,backfill
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --job-name=healthcareeng_bionemo-download-dclm-shard
#SBATCH --mail-type=FAIL
#SBATCH --exclusive
set -euxo pipefail

# ============================================================================
# Download additional DCLM shards from Hugging Face.
# Usage:
#   sbatch download_dclm_shard.sh                  # downloads shard 02
#   SHARD=03 sbatch download_dclm_shard.sh         # downloads shard 03
#   SHARD="02 03 04" sbatch download_dclm_shard.sh # downloads shards 02-04
# ============================================================================

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
DATA_DIR="${SCRATCH}/data"

# Which shard(s) to download (default: 02)
: "${SHARD:=02}"

MOUNTS="${DATA_DIR}:/workspace/data"

# Build --include args for each shard
INCLUDE_ARGS=""
for S in $SHARD; do
    INCLUDE_ARGS="${INCLUDE_ARGS} --include global-shard_${S}_of_10/*"
done

: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

read -r -d '' COMMAND <<'OUTER_EOF' || true
set -euxo pipefail

echo "========================================="
echo "Downloading DCLM shard(s): ${SHARD}"
echo "========================================="

huggingface-cli download mlfoundations/dclm-baseline-1.0 \
    --repo-type dataset \
    ${INCLUDE_ARGS} \
    --local-dir /workspace/data/dclm-baseline

echo "========================================="
echo "Download complete! Contents:"
ls -la /workspace/data/dclm-baseline/
echo "========================================="
OUTER_EOF

# Inject credentials and variables into the command.
COMMAND="export SHARD=\"${SHARD}\"; export INCLUDE_ARGS=\"${INCLUDE_ARGS}\"; export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; ${COMMAND}"

echo "Downloading DCLM shard(s): ${SHARD}"

srun \
  --output "${DATA_DIR}/download-shard-%j.out" \
  --error  "${DATA_DIR}/download-shard-%j.err" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"
