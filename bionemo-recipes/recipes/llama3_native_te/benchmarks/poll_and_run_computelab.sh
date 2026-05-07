#!/bin/bash
# ============================================================================
# ComputeLab: Poll for an 8-GPU B300 node, then run memory benchmarks.
#
# Run this inside screen/tmux on the ComputeLab login node:
#   screen -S benchmarks
#   bash poll_and_run_computelab.sh
#   # Ctrl-A D to detach, screen -r benchmarks to reattach
#
# What it does:
#   1. Polls sinfo every 60s for an idle 8-GPU node
#   2. When found, allocates the node via srun
#   3. Runs all 6 memory profiler benchmarks (3x 8B + 3x 70B)
# ============================================================================
set -euo pipefail

POLL_INTERVAL=60  # seconds between checks
PARTITION="b300"  # adjust if your partition name differs
GPUS_PER_NODE=8
TIME_LIMIT="04:00:00"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/run_memory_benchmarks_computelab.sh"

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN}"

echo "Polling for idle 8-GPU node in partition '${PARTITION}'..."
echo "Poll interval: ${POLL_INTERVAL}s"
echo "Benchmark script: ${BENCHMARK_SCRIPT}"
echo ""

while true; do
  # Check for idle nodes with 8 GPUs
  IDLE_NODES=$(sinfo -p "${PARTITION}" -t idle -N --noheader -o "%N %G" 2>/dev/null | grep "gpu:8" | head -1 | awk '{print $1}')

  if [ -n "${IDLE_NODES}" ]; then
    echo ""
    echo "$(date): Found idle node: ${IDLE_NODES}"
    echo "Allocating and running benchmarks..."

    # Run benchmarks on the node
    srun --partition="${PARTITION}" \
         --nodes=1 \
         --ntasks=1 \
         --gpus-per-node="${GPUS_PER_NODE}" \
         --time="${TIME_LIMIT}" \
         --exclusive \
         --nodelist="${IDLE_NODES}" \
         bash "${BENCHMARK_SCRIPT}"

    echo ""
    echo "$(date): All benchmarks complete!"
    break
  fi

  echo "$(date): No idle 8-GPU node available. Retrying in ${POLL_INTERVAL}s..."
  sleep "${POLL_INTERVAL}"
done
