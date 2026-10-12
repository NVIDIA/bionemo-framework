#!/usr/bin/env bash
# Run the example's acceptance matrix inside the built image. Each step echoes
# a marker line; a missing marker is a failed step. Usage (host):
#   bash examples/llava/scripts/run_matrix.sh [container-name]
set -uo pipefail

CONTAINER=${1:-llava-example-dev}

# Hosts whose driver predates CUDA 13 need the forward-compat shim on the
# link path; override with CUDA_COMPAT_DIR='' on newer drivers. PREPEND only:
# the image's own LD_LIBRARY_PATH carries the driver venv's z3 and cuDNN lib
# dirs, which the x86_64 tilelang/mamba import chain needs inside Ray workers.
CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR:-}

run() {
  echo "=== $1 ==="
  docker exec "$CONTAINER" bash -c "
    cd /opt/llava-example
    export LD_LIBRARY_PATH=${CUDA_COMPAT_DIR:+$CUDA_COMPAT_DIR:}\${LD_LIBRARY_PATH} \
           PATH=/usr/local/cuda/bin:\$PATH \
           CUDA_HOME=/usr/local/cuda \
           PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    $2" 2>&1
}

run prepare 'rm -rf outputs/data && python -m llava_example && echo MARK-PREPARE-OK'
run align-1gpu 'python -m nemo_automodel.cli.app configs/alignment.yaml --nproc-per-node 1 && echo MARK-ALIGN-1GPU-OK'
run sft-1gpu 'python -m nemo_automodel.cli.app configs/sft.yaml --nproc-per-node 1 && echo MARK-SFT-1GPU-OK'
# The two-rank runs re-prove the stages independently; wipe per-stage outputs
# first so AutoModel does not try to resume the one-rank checkpoints.
run align-2gpu 'rm -rf outputs/alignment && torchrun --nproc-per-node 2 --master_port 29561 -m nemo_automodel.cli.app configs/alignment.yaml && echo MARK-ALIGN-2GPU-OK'
run sft-2gpu 'rm -rf outputs/sft && torchrun --nproc-per-node 2 --master_port 29561 -m nemo_automodel.cli.app configs/sft.yaml && echo MARK-SFT-2GPU-OK'

# Each GRPO variant must train fresh: NeMo RL auto-resumes a completed
# outputs/grpo checkpoint, which would leave later variants with no steps.
run grpo-1gpu 'rm -rf outputs/grpo && CUDA_VISIBLE_DEVICES=0 python -m nemotron_stitch.nemo_rl.runner --config configs/grpo.yaml && echo MARK-GRPO-1GPU-OK'
run grpo-2gpu 'rm -rf outputs/grpo && python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-2gpu.yaml && echo MARK-GRPO-2GPU-OK'
# Eight-rank FSDP2: global batch 16 needs local_batch_size 2 to stay divisible
# by world size (AutoModel derives gradient accumulation from world size).
run align-8gpu 'rm -rf outputs/alignment && torchrun --nproc-per-node 8 --master_port 29562 -m nemo_automodel.cli.app configs/alignment.yaml --step_scheduler.local_batch_size=2 && echo MARK-ALIGN-8GPU-OK'
run sft-8gpu 'rm -rf outputs/sft && torchrun --nproc-per-node 8 --master_port 29562 -m nemo_automodel.cli.app configs/sft.yaml --step_scheduler.local_batch_size=2 && echo MARK-SFT-8GPU-OK'
run grpo-8gpu 'rm -rf outputs/grpo && python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-8gpu.yaml && echo MARK-GRPO-8GPU-OK'
