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
# SETUP (one-time on prenyx):
#   1. Clone TE:
#      cd /lustre/fsw/healthcareeng_bionemo/savithas
#      git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
#   2. Build TE interactively (allocate a node first):
#      salloc --account=healthcareeng_bionemo --nodes=1 --ntasks-per-node=1 --time=01:00:00 --partition=batch
#      srun --container-image=<sqsh> --container-mounts="<TE_DIR>:/workspace/transformer_engine" \
#        --container-writable bash -c '
#        pip uninstall transformer-engine transformer-engine-torch -y;
#        cd /workspace/transformer_engine && rm -rf build/cmake;
#        NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="103a" NVTE_BUILD_THREADS_PER_JOB=4 \
#        pip install -v --no-build-isolation -e .'
# ============================================================================

CONTAINER="/lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te.sqsh"
CODE_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework"
DATA_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/data"
TE_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/TransformerEngine"

export EXP_NAME="${EXP_NAME:-lingua_7b_mxfp8_qinit_v4_te_main_8n_prenyx}"
RESULTS_DIR="/lustre/fsw/healthcareeng_bionemo/savithas/results/${EXP_NAME}"
CKPT_ROOT="/lustre/fsw/healthcareeng_bionemo/savithas/checkpoints/${EXP_NAME}"

mkdir -p "${RESULTS_DIR}" "${CKPT_ROOT}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.bashrc}"
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

CONTAINER_WORKDIR="/workspace/bionemo"
TE_MOUNT="/workspace/transformer_engine"
MOUNTS="${CODE_DIR}:${CONTAINER_WORKDIR},${DATA_DIR}:/workspace/data,${RESULTS_DIR}:${CONTAINER_WORKDIR}/results,${CKPT_ROOT}:${CONTAINER_WORKDIR}/checkpoints,${TE_DIR}:${TE_MOUNT}"

read -r -d '' COMMAND <<EOF || true
export EXP_NAME="${EXP_NAME}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"

set -euxo pipefail

echo "========================================="
echo "Lingua 7B MXFP8 Quantized Model Init (8 nodes, prenyx)"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_JOB_NUM_NODES}"
echo "========================================="

# Use pre-built TE from mounted source (built once via interactive --build-te step).
# Don't uninstall container TE — its pip metadata is needed by sanity_checks_for_pypi_installation().
# Instead, prepend PYTHONPATH so our mounted TE code takes precedence over container's v2.13.
cp ${TE_MOUNT}/transformer_engine_torch*.so ${TE_MOUNT}/transformer_engine/ 2>/dev/null || true
cp ${TE_MOUNT}/transformer_engine_cu12*.so ${TE_MOUNT}/transformer_engine/ 2>/dev/null || true
export PYTHONPATH="${TE_MOUNT}:\${PYTHONPATH:-}"

# Verify TE has QuantizedTensor support in FusedAdam (PR #2753)
python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')"
python -c "from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam; import inspect; assert 'QuantizedTensor' in inspect.getsource(FusedAdam.step), 'PR #2753 not found!'; print('FusedAdam has QuantizedTensor support')"

cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te

python train_fsdp2.py --config-name L2_lingua_7b_mxfp8_qinit \
  dataset.micro_batch_size=2 \
  dataset.use_stateful_dataloader=false \
  grad_acc_steps=2 \
  checkpoint.ckpt_dir=/workspace/bionemo/checkpoints \
  checkpoint.save_every_n_steps=1500 \
  checkpoint.resume_from_checkpoint=true \
  logger.frequency=100 \
  wandb.name=\${EXP_NAME} \
  wandb.id=\${EXP_NAME} \
  wandb.project=lingua-7b

echo "========================================="
echo "Training complete!"
echo "========================================="
EOF

echo "Launching: ${EXP_NAME}"
srun \
  --output "${RESULTS_DIR}/slurm-%j-%n.out" \
  --error  "${RESULTS_DIR}/error-%j-%n.out" \
  --container-image "${CONTAINER}" \
  --container-mounts "${MOUNTS}" \
  bash -c "${COMMAND}"

# Auto-chain: resubmit so training resumes from checkpoint. scancel to stop.
echo "Resubmitting for next chain..."
sbatch --dependency=singleton "${BASH_SOURCE[0]}"

echo "Job finished! Check: ${RESULTS_DIR}"
