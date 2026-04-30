#!/bin/bash
# Launch an interactive node for TE MXFP8 bug debugging.
# Usage: bash interactive_te_debug.sh

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
TE_DIR="${SCRATCH}/TransformerEngine"

srun --account=healthcareeng_bionemo \
  --job-name=healthcareeng_bionemo-te.debug \
  --nodes=1 --ntasks-per-node=1 \
  --time=02:00:00 --partition=batch \
  --container-image="${CONTAINER}" \
  --container-mounts="${CODE_DIR}:/workspace/bionemo,${TE_DIR}:/workspace/transformer_engine" \
  --pty bash
