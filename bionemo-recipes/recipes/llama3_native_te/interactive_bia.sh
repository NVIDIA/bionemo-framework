#!/bin/bash
# Build the sqsh once on a machine with Docker + enroot (e.g. pstjohn-bo):
#   enroot import -o llama3_native_te.sqsh dockerd://llama3_native_te_base
#   aws s3 cp llama3_native_te.sqsh s3://general-purpose/savithas/containers/llama3_native_te.sqsh --endpoint-url https://pbss.s8k.io
#
# Then on BIA login node, download it:
#   aws s3 cp s3://general-purpose/savithas/containers/llama3_native_te.sqsh \
#     /lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te.sqsh --endpoint-url https://pbss.s8k.io

SQSH="/lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te.sqsh"
MOUNTS="/lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework:/workspace/bionemo,/lustre/fsw/healthcareeng_bionemo/savithas/results:/workspace/results,/lustre/fsw/healthcareeng_bionemo/savithas/checkpoints:/workspace/checkpoints,/lustre/fsw/healthcareeng_bionemo/savithas/.claude:/workspace/.claude"

srun --account=healthcareeng_bionemo \
  --job-name=healthcareeng_bionemo-lingua.test \
  --partition=batch \
  --nodes=1 --ntasks-per-node=8 \
  --time=01:00:00 --mem=0 --exclusive --pty \
  --container-image="${SQSH}" \
  --container-writable \
  --container-mounts="${MOUNTS}" \
  --container-env=WANDB_API_KEY,HUGGING_FACE_HUB_TOKEN,ANTHROPIC_BASE_URL,ANTHROPIC_AUTH_TOKEN \
  /bin/bash -l
