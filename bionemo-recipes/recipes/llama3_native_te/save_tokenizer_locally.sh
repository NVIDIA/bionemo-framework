#!/bin/bash
# One-time script to save the Meta-Llama-3-8B tokenizer to a local directory
# so training jobs don't need to hit the HF API (avoids 429 rate limits).
#
# Usage: bash save_tokenizer_locally.sh
set -euxo pipefail

SCRATCH="/lustre/fsw/healthcareeng_bionemo/savithas"
CONTAINER="${SCRATCH}/enroot/llama3_native_te_te-main-26.03.sqsh"
CODE_DIR="${SCRATCH}/bionemo-framework"
TOKENIZER_DIR="${CODE_DIR}/bionemo-recipes/recipes/llama3_native_te/tokenizers/Meta-Llama-3-8B"

: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN in ~/.bashrc}"

mkdir -p "$(dirname "${TOKENIZER_DIR}")"

srun --account=healthcareeng_bionemo --nodes=1 --ntasks-per-node=1 \
  --time=00:10:00 --partition=batch \
  --container-image="${CONTAINER}" \
  --container-mounts="${CODE_DIR}:/workspace/bionemo" \
  bash -c "export HF_TOKEN='${HUGGING_FACE_HUB_TOKEN}'; python -c \"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
tok.save_pretrained('/workspace/bionemo/bionemo-recipes/recipes/llama3_native_te/tokenizers/Meta-Llama-3-8B')
print('Tokenizer saved successfully!')
\""

echo "Done! Tokenizer saved to: ${TOKENIZER_DIR}"
