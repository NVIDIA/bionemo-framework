#!/usr/bin/env bash

# ------------------------------------------------------------------------
# (0) preamble
# ------------------------------------------------------------------------
MESSAGE_TEMPLATE='********build_dev_image_br.sh: %s\n'
DATE_OF_SCRIPT=$(date +'%Y%m%dT%H%M%S')
SCRIPT_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"
printf "${MESSAGE_TEMPLATE}" "SCRIPT_DIR=${SCRIPT_DIR}"
printf "${MESSAGE_TEMPLATE}" "hostname=$(hostname)"
printf "${MESSAGE_TEMPLATE}" "whoami=$(whoami)"
printf "${MESSAGE_TEMPLATE}" "uid=$(id -u)"
printf "${MESSAGE_TEMPLATE}" "gid=$(id -g)"

set -euo pipefail

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)
DATE=$(date --iso-8601=seconds -u)

set -x
DOCKER_BUILDKIT=1 docker buildx build \
  -t "nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev-${BRANCH}-${DATE_OF_SCRIPT}-${COMMIT}" \
  --ulimit 'nofile=65535:65535' \
  --target="development" \
  --load \
  --cache-from nvcr.io/nvidia/clara/bionemo-framework:nightly \
  --cache-to type=inline \
  --label com.nvidia.bionemo.git_sha=${COMMIT} \
  --label com.nvidia.bionemo.created_at=${DATE} \
  -f ./Dockerfile \
  .


# ----------------------
# (-1) post-amble
# --------------------------
printf "${MESSAGE_TEMPLATE}" "end script"