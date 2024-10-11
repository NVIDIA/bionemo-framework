#!/usr/bin/env bash

set -euo pipefail

COMMIT=$(git rev-parse HEAD)
DATE=$(date --iso-8601=seconds -u)

set -x
DOCKER_BUILDKIT=1 docker buildx build \
  -t "nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev-bionemo2-${COMMIT}" \
  --target="development" \
  --load \
  --cache-to type=inline \
  --label com.nvidia.bionemo.git_sha=${COMMIT} \
  --label com.nvidia.bionemo.created_at=${DATE} \
  -f ./Dockerfile \
  .
