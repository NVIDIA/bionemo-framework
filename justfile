#!/usr/bin/env -S just --justfile

# https://github.com/casey/just?tab=readme-ov-file#dotenv-settings
set dotenv-load

# https://github.com/casey/just?tab=readme-ov-file#export
set export

# don't fail fast here --> the `setup` command will check this!
COMMIT := `git rev-parse HEAD || true`
IMAGE_TAG := "bionemo2-" + COMMIT

setup:
  ./internal/scripts/check_preconditions.sh
  ./internal/scripts/setup_env_file.sh

build: setup
  @echo "Pulling updated cache..."
  docker pull ${IMAGE_REPO}:${CACHE_TAG} || true
  DOCKER_BUILDKIT=1 docker buildx build \
  -t ${IMAGE_REPO}:${IMAGE_TAG} \
  --cache-to type=inline \
  --cache-from ${IMAGE_REPO}:${CACHE_TAG} \
  --label com.nvidia.bionemo.git_sha=${COMMIT} \
  --label com.nvidia.bionemo.created_at=$(date --iso-8601=seconds -u) \
  -f ./Dockerfile \
  .
