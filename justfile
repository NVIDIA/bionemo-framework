#!/usr/bin/env -S just --justfile

setup:
  ./internal/scripts/check_docker_version.sh
  source ./internal/scripts/setup_env_file.sh

build: setup
  @echo "Pulling updated cache..."
  docker pull ${IMAGE_REPO}:${CACHE_TAG} || true
  DOCKER_BUILDKIT=1 docker buildx build \
    -t ${IMAGE_REPO}:${IMAGE_TAG} \
    --cache-to type=inline \
	  --cache-from ${IMAGE_REPO}:${CACHE_TAG} \
    --label com.nvidia.bionemo.git_sha=${COMMIT} \
    --label com.nvidia.bionemo.created_at=$(date --iso-8601=seconds -u) \
    f ./Dockerfile \
    .
