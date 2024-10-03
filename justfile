#!/usr/bin/env -S just --justfile

# https://github.com/casey/just?tab=readme-ov-file#dotenv-settings
set dotenv-load

# https://github.com/casey/just?tab=readme-ov-file#export
set export

# don't fail fast here --> the `setup` command will check this!
COMMIT := `git rev-parse HEAD || true`
IMAGE_TAG := "bionemo2-" + COMMIT
DATE := `date --iso-8601=seconds -u`

default:
  @just --list

setup:
  ./internal/scripts/check_preconditions.sh
  ./internal/scripts/setup_env_file.sh
  @echo "Pulling updated cache..."
  docker pull ${IMAGE_REPO}:${CACHE_TAG} || true

[private]
build image_tag target: setup
  DOCKER_BUILDKIT=1 docker buildx build \
  -t ${IMAGE_REPO}:{{image_tag}} \
  --target={{target}} \
  --cache-to type=inline \
  --cache-from ${IMAGE_REPO}:${CACHE_TAG} \
  --label com.nvidia.bionemo.git_sha=${COMMIT} \
  --label com.nvidia.bionemo.created_at=${DATE} \
  -f ./Dockerfile \
  .

build-release:
  @just build ${IMAGE_TAG} release

build-dev:
  @just build "dev-${IMAGE_TAG}" development
