#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

source "$(dirname "$0")/utils.sh"

# Display help message
display_help() {
    cat <<EOF
Usage: $0 [-container-registry-path <path>] [-dockerfile-path <path>] [-use-cache] [-image-tag <string>] [-push] [-print-image-name] [-cache-args <string>] [-label-args <string>] [-help]

Options:
  -container-registry-path <path>   Path to Docker container registry. Used for image name and cache retrieval if -use-cache is enabled.
  -dockerfile-path <path>           Optional. Path to the Dockerfile. Default: setup/Dockerfile.
  -use-cache                        Enable Docker image caching for faster builds.
  -image-tag <string>               Optional. Custom image tag in the format CONTAINER_REGISTRY_PATH:IMAGE_TAG. Default: <GIT_BRANCH_NAME>--<GIT_COMMIT_SHA>.
  -image-name <string>              Optional. Custom image name in the format CONTAINER_REGISTRY_PATH:IMAGE_TAG. Default: <CONTAINER_REGISTRY_PATH>:<GIT_BRANCH_NAME>--<GIT_COMMIT_SHA>.
  -push                             Push the built Docker image to the registry.
  -print-image-name-only            Print only the image name associated with the repository state.
  -cache-args <string>              Optional. Custom cache arguments for building the image.
  -label-args <string>              Optional. Custom label arguments for the Docker image.
  -extra-args <string>              Optional. Extra arguments passed to the docker buildx build method.
  -nightly-cache                    Optional. Use bionemo1--nightly docker image as cache tag of BioNeMo FW to build docker image from. Dy default using the latest released docker image.
  -regular-docker-builder           Optional. By default the docker image is built using insecure-builder - a tool for Docker that allows you to build images with additional features like multi-platform builds, better caching mechanisms,
                                    and advanced configurations. It uses BuildKit under the hood, which is a modern build engine with improved performance and flexibility but requires more a flexible security policy.
                                    To enable a regular docker builder, use this flag. For details visit https://docs.docker.com/reference/cli/docker/buildx/build/
  -help                             Display this help message.

Examples:
  To build a Docker image using caching and push it to the container registry:
    ./ci/scripts/build_docker_image.sh -container-registry-path <CONTAINER_REGISTRY_PATH> -use-cache -push

  To build and tag a docker image with a custom image tag:
    ./ci/scripts/build_docker_image.sh --container-registry-path <CONTAINER_REGISTRY_PATH> -image-tag <IMAGE_TAG>

  To print only the default docker image name specific to the repository state:
    ./ci/scripts/build_docker_image.sh -container-registry-path <CONTAINER_REGISTRY_PATH> -print-image-name-only

Warning:
  This script requires GITLAB_TOKEN to be setup.

  This script assumes that Docker is logged into the registry specified by CONTAINER_REGISTRY_PATH, using the following command:
    docker login CONTAINER_REGISTRY_URL --username <USERNAME> --password <ACCESS_TOKEN>

  If Docker's caching mechanism is enabled and the default configuration is used, ensure you are also logged into nvcr.io by running:
    docker login nvcr.io --username <USERNAME> --password $NGC_CLI_API_KEY

EOF
    exit 1
}
USE_CACHE=false
ONLY_IMAGE_NAME=false
PUSH_IMAGE=false
USE_NIGHTLY_CACHE=false
ALLOW_INSECURE_DOCKER_BUILDER=true

LABELS_ARGS=""
EXTRA_ARGS=""
CACHE_ARGS=""

DEFAULT_BRANCH_NAME="bionemo1"
DEFAULT_DOCKERFILE_PATH="setup/Dockerfile"
# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -container-registry-path) CONTAINER_REGISTRY_PATH="$2"; shift 2 ;;
        -dockerfile-path) DOCKERFILE_PATH="$2"; shift 2 ;;
        -use-cache) USE_CACHE=true; shift ;;
        -nightly-cache) USE_NIGHTLY_CACHE=true; shift ;;
        -image-tag) IMAGE_TAG="$2"; shift 2 ;;
        -image-name) IMAGE_NAME="$2"; shift 2 ;;
        -cache-args) CACHE_ARGS="$2"; shift 2 ;;
        -label-args) LABELS_ARGS="$2"; shift 2 ;;
        -extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        -push) PUSH_IMAGE=true; shift ;;
        -print-image-name) ONLY_IMAGE_NAME=true; shift ;;
        -regular-docker-builder) ALLOW_INSECURE_DOCKER_BUILDER=false; shift ;;
        -help) display_help ;;
        *) echo "Unknown parameter: $1"; display_help ;;
    esac
done

# Ensure required parameters are set
if [ -z "$CONTAINER_REGISTRY_PATH" ] && { [ -z "$IMAGE_NAME" ] || { [ "$USE_CACHE" = true ] && [ -z "$CACHE_ARGS" ]; }; }; then
    echo "Error: The container registry path is required. Use -container-registry-path <path>. Run 'ci/scripts/build_docker_image.sh -help' for more details."
    exit 1
else
    CONTAINER_REGISTRY_PATH=""
fi

if [ -z "$GITLAB_TOKEN" ]; then
  echo "Error: The docker build requires GITLAB_TOKEN to be set up. Run 'ci/scripts/build_docker_image.sh -help' for more details."
  exit 1
fi

# Ensure repository is clean
git config --global --add safe.directory $(pwd)
if ! set_bionemo_home; then
    exit 1
fi

# Get Git commit SHA and sanitized branch name
COMMIT_SHA=$(git rev-parse HEAD)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
SANITIZED_BRANCH_NAME=$(echo "$BRANCH_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g' | sed -E 's/^-+|-+$//g' | cut -c1-128)
# Set defaults if not provided
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${DEFAULT_DOCKERFILE_PATH}}"
# Set default image tag if not provided
IMAGE_TAG="${IMAGE_TAG:-${SANITIZED_BRANCH_NAME}}"
IMAGE_NAME="${IMAGE_NAME:-${CONTAINER_REGISTRY_PATH}:${IMAGE_TAG}--${COMMIT_SHA}}"
echo "Docker image name: ${IMAGE_NAME}"

if [ "$ONLY_IMAGE_NAME" = true ]; then
    exit 0
fi


# Set cache arguments if USE_CACHE is enabled
if [ "$USE_CACHE" = true ]; then
    if [ -z "$CACHE_ARGS" ]; then
        if [ "$USE_NIGHTLY_CACHE" = true ]; then
          IMAGE_TAG_BIONEMO_CACHE="${DEFAULT_BRANCH_NAME}--nightly"
        else
          BIONEMO_VERSION=$(awk '{gsub(/^[[:space:]]+|[[:space:]]+$/, ""); printf "%s", $0}' ./VERSION)
          IMAGE_TAG_BIONEMO_CACHE="${BIONEMO_VERSION}"
        fi
        CONTAINER_REGISTRY_PATH_NGC="nvcr.io/nvidia/clara/bionemo-framework"
        IMAGE_NAME_CACHE="${CONTAINER_REGISTRY_PATH}:${IMAGE_TAG}--cache"
        CACHE_ARGS="--cache-from=type=registry,ref=${CONTAINER_REGISTRY_PATH_NGC}:${IMAGE_TAG_BIONEMO_CACHE} \
                    --cache-from=type=registry,ref=${IMAGE_NAME_CACHE} \
                    --cache-from=type=registry,ref=${IMAGE_NAME} \
                    --cache-to=type=registry,mode=max,image-manifest=true,ref=${IMAGE_NAME_CACHE}"
    fi
    echo "Using cache with configuration: ${CACHE_ARGS}"
fi

# Set default label arguments if not provided
if [ -z "$LABELS_ARGS" ]; then
    current_date=$(date +%Y-%m-%d)
    LABELS_ARGS="--label com.nvidia.bionemo.branch=${BRANCH_NAME} \
    --label com.nvidia.bionemo.git_sha=${COMMIT_SHA} \
    --label com.nvidia.bionemo.created_at=${current_date}"
fi
echo "Using docker labels with configuration: ${LABELS_ARGS}"


# Push option
PUSH_OPTION=""
if [ "$PUSH_IMAGE" = true ]; then
    echo "The image ${IMAGE_NAME} will be pushed to the registry."
    PUSH_OPTION="--push"
fi

## Ensure the requirements of docker and docker buildx versions are satisfied
if ! verify_required_docker_version; then
    exit 1
fi

set -x
# Setup docker build buildx
docker buildx version
if [ "$ALLOW_INSECURE_DOCKER_BUILDER" = true ]; then
  docker buildx create --use \
      --name insecure-builder --driver-opt network=host \
      --buildkitd-flags '--allow-insecure-entitlement security.insecure'
  EXTRA_ARGS="${EXTRA_ARGS} --allow security.insecure"
fi
docker context ls

set -u
# Build the Docker image
GITLAB_TOKEN=$GITLAB_TOKEN docker buildx build $EXTRA_ARGS \
  --provenance=false --progress plain \
  --secret id=GITLAB_TOKEN,env=GITLAB_TOKEN $LABELS_ARGS $CACHE_ARGS $PUSH_OPTION \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE_PATH}" .
set +x
echo "Docker build completed. Image name: "
# Printing standalone image name is required by CI since docker image name is captured by IMAGE_NAME=$(ci/scripts/build_docker_image.sh)
echo "${IMAGE_NAME}"
