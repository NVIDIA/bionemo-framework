#!/bin/bash

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
  -push                             Push the built Docker image to the registry.
  -print-image-name-only            Print only the image name associated with the repository state.
  -cache-args <string>              Optional. Custom cache arguments for building the image.
  -label-args <string>              Optional. Custom label arguments for the Docker image.
  -set-secret                       Optional. Set Docker build secret during image construction. Requires SECRET_VAR_NAME and SECRET_VAR_VALUE to be set.
  -nightly-cache                    Optional. Use bionemo1--nightly docker image as cache tag of BioNeMo FW to build docker image from. Dy default using the latest released docker image.
  -help                             Display this help message.

Examples:
  To build a Docker image using caching and push it to the container registry:
    ./ci/scripts/build_docker_image.sh -container-registry-path <CONTAINER_REGISTRY_PATH> -use-cache -push

  To build and tag a docker image with a custom image tag:
    ./ci/scripts/build_docker_image.sh --container-registry-path <CONTAINER_REGISTRY_PATH> -image-tag <IMAGE_TAG>

  To print only the default docker image name specific to the repository state:
    ./ci/scripts/build_docker_image.sh -container-registry-path <CONTAINER_REGISTRY_PATH> -print-image-name-only

Warning:
  This script assumes that Docker is logged into the registry specified by CONTAINER_REGISTRY_PATH, using the following command:
    docker login CONTAINER_REGISTRY_URL --username <USERNAME> --password <ACCESS_TOKEN>
  If Docker's caching mechanism is enabled and the default configuration is used, ensure you are also logged into nvcr.io by running:
    docker login nvcr.io --username <USERNAME> --password $NGC_CLI_API_KEY

EOF
    exit 1
}


# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -container-registry-path) CONTAINER_REGISTRY_PATH="$2"; shift 2 ;;
        -dockerfile-path) DOCKERFILE_PATH="$2"; shift 2 ;;
        -use-cache) USE_CACHE=true; shift ;;
        -nightly-cache) USE_NIGHTLY_CACHE=true; shift ;;
        -image-tag) IMAGE_TAG="$2"; shift 2 ;;
        -cache-args) CACHE_ARGS="$2"; shift 2 ;;
        -label-args) LABELS_ARGS="$2"; shift 2 ;;
        -push) PUSH_IMAGE=true; shift ;;
        -print-image-name) ONLY_IMAGE_NAME=true; shift ;;
        -set-secret) SET_SECRET=true; shift ;;
        -help) display_help ;;
        *) echo "Unknown parameter: $1"; display_help ;;
    esac
done


# Ensure required parameters are set
if [ -z "$CONTAINER_REGISTRY_PATH" ]; then
    echo "Error: The container registry path is required. Use -container-registry-path <path>. Run 'ci/scripts/build_docker_image.sh -help' for more details."
    exit 1
fi

if [[ "$SET_SECRET" = true && ( -z "$SECRET_VAR_NAME" || -z "$SECRET_VAR_VALUE" ) ]]; then
  echo "Error: The -set-secret flag requires both SECRET_VAR_NAME and SECRET_VAR_VALUE to be defined. Run 'ci/scripts/build_docker_image.sh -help' for more details."
  exit 1
fi

# Ensure repository is clean
if ! set_bionemo_home; then
    exit 1
fi

# Get Git commit SHA and sanitized branch name
COMMIT_SHA=$(git rev-parse HEAD)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
SANITIZED_BRANCH_NAME=$(echo "$BRANCH_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g' | sed -E 's/^-+|-+$//g' | cut -c1-128)

# Set default image tag if not provided
IMAGE_TAG="${IMAGE_TAG:-${SANITIZED_BRANCH_NAME}--${COMMIT_SHA}}"
IMAGE_NAME="${CONTAINER_REGISTRY_PATH}:${IMAGE_TAG}"
echo "Docker image name: ${IMAGE_NAME}"

if [ "$ONLY_IMAGE_NAME" = true ]; then
    exit 0
fi

# Set defaults if not provided
DOCKERFILE_PATH="${DOCKERFILE_PATH:-setup/Dockerfile}"
# Set cache arguments if USE_CACHE is enabled
if [ "$USE_CACHE" = true ]; then
    if [ -z "$CACHE_ARGS" ]; then
        if [ "$USE_NIGHTLY_CACHE" = true ]; then
          IMAGE_TAG_BIONEMO_CACHE="bionemo1--nightly"
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
else
   CACHE_ARGS=""
fi

# Set default label arguments if not provided
if [ -z "$LABELS_ARGS" ]; then
    current_date=$(date +%Y-%m-%d)
    LABELS_ARGS="--label com.nvidia.bionemo.branch=${BRANCH_NAME} \
                 --label com.nvidia.bionemo.git_sha=${COMMIT_SHA} \
                 --label com.nvidia.bionemo.created_at=${current_date}"
else
    LABELS_ARGS=""
fi

SECRET_ARGS=""
if [ "$SET_SECRET" = true ]; then
    echo "Adding GitLab token secret to the build"
    SECRET_ARGS="--secret id=${SECRET_VAR_NAME},env=${SECRET_VAR_VALUE}"
fi

# Push option
PUSH_OPTION=""
if [ "$PUSH_IMAGE" = true ]; then
    echo "The image ${IMAGE_NAME} will be pushed to the registry."
    PUSH_OPTION="--push"
fi

# Build the Docker image
docker buildx build \
  --allow security.insecure \
  --provenance=false \
  --progress plain \
  "${LABELS_ARGS}" \
  "${CACHE_ARGS}" \
  "${SECRET_ARGS}" \
  "${PUSH_OPTION}" \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE_PATH}" .

echo "Docker build completed. Image name: ${IMAGE_NAME}"