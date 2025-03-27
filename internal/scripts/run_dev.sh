#!/usr/bin/env bash

set -euo pipefail

source .env

LOCAL_REPO_PATH="$(realpath $(pwd))"

if [[ "$(basename ${LOCAL_REPO_PATH})" != "bionemo-framework" ]]; then
    echo "ERROR: must run this script from the bionemo repository root!"
    exit 1
fi

COMMIT=$(git rev-parse HEAD)

DOCKER_REPO_PATH="/workspace/bionemo2"

DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2}')
DOCKER_VERSION_WITH_GPU_SUPPORT='19.03.0'
if [ "$DOCKER_VERSION_WITH_GPU_SUPPORT" == "$(echo -e "$DOCKER_VERSION\n$DOCKER_VERSION_WITH_GPU_SUPPORT" | sort -V | head -1)" ]; then
    PARAM_RUNTIME="--gpus all"
else
    PARAM_RUNTIME="--runtime=nvidia"
fi

echo "docker run ... nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev-bionemo2-${COMMIT} bash"
echo '---------------------------------------------------------------------------------------------'
# DO NOT set -x: we **DO NOT** want to leak credentials to STDOUT! (API_KEY)
docker run \
    --rm \
    -it \
    --network host \
    ${PARAM_RUNTIME} \
    -p ${JUPYTER_PORT}:8888 \
    --shm-size=4g \
    -e TMPDIR=/tmp/ \
    -e NUMBA_CACHE_DIR=/tmp/ \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e NGC_CLI_API_KEY=$NGC_CLI_API_KEY \
    -e NGC_CLI_ORG=$NGC_CLI_ORG \
    -e NGC_CLI_TEAM=$NGC_CLI_TEAM \
    -e NGC_CLI_FORMAT_TYPE=$NGC_CLI_FORMAT_TYPE \
    -e AWS_ENDPOINT_URL \
    -e AWS_REGION \
    -e AWS_SECRET_ACCESS_KEY \
    -e AWS_ACCESS_KEY_ID \
    -e HOME=${DOCKER_REPO_PATH} \
    -w ${DOCKER_REPO_PATH} \
    -v ${LOCAL_RESULTS_PATH}:${DOCKER_RESULTS_PATH} \
    -v ${LOCAL_DATA_PATH}:${DOCKER_DATA_PATH} \
    -v ${LOCAL_MODELS_PATH}:${DOCKER_MODELS_PATH} \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -v ${HOME}/.ssh:${DOCKER_REPO_PATH}/.ssh:ro \
    -v ${LOCAL_REPO_PATH}/htmlcov:/${DOCKER_REPO_PATH}/htmlcov \
    -u $(id -u):$(id -g) \
    -v ${LOCAL_REPO_PATH}:${DOCKER_REPO_PATH} \
    "nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev-bionemo2-${COMMIT}" \
    bash
