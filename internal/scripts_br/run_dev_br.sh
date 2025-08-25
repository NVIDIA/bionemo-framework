#!/usr/bin/env bash

# ------------------------------------------------------------------------
# (0) preamble
# ------------------------------------------------------------------------
MESSAGE_TEMPLATE='********run_dev_br.sh: %s\n'
DATE_OF_SCRIPT=$(date +'%Y%m%dT%H%M%S')
SCRIPT_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"
printf "${MESSAGE_TEMPLATE}" "SCRIPT_DIR=${SCRIPT_DIR}"
printf "${MESSAGE_TEMPLATE}" "hostname=$(hostname)"
printf "${MESSAGE_TEMPLATE}" "whoami=$(whoami)"
printf "${MESSAGE_TEMPLATE}" "uid=$(id -u)"
printf "${MESSAGE_TEMPLATE}" "gid=$(id -g)"


#set -euo pipefail

source .env


# --------------------------------
# (1) user paramerters
# -----------------------------------------------------
COMMIT_AT_START=$(git rev-parse --short HEAD)
BRANCH_AT_START=$(git rev-parse --abbrev-ref HEAD)
IMAGE_REPO='nvcr.io/nvidian/cvai_bnmo_trng/bionemo'
IMAGE_TAG='dev-br_bnm2533_fix_evo2_tests_a-20250825T162355-28586e55'
IMAGE_NAME="${IMAGE_REPO}:${IMAGE_TAG}"

DOCKER_REPO_PATH="/workspace/bionemo2"

# -----------------------------------------------------
# (2) santity checks
# ----------------------------------------------------
LOCAL_REPO_PATH="$(realpath $(pwd))"
if [[ "$(basename ${LOCAL_REPO_PATH})" != "bionemo-framework" ]]; then
    echo "ERROR: must run this script from the bionemo repository root!"
    exit 1
fi

# -----------------------------------------------------
# (3) make expected directories as user, not as docker
# ----------------------------------------------------
expected_local_dirs=("${LOCAL_RESULTS_PATH}" "${LOCAL_DATA_PATH}" "${LOCAL_MODELS_PATH}")
for expected_local_dir in "${expected_local_dirs[@]}"; do
    printf "${MESSAGE_TEMPLATE}" "expected_local_dir=${expected_local_dir}"
    mkdir -p "${expected_local_dir}"
done

# -----------------------------------------------------
# (4) assemble docker run command
# ----------------------------------------------------

# echo "docker run ... nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev-bionemo2-${COMMIT_AT_START} bash"
# echo '---------------------------------------------------------------------------------------------'
# DO NOT set -x: we **DO NOT** want to leak credentials to STDOUT! (API_KEY)


printf "${MESSAGE_TEMPLATE}" "create DOCKER_RUN_COMMAND"


DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2}')
DOCKER_VERSION_WITH_GPU_SUPPORT='19.03.0'
if [ "$DOCKER_VERSION_WITH_GPU_SUPPORT" == "$(echo -e "$DOCKER_VERSION\n$DOCKER_VERSION_WITH_GPU_SUPPORT" | sort -V | head -1)" ]; then
    PARAM_RUNTIME="--gpus all"
else
    PARAM_RUNTIME="--runtime=nvidia"
fi

read -r -d '' SECRETS<<EOF
    -e WANDB_API_KEY=$WANDB_API_KEY
EOF
read -r -d '' DOCKER_RUN_OPTIONS <<EOF
    -u $(id -u):$(id -g) \\
    --rm \\
    -it \\
    --network host \\
    ${PARAM_RUNTIME} \\
    -p ${JUPYTER_PORT}:8888 \\
    --shm-size=4g \\
    -e TMPDIR=/tmp/ \\
    -e BRANCH_AT_START=${BRANCH_AT_START} \\
    -e COMMIT_AT_START=${COMMIT_AT_START} \\
    -e NUMBA_CACHE_DIR=/tmp/ \\
    -e HOME=${DOCKER_REPO_PATH} \\
    -w ${DOCKER_REPO_PATH} \\
    -v ${LOCAL_RESULTS_PATH}:${DOCKER_RESULTS_PATH} \\
    -v ${LOCAL_DATA_PATH}:${DOCKER_DATA_PATH} \\
    -v ${LOCAL_MODELS_PATH}:${DOCKER_MODELS_PATH} \\
    -v /etc/passwd:/etc/passwd:ro \\
    -v /etc/group:/etc/group:ro \\
    -v /etc/shadow:/etc/shadow:ro \\
    -v ${HOME}/.ssh:${DOCKER_REPO_PATH}/.ssh:ro \\
    -v ${LOCAL_REPO_PATH}/htmlcov:/${DOCKER_REPO_PATH}/htmlcov \\
    -v ${LOCAL_REPO_PATH}:${DOCKER_REPO_PATH} \\
    -e NGC_CLI_ORG \\
    -e NGC_CLI_TEAM \\
    -e NGC_CLI_FORMAT_TYPE \\
    -e NGC_CLI_API_KEY \\
    -e AWS_ENDPOINT_URL \\
    -e AWS_REGION \\
    -e AWS_ACCESS_KEY_ID \\
    -e AWS_SECRET_ACCESS_KEY
EOF
read -r -d '' DOCKER_RUN_WITHOUT_SECRETS <<EOF
docker run \\
    ${DOCKER_RUN_OPTIONS} \\
    ${IMAGE_NAME} \\
    bash
EOF

read -r -d '' DOCKER_RUN_WITH_SECRETS <<EOF
docker run \\
    ${DOCKER_RUN_OPTIONS} \\
    ${SECRETS} \\
    ${IMAGE_NAME} \\
    bash
EOF

# -----------------------------------------------------
# (5) run docker run command
# ----------------------------------------------------
printf "${MESSAGE_TEMPLATE}" "DOCKER_RUN_WITHOUT_SECRETS=${DOCKER_RUN_WITHOUT_SECRETS}"
eval "$DOCKER_RUN_WITH_SECRETS"

# -----------------------------------------------------
# (-1) summarize
# ----------------------------------------------------

printf "${MESSAGE_TEMPLATE}" "run_dev_br.sh: end script----"
