#!/usr/bin/env bash
# This above shebang is supposed to be more portable than /bin/bash

#Name of the Docker image
IMAGE_NAME=${IMAGE_NAME:=nvcr.io/nvidian/cvai_bnmo_trng/bionemo}
COMMIT=$(git rev-parse HEAD)
IMAGE_TAG="bionemo2-${COMMIT}"
CACHE_TAG="bionemo2-latest"

# Defaults for `.env` file
export LOCAL_REPO_PATH=$(realpath -s $(pwd))
export DOCKER_REPO_PATH=${DOCKER_REPO_PATH:=/workspace/bionemo2}
export LOCAL_RESULTS_PATH=${LOCAL_RESULTS_PATH:=${LOCAL_REPO_PATH}/results}
export DOCKER_RESULTS_PATH=${DOCKER_RESULTS_PATH:=${DOCKER_REPO_PATH}/results}
export LOCAL_DATA_PATH=${LOCAL_DATA_PATH:=${LOCAL_REPO_PATH}/data}
export DOCKER_DATA_PATH=${DOCKER_DATA_PATH:=${DOCKER_REPO_PATH}/data}
export LOCAL_MODELS_PATH=${LOCAL_MODELS_PATH:=${LOCAL_REPO_PATH}/models}
export DOCKER_MODELS_PATH=${DOCKER_MODELS_PATH:=${DOCKER_REPO_PATH}/models}
export WANDB_API_KEY=${WANDB_API_KEY:=NotSpecified}
export JUPYTER_PORT=${JUPYTER_PORT:=8888}
export REGISTRY=${REGISTRY:=nvcr.io}
export REGISTRY_USER=${REGISTRY_USER:='$oauthtoken'}
export DEV_CONT_NAME=${DEV_CONT_NAME:=bionemo2}
export NGC_CLI_API_KEY=${NGC_CLI_API_KEY:=NotSpecified}
export NGC_CLI_ORG=${NGC_CLI_ORG:=nvidian}
export NGC_CLI_TEAM=${NGC_CLI_TEAM:=NotSpecified}
export NGC_CLI_FORMAT_TYPE=${NGC_CLI_FORMAT_TYPE:=ascii}
export GITLAB_TOKEN=${GITLAB_TOKEN:=NotSpecified}
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:="team-bionemo"}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:=NotSpecified}
export AWS_REGION=${AWS_REGION:="us-east-1"}
export AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL:="https://pbss.s8k.io"}
# NOTE: Some variables need to be present in the environment of processes this script kicks off.
#       Most notably, `docker build` requires the GITLAB_TOKEN env var. Otherwise, building fails.
#
#       For uniformity of behavior between externally setting an environment variable before
#       executing this script and using the .env file, we make sure to explicitly `export` every
#       environment variable that we use and may define in the .env file.
#
#       This way, all of these variables and their values will always guarenteed to be present
#       in the environment of all processes forked from this script's.


# if $LOCAL_ENV file exists, source it to specify my environment
LOCAL_ENV=.env
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
    write_env=0
else
    echo $LOCAL_ENV does not exist. Writing deafults to $LOCAL_ENV
    write_env=1
fi

# If $LOCAL_ENV was not found, write out a template for user to edit
if [ $write_env -eq 1 ]; then
    echo LOCAL_REPO_PATH=${LOCAL_REPO_PATH} \# This needs to be set to BIONEMO_HOME for local \(non-dockerized\) use >> $LOCAL_ENV
    echo DOCKER_REPO_PATH=${DOCKER_REPO_PATH} \# This is set to BIONEMO_HOME in container >> $LOCAL_ENV
    echo LOCAL_RESULTS_PATH=${LOCAL_RESULTS_PATH} >> $LOCAL_ENV
    echo DOCKER_RESULTS_PATH=${DOCKER_RESULTS_PATH} >> $LOCAL_ENV
    echo LOCAL_DATA_PATH=${LOCAL_DATA_PATH} >> $LOCAL_ENV
    echo DOCKER_DATA_PATH=${DOCKER_DATA_PATH} >> $LOCAL_ENV
    echo LOCAL_MODELS_PATH=${LOCAL_MODELS_PATH} >> $LOCAL_ENV
    echo DOCKER_MODELS_PATH=${DOCKER_MODELS_PATH} >> $LOCAL_ENV
    echo WANDB_API_KEY=${WANDB_API_KEY} >> $LOCAL_ENV
    echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV
    echo REGISTRY=${REGISTRY} >> $LOCAL_ENV
    echo REGISTRY_USER=${REGISTRY_USER} >> $LOCAL_ENV
    echo DEV_CONT_NAME=${DEV_CONT_NAME} >> $LOCAL_ENV
    echo NGC_CLI_API_KEY=${NGC_CLI_API_KEY} >> $LOCAL_ENV
    echo NGC_CLI_ORG=${NGC_CLI_ORG} >> $LOCAL_ENV
    echo NGC_CLI_TEAM=${NGC_CLI_TEAM} >> $LOCAL_ENV
    echo NGC_CLI_FORMAT_TYPE=${NGC_CLI_FORMAT_TYPE} >> $LOCAL_ENV
    echo GITLAB_TOKEN=${GITLAB_TOKEN} \# This needs to be created via your gitlab account as a personal access token with API access enabled. >> $LOCAL_ENV
fi

# Default paths for framework. We switch these depending on whether or not we are inside
# a docker environment. It is assumed that if we are in a docker environment, then it's the
# bionemo image built with `setup/Dockerfile`.


if [ -f /.dockerenv ]; then
    echo "Running inside a Docker container, using DOCKER paths from .env file."
    RESULT_PATH=${DOCKER_RESULTS_PATH}
    DATA_PATH=${DOCKER_DATA_PATH}
    MODEL_PATH=${DOCKER_MODELS_PATH}
    BIONEMO_HOME=${DOCKER_REPO_PATH}
else
    echo "Not running inside a Docker container, using LOCAL paths from .env file."
    RESULT_PATH=${LOCAL_RESULTS_PATH}
    DATA_PATH=${LOCAL_DATA_PATH}
    MODEL_PATH=${LOCAL_MODELS_PATH}
    BIONEMO_HOME=${LOCAL_REPO_PATH}
fi


#Function to build the Docker image
build() {
    version_ge() {
        # Returns 0 (true) if $1 >= $2, 1 (false) otherwise
        [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
    }

    # Check Docker version
    docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    required_docker_version="23.0.1"

    if ! version_ge "$docker_version" "$required_docker_version"; then
        echo "Error: Docker version $required_docker_version or higher is required. Current version: $docker_version"
        exit 1
    fi

    # Check Buildx version
    buildx_version=$(docker buildx version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    required_buildx_version="0.10.2"

    if ! version_ge "$buildx_version" "$required_buildx_version"; then
        echo "Error: Docker Buildx version $required_buildx_version or higher is required. Current version: $buildx_version"
        exit 1
    fi

    echo "Pulling updated cache"
    docker pull ${IMAGE_NAME}:${CACHE_TAG}
    local created_at="$(date --iso-8601=seconds -u)"
    echo "Building Docker image..."
    local DOCKER_BUILD_CMD="docker buildx build \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        --cache-to type=inline \
	    --cache-from ${IMAGE_NAME}:${CACHE_TAG} \
        --label com.nvidia.bionemo.git_sha=${COMMIT} \
        --label com.nvidia.bionemo.created_at=${created_at} \
        -f ./Dockerfile"
    echo "$DOCKER_BUILD_CMD"
    $DOCKER_BUILD_CMD .
    echo "Docker build completed successfully."
}

# Function to update our Caches locally and on CI
#  this will push a match to the local ${IMAGE_NAME}:${CACHE_TAG}
update_build_cache() {
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:${CACHE_TAG}
    docker push ${IMAGE_NAME}:${CACHE_TAG}
}

# Function to both build and cache
build_and_update_cache() {
    build
    update_build_cache
}

# Compare Docker version to find Nvidia Container Toolkit support.
# Please refer https://github.com/NVIDIA/nvidia-docker
DOCKER_VERSION_WITH_GPU_SUPPORT="19.03.0"
if [ -x "$(command -v docker)" ]; then
    DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2'})
fi

PARAM_RUNTIME="--runtime=nvidia"
if [ "$DOCKER_VERSION_WITH_GPU_SUPPORT" == "$(echo -e "$DOCKER_VERSION\n$DOCKER_VERSION_WITH_GPU_SUPPORT" | sort -V | head -1)" ];
then
    PARAM_RUNTIME="--gpus all"
fi


# NOTE: --shm-size=4g is needed for "ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)."
DOCKER_CMD="docker run \
    --network host \
    ${PARAM_RUNTIME} \
    -p ${JUPYTER_PORT}:8888 \
    --shm-size=4g \
    -e TMPDIR=/tmp/ \
    -e NUMBA_CACHE_DIR=/tmp/ \
    -e AWS_SECRET_ACCESS_KEY \
    -e AWS_ACCESS_KEY_ID "

setup() {
    # mkdir -p ${DATA_PATH}
    # mkdir -p ${RESULT_PATH}
    # mkdir -p ${MODEL_PATH}

    if [ ! -z "${NEMO_HOME}" ];
    then
        # NOTE: If we change the Python version, we will have a different mount path!
        #       The python3.X part of the path changes.
        echo "Making a volume mount for NeMo!" \
             "Mounting package (\$NEMO_HOME/nemo) in Python environment (/usr/local/lib/python3.10/dist-packages/nemo)" \
             "and NEMO_HOME (${NEMO_HOME}) to /workspace/NeMo"
        DOCKER_CMD="${DOCKER_CMD} -v ${NEMO_HOME}/nemo:/usr/local/lib/python3.10/dist-packages/nemo -v ${NEMO_HOME}:/workspace/NeMo"
    fi

    if [ ! -z "${MEGATRON_HOME}" ];
    then
        # NOTE: If we change the Python version, we will have a different mount path!
        #       The python3.X part of the path changes.
        echo "Making a volume mount for megatron!" \
             "Mounting package (\$MEGATRON_HOME/nemo) in Python environment (/usr/local/lib/python3.10/dist-packages/nemo)" \
             "and MEGATRON_HOME (${MEGATRON_HOME}) to /workspace/Megatron-LM"
        DOCKER_CMD="${DOCKER_CMD} -v ${MEGATRON_HOME}/megatron:/usr/local/lib/python3.10/dist-packages/megatron -v ${MEGATRON_HOME}:/workspace/Megatron-LM"
    fi


    # Note: For BIONEMO_HOME, if we are invoking docker, this should always be
    # the docker repo path.
    DOCKER_CMD="${DOCKER_CMD} --env BIONEMO_HOME=$DOCKER_REPO_PATH"
    DOCKER_CMD="${DOCKER_CMD} --env WANDB_API_KEY=$WANDB_API_KEY"
    DOCKER_CMD="${DOCKER_CMD} --env NGC_CLI_API_KEY=$NGC_CLI_API_KEY"
    DOCKER_CMD="${DOCKER_CMD} --env NGC_CLI_ORG=$NGC_CLI_ORG"
    DOCKER_CMD="${DOCKER_CMD} --env NGC_CLI_TEAM=$NGC_CLI_TEAM"
    DOCKER_CMD="${DOCKER_CMD} --env NGC_CLI_FORMAT_TYPE=$NGC_CLI_FORMAT_TYPE"

    # For development work
    DOCKER_CMD="${DOCKER_CMD} -v $LOCAL_REPO_PATH:$DOCKER_REPO_PATH"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_RESULTS_PATH}:${DOCKER_RESULTS_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_DATA_PATH}:${DOCKER_DATA_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_MODELS_PATH}:${DOCKER_MODELS_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v /etc/passwd:/etc/passwd:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/group:/etc/group:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/shadow:/etc/shadow:ro "
    DOCKER_CMD="${DOCKER_CMD} -u $(id -u):$(id -g) "
    DOCKER_CMD="${DOCKER_CMD} -e HOME=/workspace/bionemo2 -w /workspace/bionemo2 "

    # For dev mode, mount the local code for development purpose
    # and mount .ssh dir for working with git
    if [[ $1 == "dev" ]]; then
        echo "Mounting ~/.ssh up for development"
        DOCKER_CMD="$DOCKER_CMD -v ${HOME}/.ssh:/workspace/bionemo2/.ssh:ro"
    fi
}


dev() {
    CMD='bash'
    setup "dev"

    local IMAGE_TO_RUN
    IMAGE_TO_RUN=$IMAGE_NAME:$IMAGE_TAG

    set -x
    ${DOCKER_CMD}  --name ${DEV_CONT_NAME}  --rm -it ${IMAGE_TO_RUN} ${CMD}
    set +x
    exit
}

pull() {
    echo "Pulling image called BIONEMO_IMAGE: ${BIONEMO_IMAGE}"
    docker_login
    set -x
    docker pull ${BIONEMO_IMAGE}
    set +x
    exit
}

#Function to run the Docker container
run() {
    echo "running docker container"
    docker run --ipc=host --net=host --shm-size=512m --gpus all --rm -it --entrypoint /bin/bash ${IMAGE_NAME}:${IMAGE_TAG}
}

case "$1" in
    build)
        build
        ;;
    run)
        run
        ;;
    dev)
        dev
        ;;
    pull)
        pull
        ;;
    update_build_cache)
        update_build_cache
        ;;
    build_and_update_cache)
        build_and_update_cache
        ;;
    *)
        echo "Usage: $0 {build|update_build_cache|build_and_update_cache|run|dev|pull}"
        exit 1
        ;;
esac
