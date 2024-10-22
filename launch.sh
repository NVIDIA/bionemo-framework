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

current_script="$0"

# Check if the script is executable
if [ ! -x "$current_script" ]; then
    # If not, make it executable
    echo "This script is not executable. Making it executable..."
    chmod +x "$current_script"
    echo "Done."
fi

###############################################################################
#
# This is my $LOCAL_ENV file
#
LOCAL_ENV=.env
#
###############################################################################

usage() {
    cat <<EOF

USAGE: launch.sh

launch utility script
----------------------------------------

launch.sh [command]

    valid commands:

    pull               - pull an existing container
    download           - download pre-trained models
    download_test_data - download data necessary for openfold, diffdock tests.
    download_all       - download_test_data and download pretrained models.
    build              - build a container, only recommended if customization is needed
    run                - launch the docker container in non-dev mode. Code is cloned from git and installed.
    push               - push a container to a registry
    dev                - launch a new container in development mode. Local copy of the code is mounted and installed.
    attach             - attach to a running container
    info               - see information about supported Docker image repositories


Getting Started tl;dr
----------------------------------------

    ./launch.sh pull # pull an existing container
    ./launch.sh dev  # launch the container in interactive mode
For more detailed info on getting started, see README.md


More Information
----------------------------------------

Note: This script looks for a file called $LOCAL_ENV in the
current directory. This file should define the following environment
variables:
    BIONEMO_IMAGE
        Container image for BioNeMo training, prepended with registry. e.g.,
        Note that this is a separate (precursor) container from any service associated containers
    LOCAL_REPO_PATH
        Path on workstation or cluster to code, e.g., /home/user/code/bionemo. Inside the bionemo container,
        this is set to /workspace/bionemo by default.
    DOCKER_REPO_PATH
        Set this to change the location of the library in the container, e.g. for development work.
        It is set to /workspace/bionemo by default and a lot of the examples expect this path to be valid.
    WANDB_API_KEY
        Weights and Balances API key to upload runs to WandB. Can also be uploaded afterwards., e.g. Dkjdf...
        This value is optional -- Weights and Biases will log data and not upload if missing.
    JUPYTER_PORT
        Port for launching jupyter lab, e.g. 8888
    REGISTRY
        Container registry URL. e.g., nvcr.io. Only required to push/pull containers.
    REGISTRY_USER
        container registry username. e.g., '$oauthtoken' for registry access. Only required to push/pull containers.
    DEV_CONT_NAME
        Docker name for development container
    NGC_CLI_API_KEY
        NGC API key -- this is required for downloading models and other assets. Run \`ngc config --help\` for details.
    NGC_CLI_ORG
        NGC organization name. Default is nvidian. Run \`ngc config --help\` for details.
    NGC_CLI_TEAM
        NGC team name.  Run \`ngc config --help\` for details.
    NGC_CLI_FORMAT_TYPE
        NGC cli format. Default is ascii.  Run \`ngc config --help\` for details.
    GITLAB_TOKEN
        gitlab access token with api access, used when build container with wheels stored in gitlab registery

EOF
    exit
}

# Defaults for `.env` file
export BIONEMO_IMAGE=${BIONEMO_IMAGE:=nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev}
export LOCAL_REPO_PATH=$(pwd)
export DOCKER_REPO_PATH=${DOCKER_REPO_PATH:=/workspace/bionemo}
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
export DEV_CONT_NAME=${DEV_CONT_NAME:=bionemo}
export NGC_CLI_API_KEY=${NGC_CLI_API_KEY:=NotSpecified}
export NGC_CLI_ORG=${NGC_CLI_ORG:=nvidian}
export NGC_CLI_TEAM=${NGC_CLI_TEAM:=NotSpecified}
export NGC_CLI_FORMAT_TYPE=${NGC_CLI_FORMAT_TYPE:=ascii}
export GITLAB_TOKEN=${GITLAB_TOKEN:=NotSpecified}
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
    echo BIONEMO_IMAGE=${BIONEMO_IMAGE} >> $LOCAL_ENV
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

# Additional variables that will be used in the script when sent in the .env file:
# BASE_IMAGE        Custom Base image for building.
# NEMO_HOME         Path to external copy of NeMo source code, which is mounted into the nemo dependency install location in the environment.
#                   This allows a different version of NeMo to be used with code.
# TOKENIZERS_PATH   Workstation directory to be mounted to /tokenizers inside container

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

DOCKER_CMD="docker run \
    --network host \
    ${PARAM_RUNTIME} \
    -p ${JUPYTER_PORT}:8888 \
    --shm-size=4g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e TMPDIR=/tmp/ \
    -e NUMBA_CACHE_DIR=/tmp/ "


download() {
    if [ $(pip list | grep -F "pydantic" | wc -l) -eq 0 ]; then
        read -p 'Pydantic module (Python) not found. Install in current environment? (y/N):' confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1
        pip install $(cat setup/requirements.txt | grep pydantic)
    fi
    mkdir -p ${MODEL_PATH}
    python download_artifacts.py --models all --source ngc --model_dir ${MODEL_PATH} --verbose
}

download_test_data() {
    echo 'Downloading all test data...'
    python $BIONEMO_HOME/download_artifacts.py --data all --source pbss --verbose
    echo 'Data download complete.'
}

ngc_api_key_is_set() {
    if [ ! -z ${NGC_CLI_API_KEY} ] && [ ${NGC_CLI_API_KEY} != 'NotSpecified' ]; then
        echo true
    else
        echo false
    fi
}

docker_login() {
    local ngc_api_key_is_set_=$(ngc_api_key_is_set)
    if [ $ngc_api_key_is_set_ == true ]; then
        docker login -u "${REGISTRY_USER}" -p ${NGC_CLI_API_KEY} ${REGISTRY}
    else
        echo 'Docker login has been skipped. Container pushing and pulling may fail.'
    fi
}


pull() {
    echo "Pulling image called BIONEMO_IMAGE: ${BIONEMO_IMAGE}"
    docker_login
    set -x
    docker pull ${BIONEMO_IMAGE}
    set +x
    exit
}

# Returns the current git sha if the repository is clean.
# Exits with status code 1 otherwise.
git_sha() {

    git diff-index --quiet HEAD --
    exit_code="$?"

    if [ "${exit_code}" == "128" ]; then
        echo "ERROR: Cannot build image if not in bionemo git repository!"
        return 1

    elif [ "${exit_code}" == "1" ]; then
        echo "ERROR: Repository is dirty! Commit all changes before building image!"
        return 2

    elif [ "${exit_code}" == "0" ]; then
        git rev-parse HEAD
        return 0

    else
        echo "ERROR: Unknown exit code for `git diff-index`: ${exit_code}"
        return 1
    fi
}

image_repo() {
    # get the text up until the last ':''
    # https://stackoverflow.com/a/13857951/362021
    echo "${BIONEMO_IMAGE}" | awk 'BEGIN{FS=OFS=":"}{NF--; print}'
}

stable_image_tag() {
    # get the text after the last ':'
    echo "${BIONEMO_IMAGE}" | rev | cut -d':' -f1 | rev
}

build() {
    if [[ "${GITLAB_TOKEN}" == "" || "${GITLAB_TOKEN}" == "NotSpecified" ]]; then
      echo "ERROR: need to set GITLAB_TOKEN to build the docker image. Please see instructions at https://confluence.nvidia.com/display/CLD/Onboarding+Guide#OnboardingGuide-GitLabDockerRegistry"
      exit 1
    fi

    local PACKAGE=0
    local CLEAN=0
    local use_stable_bionemo_image_name=0

    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--pkg)
                PACKAGE=1
                shift
                ;;
            -c|--clean)
                CLEAN=1
                shift
                ;;
            -b|--base-image)
                BASE_IMAGE=$2
                shift
                shift
                ;;
            -s|--stable)
                shift
                use_stable_bionemo_image_name=1
                ;;
            *)
                echo "Unknown option $1. Please --version to specify a version."
                exit 1
                ;;
        esac
    done

    # DO NOT DO THIS:
    # local BIONEMO_GIT_HASH=
    # For some reason...using local X=$(cmd) throws-away the return code from the function (cmd) !
    # https://www.shellcheck.net/wiki/SC2155
    local BIONEMO_GIT_HASH
    local exit_code_sha
    BIONEMO_GIT_HASH=$(git_sha)
    exit_code_sha="$?"

    if [[ "${exit_code_sha}" != "0" ]]; then
        # not actually the hash! an error ocurred and this is the message
        echo "${BIONEMO_GIT_HASH}"

        if [[ "${exit_code_sha}" == "2" && "${use_stable_bionemo_image_name}" == "1" ]]; then
            # Only dirty commit state and we're not using the commit as the image tag.
            echo "NOTICE: Ignoring dirty git repository state because using stable image tag for build."
            BIONEMO_GIT_HASH=$(git rev-parse HEAD)
        else
            exit 1
        fi
    fi
    # if we get the full sha then we know that the --short will work w/o fail
    local BIONEMO_SHORT_GIT_HASH=$(git rev-parse --short HEAD)

    # local IMG_NAME=($(echo ${BIONEMO_IMAGE} | tr ":" "\n"))
    local IMAGE_NAME=$(image_repo)

    local IMAGE_TAG
    if [[ "${use_stable_bionemo_image_name}" == "1" ]]; then
        IMAGE_TAG=$(stable_image_tag)
        echo "Using stable BIONEMO_IMAGE tag (${IMAGE_TAG}) instead of current git commit (${BIONEMO_GIT_HASH})"
    else
        echo "Using current git commit (${BIONEMO_GIT_HASH}) as tag instead of BIONEMO_IMAGE"
        IMAGE_TAG="${BIONEMO_GIT_HASH}"
    fi

    if [ ${PACKAGE} -eq 1 ]
    then
        set -e
        download
        set +e
    fi

    # NOTE: It is **extremely important** to **never** pass in a secret / password / API key as either:
    #         -- an environment variable
    #         -- a file
    #       Into a docker build process. This includes passing in an env var via --build-args.
    #
    #       This is to ensure that the secret's value is never leaked: doing any of the above means
    #       that the secret value will be **persisted in the image**. Thus, when a user creates a container
    #       from the image, they will have access to this secret value (if it's a file or ENV). Or, they will
    #       be able to `docker inspect` the image and glean the secret value from looking at the layer creations
    #       (occurs when the value is an ARG).
    #
    #       Known bionemo build-time secrets:
    #         - GITLAB_TOKEN
    #
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

    local created_at="$(date --iso-8601=seconds -u)"
    DOCKER_BUILD_CMD="docker buildx build --network host \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        --cache-to type=inline \
        --cache-from=gitlab-master.nvidia.com:5005/clara-discovery/bionemo:cache\
        --secret id=GITLAB_TOKEN,env=GITLAB_TOKEN \
        --label com.nvidia.bionemo.short_git_sha=${BIONEMO_SHORT_GIT_HASH} \
        --label com.nvidia.bionemo.git_sha=${BIONEMO_GIT_HASH} \
        --label com.nvidia.bionemo.created_at=${created_at} \
        -f setup/Dockerfile"

    if [ ${CLEAN} -eq 1 ]
    then
        DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --no-cache"
    fi

    if [ ! -z "${BASE_IMAGE}" ];
    then
        DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --build-arg BASE_IMAGE=${BASE_IMAGE}"
    fi

    echo "[${created_at}] Building BioNeMo framework container..."
    set -x
    DOCKER_BUILDKIT=1 ${DOCKER_BUILD_CMD} .
    set +x
    echo "[$(date --iso-8601=seconds -u)] Finished building ${IMAGE_NAME}:${IMAGE_TAG}"
    exit
}


push() {
    local use_stable_bionemo_image_name=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift
                shift
                ;;
            -a|--additional_copies)
                ADDITIONAL_IMAGES="$2"
                shift
                shift
                ;;
            -s|--stable)
                shift
                use_stable_bionemo_image_name=1
                ;;
            *)
                echo "Unknown option $1. Please use --version to specify an additional tag image override. Or supply --additional_copies to supply a comma-delimited list of new complete image names for tagging."
                exit 1
                ;;
        esac
    done

    # local IMG_NAME=($(echo ${BIONEMO_IMAGE} | tr ":" "\n"))
    local IMAGE_NAME=$(image_repo)
    local IMAGE_TAG

    if [[ "${use_stable_bionemo_image_name}" == "1" ]]; then
        IMAGE_TAG=$(stable_image_tag)
    else
        local BIONEMO_GIT_HASH
        BIONEMO_GIT_HASH=$(git_sha)
        if [[ "$?" != "0" ]]; then
            # not actually the hash! an error ocurred and this is the message
            echo "${BIONEMO_GIT_HASH}"
            exit 1
        fi
        IMAGE_TAG="${BIONEMO_GIT_HASH}"
    fi

    docker_login
    # docker push ${IMG_NAME[0]}:latest
    # docker push ${BIONEMO_IMAGE}
    echo "Pushing image: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker push "${IMAGE_NAME}:${IMAGE_TAG}"

    if [ ! -z "${VERSION}" ];
    then
        echo "Tagging ${IMAGE_TAG} as ${VERSION} & pushing to ${IMAGE_NAME}"
        # docker tag ${BIONEMO_IMAGE} ${IMG_NAME[0]}:${VERSION}
        # docker push ${IMG_NAME[0]}:${VERSION}
        docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_NAME}:${VERSION}"
        docker push "${IMAGE_NAME}:${VERSION}"
    fi

    if [ ! -z "${ADDITIONAL_IMAGES}" ];
    then
        IFS=',' read -ra IMAGES <<< ${ADDITIONAL_IMAGES}
        for IMAGE in "${IMAGES[@]}"; do
            # docker tag ${BIONEMO_IMAGE} ${IMAGE}
            echo "Tagging ${IMAGE_NAME}:${IMAGE_TAG} as ${IMAGE} & pushing"
            docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE}"
            docker push "${IMAGE}"
        done
    fi

    exit
}


setup() {
    mkdir -p ${DATA_PATH}
    mkdir -p ${RESULT_PATH}
    mkdir -p ${MODEL_PATH}

    if [ ! -z "${NEMO_HOME}" ];
    then
        # NOTE: If we change the Python version, we will have a different mount path!
        #       The python3.X part of the path changes.
        echo "Making a volume mount for NeMo!" \
             "Mounting package (\$NEMO_HOME/nemo) in Python environment (/usr/local/lib/python3.10/dist-packages/nemo)" \
             "and NEMO_HOME (${NEMO_HOME}) to /workspace/nemo"
        DOCKER_CMD="${DOCKER_CMD} -v ${NEMO_HOME}/nemo:/usr/local/lib/python3.10/dist-packages/nemo -v ${NEMO_HOME}:/workspace/nemo"
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
    echo "Mounting ${LOCAL_REPO_PATH} at ${DOCKER_REPO_PATH} for development"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_REPO_PATH}:${DOCKER_REPO_PATH} -e HOME=${DOCKER_REPO_PATH} -w ${DOCKER_REPO_PATH} "
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_RESULTS_PATH}:${DOCKER_RESULTS_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_DATA_PATH}:${DOCKER_DATA_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_MODELS_PATH}:${DOCKER_MODELS_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v /etc/passwd:/etc/passwd:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/group:/etc/group:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/shadow:/etc/shadow:ro "
    DOCKER_CMD="${DOCKER_CMD} -u $(id -u):$(id -g) "

    # For dev mode, mount the local code for development purpose
    # and mount .ssh dir for working with git
    if [[ $1 == "dev" ]]; then
        echo "Mounting ~/.ssh up for development"
        DOCKER_CMD="$DOCKER_CMD -v ${HOME}/.ssh:${HOME}/.ssh:ro"
    fi
}


image_to_run() {
    # note: `>&2 echo`  means "write to STDERR"
    # https://stackoverflow.com/a/23550347/362021

    local use_stable_bionemo_image_name="${1}"

    local IMAGE_TO_RUN
    if [[ "${use_stable_bionemo_image_name}" == "1" ]]; then
        >&2 echo "Start development container for BIONEMO_IMAGE: ${BIONEMO_IMAGE}"
        IMAGE_TO_RUN="${BIONEMO_IMAGE}"

    elif [[ "${use_stable_bionemo_image_name}" == "0" ]]; then
        local COMMIT=$(git rev-parse HEAD)
        >&2 echo "Starting development container from latest working code ${COMMIT}"

        git diff-index --quiet HEAD --
        if [[ "$?" != "0" ]]; then
            >&2 echo "WARNING! Dirty git repository detected! Image will be out-of-sync. " \
                 "Volume mount of local code files (${LOCAL_REPO_PATH}) will keep _most_ things up to date.\n" \
                 "Rebuild the image with './launch.sh build' if you encounter bugs due to other things being out-of-sync."
        fi

        IMAGE_TO_RUN="$(image_repo):${COMMIT}"

        if [[ "$(docker images -q ${IMAGE_TO_RUN})" == "" ]]; then
            >&2 echo "ERROR: No image made for commit ${COMMIT}! Falling-back to BIONEMO_IMAGE: ${BIONEMO_IMAGE}"
            IMAGE_TO_RUN="${BIONEMO_IMAGE}"
        fi
    else
        echo "ERROR: invalid! Provide either 0 or 1 for first argument!"
        return 1
    fi
    echo "${IMAGE_TO_RUN}"
}


dev() {
    local use_stable_bionemo_image_name=0
    CMD='bash'
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--additional-args)
                DOCKER_CMD="${DOCKER_CMD} $2"
                shift
                shift
                ;;
            -t|--tmp)
                DEV_CONT_NAME="${DEV_CONT_NAME}_$2"
                shift
                shift
                ;;
            -d|--demon)
                DOCKER_CMD="${DOCKER_CMD} -d"
                shift
                ;;
            -s|--stable)
                shift
                use_stable_bionemo_image_name=1
                ;;
            -c|--cmd)
                shift
                CMD="$@"
                break
                ;;
            *)
                echo "Unknown option '$1'.
Available options are -a(--additional-args), -i(--image), -d(--demon), -s(--stable), and -c(--cmd)"
                exit 1
                ;;
        esac
    done


    setup "dev"

    local IMAGE_TO_RUN
    IMAGE_TO_RUN=$(image_to_run $use_stable_bionemo_image_name)
    if [[ "$?" != "0" ]]; then
        echo $IMAGE_TO_RUN
        exit 1
    fi

    set -x
    ${DOCKER_CMD} --rm -it --name ${DEV_CONT_NAME} ${IMAGE_TO_RUN} ${CMD}
    set +x
    exit
}

run() {
    CMD='bash'
    local use_stable_bionemo_image_name=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--stable)
                shift
                use_stable_bionemo_image_name=1
                ;;
            -c|--cmd)
                shift
                CMD="$@"
                break
                ;;
            *)
                echo "Unknown option '$1'. Only available option is -s(--stable) and -c(--cmd)"
                exit 1
                ;;
	    esac
    done

    local IMAGE_TO_RUN
    IMAGE_TO_RUN=$(image_to_run $use_stable_bionemo_image_name)
    if [[ "$?" != "0" ]]; then
        echo $IMAGE_TO_RUN
        exit 1
    fi

    set -x
    ${DOCKER_CMD} --rm -it --gpus all -e HOME=${DOCKER_REPO_PATH} -w ${DOCKER_REPO_PATH} --name ${DEV_CONT_NAME} ${IMAGE_TO_RUN} ${CMD}
    set +x
    exit
}


attach() {
    set -x
    DOCKER_CMD="docker exec"
    CONTAINER_ID=$(docker ps | grep ${DEV_CONT_NAME} | cut -d' ' -f1)
    ${DOCKER_CMD} -it ${CONTAINER_ID} /bin/bash
    exit
}

info() {
    echo "BioNeMo Image Repository Information:"
    echo '----------------------------------------------------------------'
    echo "GitLab:  gitlab-master.nvidia.com:5005/clara-discovery/bionemo"
    echo "NVCR:    nvcr.io/nvidian/cvai_bnmo_trng/bionemo"
    echo '----------------------------------------------------------------'
    echo "To change the stable image (-s), set BIONEMO_IMAGE in either the environment or the .env file."
    echo "NOTE: This value requires both the image repository name and the tag, separated by ':'."
    exit 0
}


case $1 in
    download)
        shift
        download "$@"
        ;;
    download_all)
        shift
        download "$@"
        download_test_data
        ;;
    build | run | push | pull | dev | attach | download_test_data | info)
        $@
        ;;
    *)
        usage
        ;;
esac
