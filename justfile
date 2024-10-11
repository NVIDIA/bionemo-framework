#!/usr/bin/env -S just --justfile

# https://github.com/casey/just?tab=readme-ov-file#dotenv-settings
set dotenv-load

# https://github.com/casey/just?tab=readme-ov-file#export
set export

# don't fail fast here --> the `setup` command will check this!
COMMIT := `git rev-parse HEAD || true`
IMAGE_TAG := "bionemo2-" + COMMIT
DEV_IMAGE_TAG := "dev-" + IMAGE_TAG
DATE := `date --iso-8601=seconds -u`
LOCAL_ENV := '.env'
DOCKER_REPO_PATH := '/workspace/bionemo2'
LOCAL_REPO_PATH := `realpath $(pwd)`

[private]
default:
  @just --list

###############################################################################

[private]
check_preconditions:
  #!/usr/bin/env bash

  version_ge() {
      # Returns 0 (true) if $1 >= $2, 1 (false) otherwise
      [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
  }

  if [[ $(command -v git) ]]; then
    commit=$(git rev-parse HEAD)
    if [[ "$?" != "0" ]]; then
      echo "ERROR: must run from within git repository!"
      exit 1
    fi
  else
    echo "ERROR: git is not installed!"
    exit 1
  fi

  if [[ ! $(command -v docker) ]]; then
    echo "ERROR: docker is not installed!"
    exit 1
  fi

  docker_version=$(docker --version | awk -F'[, ]' '{print $3}')
  required_docker_version='23.0.1'

  if ! version_ge "$docker_version" "$required_docker_version"; then
      echo "Error: Docker version $required_docker_version or higher is required. Current version: $docker_version"
      exit 1
  fi


# Checks for installed programs (docker, git, etc.), their versions, and grabs the latest cache image.
setup: check_preconditions
  ./internal/scripts/setup_env_file.sh
  @echo "Pulling updated cache..."
  docker pull ${IMAGE_REPO}:${CACHE_TAG} || true


[private]
assert_clean_git_repo:
  #!/usr/bin/env bash

  git diff-index --quiet HEAD --
  exit_code="$?"

  if [[ "${exit_code}" == "128" ]]; then
      echo "ERROR: Cannot build image if not in bionemo git repository!"
      exit 1

  elif [[ "${exit_code}" == "1" ]]; then
      echo "ERROR: Repository is dirty! Commit all changes before building image!"
      exit  2

  elif [[ "${exit_code}" == "0" ]]; then
      echo "ok" 2> /dev/null

  else
      echo "ERROR: Unknown exit code for `git diff-index`: ${exit_code}"
      exit 1
  fi

###############################################################################

[private]
build image_tag target: setup assert_clean_git_repo
  DOCKER_BUILDKIT=1 docker buildx build \
  -t ${IMAGE_REPO}:{{image_tag}} \
  --target={{target}} \
  --load \
  --cache-to type=inline \
  --cache-from ${IMAGE_REPO}:${CACHE_TAG} \
  --label com.nvidia.bionemo.git_sha=${COMMIT} \
  --label com.nvidia.bionemo.created_at=${DATE} \
  -f ./Dockerfile \
  .

# Builds the release image.
build-release:
  @just build ${IMAGE_TAG} release

# Builds the development image.
build-dev:
  @just build ${DEV_IMAGE_TAG} development

###############################################################################

[private]
run is_dev is_interactive image_tag cmd: setup
  #!/usr/bin/env bash

  DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2}')
  DOCKER_VERSION_WITH_GPU_SUPPORT='19.03.0'
  if [ "$DOCKER_VERSION_WITH_GPU_SUPPORT" == "$(echo -e "$DOCKER_VERSION\n$DOCKER_VERSION_WITH_GPU_SUPPORT" | sort -V | head -1)" ]; then
      PARAM_RUNTIME="--gpus all"
  else
      PARAM_RUNTIME="--runtime=nvidia"
  fi

  docker_cmd="docker run \
  --rm \
  --network host \
  ${PARAM_RUNTIME} \
  -p ${JUPYTER_PORT}:8888 \
  --shm-size=4g \
  -e TMPDIR=/tmp/ \
  -e NUMBA_CACHE_DIR=/tmp/ \
  -e BIONEMO_HOME=$DOCKER_REPO_PATH \
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
  -u $(id -u):$(id -g)"

  if [[ "{{is_dev}}" == "true" ]]; then
    docker_cmd="${docker_cmd} -v ${LOCAL_REPO_PATH}:${DOCKER_REPO_PATH}"
  fi

  if [[ "{{is_interactive}}" == "true" ]]; then
    docker_cmd="${docker_cmd} -it"
  fi

  docker_cmd="${docker_cmd} ${IMAGE_REPO}:{{image_tag}} {{cmd}}"

  set -xeuo pipefail
  DOCKER_BUILDKIT=1 ${docker_cmd}

[private]
ensure-dev-or-build:
  #!/usr/bin/env bash
  if [[ $(docker images -q "${IMAGE_REPO}:${DEV_IMAGE_TAG}" 2> /dev/null) == "" ]]; then
    echo "Building development image: ${IMAGE_REPO}:${DEV_IMAGE_TAG}"
    just build-dev
  else
    echo "Development image exists:   ${IMAGE_REPO}:${DEV_IMAGE_TAG}"
  fi

# run-dev lets us work with a dirty repository,
# beacuse this is a common state during development
# **AND** we're volume mounting the code, so we'll have the latest state.
# **BUT** we can only do this if we have built an image for DEV_IMAGE_TAG already,
# so we use ensure-dev-or-build which will build the image if it is necessary.
# Image building requires that the git repo state is clean!
#
# Runs an interactive program in the development bionemo image.
run-dev cmd='bash': ensure-dev-or-build
  @just run true true ${DEV_IMAGE_TAG} {{cmd}}

# in contrast, run-release requires a clean repository,
# because users want to know that they're running the **exact** version they expect
# and we're **NOT** volume mounting the code
#
# Runs an interactive program in the release bionemo image.
run-release cmd='bash': build-release assert_clean_git_repo
  @just run false true ${IMAGE_TAG} {{cmd}}


###############################################################################

# Executes pytest in the release image.
test: build-release
  @just run true false ${IMAGE_TAG} 'pytest -v --nbval-lax --cov=bionemo --cov-report term --cov-report=html docs/ scripts/ sub-packages/'
