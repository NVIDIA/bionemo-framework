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

default:
  @just --list

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


[private]
build image_tag target: setup assert_clean_git_repo
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
  @just build ${DEV_IMAGE_TAG} development


[private]
run image_tag cmd: setup
  #!/usr/bin/env bash

  docker_cmd="docker run \
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
  -e HOME=${DOCKER_REPO_PATH} \
  -w ${DOCKER_REPO_PATH} \
  -v ${LOCAL_RESULTS_PATH}:${DOCKER_RESULTS_PATH} \
  -v ${LOCAL_DATA_PATH}:${DOCKER_DATA_PATH} \
  -v ${LOCAL_MODELS_PATH}:${DOCKER_MODELS_PATH} \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /etc/shadow:/etc/shadow:ro \
  -v ${HOME}/.ssh:${DOCKER_REPO_PATH}/.ssh:ro"

  if [[ "${IS_DEV}" == "1" ]]; then
    docker_cmd="${docker_cmd} -v ${LOCAL_REPO_PATH}:${DOCKER_REPO_PATH}"
  fi

  docker_cmd="${docker_cmd} ${IMAGE_REPO}:{{image_tag}} {{cmd}}"

  set -xeuo pipefail
  DOCKER_BUILDKIT=1 ${docker_cmd} .

# run-dev lets us work with a dirty repository,
# beacuse this is a common state during development
# **AND** we're volume mounting the code, so we'll have the latest state
run-dev cmd: build-dev
  @just run ${IMAGE_TAG} {{cmd}}

# in contrast, run-release requires a clean repository,
# because users want to know that they're running the **exact** version they expect
# and we're **NOT** volume mounting the code
run-release cmd: build-release assert_clean_git_repo
  @just run ${IMAGE_TAG} {{cmd}}
