#!/usr/bin/env bash

if [[ $(command -v git) ]]; then
  COMMIT=$(git rev-parse HEAD)
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

#######################################################################################################################
# don't let these get set in the .env file
LOCAL_ENV='.env'
DOCKER_REPO_PATH='/workspace/bionemo2'

LOCAL_REPO_PATH=$(realpath $(pwd))
if [[ $(basename "${LOCAL_REPO_PATH}") != "bionemo-framework" ]]; then
  echo "ERROR: must run from the root of the bionemo repository!"
  echo "ERROR: invalid path: ${LOCAL_DATA_PATH}"
  exit 1
fi
# NOTE: do allow IMAGE_TAG to be overridden by env var, but DO NOT set this in the .env file!
IMAGE_TAG=${IMAGE_TAG:="bionemo2-${COMMIT}"}
#######################################################################################################################

env_file_setup() {
  # if $LOCAL_ENV file exists, source it to specify my environment
  local write_env
  if [[ -e ./$LOCAL_ENV ]]; then
      echo "Sourcing environment from ./$LOCAL_ENV"
      . ./$LOCAL_ENV
      write_env="0"
  else
      echo "$LOCAL_ENV does not exist. Writing defaults to $LOCAL_ENV"
      write_env="1"
  fi

  # these are set via the .env file
  CACHE_TAG=${CACHE_TAG:='bionemo2-latest'}
  IMAGE_REPO=${IMAGE_REPO:=nvcr.io/nvidian/cvai_bnmo_trng/bionemo}
  LOCAL_RESULTS_PATH=${LOCAL_RESULTS_PATH:=${LOCAL_REPO_PATH}/results}
  DOCKER_RESULTS_PATH=${DOCKER_RESULTS_PATH:=${DOCKER_REPO_PATH}/results}
  LOCAL_DATA_PATH=${LOCAL_DATA_PATH:=${LOCAL_REPO_PATH}/data}
  DOCKER_DATA_PATH=${DOCKER_DATA_PATH:=${DOCKER_REPO_PATH}/data}
  LOCAL_MODELS_PATH=${LOCAL_MODELS_PATH:=${LOCAL_REPO_PATH}/models}
  DOCKER_MODELS_PATH=${DOCKER_MODELS_PATH:=${DOCKER_REPO_PATH}/models}
  WANDB_API_KEY=${WANDB_API_KEY:=NotSpecified}
  JUPYTER_PORT=${JUPYTER_PORT:=8888}
  REGISTRY=${REGISTRY:=nvcr.io}
  REGISTRY_USER=${REGISTRY_USER:='$oauthtoken'}
  DEV_CONT_NAME=${DEV_CONT_NAME:=bionemo2}
  NGC_CLI_API_KEY=${NGC_CLI_API_KEY:=NotSpecified}
  NGC_CLI_ORG=${NGC_CLI_ORG:=nvidian}
  NGC_CLI_TEAM=${NGC_CLI_TEAM:=NotSpecified}
  NGC_CLI_FORMAT_TYPE=${NGC_CLI_FORMAT_TYPE:=ascii}

  # If $LOCAL_ENV was not found, write out a template for user to edit
  if [[ "${write_env}"  == "1" ]]; then
      echo "CACHE_TAG=${CACHE_TAG}" >> $LOCAL_ENV
      echo "IMAGE_REPO=${IMAGE_REPO}" >> $LOCAL_ENV
      echo "LOCAL_RESULTS_PATH=${LOCAL_RESULTS_PATH}" >> $LOCAL_ENV
      echo "DOCKER_RESULTS_PATH=${DOCKER_RESULTS_PATH}" >> $LOCAL_ENV
      echo "LOCAL_DATA_PATH=${LOCAL_DATA_PATH}" >> $LOCAL_ENV
      echo "DOCKER_DATA_PATH=${DOCKER_DATA_PATH}" >> $LOCAL_ENV
      echo "LOCAL_MODELS_PATH=${LOCAL_MODELS_PATH}" >> $LOCAL_ENV
      echo "DOCKER_MODELS_PATH=${DOCKER_MODELS_PATH}" >> $LOCAL_ENV
      echo "WANDB_API_KEY=${WANDB_API_KEY}" >> $LOCAL_ENV
      echo "JUPYTER_PORT=${JUPYTER_PORT}" >> $LOCAL_ENV
      echo "REGISTRY=${REGISTRY}" >> $LOCAL_ENV
      echo "REGISTRY_USER='${REGISTRY_USER}'" >> $LOCAL_ENV
      echo "DEV_CONT_NAME=${DEV_CONT_NAME}" >> $LOCAL_ENV
      echo "NGC_CLI_API_KEY=${NGC_CLI_API_KEY}" >> $LOCAL_ENV
      echo "NGC_CLI_ORG=${NGC_CLI_ORG}" >> $LOCAL_ENV
      echo "NGC_CLI_TEAM=${NGC_CLI_TEAM}" >> $LOCAL_ENV
      echo "NGC_CLI_FORMAT_TYPE=${NGC_CLI_FORMAT_TYPE}" >> $LOCAL_ENV

      echo "------------------------------------------------------------"
      echo "First time setup complete! Re-run!"
      exit 1
  fi
}

# setup the .env file & set environment variables
# NOTE: these env vars are needed by the justfile --> it **sources** this file!
env_file_setup
