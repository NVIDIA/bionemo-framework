#!/bin/bash
#
# title: virtual_env_create.sh
# usage:
#   cd <repo root>; ./scripts/virtual_env_create.sh
#
#   create a virtual environment for the benchmarking repo
#
MESSAGE_TEMPLATE='********virtual_env_create.sh: %s\n'
DATE_OF_SCRIPT=$(date +'%Y%m%dT%H%M%S')
SCRIPT_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"
GIT_BRANCH=$(git branch --show-current)

# -----------------------------------------------
# (1) set script-level parameters
# ------------------------------------------------
ENV_DIR=./venv_bionemo_fw

# -----------------------------------------------
# (2) create venv
# ------------------------------------------------
printf "${MESSAGE_TEMPLATE}" "attempt to created a virtual env in directory ${ENV_DIR}"

# --------------------------------------------------
# on computelab run
#   (1) cannot run as sudo
# -------------------------------------------------
#apt update
#apt install -y python3 python3-pip python3.10-venv


sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev


python3 -m venv ${ENV_DIR}


# -----------------------------------------------
# (3) enter venv
# ------------------------------------------------
source ${ENV_DIR}/bin/activate
printf "${MESSAGE_TEMPLATE}" "you are in virtual env in directory ${ENV_DIR}"

# -----------------------------------------------
# (3) install pip to virtual environment
# ------------------------------------------------
if [[ "$(hostname)" == *viking-prod* ]]; then
    printf "${MESSAGE_TEMPLATE}" "installing pip inside virtual-environment on viking host"
    sudo apt update
    sudo apt install -y python3-pip
else
    printf "${MESSAGE_TEMPLATE}" "installing pip inside virtual-environment"
    apt update python3-pip python3.10-venv
    apt install -y python3-pip
fi


# -----------------------------------------------
# (4) pip install
# ------------------------------------------------
pip install pre-commit==4.1.0

printf "${MESSAGE_TEMPLATE}" "to enter this virtual env, source ${ENV_DIR}/bin/activate"
printf "${MESSAGE_TEMPLATE}" "to exit this virtual env, deactivate"
printf "${MESSAGE_TEMPLATE}" "exiting"