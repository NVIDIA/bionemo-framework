#!/bin/bash

# ----------------------------------------
# (0) preamble
# ----------------------------------------
MESSAGE_TEMPLATE='********run_evo2_train.sh: %s\n'
DATE_OF_SCRIPT=$(date +'%Y%m%dT%H%M')
WHOAMI="$(whoami)"
SCRIPT_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"
printf "${MESSAGE_TEMPLATE}" "begin"
printf "${MESSAGE_TEMPLATE}" "DATE_OF_SCRIPT=${DATE_OF_SCRIPT}"
printf "${MESSAGE_TEMPLATE}" "WHOAMI=${WHOAMI}"

# ----------------------------------------
# (1) set some user parameters
# ----------------------------------------
RESULTS_DIR="./results"  # i.e. /workspace/bionemo2/results
RESULTS_THIS_APP_DIR="${RESULTS_DIR}/run_evo2_train"

RUN_LABEL_PREFIX="bionemo_evo2_train"
PYTHON_SCRIPT_PATH=sub-packages/bionemo-evo2/src/bionemo/evo2/run/train.py

TRAIN_ARGS_ARRAY=(
    "--mock-data" 
    "--seq-length"
    "256" 
    "--micro-batch-size"
    "1" 
    "--model-size"
    "test"
    "--max-steps"
    "40" 
    "--context-parallel-size"
    "1"
    "--devices"
    "1"
)

# ----------------------------------------
# (2) dump parameters
# ----------------------------------------
printf "${MESSAGE_TEMPLATE}" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ---------------------------------------------------
# (3) purge training app state
#   - delete / move dirs with training application state
# ----------------------------------------------------
if [[ -d "${RESULTS_DIR}/tmp/evo2" ]]; then
    rm -rf "${RESULTS_DIR}/tmp/evo2"
    printf "${MESSAGE_TEMPLATE}" "rm -rf ${RESULTS_DIR}/tmp/evo2, end"
fi
if [[ -d "${RESULTS_DIR}/evo2" ]]; then
    mv "${RESULTS_DIR}/evo2" "${RESULTS_DIR}/tmp/"
    printf "${MESSAGE_TEMPLATE}" "mv ${RESULTS_DIR}/evo2 ${RESULTS_DIR}/tmp/, end"
fi
# ----------------------------------------
# (4) create output dirs and file names
# ----------------------------------------
run_label_arr=(
    ${RUN_LABEL_PREFIX}
    ${BRANCH_AT_START}
    ${DATE_OF_SCRIPT}
    ${COMMIT_AT_START}
)
RUN_LABEL=$(IFS='_'; echo "${run_label_arr[*]}")
printf "${MESSAGE_TEMPLATE}" "RUN_LABEL=${RUN_LABEL}"

RESULTS_THIS_APP_THIS_RUN_DIR="${RESULTS_THIS_APP_DIR}/${RUN_LABEL}"
mkdir -p ${RESULTS_THIS_APP_THIS_RUN_DIR}
chmod a+rw ${RESULTS_THIS_APP_THIS_RUN_DIR}

LOG_FILE="${RESULTS_THIS_APP_THIS_RUN_DIR}/${RUN_LABEL}.log"


# ----------------------------------------
# (5) create python training script comman
# ---------------------------------------
read -r -d '' PY_COMMAND <<EOF
python -u ${PYTHON_SCRIPT_PATH} ${TRAIN_ARGS_ARRAY[@]} 2>&1 | tee -a ${LOG_FILE}
EOF

printf "${MESSAGE_TEMPLATE}" "PY_COMMAND=${PY_COMMAND}"
eval "${PY_COMMAND}"

# ----------------------------------------
# (-1) post-amble
# ----------------------------------------
printf "${MESSAGE_TEMPLATE}" "LOG_FILE=${LOG_FILE}"
printf "${MESSAGE_TEMPLATE}" "end with success"