#!/bin/bash
#
#
# nsys option like --pytorch function-trace: 
#   nvtx markers for common torch operations at the pytorch level like torch.Tensor.to
#
# nsys option like --pytorch autograd-shapes-nvtx: 
#   nvtx markers for common torch operations at the kernel level like "to", "to_copy"
#


# ----------------------------------------
# (0) preamble
# ----------------------------------------
MESSAGE_TEMPLATE='********run_evo2_train.sh: %s\n'
DATE_OF_SCRIPT=$(date +'%Y%m%dT%H%M')
SCRIPT_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"
printf "${MESSAGE_TEMPLATE}" "begin"
printf "${MESSAGE_TEMPLATE}" "DATE_OF_SCRIPT=${DATE_OF_SCRIPT}"

# ----------------------------------------
# (1) set some user parameters
# ----------------------------------------
RESULTS_DIR="./results/run_nsys_with_evo2_train"

read -r -d '' NSYS_PROFILE_OPTIONS <<EOF
    -s none \\
    --trace=cuda,nvtx \\
    --pytorch autograd-shapes-nvtx,functions-trace \\
    --force-overwrite true
EOF

TRAIN_ARGS_ARRAY=(
    "--nsys-profiling"
    "--nsys-start-step"
    "20"
    "--nsys-end-step"
    "28"
    "--mock-data"
    "--seq-length"
    "256"
    "--micro-batch-size"
    "1"
    "--model-size"
    "test"
    "--max-steps"
    "30"
    "--context-parallel-size"
    "1"
    "--devices"
    "1"
)

RUN_LABEL_PREFIX="nsys_bionemo_evo2_train"
PYTHON_SCRIPT_PATH=sub-packages/bionemo-evo2/src/bionemo/evo2/run/train.py

# ----------------------------------------
# (2) dump parameters
# ----------------------------------------
printf "${MESSAGE_TEMPLATE}" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ----------------------------------------
# (3) create log file name and report filename
# ----------------------------------------
run_label_arr=(
    ${RUN_LABEL_PREFIX}
    "mock-data"
    ${BRANCH_AT_START}
    ${DATE_OF_SCRIPT}
    ${COMMIT_AT_START}
)
RUN_LABEL="$(IFS='_'; echo "${run_label_arr[*]}")"

RESULTS_SUBDIR="${RESULTS_DIR}/"nsys"/${RUN_LABEL}"
mkdir -p "${RESULTS_SUBDIR}"
chmod a+r "${RESULTS_SUBDIR}"

LOG_FILE="${RESULTS_SUBDIR}/${RUN_LABEL}.log"
REPORT_FILE="${RESULTS_SUBDIR}/${RUN_LABEL}.nsys-rep"

# ----------------------------------------
# (4) create command
# ----------------------------------------
APPLICATION_TO_PROFILE="python ${PYTHON_SCRIPT_PATH} ${TRAIN_ARGS_ARRAY[@]}"

read -r -d '' NSYS_PROFILE_CMD <<EOF
nsys profile \\
    -o ${REPORT_FILE} \\
    ${NSYS_PROFILE_OPTIONS} \\
    ${APPLICATION_TO_PROFILE} | tee -a ${LOG_FILE}
EOF
# ----------------------------------------
# (5) run command
# ----------------------------------------
printf "\n"
printf "${MESSAGE_TEMPLATE}" "nsys version: $(nsys --version)"

printf "\n"
printf "${MESSAGE_TEMPLATE}" "APPLICATION_TO_PROFILE=${APPLICATION_TO_PROFILE}"
printf "\n"
printf "${MESSAGE_TEMPLATE}" "NSYS_PROFILE_CMD"
echo "${NSYS_PROFILE_CMD}"
eval "${NSYS_PROFILE_CMD}"

# ----------------------------------------
# (-1) post-amble
# ----------------------------------------
printf "${MESSAGE_TEMPLATE}" "end with success"