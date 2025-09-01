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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TAP_NSIGHT_LOCATION='/usr/local/cuda/bin/nsys'
export TAP_LOG_LEVEL='debug'
export TAP_MODE='nsight'     # '', nsight, or anna..... '' means ignore all profiling
export TAP_NVTX='pytorch'       # pytorch, apex, python
export TAP_BACKWARD_NVTX='false'   # true or false
export TAP_PROFILE_MEMORY='false'
export TAP_WAIT_STEPS='1'       # 2 is my default
export TAP_WARMUP_STEPS='1'    # 12 is my default, 
export TAP_ACTIVE_STEPS='4'     # 1 is my default
export TAP_WAIT_EPOCHS='1'
#!/usr/bin/env sh


export TAP_EXIT_ON_STOP=true

#export APP_NVTX_CATEGORIES='main,lit_module,dataset'
export TAP_NSIGHT_FLAGS='--trace nvtx,cuda'
#export TAP_MAX_DEPTH=14    # minimal is 4, since torch compile adds a ldevel,  default is 14


RESULTS_DIR="./results/run_tap_with_evo2_train"

TRAIN_ARGS_ARRAY=(
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
    "--val-check-interval" "0"
)

RUN_LABEL_PREFIX="tap_bionemo_evo2_train"
PYTHON_SCRIPT_PATH=sub-packages/bionemo-evo2/src/bionemo/evo2/run/train.py

# ----------------------------------------
# (2) dump parameters
# ----------------------------------------
printf "${MESSAGE_TEMPLATE}" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ----------------------------------------
# (3) create output dir names and file names
#       - create log file name and report filename
# ----------------------------------------
run_label_arr=(
    ${RUN_LABEL_PREFIX}
    "mock-data"
    ${BRANCH_AT_START}
    ${DATE_OF_SCRIPT}
    ${COMMIT_AT_START}
)
RUN_LABEL="$(IFS='_'; echo "${run_label_arr[*]}")"

RESULTS_SUBDIR="${RESULTS_DIR}/${RUN_LABEL}"
export TAP_SAVE_DIR="${RESULTS_SUBDIR}"
mkdir -p "${RESULTS_SUBDIR}"
chmod a+r "${RESULTS_SUBDIR}"

LOG_FILE="${RESULTS_SUBDIR}/${RUN_LABEL}.log"
REPORT_FILE="${RESULTS_SUBDIR}/${RUN_LABEL}.nsys-rep"

# ----------------------------------------
# (4) create command
# ----------------------------------------
APPLICATION_TO_PROFILE="python ${PYTHON_SCRIPT_PATH} ${TRAIN_ARGS_ARRAY[@]}"
#APPLICATION_TO_PROFILE="python -c 'import torch; x= torch.ones(500)'"

read -r -d '' TAP_PROFILE_CMD <<EOF
${APPLICATION_TO_PROFILE} 2>&1 | tee -a ${LOG_FILE}
EOF
# ----------------------------------------
# (5) run command
# ----------------------------------------
printf "\n"
printf "${MESSAGE_TEMPLATE}" "nsys version: $(nsys --version)"

printf "\n"
printf "${MESSAGE_TEMPLATE}" "APPLICATION_TO_PROFILE=${APPLICATION_TO_PROFILE}"
printf "\n"
printf "${MESSAGE_TEMPLATE}" "TAP_PROFILE_CMD"
echo "${TAP_PROFILE_CMD}"
eval "${TAP_PROFILE_CMD}"

if [[ -f '/tmp/.tap_dummy_nsight_report.nsys-rep' ]]; then
    cp /tmp/.tap_dummy_nsight_report.nsys-rep ${RESULTS_SUBDIR}/tap_dummy_nsight_report.nsys-rep
fi 
# ----------------------------------------
# (-1) post-amble
# ----------------------------------------
printf "${MESSAGE_TEMPLATE}" "TAP_SAVE_DIR=${TAP_SAVE_DIR}"
printf "${MESSAGE_TEMPLATE}" "end script"