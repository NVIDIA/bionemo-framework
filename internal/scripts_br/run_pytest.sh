#!/bin/bash

# ----------------------------------------
# (0) preamble
# ----------------------------------------
MESSAGE_TEMPLATE='********run_pytest.sh: %s\n'
DATE_OF_SCRIPT=$(date +'%Y%m%dT%H%M')
SCRIPT_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"
printf "${MESSAGE_TEMPLATE}" "begin"
printf "${MESSAGE_TEMPLATE}" "DATE_OF_SCRIPT=${DATE_OF_SCRIPT}"

# ----------------------------------------
# (1) set some user parameters
# ----------------------------------------
#CUDA_VISIBLE_DEVICE_LIST=MIG-0e9a0f4b-dfee-5517-a54e-a73d5c450f24 # 20gb
#CUDA_VISIBLE_DEVICE_LIST=MIG-08fb5198-a9d8-5984-b31a-a8e7044320d0 # 40gb
export CUDA_VISIBLE_DEVICE_LIST=GPU-6f9dcb23-36a0-81a9-8942-78e5f07e3817 # gpu 0 with 80gb
PYTEST_LOG_FILE_PREFIX="pytests_pr1058_unskip_evo2_tests"
PYTEST_MARKERS=("not slow")
#PYTEST_MARKERS=("not slow" "slow")
#TEST_PATH=$(pwd)
TEST_PATH=sub-packages/bionemo-evo2/tests/bionemo/evo2/test_evo2.py
#TEST_PATH=sub-packages/bionemo-evo2/tests/bionemo/evo2/test_evo2.py::test_golden_values_top_k_logits_and_cosine_similarity_7b

# ----------------------------------------
# (2) dump parameters
# ----------------------------------------
printf "${MESSAGE_TEMPLATE}" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ----------------------------------------
# (3) create log file name
# ----------------------------------------
TEST_PATH_LABEL="${TEST_PATH//\//-}"  # replace forward slashes with hypthens
TEST_PATH_LABEL="${TEST_PATH_LABEL//::/__}"  # replace forward slashes with hypthens
TEST_PATH_LABEL="${TEST_PATH_LABEL//.py/' '}"  # remove .py 

for PYTEST_MARKER in "${PYTEST_MARKERS[@]}"; do

    PYTEST_MARKER_LABEL="${PYTEST_MARKER// /}"
    PYTEST_LOG_FILE="test_logs_for_evo2/${PYTEST_LOG_FILE_PREFIX}_${TEST_PATH_LABEL}_${PYTEST_MARKER_LABEL}_${BRANCH_AT_START}_${DATE_OF_SCRIPT}_${COMMIT_AT_START}.log"
    PYTEST_COMMAND="pytest -s -v -m '${PYTEST_MARKER}' ${TEST_PATH} | tee -a ${PYTEST_LOG_FILE}"
    printf "${MESSAGE_TEMPLATE}" "PYTEST_COMMAND=${PYTEST_COMMAND}"
    eval "${PYTEST_COMMAND}"

done


# ----------------------------------------
# (-1) post-amble
# ----------------------------------------
printf "${MESSAGE_TEMPLATE}" "end with success"