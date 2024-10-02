##!/bin/bash
##
## SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: LicenseRef-NvidiaProprietary
##
## NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
## property and proprietary rights in and to this material, related
## documentation and any modifications thereto. Any use, reproduction,
## disclosure or distribution of this material and related documentation
## without an express license agreement from NVIDIA CORPORATION or
## its affiliates is strictly prohibited.
#
#export PYTHONDONTWRITEBYTECODE=1
#source "$(dirname "$0")/utils.sh"
#
#if ! set_bionemo_home; then
#    exit 1
#fi
#
#test_tag="needs_fork"
#test_dirs="tests/|examples/"  # (Py)Test directories separated by | for extended `grep`.
#test_files=$(pytest --collect-only -q -m "${test_tag}" | grep -E "^(${test_dirs}).*.py" | sed 's/\.py.*$/.py/' | awk '{$1=$1;print}' | sort | uniq)
#n_test_files=$(echo "$test_files" | wc -l)
#echo "Forked PyTest Files: ${test_files}"
#echo "Number of Forked PyTest Files: ${n_test_files}"
#counter=1
## the overall test status collected from all pytest commands with test_tag
#status=0
#
## List to store files with non-zero test_status
#failed_files=()
#
#for testfile in $test_files; do
#  rm -rf ./.pytest_cache/
#  set -x
#  if [[ $testfile != examples/* && $testfile != tests/* ]]; then
#    testfile="tests/$testfile"
#  fi
#  echo "Running test ${counter} / ${n_test_files} : ${testfile}"
#
#  pytest -m "${test_tag}" -vv --durations=0 --cov-append --cov=bionemo ${testfile}
#  test_status=$?
#  # Exit code 5 means no tests were collected: https://docs.pytest.org/en/stable/reference/exit-codes.html
#  test_status=$(($test_status == 5 ? 0 : $test_status))
#  # If test_status is not zero, add the file to the failed_files list
#  if [ "$test_status" -ne 0 ]; then
#      failed_files+=("$testfile")
#  fi
#  # Updating overall status of tests
#  status=$(($test_status > $status ? $test_status : $status))
#  set +x
#  ((counter++))
#done
#
#echo "Waiting for the tests to finish..."
#wait
#
#echo "Completed"
#
## Print the list of failed files at the end
#if [ ${#failed_files[@]} -gt 0 ]; then
#    echo "The following files had a non-zero test status:"
#    for failed_file in "${failed_files[@]}"; do
#        echo "$failed_file"
#    done
#else
#    echo "All tests passed."
#fi
#
#exit $status
