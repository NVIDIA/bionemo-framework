#!/bin/bash
test_tag="needs_fork"
test_files=$(pytest --collect-only -m "${test_tag}" | grep '^[[:space:]]*<Module' | sed 's/^[[:space:]]*<Module//'| sed 's/\.py.*$/.py/' | awk '{$1=$1;print}' | sort | uniq)
n_test_files=$(echo "$test_files" | wc -l)
counter=1
# the overall test status collected from all pytest commands with test_tag
status=0

for testfile in $test_files; do
  rm -rf ./.pytest_cache/
  set -x
  if [[ $testfile != examples/* && $testfile != tests/* ]]; then
    testfile="tests/$testfile"
  fi
  echo "Running test ${counter} / ${n_test_files} : ${testfile}"

  pytest -m "${test_tag}" -vv --durations=0 --cov-append --cov=bionemo ${testfile}
  test_status=$?
  # Exit code 5 means no tests were collected: https://docs.pytest.org/en/stable/reference/exit-codes.html
  test_status=$(($test_status == 5 ? 0 : $test_status))
  # Updating overall status of tests
  status=$(($test_status > $status ? $test_status : $status))
  set +x
  ((counter++))
done

echo "Waiting for the tests to finish..."
wait

echo "Completed"

exit $status
