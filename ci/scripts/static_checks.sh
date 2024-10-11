#!/usr/bin/env bash

set -xueo pipefail

REPOSITORY_ROOT=$(git rev-parse --show-toplevel)
cd $REPOSITORY_ROOT

echo "Running ruff checks"
set +e
ruff check scripts/ sub-packages/ docs/
E_RUFF_CHECK="$?"
set -e

echo "Running tach checks"
set +e
tach check
E_TACH_CHECK="$?"
set -e

echo "Running pre-commit checks"
set +e
pre-commit run --all-files --show-diff-on-failure --color always
E_PRE_COMMIT="$?"
set -e

set +e
ANY_FAILURE=0
if [[ "${E_PRE_COMMIT}" != "0" ]]; then
    ANY_FAILURE=1
    echo "ERROR: pre-commit hooks failed! (exit: ${E_PRE_COMMIT})"
fi
if [[ "${E_RUFF_CHECK}" != "0" ]]; then
  ANY_FAILURE=1
  echo "ERROR: ruff check failed! (exit: ${E_RUFF_CHECK})"
fi
if [[ "${E_TACH_CHECK}" != "0" ]]; then
  ANY_FAILURE=1
  echo "ERROR: tach check failed! (exit: ${E_TACH_CHECK})"
fi
if [[ "${ANY_FAILURE}" != "0" ]]; then
  exit 1
else
  exit 0
fi
