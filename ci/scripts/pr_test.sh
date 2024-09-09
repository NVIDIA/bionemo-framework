#!/usr/bin/env bash

set -xueo pipefail

REPOSITORY_ROOT=$(git rev-parse --show-toplevel)
cd $REPOSITORY_ROOT

export PYTHONDONTWRITEBYTECODE=1

echo "Running tests"

# TODO(pstjohn): replace the following two calls with the commented-out version, once all notebooks are runnable from
# the repositry root.
pytest -v scripts/ sub-packages/bionemo-*
pytest -v --nbval-lax docs/
# pytest -v --nbval-lax docs/ scripts/ sub-packages/bionemo-*
