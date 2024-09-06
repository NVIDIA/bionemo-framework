#!/usr/bin/env bash

set -xueo pipefail

REPOSITORY_ROOT=$(git rev-parse --show-toplevel)
cd $REPOSITORY_ROOT

# To start, just run the unit tests and static checks.
bash ./scripts/ci/static_checks.sh
bash ./scripts/ci/pr_test.sh
