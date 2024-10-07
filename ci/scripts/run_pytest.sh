#!/bin/bash
#


set -xueo pipefail
export PYTHONDONTWRITEBYTECODE=1

source "$(dirname "$0")/utils.sh"

if ! set_bionemo_home; then
    exit 1
fi

echo "Running pytest tests"
pytest -v --nbval-lax docs/ scripts/ sub-packages/
