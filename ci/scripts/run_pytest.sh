#!/bin/bash
set -xueo pipefail
export PYTHONDONTWRITEBYTECODE=1

if ! set_bionemo_home; then
    echo "Exiting script due to error in set_bionemo_home."
    exit 1
fi

echo "Running pytest tests"
pytest -m "not internal and not needs_fork and not needs_80gb_memory_gpu" -vv --durations=0 --cov=bionemo --cov-report term --cov-report xml:coverage-pytest.xml -k "not test_model_training" -s