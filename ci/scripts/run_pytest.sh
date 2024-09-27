#!/bin/bash
set -e
export PYTHONDONTWRITEBYTECODE=1

if [ -z "$BIONEMO_HOME" ]; then
    echo "\$BIONEMO_HOME is unset. Setting \$BIONEMO_HOME to repository root "
    REPOSITORY_ROOT=$(git rev-parse --show-toplevel)
    BIONEMO_HOME="${REPOSITORY_ROOT}"
fi
cd "${BIONEMO_HOME}" || exit 1

echo "Running pytest tests"
pytest -m "not internal and not needs_fork and not needs_80gb_memory_gpu" -vv --durations=0 --cov=bionemo --cov-report term --cov-report xml:coverage-pytest.xml -k "not test_model_training" -s