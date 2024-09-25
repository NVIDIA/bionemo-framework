#!/bin/bash

if [ -z "$BIONEMO_HOME" ]; then
    echo "\$BIONEMO_HOME is unset. Exiting."
    exit 1
fi
cd "${BIONEMO_HOME}" || exit 1

pytest -m "not internal and not needs_fork and not needs_80gb_memory_gpu" -vv --durations=0 --cov=bionemo --cov-report term --cov-report xml:coverage-pytest.xml -k "not test_model_training" -s