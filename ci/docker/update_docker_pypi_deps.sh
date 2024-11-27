#!/bin/bash

REPO_ROOT=$(git rev-parse --show-toplevel)
docker build $REPO_ROOT -t bionemo-deps -f $REPO_ROOT/ci/docker/Dockerfile.pip_deps

# Run the container to update the dependencies
docker run --rm -it -v $REPO_ROOT:/workspace -v $HOME/.cache:/root/.cache bionemo-deps /bin/bash -c "
    set -eo pipefail
    uv pip freeze > /pre-install-packages.txt
    uv pip install --no-build-isolation -r /workspace/requirements-docker.txt
    uv pip freeze > /post-install-packages.txt
    grep -vxFf /pre-install-packages.txt /post-install-packages.txt > /workspace/requirements-docker.txt
"
