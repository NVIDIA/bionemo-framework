#!/bin/bash -x

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json

export UV_LINK_MODE=copy
uv venv --system-site-packages

# 2. Activate the environment
source .venv/bin/activate

# 3. Install dependencies and ensure that constraints are not violated
pip freeze | grep transformer_engine > pip-constraints.txt
uv pip install -r build_requirements.txt --no-build-isolation  # some extra requirements are needed for building
uv pip install -c pip-constraints.txt -e . --no-build-isolation
