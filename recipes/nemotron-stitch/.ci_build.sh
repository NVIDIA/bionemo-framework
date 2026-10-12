#!/usr/bin/env bash

set -euo pipefail

# The CI image already contains the expensive, pinned framework stack. Install
# the checked-out package without resolving dependencies so every interpreter
# exercises the pull request rather than the copy baked into the image.
for python in /opt/nemo_rl_venv/bin/python /opt/ray_venvs/*/bin/python; do
    uv pip install --python "$python" --no-deps --editable .
done
