#!/bin/bash
set -euo pipefail

# The image build installs framework sub-packages editable against the copied
# repo contents. Once the devcontainer mounts the live workspace on top of that
# path, re-install the editable packages so they point at the mounted checkout.

shopt -s nullglob

packages=(./sub-packages/bionemo-*)
if ((${#packages[@]} == 0)); then
    echo "No framework sub-packages found to install."
    exit 0
fi

for sub in "${packages[@]}"; do
    uv pip install --system --no-deps --no-build-isolation --editable "$sub"
done
