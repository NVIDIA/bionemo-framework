#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Enable strict mode with better error handling
set -euo pipefail

# Mirror the framework image install pattern against the lighter recipes CI
# base image: install shared test dependencies and all framework packages, then
# overlay editable installs from the checked-out workspace.

if ! command -v cargo >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
            apt-get update -qy
            apt-get install -qyy curl
        else
            echo "curl is required to install the Rust toolchain" >&2
            exit 1
        fi
    fi

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.82.0
fi

if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

python -m pip install --ignore-installed pip setuptools wheel uv maturin

uv pip install --system --no-build-isolation \
    -r requirements-cve.txt \
    -r requirements-test.txt \
    ./sub-packages/bionemo-*

for sub in ./sub-packages/bionemo-*; do
    uv pip install --system --no-deps --no-build-isolation --editable "$sub"
done

python --version
uv --version
rustc --version
cargo --version
