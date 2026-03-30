#!/bin/bash
set -euo pipefail

# bionemo-noodles is a Rust/Python hybrid (maturin + pyo3).
# The Rust toolchain is required to build from source.

if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain 1.82.0
fi

if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

PIP_CONSTRAINT= pip install maturin
PIP_CONSTRAINT= pip install -e ../bionemo-core
PIP_CONSTRAINT= pip install -e ".[test]"
