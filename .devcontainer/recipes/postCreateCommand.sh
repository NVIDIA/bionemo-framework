#!/usr/bin/env bash
set -euo pipefail
# Run via uv to avoid relying on updated PATH in this shell
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  uvx pre-commit install
fi
