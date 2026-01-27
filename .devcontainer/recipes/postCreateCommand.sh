#!/usr/bin/env bash
set -euo pipefail
# Run via uv to avoid relying on updated PATH in this shell
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  uvx pre-commit install
fi

# Set up Claude environment and proxy server
source .devcontainer/recipes/setup_claude_env.sh
