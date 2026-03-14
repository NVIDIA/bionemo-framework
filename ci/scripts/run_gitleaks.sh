#!/usr/bin/env bash
# Wrapper to run gitleaks on individual files passed by pre-commit.
# gitleaks dir only accepts a single path, so we create a temp directory
# with symlinks to the target files and scan that.
set -euo pipefail

if [ $# -eq 0 ]; then
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

for f in "$@"; do
    mkdir -p "$tmp/$(dirname "$f")"
    ln -s "$(pwd)/$f" "$tmp/$f"
done

gitleaks dir "$tmp" --redact --follow-symlinks --no-banner --log-level warn --verbose
