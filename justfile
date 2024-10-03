#!/usr/bin/env -S just --justfile

setup:
  ./internal/scripts/check_docker_version.sh
  source ./internal/scripts/setup_env_file.sh

build: setup
