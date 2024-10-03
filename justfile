#!/usr/bin/env -S just --justfile

build:
  version_ge() {
      # Returns 0 (true) if $1 >= $2, 1 (false) otherwise
      [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
  }
  local docker_version=$(docker --version | awk -F'[, ]' '{print $3}')
  local required_docker_version='23.0.1'

  if ! version_ge "$docker_version" "$required_docker_version"; then
      echo "Error: Docker version $required_docker_version or higher is required. Current version: $docker_version"
      exit 1
  fi
  echo "ok"
