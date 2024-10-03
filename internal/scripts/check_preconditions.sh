#!/usr/bin/env bash

version_ge() {
    # Returns 0 (true) if $1 >= $2, 1 (false) otherwise
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

if [[ $(command -v git) ]]; then
  COMMIT=$(git rev-parse HEAD)
  if [[ "$?" != "0" ]]; then
    echo "ERROR: must run from within git repository!"
    exit 1
  fi
else
  echo "ERROR: git is not installed!"
  exit 1
fi

if [[ ! $(command -v docker) ]]; then
  echo "ERROR: docker is not installed!"
  exit 1
fi

docker_version=$(docker --version | awk -F'[, ]' '{print $3}')
required_docker_version='23.0.1'

if ! version_ge "$docker_version" "$required_docker_version"; then
    echo "Error: Docker version $required_docker_version or higher is required. Current version: $docker_version"
    exit 1
fi
