#!/bin/bash

ARCH=$(uname -m)

SYSTEM_ARCH=""

if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "arm64" ]; then
    SYSTEM_ARCH="arm64"
elif [ "${ARCH}" = "x86_64" ]; then
    SYSTEM_ARCH="amd64"
else
    echo "Unsupported architecture: ${ARCH}"
    exit 1
fi

echo "${SYSTEM_ARCH}"
