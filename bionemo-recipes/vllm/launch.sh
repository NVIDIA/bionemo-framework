#!/bin/bash

# Usage: ./launch.sh <container_name> [--mount_dir] [--headless]
# Example: ./launch.sh vllm --mount_dir --headless

MOUNT_DIR=false
HEADLESS=false
CONTAINER=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --mount_dir)
            MOUNT_DIR=true
            ;;
        --headless)
            HEADLESS=true
            ;;
        *)
            # First non-flag argument is the container name
            if [ -z "$CONTAINER" ]; then
                CONTAINER="$arg"
            fi
            ;;
    esac
done

if [ -z "$CONTAINER" ]; then
    echo "Usage: $0 <container_name> [--mount_dir] [--headless]"
    echo "Example: $0 vllm --mount_dir --headless"
    exit 1
fi

# Build docker run command
if [ "$HEADLESS" = true ]; then
    DOCKER_CMD="docker run -itd --gpus all --network host --ipc=host -e HF_TOKEN --rm --name vllm_dev"
else
    DOCKER_CMD="docker run -it --gpus all --network host --ipc=host -e HF_TOKEN --rm --name vllm_dev"
fi

if [ "$MOUNT_DIR" = true ]; then
    # Mount the project root (two levels up from this script)
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    DOCKER_CMD="$DOCKER_CMD -v ${PROJECT_ROOT}:/workspace/bionemo-framework"
fi

DOCKER_CMD="$DOCKER_CMD $CONTAINER /bin/bash"

exec $DOCKER_CMD
