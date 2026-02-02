#!/bin/bash

# Usage: ./launch.sh <container_name> [--mount_dir]
# Example: ./launch.sh vllm --mount_dir

MOUNT_DIR=false
CONTAINER=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --mount_dir)
            MOUNT_DIR=true
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
    echo "Usage: $0 <container_name> [--mount_dir]"
    echo "Example: $0 vllm --mount_dir"
    exit 1
fi

# Build docker run command
DOCKER_CMD="docker run -it --gpus all --network host --ipc=host -e HF_TOKEN --rm"

if [ "$MOUNT_DIR" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -v ${PWD}:/workspace/bionemo"
fi

DOCKER_CMD="$DOCKER_CMD $CONTAINER /bin/bash"

exec $DOCKER_CMD
