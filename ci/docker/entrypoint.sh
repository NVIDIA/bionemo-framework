#!/bin/bash
set -e

# Get host UID/GID from environment variables. These need to be passed in explicitly when invoking the container; i.e.,
# docker run -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) ...
HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}

# Update the UID/GID of the container user
groupmod -g $HOST_GID ubuntu > /dev/null
usermod -u $HOST_UID ubuntu > /dev/null

# Execute the main container command
exec gosu ubuntu "$@"
