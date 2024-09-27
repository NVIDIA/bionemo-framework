#!/bin/bash

# Function to check if Git repository is clean
check_git_repository() {
    if ! git diff-index --quiet HEAD --; then
        if [ $? -eq 128 ]; then
            echo "ERROR: Not in a git repository!" >&2
        else
            echo "ERROR: Repository is dirty! Commit all changes before building the image!" >&2
        fi
        return 1
    fi
}

set_bionemo_home() {
    set +u
    if [ -z "$BIONEMO_HOME" ]; then
        echo "\$BIONEMO_HOME is unset. Setting \$BIONEMO_HOME to repository root."

        # Ensure repository is clean
        if ! check_git_repository; then
            echo "Failed to set \$BIONEMO_HOME due to repository state." >&2
            return 1
        fi

        REPOSITORY_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
        if [ $? -ne 0 ]; then
            echo "ERROR: Could not determine the repository root. Ensure you're in a Git repository." >&2
            return 1
        fi

        BIONEMO_HOME="${REPOSITORY_ROOT}"
        echo "Setting \$BIONEMO_HOME to: $BIONEMO_HOME"
    fi
    set -u

    # Change directory to BIONEMO_HOME or exit if failed
    cd "${BIONEMO_HOME}" || { echo "ERROR: Could not change directory to \$BIONEMO_HOME: $BIONEMO_HOME" >&2; return 1; }
}