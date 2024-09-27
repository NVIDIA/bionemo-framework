#!/bin/bash

# Source your script (replace with your actual script name)
source "$(dirname "$0")/utils.sh"


# Test function
test_set_bionemo_home() {
    echo "Testing set_root_directory function..."

    # Unset BIONEMO_HOME to simulate the case where it is not set
    unset BIONEMO_HOME

    # Run the set_root_directory function
    set_bionemo_home

    # Check if BIONEMO_HOME was set correctly
    if [ -n "$BIONEMO_HOME" ]; then
        echo "\$BIONEMO_HOME is set to: $BIONEMO_HOME"
    else
        echo "ERROR: \$BIONEMO_HOME was not set!"
        return 1
    fi

    # Check if we are in the right directory
    if [ "$(pwd)" == "$BIONEMO_HOME" ]; then
        echo "SUCCESS: Current directory matches \$BIONEMO_HOME"
    else
        echo "ERROR: Current directory does not match \$BIONEMO_HOME"
        return 1
    fi
}

# Call the test function
test_set_bionemo_home
