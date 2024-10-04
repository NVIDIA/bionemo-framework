#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
