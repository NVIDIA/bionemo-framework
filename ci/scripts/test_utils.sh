#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
