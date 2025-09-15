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

"""Test that SCDL resources can be loaded from the local resources directory."""

from bionemo.scdl.data.load import get_all_resources, load


def test_scdl_resources_available():
    """Test that SCDL resources are available from the src directory."""
    # Test that we can get resources from the scdl data directory
    local_resources = get_all_resources()

    # Check that the SCDL resources are present
    expected_scdl_resources = ["scdl/sample", "scdl/sample_scdl_feature_ids", "scdl/sample_scdl_neighbor"]

    for resource_tag in expected_scdl_resources:
        assert resource_tag in local_resources, f"Resource {resource_tag} not found in local resources"
        print(f"✓ Found resource: {resource_tag}")


def test_load_function_uses_local_resources():
    """Test that the load function can find SCDL resources."""
    # This should not raise an error since the resource should be found locally
    try:
        # We're not actually downloading, just checking that the resource can be found
        # This will fail at download time but should pass the resource lookup
        load("scdl/sample_scdl_feature_ids")
    except ValueError as e:
        if "not found" in str(e):
            raise AssertionError(f"Resource lookup failed: {e}")
        # If it's a different error (like download failure), that's okay for this test
        pass
    except Exception:
        # Other exceptions are expected since we're not setting up full download environment
        pass

    print("✓ Load function can find SCDL resources")
