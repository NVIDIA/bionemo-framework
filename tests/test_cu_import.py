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


def test_cu_import():
    # guard imports behind test execution:
    # on ImportError, will not cause entire test module to fail to load
    import cudf
    import cugraph
    import cuml

    # check that each module has a non-empty version string
    assert len(cudf.__version__) > 0
    assert len(cuml.__version__) > 0
    assert len(cugraph.__version__) > 0
