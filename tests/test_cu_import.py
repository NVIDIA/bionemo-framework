# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


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
