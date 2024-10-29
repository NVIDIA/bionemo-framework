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


import numpy as np
import torch


def recursive_detach(x):
    """Detach all tensors in a nested structure."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return type(x)(recursive_detach(item) for item in x)
    elif isinstance(x, dict):
        return {key: recursive_detach(value) for key, value in x.items()}
    else:
        return x


def recursive_assert_approx_equal(x, y, atol=1e-4, rtol=1e-4):
    """Assert that all tensors in a nested structure are approximately equal."""
    if isinstance(x, torch.Tensor):
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
    elif isinstance(x, np.ndarray):
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
        for x_item, y_item in zip(x, y):
            recursive_assert_approx_equal(x_item, y_item)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for key in x:
            recursive_assert_approx_equal(x[key], y[key])
    else:
        assert x == y
