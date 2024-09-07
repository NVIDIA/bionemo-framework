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

from math import isclose

import pytest
import torch

from bionemo.size_aware_batching.memory_model import collect_cuda_peak_alloc, polynomial_regression


@pytest.mark.parametrize("feature_fn", [None, lambda x: x.sum()])
def test_collect_cuda_peak_alloc(dataset, model, feature_fn):
    model, alloc_peak_expected = model
    features, alloc_peaks = collect_cuda_peak_alloc(model, dataset, dataset.device, feature_fn=feature_fn)
    assert len(features) == len(dataset)
    assert len(alloc_peaks) == len(dataset)
    alloc_peaks_tensor = torch.tensor(alloc_peaks)

    try:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data_batch = next(iter(dataloader))
    except Exception as e:
        pytest.skip(f"Skipping memory allocation check because dataloading failed: {e}")
    else:
        assert isinstance(data_batch, torch.Tensor)
        alloc_peaks0 = alloc_peaks_tensor[0].repeat(alloc_peaks_tensor.numel())
        rtol = 1e-1
        atol = 1
        torch.testing.assert_close(
            alloc_peaks_tensor,
            alloc_peaks0,
            rtol=rtol,
            atol=atol,
            msg=lambda msg: f"Uniform data size results in variation of CUDA memory consumption\n\n {msg}",
        )
        assert isclose(float(alloc_peaks[0]), float(alloc_peak_expected), rel_tol=rtol), (
            f"Peak CUDA memory allocation is {alloc_peaks[0] / (1024**2)} MB, "
            f"which is not within {rtol} of the expected {alloc_peak_expected / (1024**2)} MB"
        )


def test_collect_cuda_peak_alloc_skip_oom(dataset, model_huge):
    features, alloc_peaks = collect_cuda_peak_alloc(model_huge, dataset, dataset.device)
    assert len(features) == 0
    assert len(alloc_peaks) == 0


@pytest.mark.parametrize("degree", [-1, 0, 1, 2, 3])
def test_polynomial_regression(degree):
    if degree < 0:
        with pytest.raises(ValueError):
            polynomial_regression(degree, torch.zeros(1), torch.zeros(1))
    else:
        x = torch.randn(100)
        polynomial = x.unsqueeze(-1).repeat(1, degree + 1)
        polynomial[:, 0] = 1
        polynomial = torch.cumprod(polynomial, dim=-1)
        expected_coeffs = torch.randn(degree + 1)
        y = polynomial @ expected_coeffs
        coeffs = polynomial_regression(degree, x, y)
        assert torch.allclose(coeffs, expected_coeffs)


def test_polynomial_regression_x_and_y_mismatched_length():
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 50)
    with pytest.raises(ValueError):
        polynomial_regression(2, x, y)


def test_polynomial_regression_invalid_input_type():
    x = "hello"
    y = [1, 2, 3]
    with pytest.raises(TypeError):
        polynomial_regression(2, x, y)
