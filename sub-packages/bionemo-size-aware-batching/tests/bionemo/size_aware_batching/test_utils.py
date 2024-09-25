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

from bionemo.size_aware_batching.utils import collect_cuda_peak_alloc


def get_work_fn(model: torch.nn.Module, data: torch.Tensor):
    def fbwd_and_sum(data):
        y = model(data)
        y.backward()
        return data.sum().item()

    return fbwd_and_sum


def get_cleanup_fn(model: torch.nn.Module):
    def cleanup():
        model.zero_grad(set_to_none=True)

    return cleanup


def test_collect_cuda_peak_alloc(dataset, model_and_alloc_peak):
    model, alloc_peak_expected = model_and_alloc_peak
    features, alloc_peaks = collect_cuda_peak_alloc(
        dataset, get_work_fn(model, dataset), dataset.device, cleanup=get_cleanup_fn(model)
    )
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


def test_collect_cuda_peak_alloc_skip_cpu(dataset, model_and_alloc_peak):
    model, _ = model_and_alloc_peak
    with pytest.raises(ValueError):
        collect_cuda_peak_alloc(dataset, get_work_fn(model, dataset), torch.device("cpu"))


def test_collect_cuda_peak_alloc_skip_oom(dataset, model_and_alloc_peak, model_huge_sample02):
    model, _ = model_and_alloc_peak
    features, alloc_peaks = collect_cuda_peak_alloc(
        dataset, get_work_fn(model, dataset), dataset.device, cleanup=get_cleanup_fn(model)
    )
    features_wo02, alloc_peaks_wo02 = collect_cuda_peak_alloc(
        dataset, get_work_fn(model_huge_sample02, dataset), dataset.device, cleanup=get_cleanup_fn(model_huge_sample02)
    )
    features_expected = [features[i] for i in range(len(features)) if not (i == 0 or i == 2)]
    alloc_peaks_expected = [alloc_peaks[i] for i in range(len(alloc_peaks)) if not (i == 0 or i == 2)]
    assert features_wo02 == features_expected
    assert alloc_peaks_wo02 == alloc_peaks_expected
