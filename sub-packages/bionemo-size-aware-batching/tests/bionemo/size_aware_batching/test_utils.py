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

import numpy as np
import pytest
import torch

from bionemo.size_aware_batching.utils import collect_cuda_peak_alloc, create_buckets


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


def test_create_buckets_with_invalid_sizes():
    # empty sizes list
    with pytest.raises(ValueError):
        create_buckets([], max_range=5, min_bucket_count=3)
    # sizes data type is not integer
    with pytest.raises(ValueError):
        create_buckets([1.0, 2.0], max_range=5, min_bucket_count=3)
    # sizes are not iterable of integers
    with pytest.raises(ValueError):
        create_buckets(np.array([[1, 2], [3, 4]]), max_range=5, min_bucket_count=3)


def test_create_buckets_with_invalid_max_range():
    # max_range should be non-negative number
    with pytest.raises(ValueError):
        create_buckets(np.array([1, 2, 3]), max_range=-1, min_bucket_count=3)


def test_create_buckets_with_invalid_min_bucket_count():
    # min_bucket_count should be positive integer
    with pytest.raises(ValueError):
        create_buckets(np.array([1, 2, 3]), max_range=2, min_bucket_count=-1)
    with pytest.raises(ValueError):
        create_buckets(np.array([1, 2, 3]), max_range=2, min_bucket_count=3.5)


def test_create_buckets_single_element():
    bucket_ranges, bucket_sizes = create_buckets(np.array([1]), max_range=2, min_bucket_count=3)
    assert np.allclose(bucket_ranges, [[1, 1]])
    assert np.allclose(bucket_sizes, [1])


def test_create_buckets_multiple_elements():
    sizes = np.array([1, 2, 3, 4, 5])
    np.random.shuffle(sizes)
    bucket_ranges, bucket_sizes = create_buckets(sizes, max_range=10, min_bucket_count=5)
    assert np.allclose(bucket_ranges, [[1, 5]])
    assert np.allclose(bucket_sizes, [5])


def test_create_buckets_multiple_buckets():
    sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    np.random.shuffle(sizes)
    bucket_ranges, bucket_sizes = create_buckets(sizes, max_range=4, min_bucket_count=10)
    assert np.allclose(bucket_ranges, [[1, 5], [6, 10]])
    assert np.allclose(bucket_sizes, [5, 5])


def test_create_buckets_with_duplicates():
    sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    np.random.shuffle(sizes)
    bucket_ranges, bucket_sizes = create_buckets(sizes, max_range=4, min_bucket_count=10)
    assert np.allclose(bucket_ranges, [[1, 3]])
    assert np.allclose(bucket_sizes, [12])


def test_create_buckets_with_max_range_zero():
    sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    np.random.shuffle(sizes)
    bucket_ranges, bucket_sizes = create_buckets(sizes, max_range=0, min_bucket_count=5)
    assert np.allclose(bucket_ranges, [[1, 1], [2, 2], [3, 3]])
    assert np.allclose(bucket_sizes, [3, 4, 5])


def test_create_buckets_with_max_range_reached():
    sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 22, 22, 22, 22])
    np.random.shuffle(sizes)
    bucket_ranges, bucket_sizes = create_buckets(sizes, max_range=20, min_bucket_count=20)
    assert np.allclose(bucket_ranges, [[1, 3], [22, 22]])
    assert np.allclose(bucket_sizes, [12, 4])


def test_create_buckets_with_min_bucket_count_reached():
    sizes = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 11, 11, 11, 11, 11])
    np.random.shuffle(sizes)
    bucket_ranges, bucket_sizes = create_buckets(sizes, max_range=20, min_bucket_count=10)
    assert np.allclose(bucket_ranges, [[1, 3], [11, 11]])
    assert np.allclose(bucket_sizes, [12, 5])
