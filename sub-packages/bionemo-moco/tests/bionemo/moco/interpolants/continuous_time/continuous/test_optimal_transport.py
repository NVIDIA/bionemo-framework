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

import pytest
import torch

from bionemo.moco.interpolants.continuous_time.continuous.optimal_transport import OTSampler


@pytest.fixture
def toy_data():
    x0 = torch.tensor(
        [
            [[1.1, 1.1, 1.1], [1.1, 1.1, 1.1], [1.1, 1.1, 1.1]],
            [[-1.1, -1.1, -1.1], [-1.1, -1.1, -1.1], [-1.1, -1.1, -1.1]],
            [[1.1, 1.1, 1.1], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        ]
    )

    x1 = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
        ]
    )
    mask = None
    # Calculate the cost in naive for-loop. For exact OT, sqaured Euclidean distance is used
    costs = torch.zeros((x0.shape[0], x1.shape[0]))
    for i in range(x0.shape[0]):
        for j in range(x0.shape[0]):
            c = torch.sum(torch.square(x0[i] - x1[j]))
            costs[i, j] = c
    return x0, x1, mask, costs


@pytest.fixture
def toy_masked_data():
    x0 = torch.tensor(
        [
            [[1.1, 1.1, 1.1], [1.1, 1.1, 1.1], [1.1, 1.1, 1.1]],
            [[-1.1, -1.1, -1.1], [-1.1, -1.1, -1.1], [-1.1, -1.1, -1.1]],
            [[1.1, 1.1, 1.1], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        ]
    )

    x1 = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    # Calculate the cost in naive for-loop. For exact OT, sqaured Euclidean distance is used
    costs = torch.zeros((x0.shape[0], x1.shape[0]))
    for i in range(x0.shape[0]):
        mm = mask[i].unsqueeze(-1)
        for j in range(x0.shape[0]):
            per_atom_cost = torch.where(mm, torch.square(x0[i] - x1[j]), 0)
            c = torch.sum(per_atom_cost)
            costs[i, j] = c
    return x0, x1, mask, costs


@pytest.fixture
def exact_ot_sampler():
    ot_sampler = OTSampler(method="exact", num_threads=1)
    return ot_sampler


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sampler", ["exact_ot_sampler"])
@pytest.mark.parametrize("data", ["toy_data", "toy_masked_data"])
def test_exact_ot_sampler_ot_matrix(request, sampler, data, device):
    # Create an indices tensor
    ot_sampler = request.getfixturevalue(sampler)
    assert ot_sampler is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ot_sampler = ot_sampler.to_device(device)
    x0, x1, mask, ground_truth_cost_matrix = request.getfixturevalue(data)

    cost_matrix = ot_sampler._calculate_cost_matrix(x0, x1, mask=mask)
    assert cost_matrix.shape == (3, 3)
    assert torch.allclose(cost_matrix, ground_truth_cost_matrix, atol=1e-8)

    ot_matrix = ot_sampler.get_ot_matrix(x0, x1, mask=mask)
    ot_truth = torch.tensor([[1 / 3, 0.0, 0.0], [0.0, 0.0, 1 / 3], [0.0, 1 / 3, 0.0]])
    assert ot_matrix.shape == (3, 3)
    assert torch.allclose(ot_matrix, ot_truth, atol=1e-8)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sampler", ["exact_ot_sampler"])
@pytest.mark.parametrize("data", ["toy_data", "toy_masked_data"])
def test_exact_ot_sampler_sample_map(request, sampler, data, device):
    # Create an indices tensor
    ot_sampler = request.getfixturevalue(sampler)
    assert ot_sampler is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    ot_sampler = ot_sampler.to_device(device)
    x0, x1, mask, ground_truth_cost_matrix = request.getfixturevalue(data)
    ot_matrix = ot_sampler.get_ot_matrix(x0, x1, mask=mask)
    correct_mapping = {0: 0, 1: 2, 2: 1}

    x0_idx, x1_idx = ot_sampler.sample_map(ot_matrix, x0.shape[0], replace=False)
    # print(x0_idx)
    # print(x1_idx)
    assert x0_idx.shape == (x0.shape[0],)
    assert x1_idx.shape == (x1.shape[0],)
    all_indices = set(range(x0.shape[0]))
    sampled_indices = set()
    for i in range(len(x0_idx)):
        sampled_indices.add(x0_idx[i].item())
        assert x1_idx[i].item() == correct_mapping[x0_idx[i].item()]
    # When replace is False, all indices should be sampled
    assert all_indices == sampled_indices

    x0_idx, x1_idx = ot_sampler.sample_map(ot_matrix, x0.shape[0], replace=True)
    assert x0_idx.shape == (x0.shape[0],)
    assert x1_idx.shape == (x1.shape[0],)
    print(x0_idx)
    print(x1_idx)
    for i in range(len(x0_idx)):
        sampled_indices.add(x0_idx[i].item())
        assert x1_idx[i].item() == correct_mapping[x0_idx[i].item()]
    # When replace is True, not all indices should be sampled
