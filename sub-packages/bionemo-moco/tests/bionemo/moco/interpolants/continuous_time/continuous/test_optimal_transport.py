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
import pytest
import torch

from bionemo.moco.interpolants.continuous_time.continuous.data_augmentation.equivariant_ot_sampler import (
    EquivariantOTSampler,
)
from bionemo.moco.interpolants.continuous_time.continuous.data_augmentation.kabsch_augmentation import (
    KabschAugmentation,
)
from bionemo.moco.interpolants.continuous_time.continuous.data_augmentation.ot_sampler import OTSampler


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


@pytest.fixture
def equivariant_ot_sampler():
    ot_sampler = EquivariantOTSampler(method="exact", num_threads=1)
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
    x0, x1 = x0.to(device), x1.to(device)
    if mask is not None:
        mask = mask.to(device)
    ot_matrix = ot_sampler.get_ot_matrix(x0, x1, mask=mask)
    correct_mapping = {0: 0, 1: 2, 2: 1}

    x0_idx, x1_idx = ot_sampler.sample_map(ot_matrix, x0.shape[0], replace=False)
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
    for i in range(len(x0_idx)):
        sampled_indices.add(x0_idx[i].item())
        assert x1_idx[i].item() == correct_mapping[x0_idx[i].item()]
    # When replace is True, not all indices should be sampled

    # Final test to check the apply_augmentation function
    # First check preserving the order of noise
    ot_sampled_x0, ot_sampled_x1, ot_sampled_mask = ot_sampler.apply_augmentation(
        x0, x1, mask=mask, replace=False, sort="x0"
    )
    for i in range(len(x0_idx)):
        # Check if x0 output from apply_augmentation follows the correct order
        assert torch.allclose(ot_sampled_x0[i], x0[i], atol=1e-7)
        # Check if x1 output from apply_augmentation matches the correct mapping
        assert torch.allclose(ot_sampled_x0[i], ot_sampled_x1[i], atol=0.1)
        # Check if mask is preserved
        if mask is not None:
            assert (ot_sampled_mask[i] == mask[i]).all()

    # Then check preserving the order of data
    ot_sampled_x0, ot_sampled_x1, ot_sampled_mask = ot_sampler.apply_augmentation(
        x0, x1, mask=mask, replace=False, sort="x1"
    )
    reverse_mapping = {v: k for k, v in correct_mapping.items()}
    for i in range(len(x0_idx)):
        # Check if x1 output from apply_augmentation follows the correct order
        assert torch.allclose(ot_sampled_x1[i], x1[i], atol=1e-7)
        # Check if x1 output from apply_augmentation matches the correct mapping
        assert torch.allclose(ot_sampled_x0[i], ot_sampled_x1[i], atol=0.1)
        # Check if mask is preserved
        if mask is not None:
            assert (ot_sampled_mask[i] == mask[reverse_mapping[i]]).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sampler", ["equivariant_ot_sampler"])
def test_equivariant_ot_sampler_kabsch_align(request, sampler, device):
    ot_sampler = request.getfixturevalue(sampler)
    assert ot_sampler is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "cuda":
        atol = 1e-2
    else:
        atol = 1e-6
    ot_sampler = ot_sampler.to_device(device)
    x0 = torch.randn(size=(32, 3), device=device)
    alpha = np.random.rand() * 2 * np.pi
    R = torch.Tensor(np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])).to(
        device
    )
    # Apply rotation and translation to x0
    x0_rotated = x0 @ R.T + torch.ones_like(x0) * 5

    R_kabsch = ot_sampler.kabsch_align(x0, x0_rotated)
    assert R_kabsch.shape == (3, 3)
    assert torch.allclose(R_kabsch, R, atol=atol)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sampler", ["equivariant_ot_sampler"])
def test_equivariant_ot_sample_map(request, sampler, device):
    ot_sampler = request.getfixturevalue(sampler)
    assert ot_sampler is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if device == "cuda":
        atol = 1e-2
    else:
        atol = 1e-6
    ot_sampler = ot_sampler.to_device(device)
    x0 = torch.tensor(
        [
            [[2, 1, 2], [2, 1, -2], [-2, -1, 2], [-2, -1, -2], [0, 0, 0]],  # mask last, rectangle
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],  # mask last 2, triangle
            [[2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, -2, 0], [0, 0, 2]],  # mask none, pyramid
        ],
        dtype=torch.float32,
    ).to(device)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool).to(device)
    Rs = []
    for i in range(x0.shape[0]):
        alpha = np.random.rand() * 2 * np.pi
        R = torch.Tensor(
            np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        ).to(device)
        Rs.append(R)

    # Define correct mapping
    mapping = {0: 1, 1: 2, 2: 0}

    # Create rotated x0
    x0_rotated = torch.zeros_like(x0)
    for i in range(len(x0)):
        x0_rotated[mapping[i]] = x0[i] @ Rs[i].T

    # Test the get_ot_matrix and sample_map functions
    ot_matrix, Rs_output = ot_sampler.get_ot_matrix(x0, x0_rotated, mask=mask)
    x0_idx, x0_rotated_idx = ot_sampler.sample_map(ot_matrix, x0.shape[0], replace=False)
    assert x0_idx.shape == (x0.shape[0],)
    assert x0_rotated_idx.shape == (x0_rotated.shape[0],)

    rotations = Rs_output[x0_idx, x0_rotated_idx]

    # Make sure the Rotation matrices are correct by checking if x0_rotated can be rotated back to x0
    for i in range(len(x0_idx)):
        assert x0_rotated_idx[i].item() == mapping[x0_idx[i].item()]
        RR = rotations[i]
        x0_rotate_back = x0_rotated[x0_rotated_idx[i]] @ RR
        assert torch.allclose(x0[x0_idx[i]], x0_rotate_back, atol=atol)

    # Final test to check the apply_augmentation function
    # First check preserving the order of noise
    realigned_x0, realigned_x0_rotated, realigned_mask = ot_sampler.apply_augmentation(
        x0, x0_rotated, mask=mask, replace=False, sort="x0"
    )
    for i in range(len(x0_idx)):
        # Check if x0 output from apply_augmentation follows the correct order
        assert torch.allclose(realigned_x0[i], x0[i], atol=atol)
        # Check if x1 output from apply_augmentation is rotated correctly
        assert torch.allclose(realigned_x0[i], realigned_x0_rotated[i], atol=atol)
        # Check if mask is preserved
        assert (realigned_mask[i] == mask[i]).all()

    # Then check preserving the order of data
    realigned_x0, realigned_x0_rotated, realigned_mask = ot_sampler.apply_augmentation(
        x0, x0_rotated, mask=mask, replace=False, sort="x1"
    )
    reverse_mapping = {v: k for k, v in mapping.items()}
    for i in range(len(x0_idx)):
        # Check if x0 output from apply_augmentation follows the correct order
        # Since the realigned_x0_rotated is rotated back to x0, we check if it is equal to x0[reverse_mapping[i]]
        assert torch.allclose(realigned_x0_rotated[i], x0[reverse_mapping[i]], atol=atol)
        # Check if x1 output from apply_augmentation is rotated correctly
        assert torch.allclose(realigned_x0[i], realigned_x0_rotated[i], atol=atol)
        # Check if mask is preserved
        assert (realigned_mask[i] == mask[reverse_mapping[i]]).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_kabsch_augmentation(request, device):
    torch.manual_seed(42)
    np.random.seed(42)
    augmentor = KabschAugmentation()
    assert augmentor is not None
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if device == "cuda":
        atol = 1e-2
    else:
        atol = 1e-6
    x0 = torch.randn(size=(32, 3), device=device)
    alpha = np.random.rand() * 2 * np.pi
    R = torch.Tensor(np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])).to(
        device
    )
    # Apply rotation and translation to x0
    x0_rotated = x0 @ R.T + torch.ones_like(x0) * 5
    R_kabsch, _ = augmentor.kabsch_align(x0, x0_rotated)
    assert R_kabsch.shape == (3, 3)
    assert torch.allclose(R_kabsch, R, atol=atol)
    x0_aligned, x0_copy = augmentor.apply_augmentation(x0_rotated, x0, align_noise_to_data=True)
    assert torch.allclose(x0, x0_copy, atol=atol)
    assert torch.allclose(x0_aligned, x0, atol=atol)

    x0_rotated_copy, x0_rotated_aligned = augmentor.apply_augmentation(x0_rotated, x0, align_noise_to_data=False)
    assert torch.allclose(x0_rotated, x0_rotated_copy, atol=atol)
    assert torch.allclose(x0_rotated_aligned, x0_rotated, atol=atol)

    # Batch wise tests
    x0 = torch.randn(size=(10, 32, 3), device=device)
    alpha = np.random.rand() * 2 * np.pi
    R = torch.Tensor(np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])).to(
        device
    )
    # Apply rotation and translation to x0
    x0_rotated = x0 @ R.T + torch.ones_like(x0) * 5
    R_kabsch, _ = augmentor.batch_kabsch_align(x0, x0_rotated)
    assert R_kabsch.shape == (10, 3, 3)
    assert torch.allclose(R_kabsch, R, atol=atol)
    x0_aligned, x0_copy = augmentor.apply_augmentation(x0_rotated, x0, align_noise_to_data=True)
    assert torch.allclose(x0, x0_copy, atol=atol)
    assert torch.allclose(x0_aligned, x0, atol=atol)  # values are close but error ranges from <1 to 2 e -6

    x0_rotated_copy, x0_rotated_aligned = augmentor.apply_augmentation(x0_rotated, x0, align_noise_to_data=False)
    assert torch.allclose(x0_rotated, x0_rotated_copy, atol=atol)
    assert torch.allclose(x0_rotated_aligned, x0_rotated, atol=atol)
