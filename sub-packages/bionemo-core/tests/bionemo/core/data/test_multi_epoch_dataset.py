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


import random
from unittest import mock

import pytest

from bionemo.core.data.multi_epoch_dataset import (
    EpochIndex,
    IdentityMultiEpochDatasetWrapper,
    MultiEpochDatasetResampler,
)


def test_multi_epoch_dataset_correct_length():
    dataset = mock.MagicMock()
    dataset.__len__.return_value = 100

    multi_epoch_dataset = MultiEpochDatasetResampler(dataset, num_epochs=3, shuffle=False)
    assert len(multi_epoch_dataset) == 300
    assert multi_epoch_dataset.num_samples == 300

    multi_epoch_dataset[0]
    multi_epoch_dataset[299]

    with pytest.raises(IndexError):
        multi_epoch_dataset[300]


def test_multi_epoch_dataset_correct_length_with_samples():
    dataset = mock.MagicMock()
    dataset.__len__.return_value = 100

    multi_epoch_dataset = MultiEpochDatasetResampler(dataset, num_samples=275, shuffle=False)
    assert len(multi_epoch_dataset) == 275
    assert multi_epoch_dataset.num_epochs == 3


def test_multi_epoch_dataset_with_samples_and_epochs_raises():
    dataset = mock.MagicMock()
    dataset.__len__.return_value = 100

    with pytest.raises(ValueError, match="Only one of num_epochs and num_samples should be provided."):
        MultiEpochDatasetResampler(dataset, num_samples=275, num_epochs=3, shuffle=False)


def test_multi_epoch_dataset_correct_length_no_epochs():
    dataset = mock.MagicMock()
    dataset.__len__.return_value = 100

    multi_epoch_dataset = MultiEpochDatasetResampler(dataset, shuffle=False)
    assert len(multi_epoch_dataset) == len(dataset)
    assert multi_epoch_dataset.num_epochs == 1


def test_multi_epoch_dataset_passes_correct_indices():
    dataset = mock.MagicMock()
    dataset.__len__.return_value = 5

    multi_epoch_dataset = MultiEpochDatasetResampler(dataset, num_epochs=3, shuffle=False)
    assert len(multi_epoch_dataset) == 15

    for epoch in range(3):
        for idx in range(len(dataset)):
            multi_epoch_dataset[epoch * len(dataset) + idx]
            dataset.__getitem__.assert_called_with(EpochIndex(epoch, idx))
            dataset.__getitem__.reset_mock()


def test_multi_epoch_dataset_shuffles_each_epoch():
    dataset = mock.MagicMock()
    dataset.__len__.return_value = 5

    multi_epoch_dataset = MultiEpochDatasetResampler(dataset, num_epochs=3, shuffle=True)

    previous_calls = set()
    for epoch in range(3):
        dataset.__getitem__.reset_mock()
        for idx in range(len(dataset)):
            multi_epoch_dataset[idx + epoch * len(dataset)]

        # Check that the dataset was called with all clusters
        assert dataset.__getitem__.call_count == len(dataset)
        dataset.__getitem__.assert_has_calls(
            [mock.call(EpochIndex(epoch, i)) for i in range(len(dataset))], any_order=True
        )

        # Ensure that the dataset was called with a different index order each epoch.
        call_order = tuple([call.args[0].idx for call in dataset.__getitem__.call_args_list])
        assert call_order not in previous_calls
        previous_calls.add(call_order)


def test_multi_epoch_dataset_wrapper_and_resampler():
    dataset = range(10)

    multi_epoch_dataset = IdentityMultiEpochDatasetWrapper(dataset)  # type: ignore
    assert len(multi_epoch_dataset) == 10

    resampled_dataset = MultiEpochDatasetResampler(multi_epoch_dataset, num_epochs=3, shuffle=False)
    assert len(resampled_dataset) == 30

    for epoch in range(3):
        for idx in range(10):
            assert resampled_dataset[epoch * 10 + idx] == dataset[idx]


def test_multi_epoch_dataset_memory_stress_test_epochs():
    dataset = range(1_000_000_000)
    multi_epoch_dataset = IdentityMultiEpochDatasetWrapper(dataset)  # type: ignore
    resampled_dataset = MultiEpochDatasetResampler(multi_epoch_dataset, num_epochs=1_000, shuffle=True)
    assert len(resampled_dataset) == 1_000_000_000_000

    for _ in range(10):
        rand_int = random.randint(0, 1_000_000_000_000)
        resampled_dataset[rand_int]


def test_multi_epoch_dataset_memory_stress_test_num_samples():
    dataset = range(1_000_000_000)
    multi_epoch_dataset = IdentityMultiEpochDatasetWrapper(dataset)  # type: ignore
    resampled_dataset = MultiEpochDatasetResampler(multi_epoch_dataset, num_samples=1_000_000_000_000, shuffle=True)
    assert len(resampled_dataset) == 1_000_000_000_000

    for _ in range(10):
        rand_int = random.randint(0, 1_000_000_000_000)
        resampled_dataset[rand_int]
