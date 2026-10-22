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

from bionemo.common.data.multi_epoch_dataset import EpochIndex, MultiEpochDatasetResampler


class IndexDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"num_epochs": 3}, {"num_samples": 4}])
def test_singleton_epochs(shuffle, kwargs):
    dataset = MultiEpochDatasetResampler(IndexDataset(1), shuffle=shuffle, **kwargs)
    assert [dataset[i] for i in range(len(dataset))] == [EpochIndex(i, 0) for i in range(len(dataset))]
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[-1]


def test_shuffled_epochs():
    dataset = MultiEpochDatasetResampler(IndexDataset(8), num_epochs=3, seed=42)
    repeated = MultiEpochDatasetResampler(IndexDataset(8), num_epochs=3, seed=42)
    for epoch in range(3):
        indices = [dataset[epoch * 8 + i] for i in range(8)]
        assert {index.epoch for index in indices} == {epoch}
        assert sorted(index.idx for index in indices) == list(range(8))
        assert indices == [repeated[epoch * 8 + i] for i in range(8)]
    assert any(dataset[i].idx != i for i in range(8))
