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
from torch.utils.data import Dataset

from bionemo.geneformer.data.singlecell.block_sampling import MapStyleScDataset, scDataset


class DummyDataset(Dataset):
    """A simple dummy dataset that returns sequential indices.

    This makes it easy to reason about the shuffling behavior.
    """

    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(idx.start, idx.stop, idx.step)]
        elif isinstance(idx, np.ndarray):
            return [self[i] for i in idx.tolist()]
        if idx < self.size:
            return idx
        else:
            return idx % self.size


def test_compare_mapstyle_v_iterstyle():
    """Compare scDataset with block sampling.

    to compare 'permute' scDataset with original scDataset we must:

    to compare our BlockSampling with scDataset we must use the bionemo_permute flag.

    show the three are equivalent.
    although really we know bionemo permute is sufficient, so not a big deal

    """
    dataset1 = DummyDataset(128)
    dataset2 = DummyDataset(128)

    scdataset1 = scDataset(
        dataset1,
        batch_size=16,
        block_size=4,
        fetch_factor=2,
        shuffle=True,
        drop_last=True,
        seed=42,
        bionemo_permute=True,
        sort_before_fetch=False,
        shuffle_before_yield=True,
    )
    result1 = list(scdataset1)
    # Groups into blocks and fetches
    # Looks like fetches are not being shuffled correctly.
    samplemapper = MapStyleScDataset(
        dataset2, block_size=4, batch_size=16, fetch_factor=2, seed=42, use_batch_queries=False
    )
    result2 = list(samplemapper)
    assert np.all(np.array(result1).flatten() == np.array(result2).flatten())


def test_block_sampling_identities():
    dataset_size = 128
    dataset2 = DummyDataset(dataset_size)

    samplemapper = MapStyleScDataset(dataset2, block_size=4, batch_size=16, fetch_factor=2, seed=42)
    result2 = list(samplemapper)

    assert np.all(np.sort(np.array(result2)) == np.arange(dataset_size))

    # If sorted by fetches, we should have block-continuity.
    fetch_sorted = np.sort(np.array(result2).reshape(-1, samplemapper.fetch_size))

    # Regroup by blocks (undoes the shuffling basically => shuffle blocks => shuffle fetches (into batches))
    block_sorted = np.array(fetch_sorted).reshape(-1, samplemapper.block_size)
    for block in block_sorted:
        assert np.all(np.diff(block) == 1)


@pytest.mark.parametrize("bionemo_permute", [True, False])
def test_scdataset_identities(bionemo_permute):
    dataset_size = 128
    dataset2 = DummyDataset(dataset_size)

    scdataset = scDataset(
        dataset2, block_size=4, batch_size=16, fetch_factor=2, seed=42, bionemo_permute=bionemo_permute
    )
    result2 = np.array(list(scdataset)).flatten()

    assert np.all(np.sort(result2) == np.arange(dataset_size))

    # If sorted by fetches, we should have block-continuity.
    fetch_sorted = np.sort(np.array(result2).reshape(-1, scdataset.fetch_size))

    # Regroup by blocks (undoes the shuffling basically => shuffle blocks => shuffle fetches (into batches))
    block_sorted = np.array(fetch_sorted).reshape(-1, scdataset.block_size)
    for block in block_sorted:
        assert np.all(np.diff(block) == 1)
