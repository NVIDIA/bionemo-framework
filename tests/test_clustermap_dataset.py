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


import tempfile
from collections import defaultdict

import numpy as np
import pytest
from torch.utils.data import Dataset

from bionemo.data.mapped_dataset import (
    Uniref90ClusterMappingDataset,
)


state_ = {}


class MockU50Dataset(Dataset):
    def __init__(self, num_clusters, num_samples):
        self.sample_mapping = np.random.choice(num_clusters, num_samples)
        self.cluster_ids = [f"u50_id{i}" for i in self.sample_mapping]

    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, index):
        return self.cluster_ids[index]


def make_fake_dataset(num_50_clusters, num_50_samples, num_maps):
    uniref50_dataset = MockU50Dataset(num_50_clusters, num_50_samples)
    cum_num_maps = 0
    cluster_map = {}
    alt_cluster_map = defaultdict(list)
    for i in range(num_50_clusters):
        cluster_map[f"u50_id{i}"] = [j + cum_num_maps for j in range(num_maps)]
        alt_cluster_map["counts"].append(num_maps)
        # append before incrementing cum_num_maps to indicate the start index
        # of the next `count` entries
        alt_cluster_map["starts"].append(cum_num_maps)
        cum_num_maps += num_maps

    uniref90_dataset = [{"sequence_id": i} for i in range(cum_num_maps)]

    tf2 = tempfile.NamedTemporaryFile(suffix="json")
    tf2_memmap_counts = tempfile.NamedTemporaryFile(suffix="json")
    tf2_memmap_starts = tempfile.NamedTemporaryFile(suffix="json")
    state_["alt_cluster_map"] = tf2
    state_["alt_cluster_map_counts"] = tf2_memmap_counts
    state_["alt_cluster_map_starts"] = tf2_memmap_starts

    counts = np.array(alt_cluster_map.pop("counts"))
    starts = np.array(alt_cluster_map.pop("starts"))
    counts_memmap = np.memmap(tf2_memmap_counts, dtype=int, mode="w+", shape=len(counts))
    starts_memmap = np.memmap(tf2_memmap_starts, dtype=int, mode="w+", shape=len(starts))
    counts_memmap[:] = counts
    starts_memmap[:] = starts
    counts_memmap.flush()
    starts_memmap.flush()
    return uniref50_dataset, uniref90_dataset, tf2_memmap_counts.name, tf2_memmap_starts.name


@pytest.mark.needs_gpu
@pytest.mark.skip
def test_dataset_construction():
    # Requires torch.distributed, failing on cpu tests, skipping for now.
    num_50_clusters = 20
    num_maps = 20
    num_50_samples = 200
    uniref50_dataset, uniref90_dataset, counts_fn, starts_fn = make_fake_dataset(
        num_50_clusters, num_50_samples, num_maps
    )
    index_mapping_dir = tempfile.TemporaryDirectory()
    alt_index_mapping_dir = tempfile.TemporaryDirectory()
    state_["index_mapping_dir"] = index_mapping_dir
    dataset = Uniref90ClusterMappingDataset(
        uniref50_dataset,
        uniref90_dataset,
        cluster_map_counts_fn=counts_fn,
        cluster_map_starts_fn=starts_fn,
        data_prefix="test_",
        index_mapping_dir=alt_index_mapping_dir.name,
        buffer_size=int(1e6),
    )

    assert len(dataset) == num_50_samples
    for i in range(len(uniref50_dataset)):
        assert (dataset[i]["sequence_id"] >= num_50_clusters * uniref50_dataset.sample_mapping[i]) and (
            dataset[i]["sequence_id"] < num_50_clusters * (uniref50_dataset.sample_mapping[i] + 1)
        )
