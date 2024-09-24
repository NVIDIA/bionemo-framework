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

#
# modified from https://github.com/rssrwn/semla-flow/blob/main/semlaflow/data/util.py
#

import random
from collections import Counter
from typing import List

import numpy as np
from torch.utils.data import Sampler


def create_buckets(sizes: List[int], max_range: int = 20, min_bucket_size: int = 10000):
    """Create buckets for a list of integers with pre-defined maximal range of interval and minimal bucket sizes.

    Args:
        sizes (List[int]): List of integers to specify the sizes of each entry.
        max_range (int, optional): Maximal size range in each bucket. Defaults to 20.
        min_bucket_size (int, optional): minimal number of entries in each bucket. Defaults to 10000.
            bucket size may be smaller than min_bucket_size if its range reaches max_range.

    Returns:
        List[List[int]], List[int]: range for each bucket, bucket sizes
    """

    counter = Counter(sizes)
    dist = np.asarray(sorted([[k, counter[k]] for k in counter]))

    buckets = []
    bucket_sizes = []
    st = 0
    ed = 1
    while st < len(dist):
        while (
            ed < len(dist) and np.sum(dist[st:ed, 1]) < min_bucket_size and (not dist[ed, 0] - dist[st, 0] > max_range)
        ):
            ed += 1
        buckets.append([dist[st, 0], dist[ed - 1, 0]])
        bucket_sizes.append(np.sum(dist[st:ed, 1]))
        st = ed
        ed = st + 1

    return buckets, bucket_sizes


class BucketSizeAwareBatchSampler(Sampler):
    def __init__(
        self,
        sizes: List[int],
        max_bucket_range: int = 20,
        min_bucket_size: int = 10000,
        item_costs: List[float] = None,
        cost_bias: float = 0.0,
        batch_cost_limit: float = None,
        batch_size: int = None,
        **kwargs,
    ):
        """Batch Sampler to create homogeneous batches, and limit the cost of each batch.

        Args:
            sizes (List[int]): size of each entry, used to create homogeneous and uniform batches.
            max_bucket_range (int, optional): Maximal size range in each bucket. Defaults to 20.
            min_bucket_size (int, optional): minimal number of entries in each bucket. Defaults to 10000.
                bucket size may be smaller than min_bucket_size if its range reaches max_bucket_range.
            item_costs (List[float], optional): List of costs for each item. Defaults to None.
                If set as None, fall back to use batch_size.
            cost_bias (float, optional): Additional cost term for each batch. Defaults to 0.
            batch_cost_limit (float, optional): Cost limit in one batch. Defaults to None.
                If set as None, fall back to use batch_size.
            batch_size (int, optional): Batch size. Defaults to None. Can't set as None if costs or batch_cost_limit is None.

        Notes:
            This batch sampler will try to create more or less uniform batches based on the `sizes` provided.
            If `item_costs`, `cost_bias` and `batch_cost_limit` are provided, it will limit the cost of each batch to make it not greater than `batch_cost_limit`.
            The cost of each batch will be the summation of cost for each item based on `item_costs` plus the `cost_bias`.
            If any of `item_costs`, `cost_bias`, `batch_cost_limit` are not provided, this batch sampler will create batches with fixed `batch_size`.
        """

        self.sizes = np.array(sizes)
        self.max_bucket_range = max_bucket_range
        self.min_bucket_size = min_bucket_size

        if batch_size is None and (item_costs is None or batch_cost_limit is None or cost_bias is None):
            raise RuntimeError("batch_size can't be None if item_costs or batch_cost_limit or cost_bias is None")
        if item_costs is not None and cost_bias is not None and batch_cost_limit is not None:
            if len(item_costs) != len(self.sizes):
                raise RuntimeError(
                    f"the number of sizes {len(self.sizes)} don't match with the number of item_costs {len(item_costs)}"
                )
            self.item_costs = np.array(item_costs)
            self.cost_bias = cost_bias
            self.batch_cost_limit = batch_cost_limit
            self.batch_size = None
            self.control_batch_cost = True
        else:
            self.item_costs = None
            self.cost_bias = None
            self.batch_cost_limit = None
            self.batch_size = batch_size
            self.control_batch_cost = False

        self.buckets: List[List[int]]  # List of 2 integer lists to specify the range in each bucket
        self.bucket_sizes: List[int]  # number of entries in each bucket
        self.bucket_indices: List[np.ndarray]  # List of entry indices for each bucket
        self.bucket_num_batches: List[int]  # number of batches for each bucket

        self.buckets, self.bucket_sizes = create_buckets(self.sizes, max_bucket_range, min_bucket_size)
        self.bucket_indices = [np.argwhere((self.sizes >= st) * (self.sizes <= ed))[:, 0] for st, ed in self.buckets]
        if self.item_costs is not None:
            self.bucket_num_batches = [
                int(np.ceil(np.sum(self.item_costs[indices]) / self.batch_cost_limit))
                for indices in self.bucket_indices
            ]
        else:
            self.bucket_num_batches = [int(np.ceil(size / self.batch_size)) for size in self.bucket_sizes]

    def __len__(self):
        return np.sum(self.bucket_num_batches)

    def __iter__(self):
        # TODO non-shuffled sampling
        for indices in self.bucket_indices:
            random.shuffle(indices)

        remaining_batches = self.bucket_num_batches[:]
        used_items = [0 for _ in self.bucket_sizes]

        while sum(remaining_batches) > 0:
            bucket_idx = random.choices(range(len(remaining_batches)), weights=remaining_batches, k=1)[0]
            if used_items[bucket_idx] == self.bucket_sizes[bucket_idx] or remaining_batches[bucket_idx] == 0:
                continue

            if self.control_batch_cost:
                batch_cost = self.cost_bias
                batch = []
                for idx in self.bucket_indices[bucket_idx][used_items[bucket_idx] :]:
                    if batch_cost + self.item_costs[idx] <= self.batch_cost_limit:
                        batch.append(idx)
                        batch_cost += self.item_costs[idx]
                    else:
                        break
            else:
                batch = self.bucket_indices[bucket_idx][
                    used_items[bucket_idx] : (used_items[bucket_idx] + self.batch_size)
                ].tolist()

            used_items[bucket_idx] += len(batch)
            remaining_batches[bucket_idx] -= 1
            yield batch
