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

import pytest

from bionemo.core.data.resamplers import PRNGResampleDataset


def test_prng_dataset_sequential_shuffler_full():
    """Test that the PRNGResampleDataset returns the same results as a random number generator when accessed sequentially."""
    dataset = list(range(10))
    seed = 42
    shuffled_dataset = PRNGResampleDataset(dataset, seed=seed, num_samples=100)
    rng = random.Random(seed)
    expected_output_full = [rng.randint(0, 9) for _ in range(100)]
    full_output = [shuffled_dataset[i] for i in range(100)]
    assert full_output == expected_output_full


@pytest.mark.parametrize("modulo_remainder", [0, 1, 4, 9])
def test_prng_dataset_sequential_shuffler_skips(modulo_remainder: int):
    """Test that the PRNGResampleDataset returns the same results as a random number generator when accessed sequentially but with
    some indices skipped, as would happen in a parallel dataloader context.
    """
    dataset = list(range(100))
    seed = 42
    shuffled_dataset = PRNGResampleDataset(dataset, seed=seed, num_samples=1000)
    rng = random.Random(seed)
    expected_output_full = [rng.randint(0, 99) for _ in range(1000)]
    every_10th_output = [shuffled_dataset[i] for i in range(1000) if i % 10 == modulo_remainder]
    assert every_10th_output == [expected_output_full[i] for i in range(1000) if i % 10 == modulo_remainder]


@pytest.mark.parametrize("modulo_remainder", [0, 1])
def test_prng_dataset_random_shuffler_skips(modulo_remainder: int):
    """Test that the PRNGResampleDataset returns the same results as a random number generator when accessed in a random order
    and with some indices skipped as well. This is what would happen if a user did an unexpected thing and called this
    on a dataset in a random order. This is expected to be slower but we still want it to work.
    """
    dataset = list(range(100))
    seed = 42
    shuffled_dataset = PRNGResampleDataset(dataset, seed=seed, num_samples=1000)
    rng = random.Random(seed)
    expected_output_full = [rng.randint(0, 99) for _ in range(1000)]
    indices_to_check = [i for i in range(1000) if i % 10 == modulo_remainder]
    rng.shuffle(indices_to_check)

    expected_shuffled = [expected_output_full[i] for i in indices_to_check]
    observed_shuffled = [shuffled_dataset[i] for i in indices_to_check]
    assert expected_shuffled == observed_shuffled


def test_repeated_lookups():
    """Test that repeated lookups of the same index return the same value."""
    dataset = list(range(100))
    seed = 42
    shuffled_dataset = PRNGResampleDataset(dataset, seed=seed, num_samples=1000)
    rng = random.Random(seed)
    expected_output_full = [rng.randint(0, 99) for _ in range(1000)]
    indices_to_check = [i for i in range(1000) if i % 10 == 3]
    rng.shuffle(indices_to_check)
    # Make some sequential and identical index lookups to check where the state is not advanced.
    indices_to_check[3] = 42
    indices_to_check[4] = 42
    indices_to_check[5] = 42

    indices_to_check[33] = 57
    indices_to_check[34] = 57
    indices_to_check[35] = 57
    indices_to_check[36] = 57

    expected_shuffled = [expected_output_full[i] for i in indices_to_check]
    observed_shuffled = [shuffled_dataset[i] for i in indices_to_check]
    assert expected_shuffled == observed_shuffled
