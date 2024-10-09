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


from collections import Counter

import numpy as np
import pytest

from bionemo.data.mapped_dataset import OnlineSampleMapping
from bionemo.utils.testing import pretty_size, timeit


# NOTE: we need to comment line below only when developing OnlineSampleMapping, else this file can be skipped
# pytestmark = pytest.mark.skip(reason="Skipping all tests in this module.")

# NOTE: To see example output call python tests/test_online_sample_mapping.py


@pytest.mark.parametrize("seed", (1, 2, 3, 4))
@pytest.mark.parametrize("shuffle", (False, True))
@pytest.mark.parametrize("block_size", (3, 5))
@pytest.mark.parametrize("dataset_size", (5, 7, 10))
@pytest.mark.parametrize("num_samples_coef", (1, 2, 3))
def test_online_upsampling(seed, shuffle, block_size, dataset_size, num_samples_coef):
    if not shuffle and seed != 1:
        pytest.skip("skipping non-shuffled test with seed != 1")

    sm = OnlineSampleMapping(
        dataset_size=dataset_size,
        num_samples=dataset_size * num_samples_coef,
        block_size=block_size,
        shuffle=shuffle,
    )
    # test that all samples are present exactly num_samples_coef times
    counter = Counter(list(sm))
    assert all(v == num_samples_coef for v in counter.values()), "incorrect number of samples"
    # test correct order when not shuffling
    if not shuffle:
        assert [i % dataset_size for i in range(sm.num_samples)] == list(sm), "incorrect order of samples"

    # test slicing
    for _ in range(5):
        i, j = np.random.randint(0, sm.num_samples, size=2)
        i, j = min(i, j), max(i, j)
        assert list(sm[i:j]) == list(sm)[i:j], "incorrect slicing"


if __name__ == "__main__":

    def print_sm(sm, with_list=True):
        print(sm)
        if with_list:
            print(list(sm))

    # print various examples of OnlineSampleMapping for small datasets

    # Upsampling (aligned dataset size and num_samples)
    print("\n## Upsampling dataset of 5 elements to 10 samples\n")
    for dataset_size, num_samples, block_size, shuffle, seed in [
        (5, 10, 10, False, 1),
        (5, 10, 10, True, 1),
        (5, 10, 10, True, 2),
        (5, 10, 3, False, 1),
        (5, 10, 3, True, 1),
        (5, 10, 3, True, 2),
    ]:
        sm = OnlineSampleMapping(
            dataset_size=dataset_size,
            num_samples=num_samples,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            truncate_to_block_boundary=False,
        )
        print_sm(sm)

    # Downsampling
    print(
        "\n## Downsampling dataset of 5 elements to 3 samples (notice samples are taken from all of of the dataset)\n"
    )
    for dataset_size, num_samples, block_size, shuffle, seed in [
        (5, 3, None, False, 1),
        (5, 3, None, True, 1),
        (5, 3, None, True, 2),
        (5, 3, None, True, 3),
    ]:
        sm = OnlineSampleMapping(
            dataset_size=dataset_size,
            num_samples=num_samples,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            truncate_to_block_boundary=False,
        )
        print_sm(sm)

    # Upsampling (unaligned dataset size, num_samples, block_size, and using truncate_to_block_boundary)
    print("\n## Upsampling dataset of 5 elements to 7 samples using block of 3 (nothing aligns)\n")
    for dataset_size, num_samples, block_size, shuffle, seed in [
        (5, 7, 3, False, 1),
        (5, 7, 3, True, 1),
        (5, 7, 3, True, 2),
    ]:
        sm = OnlineSampleMapping(
            dataset_size=dataset_size,
            num_samples=num_samples,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            truncate_to_block_boundary=False,
        )
        print_sm(sm)

    print(
        "\n## Upsampling dataset of 5 elements to 7 samples using block of 3 (nothing aligns) with truncation (to block aligned 6 elements)\n"
    )
    for dataset_size, num_samples, block_size, shuffle, seed in [
        (5, 7, 3, False, 1),
        (5, 7, 3, True, 1),
        (5, 7, 3, True, 3),
    ]:
        sm = OnlineSampleMapping(
            dataset_size=dataset_size,
            num_samples=num_samples,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            truncate_to_block_boundary=True,
        )
        print_sm(sm)

    # speed test of massive upsampling
    print(
        "\n## Upsampling dataset of 100M elements to 1T samples using block of 1M (num_samples aligns with block_size)\n"
    )

    with timeit("Construction of OnlineSampleMapping: "):
        sm = OnlineSampleMapping(
            dataset_size=int(100e6),
            num_samples=int(1e12),
            block_size=int(1e6),
            shuffle=True,
            seed=1,
            truncate_to_block_boundary=False,
        )

    print_sm(sm, False)
    print(f"Memory usage of OnlineSampleMapping: {pretty_size(sm)}\n")

    with timeit("Construction of a block: "):
        sample_block = sm.get_sample_block(1)
    with timeit("Construction of a block 2nd time (cached): "):
        sample_block = sm.get_sample_block(1)

    print(f"sample_block = {sample_block}")

    with timeit("Fetching sample mapping index 0 + block construction: "):
        sm[0]

    with timeit("Fetching sample mapping index 0 + cached block: "):
        sm[0]

    with timeit("Fetching sample mapping index 1M + block construction: "):
        sm[1000000]

    with timeit("Fetching sample mapping index 1M+1 + cached block: "):
        sm[1000001]

    # test slicing speed
    with timeit("Slicing 1000 samples [100:1100]: "):
        sm[100:1100]

    print(f"Memory usage of OnlineSampleMapping (with cached blocks): {pretty_size(sm)}\n")

    print(
        "\n## Upsampling dataset of 100M elements to 1T samples using block of 1M (num_samples  DOES NOT ligns with block_size)\n"
    )

    with timeit("Construction of OnlineSampleMapping: "):
        sm = OnlineSampleMapping(
            dataset_size=int(100e6),
            num_samples=int(1e12) + 1,
            block_size=int(1e6),
            shuffle=True,
            seed=1,
            truncate_to_block_boundary=False,
        )

    print_sm(sm, False)
    print(f"Memory usage of OnlineSampleMapping: {pretty_size(sm)}\n")

    with timeit("Construction of a block: "):
        sample_block = sm.get_sample_block(1)
    with timeit("Construction of a block 2nd time (cached): "):
        sample_block = sm.get_sample_block(1)

    print(f"sample_block = {sample_block}")

    with timeit("Fetching sample mapping index 0 + block construction: "):
        sm[0]

    with timeit("Fetching sample mapping index 0 + cached block: "):
        sm[0]

    with timeit("Fetching sample mapping index 1M + block construction: "):
        sm[1000000]

    with timeit("Fetching sample mapping index 1M+1 + cached block: "):
        sm[1000001]

    print(f"Memory usage of OnlineSampleMapping (with cached blocks): {pretty_size(sm)}\n")

    # test slicing speed
    with timeit("Slicing 1000 samples [100:1100]: "):
        sm[100:1100]
