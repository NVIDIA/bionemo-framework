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


import math
import random
from typing import Optional, Sequence, TypeVar, Union

from torch.utils.data import Dataset


__all__: Sequence[str] = "PRNGResampleDataset"

T_co = TypeVar("T_co", covariant=True)


class PRNGResampleDataset(Dataset[T_co]):
    """A thread-safe dataset shuffler that uses a pseudo-random number generator (PRNG) to shuffle the dataset.

    PRNGResampleDataset shuffles a given dataset using a pseudo-random number generator (PRNG). This allows for
    reproducible shuffling by controlling the random seed, while not ever storing the list of indices in memory. It
    works by generating random indices assuming that the requesting function asks for them sequentially. Although random
    lookups are supported, random lookups will involve recomputing state which is slow, and involves linearly advancing
    from 0 if the last requested index was greater than or equal to this requested index. This should work well with the
    megatron sampler which is sequential. It handles skipped lookups as will happen with multiple workers by not
    generating those numbers.

    !!! warning "Prefer bionemo.core.data.multi_epoch_dataset.MultiEpochDatasetResampler"

        This class performs sampling with replacement of an underlying dataset. It is recommended to use the epoch-based
        sampling provided by `bionemo.core.data.multi_epoch_dataset.MultiEpochDatasetResampler` instead, which ensures
        that each sample is seen exactly once per epoch. This dataset is useful for cases where the dataset is too large
        for the shuffled list of indices to fit in memory and exhaustive sampling is not required.
    """

    def __init__(self, dataset: Dataset[T_co], seed: int = 42, num_samples: Optional[int] = None):
        """Initializes the PRNGResampleDataset.

        Args:
            dataset: The dataset to be shuffled.
            seed: The seed value for the PRNG. Default is 42.
            num_samples: The number of samples to draw from the dataset.
                If None, the length of the dataset is used. Default is None.
        """
        self.initial_seed = seed
        self.rng = random.Random(seed)
        self.dataset_len = len(dataset)  # type: ignore
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.dataset = dataset
        # Store the last accessed index. On this first pass this is initialized to infinity, which will trigger a reset since
        #  index - inf < 0 for all values of index. This will lead to `self.advance_state(index)` being called which will advance
        #  the state to the correct starting index. The last_index will be then be replaced by `index` in that case and the algorithm
        #  will proceed normally.
        self.last_index: Union[int, math.inf] = math.inf
        self.last_rand_index: Optional[int] = None

    def rand_idx(self) -> int:
        """Generates a random index within the range of the dataset size."""
        return self.rng.randint(0, self.dataset_len - 1)

    def advance_state(self, num_to_advance: int):
        """Advances the PRNG state by generating n_to_advance random indices.

        Args:
            num_to_advance: The number of random state steps to advance.
        """
        for _ in range(num_to_advance):
            self.rand_idx()

    def __getitem__(self, index: int) -> T_co:
        """Returns the item from the dataset at the specified index.

        Args:
            index: The index of the item to retrieve.

        Returns:
            The item from the dataset at the specified index.

        Note:
            If the requested index is before the last accessed index, the PRNG state is reset to the initial seed
            and advanced to the correct state. This is less efficient than advancing forward.
        """
        idx_diff = index - self.last_index
        if idx_diff < 0:
            # We need to go backwards (or it is the first call), which involves resetting to the initial seed and
            #   then advancing to just before the correct index, which is accomplished with `range(index)`.
            self.rng = random.Random(self.initial_seed)
            self.advance_state(index)
        elif idx_diff == 0:
            # If the index is the same as the last index, we can just return the last random index that was generated.
            #  no state needs to be updated in this case so just return.
            return self.dataset[self.last_rand_index]
        else:
            # We need to advance however many steps were skipped since the last call. Since i+1 - i = 1, we need to advance
            #  by `idx_diff - 1` to accomodate for skipped indices.
            self.advance_state(idx_diff - 1)
        self.last_index = index
        self.last_rand_index = (
            self.rand_idx()
        )  # store the last index called incase the user wants to requrest this index again.
        return self.dataset[self.last_rand_index]  # Advances state by 1

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples
