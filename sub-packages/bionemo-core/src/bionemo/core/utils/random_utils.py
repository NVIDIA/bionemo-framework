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
from contextlib import contextmanager
from typing import Iterator, Sequence, Type

import numpy as np


__all__: Sequence[str] = (
    "get_seed_from_rng",
    "random_numpy_context",
)


@contextmanager
def random_numpy_context(seed: int = 42) -> Iterator[None]:
    """Context manager for setting numpy random state.

    The state is saved on entry and restored on exit to what it was. This way you can run code that needs random state
    in a `with` context using this function, and get back to whatever state was there before. This is useful for testing
    where you don't want the random state from one test to impact other tests.

    Example:
        >>> import numpy as np
        >>> from bionemo.core.utils.random_utils import random_numpy_context
        >>> ori_state = np.random.get_state()
        >>> with random_numpy_context(45):
            np.random.randint(5) # this will change the state
        >>> new_state = np.random.get_state()
        >>> assert ori_state == new_state
    """
    state = np.random.get_state()  # just fail if this fails
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def get_seed_from_rng(rng: np.random.Generator, dtype: Type[np.signedinteger] = np.int64) -> int:
    """Generates a deterministic random seed from an existing random generator.

    This is useful in particular because setting the torch seed doesn't want to accept a tuple of numbers, we we often
    do in initializing a numpy random generator with epoch, index, and global seeds.

    Used to seed a torch random generator from a numpy random generator.
    """
    return int(rng.integers(np.iinfo(dtype).max))
