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
import torch
import torch.distributed as dist
import torch.multiprocessing.spawn

from bionemo.testing.distributed import dist_environment


REQUIRED_WORLD_SIZE = 2


def _test_all_reduce_sum(rank: int, world_size: int):
    with dist_environment(rank=rank, world_size=world_size):
        device = torch.device(f"cuda:{rank}")
        tensor = torch.tensor([rank + 1], device=device)
        dist.all_reduce(tensor)
        assert tensor.item() == world_size * (world_size + 1) / 2


@pytest.mark.skipif(
    torch.cuda.device_count() < REQUIRED_WORLD_SIZE,
    reason=f"Requires {REQUIRED_WORLD_SIZE} devices but got {torch.cuda.device_count()}",
)
def test_all_reduce_sum(world_size: int = REQUIRED_WORLD_SIZE):
    torch.multiprocessing.spawn(
        fn=_test_all_reduce_sum,
        args=(world_size,),
        nprocs=world_size,
    )
