# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
import numpy.testing as npt
import torch
from nemo.core.optim.lr_scheduler import get_scheduler
from torch.nn import Parameter
from torch.optim import Adam


# needed to register the scheduler


exp_values = [
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.1,
    0.1,
    0.1,
    0.05,
    0.025,
    0,
    0,
]


def test_linear_warmup_hold_decay_scheduler():
    scheduler_cls = get_scheduler('LinearWarmupHoldDecayPolicy')
    n_steps = 12
    lr = 0.1

    model = [Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = Adam(lr=lr, params=model)

    warmup_steps = 4

    scheduler = scheduler_cls(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        constant_steps=3,
        max_steps=10,
        min_lr=0,
    )

    lrs = np.zeros(n_steps)
    for i in range(n_steps):
        lrs[i] = scheduler.get_lr()[0]
        scheduler.step()

    npt.assert_allclose(lrs, exp_values)
