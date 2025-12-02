# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_annealing_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2_000,
    num_decay_steps=500_000,
    last_epoch=-1,
):
    """Cosine annealing scheduler with warmup.

    The learning rate is linearly warmed up from 0 to peak over num_warmup_steps,
    then follows a cosine annealing schedule from peak to 0 over num_decay_steps.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Warmup phase: linearly increase learning rate from 0 to 1
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine annealing phase: decay from 1 to 0 using cosine schedule
            progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
