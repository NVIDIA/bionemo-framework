# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
This file tests the training-related utilities for ESM2.
"""

import torch
from nemo.core.optim.lr_scheduler import WarmupAnnealDecayHold


def test_warmup_anneal_decay_hold_scheduler_exists():
    optim = torch.optim.Adam(torch.nn.Linear(10, 1).parameters(), lr=4e-4, weight_decay=0.01, betas=[0.9, 0.98])
    scheduler = WarmupAnnealDecayHold(
        optimizer=optim, warmup_steps=2000, constant_steps=50000, min_lr=4e-5, max_steps=500000, max_lr=4e-4
    )
    assert scheduler is not None


def test_warmup_anneal_decay_hold_scheduler_initializes():
    optim = torch.optim.Adam(torch.nn.Linear(10, 1).parameters(), lr=4e-4, weight_decay=0.01, betas=[0.9, 0.98])
    scheduler = WarmupAnnealDecayHold(
        optimizer=optim, warmup_steps=10, constant_steps=50, min_lr=0.01, max_steps=100, max_lr=0.1
    )

    assert scheduler.max_steps == 100
    assert scheduler.warmup_steps == 10
    assert scheduler.max_lr == 0.1
    assert scheduler.min_lr == 0.01


def test_warmup_anneal_decay_hold_works():
    optim = torch.optim.Adam(torch.nn.Linear(10, 1).parameters(), lr=4e-4, weight_decay=0.01, betas=[0.9, 0.98])
    max_lr = 0.1
    min_lr = 0.01
    anneal_percentage = 0.50
    constant_value = anneal_percentage * max_lr
    scheduler = WarmupAnnealDecayHold(
        optimizer=optim,
        warmup_steps=20,
        min_lr=min_lr,
        max_steps=100,
        max_lr=max_lr,
        anneal_percentage=anneal_percentage,
    )

    assert scheduler.get_lr()[0] == min_lr
    # Check initial LR
    for _ in range(20):
        scheduler.step()
    # Check warmup phase
    assert scheduler.get_lr()[0] == max_lr

    # Check decay is lower than max
    for _ in range(20):
        scheduler.step()

    decay_lr = scheduler.get_lr()[0]
    # Check decay is lower than last decay
    assert decay_lr < max_lr

    # Keep decay stepping
    for _ in range(20):
        scheduler.step()

    decay_low = scheduler.get_lr()[0]
    assert decay_low < decay_lr
    assert decay_low == constant_value

    for _ in range(30):
        scheduler.step()

    assert scheduler.get_lr()[0] == constant_value

    # Check hold phase. Run it much longer and confirm
    for _ in range(300):
        scheduler.step()

    assert scheduler.get_lr()[0] == constant_value


def test_config_creates_proper_optimizer():
    # TODO: Create the optimizer from the config file and make sure it works.
    pass
