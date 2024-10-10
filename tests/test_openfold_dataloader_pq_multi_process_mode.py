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


"""Test module to check correct functionality of a priority queue dataloader
for the openfold model.

Check correct behavior of priority queue dataloader for
    - real dataset and sampler
    - multiprocess mode
See also test_openfold_dataloader_pq_single_process_mode.py.
"""

import multiprocessing

import pytest
from omegaconf import DictConfig

from bionemo.utils.logging_utils import log_with_nemo_at_level
from tests.test_openfold_data import initial_training_cfg  # noqa
from tests.test_openfold_dataloader_pq_single_process_mode import create_and_step_through_dataloader_with_openfold_data


@pytest.mark.skip(reason="As of 2024-05-13, on ci machine, hit timeouts for 2 and 20.")
@pytest.mark.parametrize(
    "num_workers,prefetch_factor",
    [
        (2, 2),
    ],
)
def test_000_get_real_data_in_multiprocessing_mode(
    initial_training_cfg: DictConfig,  # noqa: F811
    num_workers: int,
    prefetch_factor: int,
):
    """Check that if the timeout_for_get is long, at least 1 record is obtained.

    Also, based on previous execution, if num_workers >= 2, we expect that
    records should be longer as the timeout is increased...

    Assume 1 node, with single gpu
        rank=0
        world_size = 1
    """

    # (0) sanity check ----------------------------------------------------
    if num_workers > (multiprocessing.cpu_count() - 2):
        return

    # (1) set params ------------------------------------------------------
    #
    timeout_values = sorted([2, 20])
    seed = 44
    device_batch_size = 1
    num_prev_steps = 0
    world_size = 1
    rank = 0
    is_distributed = True
    num_steps_in_one_epoch = 5

    # (2) run the dataloader-----------------------------------------------
    #
    #   on dev-machine, with:
    #       AMD Ryzen Threadripper PRO 5975WX 32-Cores, 126 GB phys mem
    #
    #       timeout=2:  training_example_indices = [24, 1, 11]
    #       timeout=20: training_example_indices = [24, 1, 11, 28, 20]
    #
    expected_training_example_indices = [24, 1, 11, 28, 20]
    training_example_indices_for_each_timeout_value = []
    for _, timeout in enumerate(timeout_values):
        training_example_indices, _ = create_and_step_through_dataloader_with_openfold_data(  # call function afresh
            initial_training_cfg=initial_training_cfg,
            num_steps_in_one_epoch=num_steps_in_one_epoch,
            seed=seed,
            num_prev_steps=num_prev_steps,
            is_distributed=is_distributed,
            world_size=world_size,
            rank=rank,
            device_batch_size=device_batch_size,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
            timeout_for_get=timeout,
        )
        log_with_nemo_at_level(
            f"""
            training_example_indices={training_example_indices}
            """
        )
        training_example_indices_for_each_timeout_value += [training_example_indices]

        # (3) check results------------------------------------------

        # (3a) if there are any records obtained within the period of the timeout,
        # they must belong to the set of records identified by the sampler
        assert set(training_example_indices_for_each_timeout_value[-1]).issubset(expected_training_example_indices)

        # (3b) check that if the timeout is long, we get at least 1 record
        if timeout >= 20:
            assert len(training_example_indices_for_each_timeout_value[-1]) > 0

    # if timeout is longer, should get at least as many records
    if len(timeout_values) >= 2:
        assert len(training_example_indices_for_each_timeout_value[-1]) >= len(
            training_example_indices_for_each_timeout_value[0]
        )
