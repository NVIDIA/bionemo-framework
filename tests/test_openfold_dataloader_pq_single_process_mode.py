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

    - simple dataset and sampler
        - single-process mode
        - multiprocess mode

    - real dataset and sampler
        - single process mode

See also test_openfold_dataloader_pq_multi_process_mode.py
"""

from functools import reduce
from typing import Iterator, List, Tuple

import pytest
from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler

from bionemo.data.protein.openfold.dataloaders_pq import TIMEOUT_FOR_PQUEUE_GET_DEFAULT, InitialTrainingDataloaderPQ
from bionemo.data.protein.openfold.helpers import collate
from bionemo.data.protein.openfold.samplers import InitialTrainingSampler
from tests.test_openfold_data import get_initial_training_dataset, initial_training_cfg  # noqa


SIMPLE_SAMPLER_LENGTH = 6
LOCAL_BATCH_SIZE_FOR_SIMPLE_DATA = 2


class Fruit:
    """A simple record, to use to test the mechanics of the priority queue
    dataloder."""

    def __init__(self, name: str = "apple"):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name


class SimpleDatasetWithRecordsAsDicts(Dataset):
    """Create a small dataset with simple records, to test the mechanics of
    the priority queue dataloader.

    The term index is equivalent to 'key', and does not imply an ordering.
    """

    def __init__(self, record_type=None):
        if record_type == "Fruit":
            self.records = {
                (10, 0): {"fruit": Fruit("apple")},
                (11, 1): {"fruit": Fruit("banana")},
                (12, 2): {"fruit": Fruit("coconut")},
                (13, 3): {"fruit": Fruit("dewberry")},
                (14, 4): {"fruit": Fruit("elderberry")},
                (15, 5): {"fruit": Fruit("fig")},
                (16, 6): {"fruit": Fruit("grape")},
            }
        else:
            self.records = {
                (10, 0): {"fruit": "apple"},
                (11, 1): {"fruit": "banana"},
                (12, 2): {"fruit": "coconut"},
                (13, 3): {"fruit": "dewberry"},
                (14, 4): {"fruit": "elderberry"},
                (15, 5): {"fruit": "fig"},
                (16, 6): {"fruit": "grape"},
            }

    def __getitem__(self, index: int):
        return self.records[index]

    def __len__(self):
        return len(self.records)

    def indices(self):
        return self.records.keys()


class SimpleSampler(Sampler[int]):
    """Single-record sampler, generates a sequence of indices."""

    def __init__(self, indices: List[int], length_of_sampler: int):
        self.length_of_sampler = length_of_sampler
        self.indices_sorted = sorted(indices)

    def __iter__(self) -> Iterator[int]:
        for i in range(self.length_of_sampler):
            yield self.indices_sorted[i % self.length_of_sampler]

    def __len__(self) -> int:
        return self.length_of_sampler


# function scope for the iteration
@pytest.fixture(scope="function")
def get_simple_dataset_and_sampler_for_builtins():
    dataset = SimpleDatasetWithRecordsAsDicts()
    sampler = SimpleSampler(
        indices=dataset.indices(),
        length_of_sampler=SIMPLE_SAMPLER_LENGTH,
    )
    return dataset, sampler


# function scope for the iteration
@pytest.fixture(scope="function")
def get_simple_dataset_and_sampler_for_fruit_class_instances():
    dataset = SimpleDatasetWithRecordsAsDicts(record_type="Fruit")
    sampler = SimpleSampler(
        indices=dataset.indices(),
        length_of_sampler=SIMPLE_SAMPLER_LENGTH,
    )
    return dataset, sampler


def test_000_simple_sampler(
    get_simple_dataset_and_sampler_for_builtins: Tuple[SimpleDatasetWithRecordsAsDicts, SimpleSampler],
    get_simple_dataset_and_sampler_for_fruit_class_instances: Tuple[SimpleDatasetWithRecordsAsDicts, SimpleSampler],
):
    """Check that calls to the sampler (a) return indices in the correct order,
    and (b) stop at SIMPLE_SAMPLER_LENGTH.

    Args:
        get_simple_dataset_and_sampler_for_builtins:
            dataset and sampler for SimpleDatasetWithRecordsAsDicts with
            records that are python builtins
        get_simple_dataset_and_sampler_for_fruit_class_instances:
            dataset and sampler for SimpleDatasetWithRecordsAsDicts with
            records that are instances of a simple class
    """
    # builtins version
    _, sampler = get_simple_dataset_and_sampler_for_builtins
    records = [x for _, (x, _) in enumerate(sampler)]
    assert 75 == sum(records)

    # class instances version
    _, sampler = get_simple_dataset_and_sampler_for_fruit_class_instances
    records = [x for _, (x, _) in enumerate(sampler)]
    assert 75 == sum(records)


def test_010_simple_dataset_and_simple_sampler_with_collate(
    get_simple_dataset_and_sampler_for_builtins: Tuple[SimpleDatasetWithRecordsAsDicts, SimpleSampler],
    get_simple_dataset_and_sampler_for_fruit_class_instances: Tuple[SimpleDatasetWithRecordsAsDicts, SimpleSampler],
):
    """Check that the records in the dataset are compatible with the collate
    function.
    """
    # builtins version
    dataset, sampler = get_simple_dataset_and_sampler_for_builtins
    sampler_iter = sampler.__iter__()  # create/reset the iterator
    records = [dataset[sampler_iter.__next__()] for _ in range(LOCAL_BATCH_SIZE_FOR_SIMPLE_DATA)]
    batch = collate(records)
    assert batch["fruit"] == ["apple", "banana"]

    # class instances version
    dataset, sampler = get_simple_dataset_and_sampler_for_fruit_class_instances
    sampler_iter = sampler.__iter__()  # create/reset the iterator
    records = [dataset[sampler_iter.__next__()] for _ in range(LOCAL_BATCH_SIZE_FOR_SIMPLE_DATA)]
    actual_batch = collate(records)
    expected_batch = {"fruit": [Fruit("apple"), Fruit("banana")]}
    assert expected_batch["fruit"] == actual_batch["fruit"]


@pytest.mark.parametrize(
    "num_workers",
    [0, 2],
)
def test_020_initial_training_dataloader_pq_with_simple_dataset_and_sampler(
    get_simple_dataset_and_sampler_for_builtins: Tuple[SimpleDatasetWithRecordsAsDicts, SimpleSampler],
    get_simple_dataset_and_sampler_for_fruit_class_instances: Tuple[SimpleDatasetWithRecordsAsDicts, SimpleSampler],
    num_workers: int,
):
    """Check that priority queue delivers the expected order of records,
    for records that are dicts with (a) builtin value, (b) user-defined class instances..

    Args:
        get_simple_dataset_and_sampler_for_builtins:
            dataset and sampler for SimpleDatasetWithRecordsAsDicts with
            records that are python builtins
        get_simple_dataset_and_sampler_for_fruit_class_instances:
            dataset and sampler for SimpleDatasetWithRecordsAsDicts with
            records that are instances of a simple class
        num_workers:  If the argument num_workers is 0, then all instance
            objects reside in current process.  If num_works > 1, then
            subprocesses are created in order to use multiple CPU cores
            for the Dataset.__getitem__ step.
    """

    # (1) dataset of dicts where values are builtins
    dataset, sampler = get_simple_dataset_and_sampler_for_builtins

    dl = InitialTrainingDataloaderPQ(
        dataset=dataset,
        sampler=sampler,
        local_batch_size=LOCAL_BATCH_SIZE_FOR_SIMPLE_DATA,
        num_workers=num_workers,
        prefetch_factor=1,
        seed=None,
        uniform_recycling_iters=None,
        num_prev_steps=0,
        use_threading=False,
        do_train_batch_properties=False,
    )
    records = [x for _, x in enumerate(dl)]

    actual_records_without_priority = [d["fruit"] for d in records]
    expected_records_without_priority = [
        ["apple", "banana"],
        ["coconut", "dewberry"],
        ["elderberry", "fig"],
    ]
    actual_flat = reduce(lambda x, y: x + y, actual_records_without_priority, [])
    expected_records_without_priority = reduce(lambda x, y: x + y, expected_records_without_priority, [])
    assert sorted(actual_flat) == sorted(expected_records_without_priority)

    # (2) dataset of dicts where values are user-defined class instances
    dataset, sampler = get_simple_dataset_and_sampler_for_fruit_class_instances

    dl = InitialTrainingDataloaderPQ(
        dataset=dataset,
        sampler=sampler,
        local_batch_size=LOCAL_BATCH_SIZE_FOR_SIMPLE_DATA,
        num_workers=num_workers,
        prefetch_factor=1,
        seed=None,
        uniform_recycling_iters=None,
        num_prev_steps=0,
        use_threading=False,
        do_train_batch_properties=False,
    )
    records = [x for _, x in enumerate(dl)]

    actual_records_without_priority = [d["fruit"] for d in records]
    expected_records_without_priority = [
        [Fruit("apple"), Fruit("banana")],
        [Fruit("coconut"), Fruit("dewberry")],
        [Fruit("elderberry"), Fruit("fig")],
    ]

    actual_flat = reduce(lambda x, y: x + y, actual_records_without_priority, [])
    expected_records_without_priority = reduce(lambda x, y: x + y, expected_records_without_priority, [])
    assert sorted(actual_flat) == sorted(expected_records_without_priority)


@pytest.mark.parametrize(
    "num_workers,prefetch_factor",
    [
        (0, 2),
    ],
)
def test_030_get_real_data_in_single_process_mode(
    initial_training_cfg: DictConfig,  # noqa: F811
    num_workers: int,
    prefetch_factor: int,
):
    """Run the priority-queue dataloader in single-process mode, with real data.  Check
    that the expecte records are delivered.

    Separate the execution in single-process-mode (num_workers=0) from the
    execution in multiprocess-mode (num_workers >= 2), otherwise, the gets
    from the priority q in the multiprocess-mode always timeout.

    In single-process mode, every one of the num_steps_in_one_epoch
    records is successfully delivered by the dataloader.

    Assume 1 node, with single gpu
        rank=0
        world_size = 1
    """

    # (1) set params ------------------------------------------------------
    #
    seed = 44
    device_batch_size = 1
    num_prev_steps = 0
    world_size = 1
    rank = 0
    is_distributed = True
    num_steps_in_one_epoch = 5
    timeout_for_get = 2

    # (2) run the dataloader-----------------------------------------------
    #
    expected_training_example_indices = [24, 1, 11, 28, 20]  # at most

    training_example_indices, _ = create_and_step_through_dataloader_with_openfold_data(
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
        timeout_for_get=timeout_for_get,
    )

    # (3) check results------------------------------------------

    # (3a) if there are any records obtained within the period of the timeout,
    # they must belong to the set of records identified by the sampler
    assert set(training_example_indices).issubset(expected_training_example_indices)

    # (3b) check that if the timeout is long, we get at least 1 record
    assert len(training_example_indices) > 0


def create_and_step_through_dataloader_with_openfold_data(
    initial_training_cfg: DictConfig,  # noqa: F811
    num_steps_in_one_epoch: int,
    seed: int,
    num_prev_steps: int,
    is_distributed: bool,
    world_size: int,
    rank: int,
    device_batch_size: int,
    prefetch_factor: int,
    num_workers: int,
    timeout_for_get: int = TIMEOUT_FOR_PQUEUE_GET_DEFAULT,
):
    """Create dataset, sampler, and priority q dataloader, for real openfold
    data, step through all records pointed to be the sampler..
    """
    ds = get_initial_training_dataset(initial_training_cfg)
    sampler = InitialTrainingSampler(
        dataset=ds,
        device_batch_size=device_batch_size,
        global_batch_size=world_size * device_batch_size,
        num_steps_in_one_epoch=num_steps_in_one_epoch,
        num_prev_steps=num_prev_steps,
        seed=seed,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    dataloader_pq = InitialTrainingDataloaderPQ(
        dataset=ds,
        sampler=sampler,
        local_batch_size=device_batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        uniform_recycling_iters=list(range(0, initial_training_cfg.model.num_recycling_iters + 1)),
        num_prev_steps=num_prev_steps,
        use_threading=False,
        timeout_for_pqueue_get=timeout_for_get,
        do_train_batch_properties=True,
    )
    records = [x for _, x in enumerate(dataloader_pq)]
    training_example_indices = [record["id"][0][1] for record in records]

    return training_example_indices, records
