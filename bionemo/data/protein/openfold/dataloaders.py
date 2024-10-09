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
from typing import Iterator, List

from nemo.utils import logging
from torch.utils.data import DataLoader

from bionemo.data.protein.openfold.datasets import FinetuningDataset, InitialTrainingDataset, ValidationDataset
from bionemo.data.protein.openfold.helpers import collate
from bionemo.data.protein.openfold.samplers import FinetuningSampler, InitialTrainingSampler, ValidationSampler
from bionemo.model.protein.openfold.utils.torch_utils import map_tensor_tree


class InitialTrainingDataloaderPT(DataLoader):
    """Dataloader for the initial training stage, PyTorch(PT) version - no custom priority queue."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        sampler: InitialTrainingSampler,
        local_batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        seed: int,
        uniform_recycling_iters: List[int],
        num_prev_steps: int,
        use_threading: bool,
        **kwargs,  # TODO: doesn't it foreshadow sampler with batch_sampler? Remove and re-check
    ) -> None:
        self.device_batch_size = local_batch_size
        self.num_prev_steps = num_prev_steps
        self.seed = seed
        self.uniform_recycling_iters = uniform_recycling_iters
        if use_threading:
            logging.warning(f"threading is not supported in {InitialTrainingDataloaderPT}")
        super(InitialTrainingDataloaderPT, self).__init__(
            dataset=dataset,
            collate_fn=collate,
            sampler=sampler,
            batch_size=local_batch_size,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            persistent_workers=bool(num_workers > 0),
        )
        self._set_train_batch_properties_fn = TrainBatchProperties(
            seed=seed,
            uniform_recycling_iters=uniform_recycling_iters,
            num_prev_steps=num_prev_steps,
        )

    def __iter__(self) -> Iterator[dict]:
        iterator = super().__iter__()
        for batch in iterator:
            yield self._set_train_batch_properties_fn(batch)


class ValidationDataloader(DataLoader):
    """Validation dataloader."""

    def __init__(
        self,
        dataset: ValidationDataset,
        sampler: ValidationSampler,
        num_workers: int,
        **kwargs,  # TODO: doesn't it foreshadow sampler with batch_sampler? Remove and re-check
    ) -> None:
        super(ValidationDataloader, self).__init__(
            dataset=dataset,
            collate_fn=collate,
            sampler=sampler,
            batch_size=1,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(4 if num_workers > 0 else None),
            persistent_workers=bool(num_workers > 0),
        )


class FinetuningDataloader(DataLoader):
    """Dataloader for the fine-tuning stage."""

    def __init__(
        self,
        dataset: FinetuningDataset,
        sampler: FinetuningSampler,
        device_batch_size: int,
        num_workers: int,
        seed: int,
        uniform_recycling_iters: List[int],
        num_prev_steps: int,
    ) -> None:
        super(FinetuningDataloader, self).__init__(
            dataset=dataset,
            collate_fn=collate,
            sampler=sampler,
            batch_size=device_batch_size,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(4 if num_workers > 0 else None),
            persistent_workers=bool(num_workers > 0),
        )
        self._set_train_batch_properties_fn = TrainBatchProperties(
            seed=seed,
            uniform_recycling_iters=uniform_recycling_iters,
            num_prev_steps=num_prev_steps,
        )

    def __iter__(self) -> Iterator[dict]:
        iterator = super().__iter__()
        for batch in iterator:
            yield self._set_train_batch_properties_fn(batch)


class TrainBatchProperties:
    """Assigns randomized global train batch properties."""

    def __init__(
        self,
        seed: int,
        uniform_recycling_iters: List[int],
        num_prev_steps: int,
    ) -> None:
        self._random_num_recycling_iters_iterator = _random_num_recycling_iters_generator(
            uniform_recycling_iters=uniform_recycling_iters,
            seed=seed,
        )
        assert num_prev_steps >= 0
        self._iteration = num_prev_steps
        self._num_recycling_iters = None
        # restore rng state by iterating through previous iterations:
        for _ in range(num_prev_steps):
            next(self._random_num_recycling_iters_iterator)

    def __call__(self, batch: dict) -> dict:
        self._iteration += 1
        self._num_recycling_iters = next(self._random_num_recycling_iters_iterator)
        assert self._num_recycling_iters is not None
        batch = map_tensor_tree(
            fn=lambda t: t[..., : self._num_recycling_iters + 1],
            tree=batch,
        )
        return batch


def _random_num_recycling_iters_generator(
    uniform_recycling_iters: List[int],
    seed: int,
) -> Iterator[int]:
    assert isinstance(uniform_recycling_iters, list)
    assert len(uniform_recycling_iters) > 0
    rng = random.Random(seed)
    while True:
        num_recycling_iters_values = uniform_recycling_iters.copy()
        rng.shuffle(num_recycling_iters_values)
        for num_recycling_iters in num_recycling_iters_values:
            yield num_recycling_iters
