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
from collections import Counter, deque
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from nemo.utils import logging
from torch import LongTensor
from torch.utils.data import Sampler

from bionemo.data.protein.openfold.datasets import (
    FinetuningDataset,
    InitialTrainingDataset,
    ValidationDataset,
)
from bionemo.data.protein.openfold.helpers import get_seed_from_string
from bionemo.utils.logging_utils import log_with_nemo_at_level


class InitialTrainingSampler(Sampler[Tuple[int, int]]):
    """An iterator that creates and steps through a list of integer-valued training example
    identifiers.  This list defines the training sample for the rank-specific
    process creating this object.

    The training **dataset** is composed of some number num_examples_in_training_dataset,
    of (featurized sequence, structure) pairs.  For OpenFold, as of 2024-04-16,

        num_examples_in_training_dataset ~= 594k

    Here, the term 'step' refers to an increment of trainer.global_step, and
    we assume this increments with each call to
        LightningModule.training_step(self, batch, batch_idx).
    That is a 'step' corresponds to one delivery of a batch on rank 0, and
    increments in sync across all ranks.

    A call to training_step(..) is different than a call to some optimizer.step().

    The training **sample** is defined as:
        (1) Assign a unique integer to each record in training dataset.
        (2) Randomly-select a sequence of the integer-valued identifiers,
            where the length of the sequence should be

                num_example_visits_in_global_epoch = mini_batch_size * num_steps_in_one_epoch
                mini_batch_size = micro_batch_size * world_size
                global_batch_size = mini_batch_size * accumulate_grad_batches

            The global_batch_size is defined in infer_global_batch_size(..),
            and is the number training example visits that contribute
            gradients to the computation of one step in weight-space, i.e. an
            optimizer step.

            This sequence is called global_sequence_of_training_example_indices
            in the code below, and is a random sample from the dataset passed to __init__.

            This definition has the effect that the number of training
            example visits in the training sample, increases with the world_size
            (number of gpu), if num_steps_in_one_epoch is fixed.

            Suppose, num_steps_in_one_epoch = 80_000, then for num_gpu=128,
            and micro_batch_size=1,

            num_example_visits_in_global_epoch = 128 * 80_000 = 10_240_000.

        (3) Each rank-specific instance of InitialTrainingSampler is assigned
            the subsequence of global_sequence_of_training_example_indices
            starting at rank, and including every world_size-th element
            (at micro_batch_size=1).

    One training **epoch**, as specified by the variable pl.Trainer.current_epoch,
    is identified as trainer.global_step % num_steps_in_one_epoch = 0, and amounts
    to one pass through the global_sequence_of_training_example_indices, where
    each rank-specific process steps through the sequence from (3).

    Note:
        (i) The PL definition of one epoch appears to be something like
            "one epoch has passed when the trainer has stepped through number of
            mini-batches equal to the value returned by the length method in the sampler,
            which is embedded in the dataloader, for one of the rank-specific process".

            This definition is relevant to variables like
                pl.Trainer.global_step
                pl.Trainer.current_epoch
                pl.Trainer.max_epochs
                pl.Trainer.max_steps

    Refer: https://github.com/Lightning-AI/pytorch-lightning/discussions/8007
    """

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        device_batch_size: int,
        global_batch_size: int,
        num_steps_in_one_epoch: int,
        num_prev_steps: int,
        seed: int,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
    ) -> None:
        """Create some instance variables based on input parameters.

        Args:
            dataset:
                An iterable over training examples, where each example is a
                (featurized sequence, structure) pair, indexed with a unique integer.
            device_batch_size:
                number of (featurized sequence, structure) pairs for each GPU,
                for 1 step of the pl.Trainer
            global_batch_size:
                number of (featurized sequence, structure) pairs over all GPUs,
                for 1 step of the pl.Trainer
            num_steps_in_one_epoch:
                The number of mini-batches in the sample of training-example
                identifiers, that defines the "sampled training dataset".
                Each rank-specific instance of InitialTrainingSampler will
                hold a list of training-example identifiers equal to this number.
            num_prev_steps:
                The number of values of trainer.global_step (i.e. occurences of
                on_train_batch_end(..)) visited in previous instances
                of train.py used to create the checkpoint providing the initial
                condition.
            seed:
                Used to create the sequence of single-example samples from dataset
            is_distributed:
                multi-gpu, currently hard-coded to True
            rank:
                global rank, has a value in 0, 1, 2, ..., (world_size - 1)
            world_size:
                number of GPUs
        """
        num_examples_in_training_dataset: int = len(dataset.get_sampler_weights())
        log_with_nemo_at_level(
            f"""
            openfold.samplers.InitialTrainingSampler.__init__(),
            rank={rank}
            num_steps_in_one_epoch={num_steps_in_one_epoch}
            num_prev_steps={num_prev_steps}
            num_examples_in_training_dataset={num_examples_in_training_dataset}
            """,
            level=logging.INFO,
        )
        # (1) check the input parameters
        #
        self.sanity_checks_on_input(
            device_batch_size,
            is_distributed,
            rank,
            world_size,
            global_batch_size,
        )

        # (2) Create a sequence of training example indices, randomly selected
        #   shape (num_samples_in_global_epoch, 1)
        global_sequence_of_training_example_indices: LongTensor = (
            self.create_global_sequence_of_training_example_indices(
                dataset, seed, num_example_visits_in_global_epoch=(num_steps_in_one_epoch * global_batch_size)
            )
        )

        # (3) Create a seed for each training example index
        #   shape: (num_samples_in_global_epoch, 1)
        global_sequence_of_seeds: LongTensor = self.create_global_sequence_of_seeds(
            seed, num_example_visits_in_global_epoch=(num_steps_in_one_epoch * global_batch_size)
        )

        # (4) Zip the global sequences of training example indices and the seeds,
        #       and create a subsequence specific to current rank
        #
        self.rank_specific_sequence_of_training_example_indices_and_seeds: List[Tuple] = (
            self.create_rank_specific_sequence_of_training_example_indices_and_seeds(
                global_sequence_of_training_example_indices,
                global_sequence_of_seeds,
                device_batch_size,
                num_steps_in_one_epoch,
                is_distributed,
                rank,
                world_size,
                num_prev_steps,
            )
        )
        num_training_example_visits_in_one_epoch_for_this_rank: int = len(
            self.rank_specific_sequence_of_training_example_indices_and_seeds
        )

        log_with_nemo_at_level(
            f"""
            InitialTrainingSampler().__init__, summary at the end
            rank={rank}
            num_steps_in_one_epoch={num_steps_in_one_epoch}
            num_prev_steps={num_prev_steps}
            num_training_example_visits_assigned_to_this_rank={num_training_example_visits_in_one_epoch_for_this_rank}
            num_examples_in_training_dataset={num_examples_in_training_dataset}
            """,
            level=logging.INFO,
        )

    def sanity_checks_on_input(
        self,
        dev_batch_size: int,
        is_distributed: bool,
        rank: int,
        world_size: int,
        global_batch_size: int,
    ):
        """Run simple checks on input"""

        if dev_batch_size != 1:
            raise NotImplementedError(f"have not yet considered the logic for dev_batch_size={dev_batch_size}")

        if is_distributed:
            assert rank is not None
            assert world_size is not None
            assert global_batch_size % dev_batch_size == 0
            assert global_batch_size % world_size == 0

    def create_global_sequence_of_training_example_indices(
        self,
        dataset: InitialTrainingDataset,
        seed: int,
        num_example_visits_in_global_epoch: int,
    ):
        """Create a 1-axis tensor containing traning-example identifiers"""
        index_generator = torch.Generator()
        index_generator.manual_seed(seed)
        return torch.multinomial(
            input=dataset.get_sampler_weights(),
            num_samples=num_example_visits_in_global_epoch,
            replacement=True,
            generator=index_generator,
        )

    def create_global_sequence_of_seeds(self, seed: int, num_example_visits_in_global_epoch: int):
        seed_generator = torch.Generator()
        seed_generator.manual_seed(seed)
        return torch.randint(
            low=0,
            high=2**63 - 1,
            size=[num_example_visits_in_global_epoch],
            generator=seed_generator,
        )

    def create_rank_specific_sequence_of_training_example_indices_and_seeds(
        self,
        global_sequence_of_training_example_indices: LongTensor,
        global_sequence_of_seeds: LongTensor,
        device_batch_size: int,
        num_steps_in_one_epoch: int,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
        num_prev_steps: int,
    ):
        """Determines the sub-sequence of global_sequence_of_training_example_indices
        to use as the training sample for this rank.

        This method accounts for the fact that previous training runs have
        completed num_prev_steps, where each step has device_batch_size learning
        examples for this rank.

        BrToDo: revisit logic for device_batch_size != 1

        Args:
            global_sequence_of_training_example_indices:
                A 1-axis tensor with integer-valued identifiers for elements
                of self.dataset

            global_sequence_of_seeds:
                A 1-axis tensor with integer-valued RNG seeds, assigned to
                each visit of an example fro self.dataset

            device_batch_size: defined_in __init__
            num_steps_in_one_epoch: defined in __init__
            is_distributed: defined in __init__
            rank: defined in __init__
            world_size: defined in __init__
            num_prev_steps: defined in __init__

        Returns:
            rank_specific_index_seed_pairs is obtained by zipping list versions of
            global_sequence_of_training_example_indices and global_sequence_of_seeds,
            and then creating a sub-sequence starting at rank and stepping every
            world_size positions.
        """
        # (0) sanity check on inputs
        assert global_sequence_of_training_example_indices.size() == global_sequence_of_seeds.size()

        # (1) create sub-sequence for this rank
        rank_specific_index_seed_pairs = list(
            zip(global_sequence_of_training_example_indices.tolist(), global_sequence_of_seeds.tolist())
        )
        if is_distributed:
            rank_specific_index_seed_pairs = rank_specific_index_seed_pairs[rank::world_size]

        num_example_visits_for_this_rank_in_one_epoch = num_steps_in_one_epoch * device_batch_size
        assert len(rank_specific_index_seed_pairs) == num_example_visits_for_this_rank_in_one_epoch

        # (2) Create cyclic permuation of rank_specific_index_seed_pairs
        #   - previous runs have visited num_prev_steps positions in rank-specific sequence
        offset = num_prev_steps * device_batch_size % num_example_visits_for_this_rank_in_one_epoch
        rank_specific_index_seed_pairs = (
            rank_specific_index_seed_pairs[offset:] + rank_specific_index_seed_pairs[0:offset]
        )

        return rank_specific_index_seed_pairs

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        assert hasattr(self, "rank_specific_sequence_of_training_example_indices_and_seeds")
        yield from self.rank_specific_sequence_of_training_example_indices_and_seeds
        # del self.index_seed_pairs # aCHCK

    def __len__(self) -> int:
        assert hasattr(self, "rank_specific_sequence_of_training_example_indices_and_seeds")
        return len(self.rank_specific_sequence_of_training_example_indices_and_seeds)


class ValidationSampler(Sampler[Tuple[int, int]]):
    """Sampler for validation dataset."""

    def __init__(
        self,
        dataset: ValidationDataset,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
    ) -> None:
        dataset_length = len(dataset)  # 180
        if is_distributed:
            assert rank is not None
            assert world_size is not None
            epoch_length = math.ceil(dataset_length / world_size)
        else:
            epoch_length = dataset_length
        # BrToDo: why is seed a function of identifier for ValidationSampler,
        #   but not for InitialTrainingSampler or FineTuningSampler
        seeds = [get_seed_from_string(pdb_chain_id) for pdb_chain_id in dataset.pdb_chain_ids]
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self._dataset_length = dataset_length
        self._epoch_length = epoch_length
        self._seeds = seeds
        indices = list(range(self._dataset_length))
        self.indices = indices[self.rank :: self.world_size]
        self.seeds = [self._seeds[index] for index in indices]

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        yield from zip(self.indices, self.seeds)

    def __len__(self) -> int:
        return len(self.indices)  # self._epoch_length


class FinetuningSampler(Sampler[Tuple[str, int, int]]):
    """Sampler for the fine-tuning dataset."""

    def __init__(
        self,
        dataset: FinetuningDataset,
        dataset_weights: Dict[str, float],
        device_batch_size: int,
        global_batch_size: int,
        num_steps_in_one_epoch: int,
        seed: int,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
        num_prev_steps: int,
    ) -> None:
        if is_distributed:
            assert rank is not None
            assert world_size is not None
            assert global_batch_size % device_batch_size == 0
            assert global_batch_size % world_size == 0
        weights = dataset.get_sampler_weights()
        num_samples_in_global_epoch = num_steps_in_one_epoch * global_batch_size
        num_samples_in_device_epoch = num_steps_in_one_epoch * device_batch_size

        # Sample datasets:
        dataset_rng = random.Random(seed)
        assert sum(dataset_weights.values()) > 0
        dataset_choices = dataset_rng.choices(
            population=list(dataset_weights.keys()),
            weights=list(dataset_weights.values()),
            k=num_samples_in_global_epoch,
        )

        # Sample indices:
        dataset_choices_cnt = Counter(dataset_choices)
        index_generators = {k: torch.Generator() for k in weights.keys()}
        for index_generator in index_generators.values():
            index_generator.manual_seed(seed)
        indices = {
            k: torch.multinomial(
                input=weights[k],
                num_samples=dataset_choices_cnt[k],
                replacement=True,
                generator=index_generators[k],
            )
            for k in index_generators.keys()
            if dataset_choices_cnt[k] > 0
        }
        indices = {k: deque(tensor.tolist()) for k, tensor in indices.items()}
        indices = [indices[dataset_choice].popleft() for dataset_choice in dataset_choices]

        # Sample seeds:
        seed_generator = torch.Generator()
        seed_generator.manual_seed(seed)
        seeds = torch.randint(
            low=0,
            high=2**63 - 1,
            size=[num_samples_in_global_epoch],
            generator=seed_generator,
        )
        seeds = seeds.tolist()

        # Create (dataset, index, seed) triples:
        assert len(dataset_choices) == len(indices) == len(seeds)
        dataset_index_seed_triples = list(zip(dataset_choices, indices, seeds))
        if is_distributed:
            dataset_index_seed_triples = dataset_index_seed_triples[rank::world_size]
        assert len(dataset_index_seed_triples) == num_samples_in_device_epoch

        # Move forward by skipping previous iterations:
        offset = num_prev_steps * device_batch_size % num_samples_in_device_epoch
        self.dataset_index_seed_triples = dataset_index_seed_triples[offset:] + dataset_index_seed_triples[0:offset]

    def __iter__(self) -> Iterator[Tuple[str, int, int]]:
        assert hasattr(self, "dataset_index_seed_triples")
        yield from self.dataset_index_seed_triples

    def __len__(self) -> int:
        assert hasattr(self, "dataset_index_seed_triples")
        return len(self.dataset_index_seed_triples)
