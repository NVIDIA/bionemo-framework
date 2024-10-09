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


import logging
import queue
import threading
import traceback
from multiprocessing.managers import SyncManager
from typing import Iterator, List

import torch

from bionemo.data.protein.openfold.dataloaders import TrainBatchProperties
from bionemo.data.protein.openfold.datasets import InitialTrainingDataset
from bionemo.data.protein.openfold.helpers import collate
from bionemo.data.protein.openfold.samplers import InitialTrainingSampler
from bionemo.utils.logging_utils import log_with_nemo_at_level


# ----------------------------------------------------------------------------
# TIMEOUT_FOR_PQUEUE_GET_DEFAULT
#
#   - We set a long default time, to lower the risk of timeout for a long training job.
#   - Most users should adopt this default setting, and should only override
#   the default value, for testing.  Override values can be set in
#   examples/protein/openfold/conf/openfold_initial_training.yaml.
#   - The featurization of each openfold training example varies by training example,
#   and by machine / os platform.
#   - For the openfold sample dataset downloaded as part of CI unit testing,
#   and with a standard desktop, the featurization time ranges from below 1 second to a couple minutes.
TIMEOUT_FOR_PQUEUE_GET_DEFAULT = 1200  # seconds, 1200 seconds is 20min
LOG_OUTPUT_CUT = 64


class InitialTrainingDataloaderPQ:
    """Dataloader for the initial training stage with non-blocking priority queue."""

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
        timeout_for_pqueue_get: int = TIMEOUT_FOR_PQUEUE_GET_DEFAULT,
        do_train_batch_properties: bool = True,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.local_batch_size = local_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.threading_enabled = use_threading
        self.timeout_for_pqueue_get = timeout_for_pqueue_get
        self.queue_maxsize = self.num_workers * self.prefetch_factor

        self._set_train_batch_properties_fn = (
            TrainBatchProperties(
                seed=seed,
                uniform_recycling_iters=uniform_recycling_iters,
                num_prev_steps=num_prev_steps,
            )
            if do_train_batch_properties
            else lambda x: x
        )

        # [optim-hub] since DAP is not implemented, dist_group_size, dist_rank_in_group, and
        # dist_producer_rank will always be None. If DAP continues to be out of scope, we can optimise the
        # code and remove these arguments

        # if torch.distributed.is_initialized():
        #     self.dist_rank = int(torch.distributed.get_rank())
        #     self.dist_world_size = int(torch.distributed.get_world_size())
        # else:
        #     self.dist_rank = None
        #     self.dist_world_size = None
        # self.dist_group_size = None
        # self.dist_rank_in_group = None
        # self.dist_group = None

    def _start_manager(self) -> None:
        self._manager = ManagerPQ()
        self._manager.start()

    def _start_multiprocessing(self) -> None:
        # create queues:
        self._index_queue = torch.multiprocessing.Queue(maxsize=self.queue_maxsize)
        self._sample_pqueue = self._manager.PriorityQueue(maxsize=self.queue_maxsize)
        self._batch_pqueue = self._manager.PriorityQueue(maxsize=self.queue_maxsize)

        # create index process:
        self._index_process = torch.multiprocessing.Process(
            target=_index_worker,
            args=(
                self.sampler,
                self._index_queue,
            ),
        )
        self._index_process.daemon = True
        self._index_process.start()

        # create sample processes:
        self._sample_processes = []
        for w in range(self.num_workers):
            sample_process = torch.multiprocessing.Process(
                target=_sample_worker,
                args=(
                    self.dataset,
                    self._index_queue,
                    self._sample_pqueue,
                ),
            )
            sample_process.daemon = True
            sample_process.start()
            self._sample_processes.append(sample_process)

        # create batch process:
        self._batch_process = torch.multiprocessing.Process(
            target=_batch_worker,
            args=(
                self._sample_pqueue,
                self._batch_pqueue,
                self.local_batch_size,
            ),
        )
        self._batch_process.daemon = True
        self._batch_process.start()

    def _start_threading(self, input_queue) -> None:
        if self.threading_enabled:
            self._batch_tqueue = queue.Queue(maxsize=1)
            self._batch_thread = threading.Thread(
                target=_batch_thread,
                args=(
                    input_queue,
                    self._batch_tqueue,
                ),
            )
            self._batch_thread.daemon = True
            self._batch_thread.start()

    def _close_manager(self) -> None:
        if hasattr(self, "_manager"):
            del self._manager

    def _close_multiprocessing(self) -> None:
        log_with_nemo_at_level(
            """InitialTrainingDataloaderPQ._close_multiprocessing, begin""",
            level=logging.WARNING,
        )
        if hasattr(self, "_batch_process"):
            self._batch_process.kill()
            del self._batch_process

        if hasattr(self, "_sample_processes"):
            for sample_process in reversed(self._sample_processes):
                sample_process.kill()
            del self._sample_processes

        if hasattr(self, "_index_process"):
            self._index_process.kill()
            del self._index_process

        if hasattr(self, "_batch_pqueue"):
            del self._batch_pqueue

        if hasattr(self, "_sample_pqueue"):
            del self._sample_pqueue

        if hasattr(self, "_index_queue"):
            del self._index_queue

        log_with_nemo_at_level(
            """InitialTrainingDataloaderPQ._close_multiprocessing, end""",
            level=logging.WARNING,
        )

    def _close_threading(self) -> None:
        if hasattr(self, "_batch_thread"):
            del self._batch_thread

        if hasattr(self, "_batch_tqueue"):
            del self._batch_tqueue

    def _close_threading(self) -> None:
        if hasattr(self, "_batch_thread"):
            del self._batch_thread

        if hasattr(self, "_batch_tqueue"):
            del self._batch_tqueue

    def _multiprocessing_iter(self) -> Iterator[dict]:
        self._start_manager()
        self._start_multiprocessing()
        self._start_threading(self._batch_pqueue)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i_dataloader in range(num_dataloader_iters):
            try:
                if self.threading_enabled:
                    priority, batch = self._batch_tqueue.get(timeout=self.timeout_for_pqueue_get)
                else:
                    priority, batch = self._batch_pqueue.get(timeout=self.timeout_for_pqueue_get)

                batch["__priority__"] = priority
                yield batch

            except Exception as e:
                message = f"""
                    InitialTrainingDataloaderPQ._multiprocessing_iter, get fails for batch queue
                    self.threading_enabled={self.threading_enabled}
                    self.timeout_for_pqueue_get={self.timeout_for_pqueue_get}
                    i_dataloader={i_dataloader}
                    e={traceback.format_exception(e)}
                """
                log_with_nemo_at_level(
                    message,
                    level=logging.WARNING,
                )

        self._close_threading()
        self._close_multiprocessing()
        self._close_manager()

    def _synchronous_iter(self) -> Iterator[dict]:
        sampler_iterator = iter(self.sampler)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            samples = []
            for j in range(self.local_batch_size):
                index = next(sampler_iterator)
                sample = self.dataset[index]
                assert isinstance(sample, dict)
                samples.append(sample)
            batch = collate(samples)
            batch["__priority__"] = i
            yield batch

    def __iter__(self) -> Iterator[dict]:
        if self.num_workers > 0:
            iterator = self._multiprocessing_iter()
        elif self.num_workers == 0:
            iterator = self._synchronous_iter()
        for i, batch in enumerate(iterator):
            yield self._set_train_batch_properties_fn(batch)

    def __del__(self) -> None:
        log_with_nemo_at_level(
            """InitialTrainingDataloaderPQ.__del__, begin""",
            level=logging.WARNING,
        )

        self._close_threading()
        self._close_multiprocessing()
        self._close_manager()

        log_with_nemo_at_level(
            """InitialTrainingDataloaderPQ.__del__, end""",
            level=logging.WARNING,
        )


def _index_worker(sampler, index_queue) -> None:
    for priority, index in enumerate(sampler):
        index_queue.put((priority, index))


def _sample_worker(dataset, index_queue, sample_pqueue) -> None:
    while True:
        # try to get
        try:
            index_queue_out = index_queue.get()
        except Exception as e:
            message = f"""
                _sample_worker(), index_queue.get() raised exception
                index_queue_out={index_queue_out}
                e={traceback.format_exception(e)}
                """
            log_with_nemo_at_level(message, level=logging.WARNING)

        # try to put
        try:
            priority, index = index_queue_out
            sample = dataset[index]
            put_value = (priority, sample)
            sample_pqueue.put(put_value)

        except (EOFError, BrokenPipeError) as e:
            short_message = """_sample_worker(), sample_pqueue.put() raised EOFError or BrokenPipeError
                This exception is likely the result of the training job reaching max_steps.
                Application is expected to hang until the slurm job time limit.
                The resume slurm job will start up, and quickly exit, because max_steps is reached."""
            message = f"""
                {short_message}
                index_queue_out={index_queue_out}
                put_value={str(put_value)[0:LOG_OUTPUT_CUT]}
                e={traceback.format_exception(e)}
                """
            log_with_nemo_at_level(message, level=logging.DEBUG)
            return

        except Exception as e:
            message = f"""
                _sample_worker(), sample_pqueue.put() raised exception.
                This exception could be the result of the training job reaching max_steps.
                index_queue_out={index_queue_out}
                put_value={str(put_value)[0:LOG_OUTPUT_CUT]}
                e={traceback.format_exception(e)}
                """
            log_with_nemo_at_level(message, level=logging.WARNING)


def _batch_worker(sample_pqueue, batch_pqueue, local_batch_size) -> None:
    while True:
        samples = []
        priorities = []
        for j in range(local_batch_size):
            # try to get
            try:
                priority, sample = sample_pqueue.get()
                samples.append(sample)
                priorities.append(priority)

            except (EOFError, BrokenPipeError) as e:
                short_message = """_batch_worker(), sample_pqueue.get() raised EOFError or BrokenPipeError
                    This exception is likely the result of the training job reaching max_steps.
                    Application is expected to hang until the slurm job time limit.
                    The resume slurm job will start up, and quickly exit, because max_steps is reached."""
                message = f"""
                    {short_message}
                    priority={priority}
                    sample={str(sample)[0:LOG_OUTPUT_CUT]}
                    e={traceback.format_exception(e)}
                    """
                log_with_nemo_at_level(message, level=logging.WARNING)
                return

            except Exception as e:
                message = f"""
                    _batch_worker(), sample_pqueue.get() raises exception
                    priority={priority}
                    sample={str(sample)[0:LOG_OUTPUT_CUT]}
                    e={traceback.format_exception(e)}
                    """
                log_with_nemo_at_level(message, level=logging.WARNING)

        # try to put
        try:
            priority_for_batch = min(priorities)
            batch = collate(samples)
            put_value_for_batch = (priority_for_batch, batch)
            batch_pqueue.put(put_value_for_batch)

        except (EOFError, BrokenPipeError) as e:
            short_message = """_batch_worker(), batch_pqueue.put() raised EOFError or BrokenPipeError
                This exception is likely the result of the training job reaching max_steps
                Application is expected to hang until the slurm job time limit.
                The resume slurm job will start up, and quickly exit, because max_steps is reached."""
            message = f"""
                {short_message}
                put_value_for_batch={str(put_value_for_batch)[LOG_OUTPUT_CUT]}
                e={traceback.format_exception(e)}
                """
            log_with_nemo_at_level(message, level=logging.WARNING)
            return

        except Exception as e:
            message = f"""
                _batch_worker(), batch_pqueue.put() raises exception.
                This exception could be the result of the training job reaching max_steps.
                put_value_for_batch={str(put_value_for_batch)[LOG_OUTPUT_CUT]}
                e={traceback.format_exception(e)}
                """
            log_with_nemo_at_level(message, level=logging.WARNING)


def _batch_thread(input_queue, batch_tqueue) -> None:
    while True:
        obj = input_queue.get()
        batch_tqueue.put(obj)


# def _batch_deepcopy(batch: dict) -> dict:
#     output = {}
#     for key, value in batch.items():
#         if isinstance(value, torch.Tensor):
#             output[key] = value.clone()
#         else:
#             output[key] = deepcopy(value)
#     return output


class ManagerPQ(SyncManager):
    pass


ManagerPQ.register("PriorityQueue", queue.PriorityQueue)
