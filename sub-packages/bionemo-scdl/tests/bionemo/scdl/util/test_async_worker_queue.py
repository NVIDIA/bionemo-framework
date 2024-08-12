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

import concurrent.futures
import time

import pytest

from bionemo.scdl.util.async_worker_queue import AsyncWorkQueue


def sample_function(x: int, y: int) -> int:
    return x + y


def sample_function_exception(x: int, y: int):
    raise ValueError("sample function ValueError")


def sleep_function(duration: int) -> int:
    time.sleep(duration)
    return duration


def test_async_worker_queue_submits_task_thread_with_expected_result():
    async_work_queue_thread = AsyncWorkQueue(max_workers=4, use_processes=False)
    future = async_work_queue_thread.submit_task(sample_function, 1, 2)
    result = future.result()
    assert result == 3
    assert len(async_work_queue_thread.get_completed_tasks()) == 1
    assert len(async_work_queue_thread.get_pending_tasks()) == 0


def test_async_worker_queue_submits_task_process_with_expected_result():
    async_work_queue_process = AsyncWorkQueue(max_workers=4, use_processes=True)
    future = async_work_queue_process.submit_task(sample_function, 1, 2)
    result = future.result()
    assert result == 3
    assert len(async_work_queue_process.get_completed_tasks()) == 1
    assert len(async_work_queue_process.get_pending_tasks()) == 0


def test_async_worker_queue_completes_tasks():
    async_work_queue_thread = AsyncWorkQueue(max_workers=4, use_processes=False)

    async_work_queue_thread.submit_task(sample_function, 1, 2)
    async_work_queue_thread.submit_task(sample_function, 3, 4)
    concurrent.futures.wait(async_work_queue_thread.tasks)
    completed_tasks = async_work_queue_thread.get_completed_tasks()
    assert len(completed_tasks) == 2


def test_async_worker_queue_gets_expected_results_multiple_tasks():
    async_work_queue_thread = AsyncWorkQueue(max_workers=4, use_processes=False)
    async_work_queue_thread.submit_task(sample_function, 1, 2)
    async_work_queue_thread.submit_task(sample_function, 3, 4)
    concurrent.futures.wait(async_work_queue_thread.tasks)
    results = async_work_queue_thread.get_task_results()
    assert results == [3, 7]


def test_async_worker_queue_raises_exception_in_task():
    async_work_queue_thread = AsyncWorkQueue(max_workers=4, use_processes=False)
    async_work_queue_thread.submit_task(sample_function_exception, 1, 2)
    concurrent.futures.wait(async_work_queue_thread.tasks)
    results = async_work_queue_thread.get_task_results()
    with pytest.raises(ValueError, match=r"sample function ValueError"):
        raise results[0]


def test_async_worker_queue_shutsdown_properly():
    async_work_queue_thread = AsyncWorkQueue(max_workers=4, use_processes=False)

    async_work_queue_thread.submit_task(sample_function, 1, 2)
    async_work_queue_thread.shutdown()
    with pytest.raises(RuntimeError, match=r"cannot schedule new futures after shutdown"):
        async_work_queue_thread.submit_task(sample_function, 3, 4)


def test_async_worker_queue_waits_then_produces_expected_results():
    async_work_queue_thread = AsyncWorkQueue(max_workers=4, use_processes=False)

    async_work_queue_thread.submit_task(sample_function, 1, 2)
    async_work_queue_thread.submit_task(sample_function_exception, 3, 4)
    results = async_work_queue_thread.wait()
    assert results[0] == 3
    with pytest.raises(ValueError, match=r"sample function ValueError"):
        raise results[1]


def test_async_worker_queue_sleeps_then_produces_expected_results():
    async_work_queue_thread = AsyncWorkQueue(max_workers=4, use_processes=False)

    future = async_work_queue_thread.submit_task(sleep_function, 10)
    result = future.result()
    assert result == 10
    assert len(async_work_queue_thread.get_completed_tasks()) == 1
    assert len(async_work_queue_thread.get_pending_tasks()) == 0


def test_async_worker_queue_with_processes_waits_then_produces_expected_results():
    async_work_queue_process = AsyncWorkQueue(max_workers=4, use_processes=True)
    future = async_work_queue_process.submit_task(sleep_function, 10)
    result = future.result()
    assert result == 10
    assert len(async_work_queue_process.get_completed_tasks()) == 1
    assert len(async_work_queue_process.get_pending_tasks()) == 0
