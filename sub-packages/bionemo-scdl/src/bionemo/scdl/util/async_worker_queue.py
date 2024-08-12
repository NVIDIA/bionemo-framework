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
import threading
from typing import Any, Callable, List, Sequence, Union


__all__: Sequence[str] = ("AsyncWorkQueue",)


class AsyncWorkQueue:
    """Implements an asynchronous queue."""

    def __init__(self, max_workers: int = 5, use_processes: bool = False) -> None:
        """Initialize the AsyncWorkQueue.

        Args:
            max_workers: The maximum number of worker threads or processes.
            use_processes: If True, use ProcessPoolExecutor; otherwise, use ThreadPoolExecutor.
        """
        self.use_processes = use_processes
        if use_processes:
            self.executor: Union[concurrent.futures.ThreadPoolExecutor, concurrent.futures.ProcessPoolExecutor] = (
                concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            )
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.tasks: List[concurrent.futures.Future] = []

    def submit_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """Submit a task to the work queue.

        Args:
            func: The function to be executed asynchronously.
            args: Positional arguments to pass to the function.
            kwargs: Keyword arguments to pass to the function.
            A Future object representing the execution of the function.

        Returns:
            Future: placeholder for the asynchronous operation.
        """
        with self.lock:
            future = self.executor.submit(func, *args, **kwargs)
            self.tasks.append(future)
            return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor and wait for the tasks to complete.

        Args:
            wait: If True, wait for all tasks to complete before shutting down.
        """
        self.executor.shutdown(wait=wait)

    def get_completed_tasks(self) -> List[concurrent.futures.Future]:
        """Get the list of completed tasks.

        Returns:
            A list of Future objects that are completed.
        """
        with self.lock:
            completed_tasks = [task for task in self.tasks if task.done()]
            return completed_tasks

    def get_pending_tasks(self) -> List[concurrent.futures.Future]:
        """Get the list of pending tasks.

        Returns:
            A list of Future objects that are not yet completed.
        """
        with self.lock:
            pending_tasks = [task for task in self.tasks if not task.done()]
            return pending_tasks

    def get_task_results(self) -> List[Any]:
        """Get the results of all completed tasks.

        Returns:
            A list of results from the completed tasks.

        Raises:
            Exception: This would be expected if the task fails to complete or
            if is cancelled.
        """
        completed_tasks = self.get_completed_tasks()
        results = []
        for task in completed_tasks:
            try:
                results.append(task.result())
            except Exception as e:
                results.append(e)
        return results

    def wait(self) -> List[Any]:
        """Wait for all submitted tasks to complete and return their results.

        Returns:
            A list of results from all completed tasks.
        """
        # Wait for all tasks to complete
        concurrent.futures.wait(self.tasks)

        # Collect results from all tasks
        results = []
        for task in self.tasks:
            try:
                results.append(task.result())
            except Exception as e:
                results.append(e)

        return results
