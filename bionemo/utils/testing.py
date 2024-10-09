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


# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import time
from contextlib import contextmanager
from typing import Any, Optional, Set


def get_size(obj: Any, seen: Optional[Set[int]] = None) -> int:
    """
    Recursively find the total size of an object and all its attributes.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    if isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_size(item, seen) for item in obj)
    elif isinstance(obj, dict):
        size += sum(get_size(key, seen) + get_size(value, seen) for key, value in obj.items())
    return size


def human_readable_size(size_in_bytes: int) -> str:
    """
    Convert size in bytes to a human-readable format.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

    return f"{size_in_bytes*1024.0:.2f} {unit}"


def pretty_size(obj: object) -> str:
    """
    Return a human-readable size of an object.
    """
    return human_readable_size(get_size(obj))


@contextmanager
def timeit(message="", logging_func=print) -> float:
    """
    This is a context manager that can be used to time a block of code.
    useful for benchmarking purposes.

    Args:
        message (str): message to be printed along with the time taken. None skips printing.
        logging_func (callable): function to be used for logging.

    Example:
    >>> with timeit("Time taken: "):
            pass
    """
    start_time = time.monotonic_ns() * 1e-9
    yield
    end_time = time.monotonic_ns() * 1e-9
    elapsed_time = end_time - start_time

    if logging_func is not None:
        if elapsed_time < 1e-12:
            time_str = f"{elapsed_time * 1e12:.3f} [picoseconds]"
        elif elapsed_time < 1e-9:
            time_str = f"{elapsed_time * 1e9:.3f} [nanoseconds]"
        elif elapsed_time < 1e-6:
            time_str = f"{elapsed_time * 1e6:.3f} [microseconds]"
        elif elapsed_time < 1e-3:
            time_str = f"{elapsed_time * 1e3:.3f} [milliseconds]"
        else:
            time_str = f"{elapsed_time:.3f} [seconds]"

        logging_func(f"{message}{time_str}")

    return elapsed_time
