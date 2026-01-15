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


# conftest.py
import gc
import os
import random
import signal
import time

import numpy as np
import pytest
import torch


def get_device_and_memory_allocated() -> str:
    """Get the current device index, name, and memory usage."""
    current_device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(current_device_index)
    message = f"""
        current device index: {current_device_index}
        current device uuid: {props.uuid}
        current device name: {props.name}
        memory, total on device: {torch.cuda.mem_get_info()[1] / 1024**3:.3f} GB
        memory, available on device: {torch.cuda.mem_get_info()[0] / 1024**3:.3f} GB
        memory allocated for tensors etc: {torch.cuda.memory_allocated() / 1024**3:.3f} GB
        max memory reserved for tensors etc: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB
        """
    return message


def pytest_sessionstart(session):
    """Called at the start of the test session."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(
            f"""
            sub-packages/bionemo-evo2/tests/bionemoe/evo2: Starting test session
            {get_device_and_memory_allocated()}
            """
        )


def pytest_sessionfinish(session, exitstatus):
    """Called at the end of the test session."""
    if torch.cuda.is_available():
        print(
            f"""
            sub-packages/bionemo-evo2/tests/bionemoe/evo2: Test session complete
            {get_device_and_memory_allocated()}
            """
        )


def _cleanup_child_processes():
    """Kill any orphaned child processes that might be holding GPU memory.

    This is particularly important for tests that spawn subprocesses via torchrun.
    """
    import subprocess

    current_pid = os.getpid()
    try:
        # Find child processes
        result = subprocess.run(
            ["pgrep", "-P", str(current_pid)], check=False, capture_output=True, text=True, timeout=5
        )
        child_pids = result.stdout.strip().split("\n")
        for pid_str in child_pids:
            if pid_str:
                try:
                    pid = int(pid_str)
                    os.kill(pid, signal.SIGTERM)
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _thorough_gpu_cleanup():
    """Perform thorough GPU memory cleanup."""
    if not torch.cuda.is_available():
        return

    # Synchronize all CUDA streams to ensure all operations are complete
    torch.cuda.synchronize()

    # Clear all cached memory
    torch.cuda.empty_cache()

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Run garbage collection multiple times to ensure all objects are collected
    for _ in range(3):
        gc.collect()

    # Another sync and cache clear after gc
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Small sleep to allow GPU memory to be fully released
    time.sleep(0.1)


def _reset_random_seeds():
    """Reset random seeds to ensure reproducibility across tests.

    Some tests may modify global random state, which can affect subsequent tests
    that depend on random splitting (like dataset preprocessing).
    """
    # Reset Python's random module
    random.seed(None)

    # Reset NumPy's random state (intentionally using legacy API to reset global state)
    np.random.seed(None)  # noqa: NPY002

    # Reset PyTorch's random state
    torch.seed()
    if torch.cuda.is_available():
        torch.cuda.seed_all()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up GPU memory and reset state after each test."""
    # Reset random seeds before the test to ensure reproducibility
    _reset_random_seeds()

    yield

    # After the test, perform thorough cleanup
    _thorough_gpu_cleanup()

    # Clean up any orphaned child processes (important for subprocess tests)
    _cleanup_child_processes()

    # Final garbage collection
    gc.collect()


def pytest_addoption(parser: pytest.Parser):
    """Pytest configuration for bionemo.evo2.run tests. Adds custom command line options for dataset paths."""
    parser.addoption("--dataset-dir", action="store", default=None, help="Path to preprocessed dataset directory")
    parser.addoption("--training-config", action="store", default=None, help="Path to training data config YAML file")
