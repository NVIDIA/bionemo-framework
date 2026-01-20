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
from pathlib import Path

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

    Note: Skips cleanup when distributed is initialized, as killing processes
    could interfere with NCCL's internal state and cause hangs.
    """
    # Don't kill child processes if distributed is active - NCCL might have
    # internal processes that should not be killed
    if torch.distributed.is_initialized():
        return

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
    """Perform thorough GPU memory cleanup.

    Note: This is intentionally gentle when torch.distributed is initialized,
    as aggressive cleanup can interfere with NCCL state and cause hangs in
    subsequent tests that reinitialize distributed.
    """
    if not torch.cuda.is_available():
        return

    # If distributed is still initialized, skip aggressive cleanup to avoid
    # interfering with NCCL's internal state. The test fixture should handle
    # distributed cleanup before this runs, but if it hasn't, we don't want
    # to cause issues.
    if torch.distributed.is_initialized():
        # Just do minimal cleanup - gc only
        gc.collect()
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

    Note: Skips CUDA seed reset when distributed is initialized, as this can
    interfere with Megatron's tensor parallel RNG tracker.
    """
    # Reset Python's random module
    random.seed(None)

    # Reset NumPy's random state (intentionally using legacy API to reset global state)
    np.random.seed(None)  # noqa: NPY002

    # Reset PyTorch's random state
    torch.seed()

    # Only reset CUDA seeds if distributed is not initialized, as the distributed
    # tests manage their own RNG state via model_parallel_cuda_manual_seed
    if torch.cuda.is_available() and not torch.distributed.is_initialized():
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


# =============================================================================
# Session-scoped checkpoint fixtures for sharing across test files
# =============================================================================


@pytest.fixture(scope="session")
def mbridge_checkpoint_1b_8k_bf16(tmp_path_factory) -> Path:
    """Session-scoped MBridge checkpoint for the 1b-8k-bf16 model.

    This fixture converts the NeMo2 checkpoint to MBridge format once per test session,
    allowing it to be shared across multiple test files (test_infer.py, test_predict.py, etc.).

    Returns:
        Path to the MBridge checkpoint iteration directory (e.g., .../iter_0000001)
    """
    from bionemo.core.data.load import load
    from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
    from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge

    try:
        nemo2_ckpt_path = load("evo2/1b-8k-bf16:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            pytest.skip(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    output_dir = tmp_path_factory.mktemp("mbridge_ckpt_1b_8k_bf16_session")
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_ckpt_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=output_dir / "evo2_1b_mbridge",
        model_size="1b",
        seq_length=8192,
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.fixture(scope="module")
def mbridge_checkpoint_path(mbridge_checkpoint_1b_8k_bf16) -> Path:
    """Module-scoped alias for the session-scoped 1b-8k-bf16 checkpoint.

    This provides backward compatibility for tests that use the name 'mbridge_checkpoint_path'.
    The actual checkpoint is shared at session scope via mbridge_checkpoint_1b_8k_bf16.

    Returns:
        Path to the MBridge checkpoint iteration directory
    """
    return mbridge_checkpoint_1b_8k_bf16
