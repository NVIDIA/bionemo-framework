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
import copy
import gc
import os
import shlex
import socket
import subprocess
from pathlib import Path

import pytest
import torch

from bionemo.core.data.load import load as bionemo_load
from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH, DEFAULT_HF_TOKENIZER_MODEL_PATH_512
from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge


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


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up GPU memory, reset state, and restore env vars after each test.

    Megatron Core's LanguageModule._set_attention_backend() mutates os.environ
    (e.g. NVTE_FUSED_ATTN, NVTE_FLASH_ATTN) when models are constructed in-process.
    Capturing and restoring the full environment prevents cross-test pollution.
    """
    saved_environ = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved_environ)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    try:
        nemo2_ckpt_path = bionemo_load("evo2/1b-8k-bf16:1.0")
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
        model_size="evo2_1b_base",
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


@pytest.fixture(scope="session")
def mbridge_eden_checkpoint(tmp_path_factory) -> Path:
    """Session-scoped MBridge checkpoint for a tiny Eden (Llama) model.

    Creates an Eden mbridge checkpoint by training a tiny model (2 layers, seq_length=64)
    for 1 step with mock data. Shared across test files that need an Eden checkpoint.

    Returns:
        Path to the MBridge checkpoint directory (parent of iter_0000001).
    """

    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    tmp_dir = tmp_path_factory.mktemp("eden_ckpt_session")
    run_dir = tmp_dir / "eden_train"
    run_dir.mkdir(parents=True, exist_ok=True)
    port = _find_free_port()

    cmd = (
        f"torchrun --nproc-per-node 1 --no-python --master_port {port} "
        f"train_evo2 "
        f"--hf-tokenizer-model-path {DEFAULT_HF_TOKENIZER_MODEL_PATH} "
        "--model-size eden_7b --num-layers 2 "
        "--max-steps 1 --eval-interval 1 --eval-iters 1 "
        f"--mock-data --result-dir {run_dir} "
        "--micro-batch-size 4 --global-batch-size 4 --seq-length 64 "
        "--tensor-model-parallel 1 --pipeline-model-parallel 1 --context-parallel 1 "
        "--mixed-precision-recipe bf16_mixed "
        "--no-activation-checkpointing "
        "--decay-steps 1000 --warmup-steps 10 "
        "--log-interval 1 --seed 41 --dataset-seed 33"
    )

    env = copy.deepcopy(os.environ)
    result = subprocess.run(shlex.split(cmd), check=False, capture_output=True, text=True, cwd=run_dir, env=env)
    if result.returncode != 0:
        print(f"Eden checkpoint creation STDOUT:\n{result.stdout}")
        print(f"Eden checkpoint creation STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"Eden checkpoint creation failed: {result.stderr[-2000:]}"

    ckpt_dir = run_dir / "evo2" / "checkpoints"
    iter_dir = ckpt_dir / "iter_0000001"
    assert iter_dir.exists(), f"Eden checkpoint not found at {iter_dir}"
    return iter_dir
