# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Test that parameter distributions are identical with and without meta device initialization.

These tests verify that when using meta device initialization (creating the model on meta device, then calling
`to_empty` and `_init_weights`), the resulting parameter distributions (mean and std) match those from normal
initialization. This is important because we previously observed differences in convergence between meta-device-init and
non-meta-device-init training, which suggested that the initialization was not being applied correctly after `to_empty`.
By explicitly calling `_init_weights` after `to_empty`, we ensure that parameters are properly initialized, leading to
consistent training behavior regardless of whether meta device initialization is used.
"""

import os
import subprocess

import pytest
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, set_seed

from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def test_meta_device_init():
    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D").to_dict())

    set_seed(42)
    with torch.device("meta"):
        model_meta_init = NVEsmForMaskedLM(config)

    # Assert parameters are actually on the meta device
    for name, parameter in model_meta_init.named_parameters():
        assert parameter.device == torch.device("meta"), f"Parameter {name} is not on the meta device"

    model_meta_init.to_empty(device="cuda")
    model_meta_init.apply(model_meta_init._init_weights)

    # Assert parameters are actually on the cuda device after to_empty
    for name, parameter in model_meta_init.named_parameters():
        assert str(parameter.device).startswith("cuda"), f"Parameter {name} is not on the cuda device"

    set_seed(42)
    model_normal_init = NVEsmForMaskedLM(config)
    model_normal_init.to("cuda")

    state_dict_meta_init = model_meta_init.state_dict()
    state_dict_normal_init = model_normal_init.state_dict()

    for key in state_dict_meta_init.keys():
        meta_tensor = state_dict_meta_init[key]
        normal_tensor = state_dict_normal_init[key]
        # Skip non-numeric tensors (e.g., Byte/uint8 tensors like _extra_state)
        if meta_tensor.dtype not in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.complex64,
            torch.complex128,
        ):
            continue
        torch.testing.assert_close(
            normal_tensor.mean(),
            meta_tensor.mean(),
            atol=1e-3,
            rtol=1e-4,
            msg=lambda x: f"Mean mismatch for parameter {key}: {x}",
        )
        torch.testing.assert_close(
            normal_tensor.std(),
            meta_tensor.std(),
            atol=1e-3,
            rtol=1e-4,
            msg=lambda x: f"Std mismatch for parameter {key}: {x}",
        )


@pytest.mark.parametrize("num_gpus", [1, pytest.param(2, marks=requires_multi_gpu)])
def test_meta_device_init_after_fully_shard(num_gpus: int):
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        os.path.relpath(__file__),
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="cuda:nccl")
    torch.cuda.set_device(torch.distributed.get_rank())

    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D").to_dict())

    set_seed(42)

    with torch.device("meta"):
        model_meta_init = NVEsmForMaskedLM(config)

    for layer in model_meta_init.esm.encoder.layers:
        fully_shard(layer)
    fully_shard(model_meta_init)

    # Assert parameters are actually on the meta device
    for name, parameter in model_meta_init.named_parameters():
        assert parameter.device == torch.device("meta"), f"Parameter {name} is not on the meta device"

    model_meta_init.to_empty(device="cuda")
    model_meta_init.apply(model_meta_init._init_weights)

    # Assert parameters are actually on the cuda device after to_empty
    for name, parameter in model_meta_init.named_parameters():
        assert str(parameter.device).startswith("cuda"), f"Parameter {name} is not on the cuda device"

    set_seed(42)
    model_normal_init = NVEsmForMaskedLM(config)

    for layer in model_normal_init.esm.encoder.layers:
        fully_shard(layer)
    fully_shard(model_normal_init)

    state_dict_meta_init = model_meta_init.state_dict()
    state_dict_normal_init = model_normal_init.state_dict()

    for key in state_dict_meta_init.keys():
        meta_tensor = state_dict_meta_init[key]
        normal_tensor = state_dict_normal_init[key]
        # Skip non-numeric tensors (e.g., Byte/uint8 tensors like _extra_state)
        if meta_tensor.dtype not in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.complex64,
            torch.complex128,
        ):
            continue

        torch.testing.assert_close(
            normal_tensor.mean(),
            meta_tensor.mean(),
            atol=1e-3,
            rtol=1e-4,
            msg=lambda x: f"Mean mismatch for parameter {key}: {x}",
        )

        if isinstance(normal_tensor, DTensor) and isinstance(meta_tensor, DTensor):
            torch.testing.assert_close(
                normal_tensor.full_tensor().std(),
                meta_tensor.full_tensor().std(),
                atol=1e-3,
                rtol=1e-4,
                msg=lambda x: f"Std mismatch for parameter {key}: {x}",
            )

        else:
            torch.testing.assert_close(
                normal_tensor.std(),
                meta_tensor.std(),
                atol=1e-3,
                rtol=1e-4,
                msg=lambda x: f"Std mismatch for parameter {key}: {x}",
            )
