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
import transformer_engine.pytorch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformer_engine.pytorch.tensor import QuantizedTensor
from transformers import AutoConfig, set_seed

from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM, NVEsmForTokenClassification


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def msg(x):
    return f"Mismatch in module {name}: {x}"


def verify_model_parameters_initialized_correctly(
    model: NVEsmForMaskedLM, atol=1e-3, rtol=1e-4, should_be_fp8: bool = False
):
    config = model.config

    for name, parameter in model.named_parameters():
        assert str(parameter.device).startswith("cuda"), f"Parameter {name} is not on the cuda device"

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            torch.testing.assert_close(module.weight.mean().item(), 0.0, atol=atol, rtol=rtol, msg=msg)
            torch.testing.assert_close(
                module.weight.std().item(), config.initializer_range, atol=atol, rtol=rtol, msg=msg
            )

        elif name == "lm_head.decoder":
            # Make sure the lm_head decoder weights are still tied to the encoder weights
            assert module.weight is model.esm.embeddings.word_embeddings.weight, "Decoder weight tying has been broken"

        elif isinstance(module, transformer_engine.pytorch.Linear):
            torch.testing.assert_close(module.weight.mean().item(), 0.0, atol=atol, rtol=rtol, msg=msg)
            torch.testing.assert_close(
                module.weight.std().item(), config.initializer_range, atol=atol, rtol=rtol, msg=msg
            )
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias), msg=msg)
            if should_be_fp8:
                assert isinstance(module.weight, QuantizedTensor), f"Module {name} weight is not a QuantizedTensor"

        elif isinstance(module, transformer_engine.pytorch.LayerNormLinear):
            torch.testing.assert_close(module.weight.mean().item(), 0.0, atol=atol, rtol=rtol, msg=msg)
            torch.testing.assert_close(
                module.weight.std().item(), config.initializer_range, atol=atol, rtol=rtol, msg=msg
            )
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias), msg=msg)
            torch.testing.assert_close(module.layer_norm_weight, torch.ones_like(module.layer_norm_weight), msg=msg)
            torch.testing.assert_close(module.layer_norm_bias, torch.zeros_like(module.layer_norm_bias), msg=msg)
            if should_be_fp8:
                assert isinstance(module.weight, QuantizedTensor), f"Module {name} weight is not a QuantizedTensor"

        elif isinstance(module, transformer_engine.pytorch.LayerNormMLP):
            torch.testing.assert_close(module.fc1_weight.mean().item(), 0.0, atol=atol, rtol=rtol, msg=msg)
            torch.testing.assert_close(
                module.fc1_weight.std().item(), config.initializer_range, atol=atol, rtol=rtol, msg=msg
            )
            torch.testing.assert_close(module.fc2_weight.mean().item(), 0.0, atol=atol, rtol=rtol, msg=msg)
            torch.testing.assert_close(
                module.fc2_weight.std().item(), config.initializer_range, atol=atol, rtol=rtol, msg=msg
            )
            torch.testing.assert_close(module.fc1_bias, torch.zeros_like(module.fc1_bias), msg=msg)
            torch.testing.assert_close(module.fc2_bias, torch.zeros_like(module.fc2_bias), msg=msg)
            torch.testing.assert_close(module.layer_norm_weight, torch.ones_like(module.layer_norm_weight), msg=msg)
            torch.testing.assert_close(module.layer_norm_bias, torch.zeros_like(module.layer_norm_bias), msg=msg)
            if should_be_fp8:
                assert isinstance(module.fc1_weight, QuantizedTensor), (
                    f"Module {name} fc1_weight is not a QuantizedTensor"
                )
                assert isinstance(module.fc2_weight, QuantizedTensor), (
                    f"Module {name} fc2_weight is not a QuantizedTensor"
                )

        elif isinstance(module, torch.nn.LayerNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight), msg=msg)
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias), msg=msg)

        elif isinstance(module, transformer_engine.pytorch.attention.rope.RotaryPositionEmbedding):
            dim = config.hidden_size // config.num_attention_heads
            expected_inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cuda") / dim))
            torch.testing.assert_close(module.inv_freq, expected_inv_freq, msg=msg)


def verify_pretrained_model_sanity(model: NVEsmForTokenClassification, atol=1e-3, rtol=1e-4):
    for name, p in model.named_parameters():
        assert p.numel() > 0, f"{name} is empty"
        assert torch.isfinite(p).all(), f"{name} has NaN/Inf"

        max_abs = p.abs().max().item()
        assert max_abs < 1e3, f"{name} extreme values: {max_abs}"

        if name == "classifier.weight":
            torch.testing.assert_close(p.mean().item(), 0.0, atol=atol, rtol=rtol, msg=msg)
            torch.testing.assert_close(p.std().item(), model.config.initializer_range, atol=atol, rtol=rtol, msg=msg)

        if name == "classifier.bias":
            torch.testing.assert_close(p, torch.zeros_like(p), msg=msg)


def test_cuda_init():
    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D").to_dict())

    set_seed(42)
    model = NVEsmForMaskedLM(config)
    model.to("cuda")

    verify_model_parameters_initialized_correctly(model)


def test_meta_init():
    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D").to_dict())

    set_seed(42)
    with torch.device("meta"):
        model = NVEsmForMaskedLM(config)

    # Assert parameters are actually on the meta device
    for name, parameter in model.named_parameters():
        assert parameter.device == torch.device("meta"), f"Parameter {name} is not on the meta device"

    # Move the model to the cuda device and initialize the parameters
    model.init_empty_weights()

    verify_model_parameters_initialized_correctly(model)


def test_cuda_fp8_init(fp8_recipe):
    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D", revision="c731040f").to_dict())

    set_seed(42)
    with transformer_engine.pytorch.fp8_model_init(recipe=fp8_recipe):
        model = NVEsmForMaskedLM(config)

    model.to("cuda")

    verify_model_parameters_initialized_correctly(model, atol=1e-2, should_be_fp8=True)


def test_meta_fp8_init(fp8_recipe):
    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D", revision="c731040f").to_dict())

    set_seed(42)
    with transformer_engine.pytorch.fp8_model_init(recipe=fp8_recipe), torch.device("meta"):
        model = NVEsmForMaskedLM(config)

    # Move the model to the cuda device and initialize the parameters
    model.init_empty_weights()

    verify_model_parameters_initialized_correctly(model, should_be_fp8=True)


def test_model_for_token_classification_init():
    config = NVEsmConfig(
        **AutoConfig.from_pretrained(
            "facebook/esm2_t6_8M_UR50D", trust_remote_code=True, revision="c731040f"
        ).to_dict()
    )

    set_seed(42)
    model = NVEsmForTokenClassification.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", config=config, dtype=torch.bfloat16, trust_remote_code=True, revision="c731040f"
    )
    model.to("cuda")

    verify_pretrained_model_sanity(model)


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

    config = NVEsmConfig(**AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D", revision="c731040f").to_dict())

    set_seed(42)

    with torch.device("meta"):
        model_meta_device = NVEsmForMaskedLM(config)

    for layer in model_meta_device.esm.encoder.layers:
        fully_shard(layer)
    fully_shard(model_meta_device)

    # Assert parameters are actually on the meta device
    for name, parameter in model_meta_device.named_parameters():
        assert parameter.device == torch.device("meta"), f"Parameter {name} is not on the meta device"

    model_meta_device.init_empty_weights()

    # Assert parameters are actually on the cuda device after to_empty
    for name, parameter in model_meta_device.named_parameters():
        assert str(parameter.device).startswith("cuda"), f"Parameter {name} is not on the cuda device"

    set_seed(42)
    model_normal_init = NVEsmForMaskedLM(config)

    for layer in model_normal_init.esm.encoder.layers:
        fully_shard(layer)
    fully_shard(model_normal_init)

    state_dict_meta_init = model_meta_device.state_dict()
    state_dict_normal_init = model_normal_init.state_dict()

    for key in state_dict_meta_init.keys():
        if key.endswith("_extra_state"):
            continue

        meta_tensor = state_dict_meta_init[key]
        normal_tensor = state_dict_normal_init[key]

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
                atol=1e-2,
                rtol=1e-4,
                msg=lambda x: f"Std mismatch for parameter {key}: {x}",
            )

        else:
            torch.testing.assert_close(
                normal_tensor.std(),
                meta_tensor.std(),
                atol=1e-2,
                rtol=1e-4,
                msg=lambda x: f"Std mismatch for parameter {key}: {x}",
            )
