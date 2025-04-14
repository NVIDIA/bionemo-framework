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

import pytest
import torch
from einops import rearrange
from nemo.collections.llm.gpt.model.hyena import HyenaNVTestConfig, HyenaTestConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer_specs import hyena_stack_spec_no_te
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_mixer import HyenaMixer

from bionemo.testing import megatron_parallel_state_utils


@pytest.fixture(params=[torch.bfloat16, torch.float32])
def hyena_test_config(request) -> HyenaTestConfig:
    config = HyenaTestConfig()
    config.params_dtype = request.param
    return config


@pytest.fixture(params=[torch.bfloat16, torch.float32])
def hyena_nv_test_config(request) -> HyenaNVTestConfig:
    config = HyenaNVTestConfig()
    config.params_dtype = request.param
    return config


@pytest.fixture
def hyena_config() -> HyenaConfig:
    config = HyenaConfig()
    config.num_groups_hyena = 4096
    config.num_groups_hyena_short = 256
    config.num_groups_hyena_medium = 256
    return config


@pytest.fixture(
    params=[
        (1, 1),  # (TP=1, CP=1)
        (2, 1),  # (TP=2, CP=1)
        (1, 2),  # (TP=1, CP=2)
        # (2, 2),  # Uncomment if you want to test TP=2, CP=2
    ]
)
def parallel_config(request):
    """Return tuple of (tensor_parallel_size, context_parallel_size)"""
    tp_size, cp_size = request.param
    # Calculate required GPU count
    required_gpus = tp_size * cp_size

    # Skip this configuration if not enough GPUs
    available_gpus = torch.cuda.device_count()
    if required_gpus > available_gpus:
        pytest.skip(
            f"Skipping test with TP={tp_size}, CP={cp_size} - requires {required_gpus} GPUs, but only {available_gpus} available"
        )
    return request.param


@pytest.fixture
def mixer(hyena_test_config: HyenaTestConfig, hyena_config: HyenaConfig, parallel_config):
    """Create a HyenaMixer instance for testing with standard config"""
    tp_size, cp_size = parallel_config
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=tp_size * cp_size, tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    ):
        # Create necessary submodules - use the mixer submodules like in the regular mixer fixture
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

        # Initialize with different operator types for testing
        yield HyenaMixer(
            transformer_config=hyena_test_config,
            hyena_config=hyena_config,
            max_sequence_length=512,
            submodules=submodules,
            layer_number=1,
            operator_type="hyena_short_conv",
            use_b2b_causal_conv1d=True,
        )


@pytest.fixture
def nv_mixer(hyena_nv_test_config: HyenaNVTestConfig, hyena_config: HyenaConfig, parallel_config):
    """Create a HyenaMixer instance for testing with NV config"""
    tp_size, cp_size = parallel_config
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=tp_size * cp_size, tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    ):
        # Create necessary submodules - use the mixer submodules like in the regular mixer fixture
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

        yield HyenaMixer(
            transformer_config=hyena_nv_test_config,
            hyena_config=hyena_config,
            max_sequence_length=512,
            submodules=submodules,
            layer_number=1,
            operator_type="hyena_short_conv",
            use_b2b_causal_conv1d=True,
        )


def b2b_torch_forward(mixer: HyenaMixer, features: torch.Tensor, _proj_use_cp: bool = False):
    features = mixer.hyena_proj_conv(features, _use_cp=_proj_use_cp)  # [B, D, L]
    x1, x2, v = rearrange(features, "b (g dg p) l -> b (g dg) p l", p=3, g=mixer.num_groups_per_tp_rank).unbind(dim=2)
    z = mixer.mixer(x1, x2, v)
    return z


def test_b2b_causal_conv1d(mixer: HyenaMixer, parallel_config):
    """Test the B2B causal conv1d layer"""
    tp_size, cp_size = parallel_config

    assert hasattr(mixer, "hyena_proj_conv")
    assert hasattr(mixer, "mixer")

    # Scale input features based on TP size
    hidden_size_per_tp = mixer.hidden_size // mixer.model_parallel_size
    # For CP, we need to adjust sequence length in some cases
    seq_len = 512
    if cp_size > 1:
        # When using CP, make sure sequence length is divisible by 2*CP for zigzag splitting
        seq_len = ((seq_len + (2 * cp_size - 1)) // (2 * cp_size)) * (2 * cp_size)

    input_features = torch.rand(
        (2, hidden_size_per_tp * 3, seq_len),
        dtype=mixer.transformer_config.params_dtype,
        device=mixer.hyena_proj_conv.short_conv_weight.device,
    )

    # Choose whether to use CP based on the parallel configuration
    use_cp = cp_size > 1
    output_features_b2b_torch = b2b_torch_forward(
        mixer, input_features, _proj_use_cp=use_cp
    )  # Always disable CP for b2b_torch_forward

    assert hasattr(mixer, "b2b_kernel")
    output_features_b2b = mixer.b2b_kernel(input_features, _use_cp=use_cp)  # Always disable CP for the CUDA kernel

    # Compare with stored expected output using parametrized tolerance
    assert torch.allclose(output_features_b2b, output_features_b2b_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.skip(reason="NV config (with conv bias) is not supported by b2b CUDA kernel yet")
def test_nv_b2b_causal_conv1d(nv_mixer: HyenaMixer, parallel_config):
    """Test the B2B causal conv1d layer with NV config"""
    tp_size, cp_size = parallel_config

    assert hasattr(nv_mixer, "hyena_proj_conv")
    assert hasattr(nv_mixer, "mixer")

    # Scale input features based on TP size
    hidden_size_per_tp = nv_mixer.hidden_size // nv_mixer.model_parallel_size
    # For CP, we need to adjust sequence length in some cases
    seq_len = 512
    if cp_size > 1:
        # When using CP, make sure sequence length is divisible by 2*CP for zigzag splitting
        seq_len = ((seq_len + (2 * cp_size - 1)) // (2 * cp_size)) * (2 * cp_size)

    input_features = torch.rand(
        (2, hidden_size_per_tp * 3, seq_len),
        dtype=nv_mixer.transformer_config.params_dtype,
        device=nv_mixer.hyena_proj_conv.short_conv_weight.device,
    )

    # Choose whether to use CP based on the parallel configuration
    use_cp = cp_size > 1
    output_features_b2b_torch = b2b_torch_forward(
        nv_mixer, input_features, _proj_use_cp=use_cp
    )  # Always disable CP for b2b_torch_forward

    assert hasattr(nv_mixer, "b2b_kernel")
    output_features_b2b = nv_mixer.b2b_kernel(input_features, _use_cp=use_cp)  # Always disable CP for the CUDA kernel

    # Compare with stored expected output using parametrized tolerance
    assert torch.allclose(output_features_b2b, output_features_b2b_torch, rtol=1e-2, atol=1e-2)
