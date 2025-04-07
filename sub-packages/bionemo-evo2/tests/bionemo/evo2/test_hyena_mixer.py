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


@pytest.fixture
def hyena_nv_test_config() -> HyenaNVTestConfig:
    config = HyenaNVTestConfig()
    return config


@pytest.fixture
def hyena_test_config() -> HyenaTestConfig:
    config = HyenaTestConfig()
    return config


@pytest.fixture
def hyena_config() -> HyenaConfig:
    config = HyenaConfig()
    config.num_groups_hyena = 4096
    config.num_groups_hyena_short = 256
    config.num_groups_hyena_medium = 256
    return config


@pytest.fixture
def mixer(hyena_test_config: HyenaTestConfig, hyena_config: HyenaConfig):
    """Create a HyenaMixer instance for testing with standard config"""
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        # Create necessary submodules
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
def nv_mixer(hyena_nv_test_config: HyenaNVTestConfig, hyena_config: HyenaConfig):
    """Create a HyenaMixer instance for testing with NV config"""
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        # Create necessary submodules
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


def test_initialization(mixer: HyenaMixer):
    """Test the initialization of the HyenaMixer"""
    assert hasattr(mixer, "hyena_proj_conv")
    assert hasattr(mixer, "mixer")


def test_nv_initialization(nv_mixer: HyenaMixer):
    """Test the initialization of the HyenaMixer with NV config"""
    assert hasattr(nv_mixer, "hyena_proj_conv")
    assert hasattr(nv_mixer, "mixer")


def b2b_torch_forward(mixer: HyenaMixer, features: torch.Tensor, _proj_use_cp: bool = False):
    features = mixer.hyena_proj_conv(features, _use_cp=_proj_use_cp)  # [B, D, L]
    x1, x2, v = rearrange(features, "b (g dg p) l -> b (g dg) p l", p=3, g=mixer.num_groups_per_tp_rank).unbind(dim=2)
    z = mixer.mixer(x1, x2, v)
    return z


def test_b2b_causal_conv1d(mixer: HyenaMixer):
    """Test the B2B causal conv1d layer"""
    data = torch.load("/workspaces/bionemo-framework/sub-packages/bionemo-evo2/tests/bionemo/evo2/test_layer1.pt")
    input_features = data["input_features"]
    output_features_b2b_torch = b2b_torch_forward(mixer, input_features)

    assert hasattr(mixer, "b2b_kernel")
    output_features_b2b = mixer.b2b_kernel(input_features)

    # Compare with stored expected output
    assert torch.allclose(output_features_b2b, output_features_b2b_torch, rtol=1e-2, atol=1e-2)


def test_nv_b2b_causal_conv1d(nv_mixer: HyenaMixer):
    """Test the B2B causal conv1d layer with NV config"""
    data = torch.load("/workspaces/bionemo-framework/sub-packages/bionemo-evo2/tests/bionemo/evo2/test_nv_layer1.pt")
    input_features = data["input_features"]
    output_features_b2b_torch = b2b_torch_forward(nv_mixer, input_features)

    assert hasattr(nv_mixer, "b2b_kernel")
    output_features_b2b = nv_mixer.b2b_kernel(input_features)

    # Compare with stored expected output
    assert torch.allclose(output_features_b2b, output_features_b2b_torch, rtol=1e-2, atol=1e-2)
