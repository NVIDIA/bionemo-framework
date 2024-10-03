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


import math

import pytest
import torch
from megatron.core.transformer.enums import AttnMaskType

from bionemo.amplify.api import AMPLIFYConfig
from bionemo.amplify.model.attention import AMPLIFYDotProductAttention
from bionemo.testing import megatron_parallel_state_utils


@pytest.fixture(scope="module")
def config() -> AMPLIFYConfig:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        yield AMPLIFYConfig(
            seq_length=20, hidden_size=4, num_attention_heads=4, attention_dropout=0.1, use_amplify_attention=True
        )


@pytest.fixture
def attention_layer(config):
    return AMPLIFYDotProductAttention(
        config=config,
        layer_number=0,
        attn_mask_type=AttnMaskType.padding,
        attention_type="normal",
    ).eval()


def test_init(attention_layer, config):
    assert attention_layer.config.use_amplify_attention
    assert attention_layer.config == config


def test_forward(attention_layer, config):
    batch_size = 2
    sequence_length = config.seq_length
    hidden_size = config.hidden_size
    device = torch.device("cuda")

    query = torch.randn(sequence_length, batch_size, 1, hidden_size, device=device)
    key = torch.randn(sequence_length, batch_size, 1, hidden_size, device=device)
    value = torch.randn(sequence_length, batch_size, 1, hidden_size, device=device)
    random_ints = torch.randint(0, 2, (batch_size, 1, sequence_length, sequence_length), device=device)
    attention_mask = ((random_ints + torch.transpose(random_ints, dim0=2, dim1=3)) / 2).to(
        dtype=torch.bool
    )  # symmetric mask tensor

    output = attention_layer(query, key, value, attention_mask)
    assert output.shape == (sequence_length, batch_size, hidden_size)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.half])
def test_attention_with_mask(attention_layer, dtype):
    sequence_length_val = 3
    sequence_length_query = 1
    batch_size = 2
    emb_dim = 4
    device = torch.device("cuda")

    # query and key such that the dot prod is an all-ones tensor
    query = torch.ones(batch_size, sequence_length_query, 1, emb_dim, device=device, dtype=dtype) / math.sqrt(emb_dim)
    key = torch.ones(batch_size, sequence_length_val, 1, emb_dim, device=device, dtype=dtype) / math.sqrt(emb_dim)

    query = query.transpose(0, 1)
    key = key.transpose(0, 1)

    attention_mask = torch.zeros(batch_size, 1, 1, sequence_length_val, device=device, dtype=dtype)
    attention_mask[0, :, :, 2:] = 1  # average first two tensors in val
    attention_mask[1, :, :, 1:] = 1  # select first item from val

    values = torch.stack([torch.arange(sequence_length_val)] * batch_size).to(device=device, dtype=dtype) + 1.0
    values = torch.stack([values] * emb_dim, dim=2).unsqueeze(2).transpose(0, 1)

    assert values.shape == (sequence_length_val, batch_size, 1, emb_dim)

    # softmax will make the the avg first 2 tensors in vals (ones + twos)/2 and second row is just ones
    output = attention_layer(query, key, values, attention_mask)
    expected_output = torch.tensor(
        [[[1.5000, 1.5000, 1.5000, 1.5000], [1.0000, 1.0000, 1.0000, 1.0000]]], device=device, dtype=dtype
    )
    assert torch.equal(output, expected_output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.half])
def test_amplify_scale_mask_softmax(attention_layer, config, dtype):
    batch_size = 2
    sequence_length = config.seq_length
    num_attention_heads = config.num_attention_heads

    input_tensor = torch.randn(batch_size, num_attention_heads, sequence_length, sequence_length, dtype=dtype)

    output = attention_layer.amplify_scale_mask_softmax(input_tensor)
    assert output.shape == input_tensor.shape
    assert output.dtype == dtype
