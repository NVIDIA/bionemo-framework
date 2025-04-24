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


import os
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from nemo.collections.llm.gpt.model.hyena import HyenaTestConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer_specs import hyena_stack_spec_no_te
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_mixer import HyenaMixer


def init_parallel_state(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1):
    """Initialize distributed training and megatron parallel state."""

    num_gpus = torch.cuda.device_count()
    required_world_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    assert num_gpus == required_world_size, (
        f"World size {num_gpus} != TP={tensor_model_parallel_size} x PP={pipeline_model_parallel_size} x CP={context_parallel_size}"
    )

    # Set up environment variables
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Get local rank
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)

    # Set up timeout
    timeout_seconds = int(os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800))
    timeout_timedelta = timedelta(seconds=timeout_seconds)

    # Initialize process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout_timedelta)
        print(f"Initialized distributed training with local rank {local_rank}")

    # Initialize parallel state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
    )

    # Verify initialization
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_world_size = parallel_state.get_context_parallel_world_size()
    print(f"CP rank: {cp_rank}, CP world size: {cp_world_size}")
    return local_rank


if __name__ == "__main__":
    # Initialize parallel state
    local_rank = init_parallel_state(context_parallel_size=2)
    # Initialize the model parallel RNG
    model_parallel_cuda_manual_seed(42)

    # Your model initialization and other code here
    hyena_config = HyenaConfig(num_groups_hyena=4096, num_groups_hyena_short=256, num_groups_hyena_medium=256)
    hyena_test_config = HyenaTestConfig(params_dtype=torch.float32)

    # Create necessary submodules - use the mixer submodules like in the regular mixer fixture
    submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

    # Define dimensions
    batch_size = 2
    seq_len = 512

    print("Creating HyenaMixer...")
    mixer = HyenaMixer(
        transformer_config=hyena_test_config,
        hyena_config=hyena_config,
        max_sequence_length=seq_len,
        submodules=submodules,
        layer_number=1,
        operator_type="hyena_short_conv",
        use_b2b_causal_conv1d=True,
    )

    # The HyenaMixer expects input in [seq_len, batch_size, hidden_dim] format
    # Create input in the correct format
    input_features = torch.rand(
        (seq_len, batch_size, mixer.input_size),
        dtype=mixer.transformer_config.params_dtype,
        device=torch.cuda.current_device(),
    )

    print(f"Input shape: {input_features.shape}")

    # Forward pass
    output_features, bias = mixer.forward(input_features)

    print(f"Output shape: {output_features.shape}")
    if bias is not None:
        print(f"Bias shape: {bias.shape}")

    # Clean up at the end
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
