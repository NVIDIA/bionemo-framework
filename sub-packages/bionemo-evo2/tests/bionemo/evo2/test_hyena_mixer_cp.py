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

"""Usage:
torchrun --nproc_per_node=2 sub-packages/bionemo-evo2/tests/bionemo/evo2/test_hyena_mixer_cp.py
"""

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
from torch.distributed.nn.functional import all_gather as functional_all_gather
from torch.nn.parallel import DistributedDataParallel as DDP


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


def zigzag_split_across_group_ranks(data, group, seq_dim=0):
    """Splits the data across group ranks along the seq dimension in a zigzag fashion.

    Arguments:
        data: tensor to split across ranks.
        group: the group to split the data across.
        seq_dim: the sequence dimension to split.

    Returns:
        data: split of the data on the current rank.
    """

    world_size = len(dist.get_process_group_ranks(group))
    # first check if we can just skip it...
    if world_size == 1:
        return data

    rank = dist.get_rank(group)

    # Zigzag-split the data
    seq_chunks = torch.chunk(data, 2 * world_size, dim=seq_dim)
    _data = [torch.cat((seq_chunks[i], seq_chunks[-(i + 1)]), dim=seq_dim) for i in range(world_size)]

    # Select the corresponding rank
    return _data[rank].contiguous()


def zigzag_gather_from_group_ranks(data, group, seq_dim=0):
    """Gathers data from all group ranks according to zigzag splitting.

    Arguments:
        data: tensor to gather across group ranks.
        seq_dim: the sequence dimension to concatenate chunks.

    Returns:
        data: gathered data from all group ranks concatenated along the seq_dim.

    """

    world_size = len(dist.get_process_group_ranks(group))
    # first check if we can just skip it...
    if world_size == 1:
        return data

    # Gather from all ranks using autograd-enabled all_gather
    gathered_data = functional_all_gather(data, group=group)

    # Initialize a list to store the original sequence chunks
    # `gathered_data` is a list of tensors from all ranks
    # Each rank's data consists of two chunks concatenated along seq_dim
    seq_chunks = [None] * (2 * world_size)

    for i, data_i in enumerate(gathered_data):
        chunk_size = data_i.size(seq_dim) // 2

        # Split the data_i back into the original two chunks
        chunk0, chunk1 = torch.split(data_i, chunk_size, dim=seq_dim)

        # Reassign the chunks to their original positions
        seq_chunks[i] = chunk0
        seq_chunks[-(i + 1)] = chunk1

    # Concatenate all chunks to reconstruct the original data
    reconstructed_data = torch.cat(seq_chunks, dim=seq_dim)

    return reconstructed_data


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
    ddp_mixer = DDP(mixer, process_group=parallel_state.get_data_parallel_group(with_context_parallel=True))

    # The HyenaMixer expects input in [seq_len, batch_size, hidden_dim] format
    # Create input in the correct format
    input_features = torch.rand(
        (seq_len, batch_size, mixer.input_size),
        dtype=mixer.transformer_config.params_dtype,
        device=torch.cuda.current_device(),
    )

    # Broadcast within each group
    cp_group = parallel_state.get_context_parallel_group()
    dist.broadcast(input_features, min(dist.get_process_group_ranks(cp_group)), group=cp_group)

    ################### Without context parallel
    output_features, bias = ddp_mixer(input_features, _hyena_use_cp=False)
    assert output_features.shape == (seq_len, batch_size, mixer.input_size)

    loss = output_features.float().mean() + bias.float().mean()
    loss.backward()
    dist.barrier()

    # Store the gradients for later comparison.
    grads_without_cp = []
    for n, p in ddp_mixer.named_parameters():
        if p.grad is not None:
            grads_without_cp.append((n, p.grad.clone()))

    ddp_mixer.zero_grad()
    dist.barrier()

    ################### With context parallel
    # Split the input features across the context parallel group
    input_features_cp = zigzag_split_across_group_ranks(input_features, group=cp_group, seq_dim=0)

    output_features_cp, bias = ddp_mixer(input_features_cp, _hyena_use_cp=True)
    assert output_features_cp.shape == (
        seq_len // parallel_state.get_context_parallel_world_size(),
        batch_size,
        mixer.input_size,
    )

    # Gather from all ranks according to zigzag splitting.
    output_features_cp_gathered = zigzag_gather_from_group_ranks(output_features_cp, group=cp_group, seq_dim=0)
    assert output_features_cp_gathered.shape == (seq_len, batch_size, mixer.input_size)

    loss_with_cp = output_features_cp_gathered.float().mean() + bias.float().mean()
    loss_with_cp.backward()
    dist.barrier()

    # Store the gradients for later comparison.
    grads_with_cp = []
    for n, p in ddp_mixer.named_parameters():
        if p.grad is not None:
            grads_with_cp.append((n, p.grad.clone()))

    ddp_mixer.zero_grad()
    dist.barrier()

    # Compute the loss difference.
    loss_abs_diff = torch.abs(loss - loss_with_cp)
    out_abs_diff = torch.abs(output_features - output_features_cp_gathered)
    torch.testing.assert_close(loss, loss_with_cp)
    # torch.testing.assert_close(output_features, output_features_cp_gathered)

    # Check gradients with and without CP.
    assert len(grads_without_cp) == len(grads_with_cp)

    gradient_mismatch = False
    for (n_without_cp, g_without_cp), (n_with_cp, g_with_cp) in zip(grads_without_cp, grads_with_cp):
        torch.testing.assert_close(g_without_cp, g_with_cp)

    # Clean up at the end
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
