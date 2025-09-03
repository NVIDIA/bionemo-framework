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

import os
from contextlib import contextmanager

import torch


@contextmanager
def initialize_distributed(cfg):
    """
    Setup the DeviceMesh for distributed training.

    Args:
        cfg: Hydra config.

    Yields:
        device_mesh: The DeviceMesh.

    Raises:
        ValueError: If the parallelism sizes are invalid.
    """
    # Initialize distributed training environment.
    torch.distributed.init_process_group()

    # Associate all future device operations in the current process
    # with a uniquely-indexed local device, e.g. "cuda:0" on Rank 0.
    local_rank = int(os.getenv("LOCAL_RANK", torch.distributed.get_rank()))
    torch.cuda.set_device(local_rank)

    # Initialize DeviceMesh. Validate parallelism sizes.
    # TODO(@cspades): Will add TE-backed context parallelism (CP) in the future, just need to
    # modify the ViT model to shard the sequence dimension after tokenization. For now, we
    # setup the CP dimension for demonstrating how to use DeviceMesh and CP with Megatron-FSDP.
    if cfg.distributed.dp_inter * cfg.distributed.dp_shard * cfg.distributed.cp != torch.distributed.get_world_size():
        raise ValueError(
            f"Invalid parallelism sizes: dp_inter({cfg.distributed.dp_inter}) * dp_shard({cfg.distributed.dp_shard}) * cp({cfg.distributed.cp}) * tp(1) != world_size({torch.distributed.get_world_size()})"
        )
    device_mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(
            cfg.distributed.dp_inter,
            cfg.distributed.dp_shard,
            cfg.distributed.cp,
            1,  # Needed to use TransformerEngine layers with Megatron-FSDP. "TP is always 1."
        ),
        mesh_dim_names=("dp_inter", "dp_shard", "cp", "tp"),
    )

    # Sub-meshes (possibly) required for Megatron-FSDP.
    # WARNING: These have a tendency to be deleted by Torch. Save references
    # or pass them to all classes or functions that use them.
    # DP: Only relevant when using HSDP, where we need the flattened DP group for data parallelism. (Otherwise, just pass dp_shard.)
    device_mesh[("dp_inter", "dp_shard")]._flatten("dp")
    # DP-Shard-CP: Only required if using CP. Otherwise, just pass dp_shard to FSDP.
    device_mesh[("dp_shard", "cp")]._flatten("dp_cp_shard")
    # HSDP (DP-CP): Only required if using HSDP. Otherwise, don't pass hybrid_fsdp_group to Megatron-FSDP.
    device_mesh[("dp_inter", "dp_shard", "cp")]._flatten("hsdp")

    # Yield DeviceMesh.
    yield device_mesh

    # Destroy process group.
    torch.distributed.destroy_process_group()
