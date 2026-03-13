# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for expert parallelism (EP) in the Mixtral MoE model.

Verifies that running with EP=2 (experts sharded across 2 GPUs) produces
the same logits and loss as EP=1 (all experts on a single GPU).
"""

import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch
from distributed_helpers import DistributedConfig, create_small_mixtral_config, get_dummy_batch

from modeling_mixtral_te import NVMixtralForCausalLM


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@requires_multi_gpu
def test_ep2_matches_ep1(unused_tcp_port):
    """Test that EP=2 produces the same logits as EP=1."""
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{unused_tcp_port}",
        str(Path(__file__).resolve()),
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        cwd=str(Path(__file__).parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"EP equivalence test failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Distributed worker executed via torchrun
# ---------------------------------------------------------------------------


def _distribute_state_dict(full_state_dict: dict, model: torch.nn.Module, device: torch.device) -> dict:
    """Distribute a full (EP=1) state dict to match a model's DTensor sharding.

    After calling ``set_ep_groups``, expert weight parameters become DTensors with
    ``Shard(0)`` placement.  This function uses ``distribute_tensor`` to automatically
    shard full expert weights according to those annotations, avoiding manual slicing.

    Args:
        full_state_dict: Complete state dict from an EP=1 model (plain tensors).
        model: Target EP model whose expert parameters are already DTensors.
        device: Device to move source tensors to before distributing.
    """
    from torch.distributed.tensor import DTensor, distribute_tensor

    distributed_state: dict = {}
    for key, value in model.state_dict().items():
        if key not in full_state_dict:
            continue
        full_value = full_state_dict[key]
        if isinstance(value, DTensor):
            distributed_state[key] = distribute_tensor(
                full_value.to(device),
                value.device_mesh,
                list(value.placements),
            )
        else:
            distributed_state[key] = full_value
    return distributed_state


def _run_ep_equivalence_test():
    """Main worker function for the EP equivalence test.

    1. Set up each rank's device, init distributed.
    2. Every rank creates an EP=1 model on its own GPU, runs forward, saves reference.
    3. Create EP=2 model with sharded expert weights, run forward, compare.
    """
    from torch.distributed.tensor.device_mesh import DeviceMesh

    # --- Setup distributed first so each rank uses its own GPU ---
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    ep_size = dist_config.world_size

    # --- Phase 1: EP=1 reference (every rank computes independently) ---
    config_ep1 = create_small_mixtral_config(expert_parallel_size=1)
    torch.manual_seed(0)
    model_ep1 = NVMixtralForCausalLM(config_ep1).to(dtype=torch.bfloat16, device=device)
    model_ep1.eval()

    batch = get_dummy_batch(config_ep1.vocab_size, seq_len=32, batch_size=2, device=device)

    with torch.no_grad():
        outputs_ep1 = model_ep1(**batch)

    logits_ep1 = outputs_ep1.logits.detach().clone().cpu()
    loss_ep1 = outputs_ep1.loss.detach().clone().cpu()

    # Save EP=1 full state dict on CPU for loading into EP model
    full_state_dict = {k: v.clone().cpu() for k, v in model_ep1.state_dict().items()}

    del model_ep1, outputs_ep1
    torch.cuda.empty_cache()

    # --- Phase 2: EP=2 distributed run ---
    config_ep2 = create_small_mixtral_config(expert_parallel_size=ep_size)
    torch.manual_seed(0)
    model_ep2 = NVMixtralForCausalLM(config_ep2).to(dtype=torch.bfloat16, device=device)

    # Set EP groups first to create DTensor annotations on expert weights
    ep_mesh = DeviceMesh("cuda", list(range(ep_size)))
    ep_group = ep_mesh.get_group()
    model_ep2.model.set_ep_groups(ep_group, ep_mesh)

    # Load EP=1 weights — distribute_tensor handles expert sharding automatically
    distributed_state = _distribute_state_dict(full_state_dict, model_ep2, device)
    model_ep2.load_state_dict(distributed_state, strict=True)
    model_ep2.eval()

    # Same batch on all ranks (EP dispatches tokens, input is replicated)
    batch_cuda = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs_ep2 = model_ep2(**batch_cuda)

    logits_ep2 = outputs_ep2.logits.detach().cpu()
    loss_ep2 = outputs_ep2.loss.detach().cpu()

    # --- Phase 3: Compare on rank 0 ---
    if dist_config.is_main_process():
        torch.testing.assert_close(
            logits_ep2,
            logits_ep1,
            atol=1e-2,
            rtol=1e-2,
            msg="EP=2 logits do not match EP=1 logits",
        )

        torch.testing.assert_close(
            loss_ep2,
            loss_ep1,
            atol=1e-3,
            rtol=1e-3,
            msg="EP=2 loss does not match EP=1 loss",
        )

        print("EP equivalence test PASSED: EP=2 logits and loss match EP=1")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _run_ep_equivalence_test()
