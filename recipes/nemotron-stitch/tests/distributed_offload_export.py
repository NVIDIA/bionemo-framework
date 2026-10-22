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

"""Two-rank CUDA/NCCL portable-artifact qualification for CPU-offloaded shards.

Run with torchrun --nproc-per-node=2 and a shared fresh artifact directory as
the first argument. All exports and restores must preserve exact tensor values
and CPU storage, including trainable extra parameters. This is an integration
check rather than a default pytest case because it requires two physical GPUs.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemotron_stitch.projector import MultimodalProjector
from nemotron_stitch.projector.artifact import load_projector_artifact, read_projector_artifact, save_projector_artifact
from nemotron_stitch.projector.tensors import full_tensor_for_export
from nemotron_stitch.projector.trainability import TrainabilityPolicy, collect_extra_state


def main():
    """Save/zero/restore real offloaded shards and compare all full tensors."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        mesh = init_device_mesh("cuda", (2,))
        torch.manual_seed(42)
        model = torch.nn.Module()
        config = {"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 4, "hidden_size": 8}
        model.config = SimpleNamespace(mm_projectors=[config])
        model.mm_projector = MultimodalProjector.from_config([config], output_size=6)
        model.register_parameter("marker", torch.nn.Parameter(torch.arange(6.0)))
        expected = {key: value.detach().clone() for key, value in model.mm_projector.state_dict().items()}
        for module in model.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is not None:
                    setattr(
                        module,
                        name,
                        torch.nn.Parameter(
                            distribute_tensor(parameter.detach().cuda(), mesh, [Shard(0)]).cpu(),
                            requires_grad=parameter.requires_grad,
                        ),
                    )
        extra = collect_extra_state(model, TrainabilityPolicy(extra_patterns=("marker",)))
        torch.testing.assert_close(extra["marker"], torch.arange(6.0), rtol=0, atol=0)
        path = Path(sys.argv[1])
        provenance = {"base_model": {"repo_id": "example/base", "revision": "cpu-offload"}}
        save_projector_artifact(model, path, provenance=provenance, stage=2, extra_state=extra)
        dist.barrier()
        artifact = read_projector_artifact(path, expected=provenance)
        for name, tensor in expected.items():
            torch.testing.assert_close(artifact.state[name], tensor, rtol=0, atol=0)
        for parameter in model.parameters():
            assert parameter.device.type == "cpu" and parameter.requires_grad
            parameter.data.to_local().zero_()
        load_projector_artifact(model, path, expected=provenance, extra_patterns=("marker",))
        for name, tensor in model.mm_projector.state_dict().items():
            torch.testing.assert_close(full_tensor_for_export(tensor), expected[name], rtol=0, atol=0)
        restored_extra = collect_extra_state(model, TrainabilityPolicy(extra_patterns=("marker",)))
        torch.testing.assert_close(restored_extra["marker"], torch.arange(6.0), rtol=0, atol=0)
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        print("OFFLOAD_ARTIFACT_EXACT_RESTORE_PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
