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

"""Artifact gathers preserve parameters while using the collective's device."""

from types import SimpleNamespace

import pytest
import torch

from nemotron_stitch.projector.tensors import full_tensor_for_export


def test_plain_export_is_contiguous_and_graph_free():
    """Portable state retains exact values without retaining a training graph."""
    source = torch.arange(12.0, requires_grad=True).reshape(3, 4).t()
    exported = full_tensor_for_export(source)
    assert exported.is_contiguous() and not exported.requires_grad
    torch.testing.assert_close(exported, source, rtol=0, atol=0)
    assert source.requires_grad and not source.is_contiguous()


@pytest.mark.parametrize("local_device,mesh_device", [("cpu", "cuda"), ("cpu", "cpu"), ("cuda", "cuda")])
def test_offloaded_shards_gather_on_mesh_device(local_device, mesh_device):
    """A CUDA mesh must gather CUDA buffers without moving the source shard."""
    events = []

    class Shard:
        """A device-sensitive collective fixture; no CUDA runtime is needed."""

        def __init__(self, device):
            self.device = torch.device(device)
            self.device_mesh = SimpleNamespace(device_type=mesh_device)

        def detach(self):
            return self

        def to(self, device):
            events.append(("move", device))
            return Shard(device)

        def full_tensor(self):
            assert self.device.type == mesh_device
            events.append(("gather", mesh_device))
            return torch.arange(4.0)

    source = Shard(local_device)
    torch.testing.assert_close(full_tensor_for_export(source), torch.arange(4.0), rtol=0, atol=0)
    assert source.device.type == local_device
    assert events == ([("move", mesh_device)] if local_device != mesh_device else []) + [("gather", mesh_device)]


def test_compact_export_only_gathers_selected_tensors():
    from nemotron_stitch.automodel.checkpoint import collect_portable_states

    class Unselected:
        def full_tensor(self):
            raise AssertionError("export must not gather the frozen decoder")

    tensors = {
        "backbone.weight": Unselected(),
        "projection.weight": torch.ones(2),
        "marker": torch.full((2,), 3.0),
        "layer.lora_A": torch.full((2,), 4.0),
    }
    model = SimpleNamespace(named_parameters=lambda: tensors.items())
    projector, extra, adapter = collect_portable_states(model, projector_prefix="projection.", extra_names=("marker",))
    assert torch.equal(projector["weight"], tensors["projection.weight"])
    assert torch.equal(extra["marker"], tensors["marker"])
    assert torch.equal(adapter["layer.lora_A"], tensors["layer.lora_A"])
