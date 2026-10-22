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

"""Portable projector/extra tensors from regular or offloaded distributed state."""

from __future__ import annotations

import torch


def full_tensor_for_export(value: torch.Tensor) -> torch.Tensor:
    """Gather one artifact tensor on its mesh device, then detach it onto CPU.

    All ranks participate for DTensors. FSDP2 CPU offload retains a CUDA mesh
    around CPU local shards; NCCL cannot gather those CPU buffers. Only this
    selected artifact tensor moves to the mesh device for the public
    ``full_tensor`` collective, preserving the source parameter/device and
    sharding. Plain CPU tensors and CPU/gloo meshes remain on CPU. The result
    is full-shaped, contiguous, graph-free state suitable for safetensors.
    """
    value = value.detach()
    if hasattr(value, "full_tensor"):
        mesh_device = value.device_mesh.device_type
        if value.device.type != mesh_device:
            value = value.to(mesh_device)
        value = value.full_tensor()
    return value.cpu().contiguous()
