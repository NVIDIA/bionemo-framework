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

from leptonai.api.v1.types.deployment import EnvValue, EnvVar, Mount


resource_shapes_by_node_group = {
    "yo-bom-lepton-001": ["h100-sxm"],
    "nv-int-multiteam-nebius-h200-01": ["h200"],
    "az-sat-lepton-001": ["a100-80gb"],
}

UNIVERSAL_CPU_RESOURCES = ["cpu.small", "cpu.medium", "cpu.large", "my.cpu.large-40gb-mem"]


def construct_mount(path: str, mount_path: str, from_: str = "node-nfs:lepton-shared-fs") -> Mount:
    """Construct a Mount object for a given path, mount_path, and source."""
    # note, the from_="node-nfs:lepton-shared-fs" is not yet documented in the API docs, but is necessary
    mount = {
        "path": path,
        "mount_path": mount_path,
        "from": from_,
    }
    return Mount(**mount)


def construct_env_var(env_var) -> EnvVar:
    """Construct an EnvVar object from a config entry, supporting both secrets and literals."""
    if "value_from" in env_var:
        return EnvVar(
            name=env_var.name,
            value_from=EnvValue(secret_name_ref=env_var.value_from),
        )
    else:
        return EnvVar(
            name=env_var.name,
            value=env_var.value,
        )


def validate_resource_shape(node_group: str, resource_shape: str) -> None:
    """Validate that the resource shape is compatible with the node group.

    Args:
        node_group: The node group name
        resource_shape: The resource shape (e.g., "gpu.2xh100-sxm" or "cpu.small")

    Raises:
        SystemExit: If node group is unknown or resource shape is incompatible
    """
    print(f"Validating resource shape: {resource_shape} for node group: {node_group}")
    # CPU resources are available on all clusters
    if resource_shape in UNIVERSAL_CPU_RESOURCES:
        return

    if node_group not in resource_shapes_by_node_group:
        known_groups = ", ".join(sorted(resource_shapes_by_node_group.keys()))
        raise SystemExit(f"Unknown node group '{node_group}'.\nKnown node groups: {known_groups}")

    # Extract GPU type from resource shape
    # Formats: "gpu.h100-sxm" (single), "gpu.2xh100-sxm" (multi)
    try:
        gpu_part = resource_shape.split(".", 1)[1]  # Get "h100-sxm" or "2xh100-sxm"
        # Check if format is "NxGPU_TYPE" (multi-GPU) or just "GPU_TYPE" (single GPU)
        if gpu_part[0].isdigit() and "x" in gpu_part:
            gpu_type = gpu_part.split("x", 1)[1]  # Get "h100-sxm" from "2xh100-sxm"
        else:
            gpu_type = gpu_part  # Single GPU: "h100-sxm"
    except (IndexError, ValueError):
        raise SystemExit(
            f"Invalid resource shape format: {resource_shape}. Expected format: gpu.GPU_TYPE or gpu.NxGPU_TYPE"
        )

    available_gpu_types = resource_shapes_by_node_group[node_group]
    if gpu_type not in available_gpu_types:
        raise SystemExit(
            f"Resource shape '{resource_shape}' (GPU type: {gpu_type}) is not available in node group '{node_group}'.\n"
            f"Available GPU types for {node_group}: {', '.join(available_gpu_types)}"
        )
