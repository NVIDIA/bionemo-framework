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

"""Shared test utilities for evo2 tests."""

import socket

import torch


def find_free_network_port(address: str = "localhost") -> int:
    """Find a free port on localhost for distributed testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_compute_capability() -> tuple[int, int]:
    """Get the compute capability of the current device."""
    if not torch.cuda.is_available():
        return (0, 0)
    # Returns a tuple, e.g., (9, 0) for H100
    return torch.cuda.get_device_capability()


def is_fp8_supported() -> bool:
    """Check if FP8 is supported on the current device.

    FP8 is supported on Ada Lovelace (8.9) and Hopper (9.0+).
    """
    cc = get_compute_capability()
    return cc >= (8, 9)


def is_fp4_supported() -> bool:
    """Check if FP4 is supported on the current device.

    Native support requires Blackwell (10.0+).
    """
    cc = get_compute_capability()
    return (10, 0) <= cc < (12, 0)


def is_mxfp8_supported() -> bool:
    """Check if MXFP8 is supported on the current device.

    Native support requires Blackwell (10.0+).
    """
    cc = get_compute_capability()
    return (10, 0) <= cc < (12, 0)


def check_fp8_support(device_id: int = 0) -> tuple[bool, str, str]:
    """Check if FP8 is supported on the current GPU.

    FP8 requires compute capability 8.9+ (Ada Lovelace/Hopper architecture or newer).

    Returns:
        Tuple of (is_supported, compute_capability_string, device_info_message).
    """
    if not torch.cuda.is_available():
        return False, "0.0", "CUDA not available"
    device_props = torch.cuda.get_device_properties(device_id)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    device_name = device_props.name
    # FP8 is supported on compute capability 8.9+ (Ada Lovelace/Hopper architecture)
    is_supported = (device_props.major > 8) or (device_props.major == 8 and device_props.minor >= 9)
    return is_supported, compute_capability, f"Device: {device_name}, Compute Capability: {compute_capability}"


def is_a6000_gpu() -> bool:
    """Check if any of the visible GPUs is an A6000."""
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        if "A6000" in device_name:
            return True
    return False
