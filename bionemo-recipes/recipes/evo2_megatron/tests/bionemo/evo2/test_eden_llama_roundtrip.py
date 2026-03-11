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

"""Test mbridge -> HF -> mbridge roundtrip fidelity for Eden (Llama) models.

Verifies that an Eden mbridge checkpoint survives the mbridge -> HF -> mbridge
round trip with bit-exact weight preservation.
"""

import copy
import os
import shlex
import subprocess
from pathlib import Path

import pytest
import torch

from bionemo.evo2.utils.checkpoint.mbridge_to_vortex import load_mbridge_state_dict

from .utils import find_free_network_port


ROUNDTRIP_HELPER = Path(__file__).parent / "_eden_roundtrip_helper.py"
PRETEST_ENV = copy.deepcopy(os.environ)


@pytest.fixture(scope="module")
def eden_ckpt(mbridge_eden_checkpoint) -> Path:
    """Module-scoped alias for the session-scoped Eden checkpoint."""
    return mbridge_eden_checkpoint


@pytest.fixture(scope="module")
def original_mbridge_sd(eden_ckpt: Path) -> dict[str, torch.Tensor]:
    """Load the original mbridge state dict from DCP (no distributed init needed)."""
    return load_mbridge_state_dict(eden_ckpt)


@pytest.fixture(scope="module")
def hf_exported_dir(eden_ckpt: Path, tmp_path_factory) -> Path:
    """Export the Eden mbridge checkpoint to HuggingFace format."""
    tmp_dir = tmp_path_factory.mktemp("eden_hf_export")
    hf_dir = tmp_dir / "hf_checkpoint"

    open_port = find_free_network_port()
    cmd = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"{ROUNDTRIP_HELPER} --mode export "
        f"--ckpt-dir {eden_ckpt} --hf-output-dir {hf_dir}"
    )
    env = copy.deepcopy(PRETEST_ENV)
    result = subprocess.run(shlex.split(cmd), check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"EXPORT STDOUT:\n{result.stdout}")
        print(f"EXPORT STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"mbridge->HF export failed: {result.stderr[-2000:]}"
    assert hf_dir.exists(), f"HF export dir not created at {hf_dir}"
    return hf_dir


@pytest.fixture(scope="module")
def roundtripped_mbridge_sd(hf_exported_dir: Path, tmp_path_factory) -> dict[str, torch.Tensor]:
    """Import HF checkpoint back into an mbridge model and return the state dict.

    Flow: HF -> mbridge (via AutoBridge).  The helper script saves the megatron
    model's state dict as a .pt file so we can compare it against the original.
    """
    tmp_dir = tmp_path_factory.mktemp("eden_mbridge_reimport")
    reimport_dir = tmp_dir / "reimported_mbridge"

    open_port = find_free_network_port()
    cmd = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"{ROUNDTRIP_HELPER} --mode import "
        f"--hf-input-dir {hf_exported_dir} --ckpt-output-dir {reimport_dir}"
    )
    env = copy.deepcopy(PRETEST_ENV)
    result = subprocess.run(shlex.split(cmd), check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"IMPORT STDOUT:\n{result.stdout}")
        print(f"IMPORT STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"HF->mbridge import failed: {result.stderr[-2000:]}"

    sd_path = reimport_dir / "state_dict.pt"
    assert sd_path.exists(), f"Roundtripped state dict not found at {sd_path}"
    return torch.load(sd_path, map_location="cpu", weights_only=True)


def _tensor_keys(sd: dict[str, object]) -> set[str]:
    """Return keys whose values are Tensors (skip _extra_state / bytes metadata)."""
    return {k for k, v in sd.items() if isinstance(v, torch.Tensor)}


@pytest.mark.slow
def test_roundtrip_mbridge_weight_equality(
    original_mbridge_sd: dict[str, torch.Tensor],
    roundtripped_mbridge_sd: dict[str, torch.Tensor],
):
    """Verify that mbridge -> HF -> mbridge produces identical weights."""
    orig_keys = _tensor_keys(original_mbridge_sd)
    rt_keys = _tensor_keys(roundtripped_mbridge_sd)

    assert orig_keys == rt_keys, (
        f"Key mismatch.\nOnly in original: {sorted(orig_keys - rt_keys)}\n"
        f"Only in roundtripped: {sorted(rt_keys - orig_keys)}"
    )

    for key in sorted(orig_keys):
        assert original_mbridge_sd[key].shape == roundtripped_mbridge_sd[key].shape, f"Shape mismatch for {key}"
        torch.testing.assert_close(
            original_mbridge_sd[key],
            roundtripped_mbridge_sd[key],
            atol=0,
            rtol=0,
            msg=lambda diff: f"Weight mismatch for {key}: {diff}",
        )
