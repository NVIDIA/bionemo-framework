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

"""Round-trip test: Savanna -> MBridge -> Vortex checkpoint conversion.

Downloads the 1b savanna and vortex checkpoints from HuggingFace, converts
savanna -> mbridge -> vortex, and compares the output against the reference
vortex checkpoint.
"""

import os

import pytest
import torch
from huggingface_hub import hf_hub_download

from bionemo.evo2.models.evo2_provider import HYENA_MODEL_OPTIONS
from bionemo.evo2.utils.checkpoint.mbridge_to_vortex import mbridge_to_vortex_state_dict
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import load_savanna_state_dict, savanna_to_mbridge_state_dict


SAVANNA_1B_REPO = "arcinstitute/savanna_evo2_1b_base"
VORTEX_1B_REPO = "arcinstitute/evo2_1b_base"
MODEL_SIZE = "evo2_1b_base"


@pytest.fixture(scope="module")
def savanna_checkpoint_path(tmp_path_factory):
    """Download the 1b savanna checkpoint from HuggingFace."""
    cache_dir = tmp_path_factory.mktemp("savanna_ckpt")
    path = hf_hub_download(
        repo_id=SAVANNA_1B_REPO,
        filename="savanna_evo2_1b_base.pt",
        local_dir=str(cache_dir),
    )
    return path


@pytest.fixture(scope="module")
def vortex_reference_path(tmp_path_factory):
    """Download the 1b vortex checkpoint from HuggingFace."""
    cache_dir = tmp_path_factory.mktemp("vortex_ref")
    path = hf_hub_download(
        repo_id=VORTEX_1B_REPO,
        filename="evo2_1b_base.pt",
        local_dir=str(cache_dir),
    )
    return path


@pytest.fixture(scope="module")
def roundtrip_vortex_sd(savanna_checkpoint_path):
    """Perform savanna -> mbridge -> vortex conversion and return the vortex state dict."""
    provider_cls = HYENA_MODEL_OPTIONS[MODEL_SIZE]
    model_provider = provider_cls()
    pattern = model_provider.hybrid_override_pattern

    savanna_sd = load_savanna_state_dict(savanna_checkpoint_path)
    mbridge_sd = savanna_to_mbridge_state_dict(savanna_sd, pattern, te_enabled=True)
    vortex_sd = mbridge_to_vortex_state_dict(mbridge_sd, model_provider, te_enabled=True)
    return vortex_sd


@pytest.fixture(scope="module")
def vortex_reference_sd(vortex_reference_path):
    """Load the reference vortex state dict from HuggingFace."""
    return torch.load(vortex_reference_path, map_location="cpu", weights_only=False)


@pytest.mark.slow
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space")
def test_roundtrip_key_set_equality(roundtrip_vortex_sd, vortex_reference_sd):
    """The round-trip output should have the same set of keys as the reference."""
    roundtrip_keys = set(roundtrip_vortex_sd.keys())
    reference_keys = set(vortex_reference_sd.keys())

    missing = reference_keys - roundtrip_keys
    extra = roundtrip_keys - reference_keys

    assert not missing, f"Keys missing from round-trip output: {sorted(missing)}"
    assert not extra, f"Extra keys in round-trip output: {sorted(extra)}"


@pytest.mark.slow
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space")
def test_roundtrip_tensor_shapes(roundtrip_vortex_sd, vortex_reference_sd):
    """All tensors should have matching shapes between round-trip and reference."""
    common_keys = set(roundtrip_vortex_sd.keys()) & set(vortex_reference_sd.keys())
    assert len(common_keys) > 0, "No common keys found"

    mismatches = []
    for key in sorted(common_keys):
        rt_shape = roundtrip_vortex_sd[key].shape
        ref_shape = vortex_reference_sd[key].shape
        if rt_shape != ref_shape:
            mismatches.append(f"{key}: roundtrip={rt_shape} vs reference={ref_shape}")

    assert not mismatches, "Shape mismatches:\n" + "\n".join(mismatches)


@pytest.mark.slow
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space")
def test_roundtrip_tensor_values(roundtrip_vortex_sd, vortex_reference_sd):
    """All tensors should have near-equal values between round-trip and reference."""
    common_keys = set(roundtrip_vortex_sd.keys()) & set(vortex_reference_sd.keys())
    assert len(common_keys) > 0, "No common keys found"

    mismatches = []
    for key in sorted(common_keys):
        rt_tensor = roundtrip_vortex_sd[key].float()
        ref_tensor = vortex_reference_sd[key].float()

        if rt_tensor.shape != ref_tensor.shape:
            mismatches.append(f"{key}: shape mismatch {rt_tensor.shape} vs {ref_tensor.shape}")
            continue

        if not torch.allclose(rt_tensor, ref_tensor, atol=1e-5, rtol=1e-4):
            max_diff = (rt_tensor - ref_tensor).abs().max().item()
            mismatches.append(f"{key}: max_diff={max_diff:.6e}")

    assert not mismatches, "Value mismatches:\n" + "\n".join(mismatches)
