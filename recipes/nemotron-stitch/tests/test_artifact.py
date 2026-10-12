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

"""Artifact schema and codec tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from nemotron_stitch.contracts import MANIFEST_FILENAME, PROJECTOR_FILENAME
from nemotron_stitch.projector import MultimodalProjector
from nemotron_stitch.projector.artifact import (
    ProjectorManifest,
    load_projector_artifact,
    read_projector_artifact,
    save_projector_artifact,
)

PROVENANCE = {"base_model": {"repo_id": "example/base", "revision": "abc"}}


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        config = {"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 4, "hidden_size": 8}
        self.config = SimpleNamespace(mm_projectors=[config])
        self.mm_projector = MultimodalProjector.from_config([config], output_size=6)


def test_package_format_round_trip_is_bit_identical(tmp_path):
    model = _Model()
    expected = {k: v.clone() for k, v in model.mm_projector.state_dict().items()}
    save_projector_artifact(model, tmp_path, provenance=PROVENANCE, stage=1)

    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert manifest["schema"] == "nemotron-add-modality/mm-projector-manifest"
    assert manifest["schema_version"] == 1
    assert manifest["stage"] == 1
    assert manifest["output_size"] == 6
    assert manifest["projector_configs"][0]["kind"] == "mlp2x_gelu"
    assert manifest["checksums"][PROJECTOR_FILENAME]

    for parameter in model.mm_projector.parameters():
        nn.init.zeros_(parameter)
    read_projector_artifact(tmp_path, expected=PROVENANCE)  # checksums verify on read
    loaded = load_projector_artifact(model, tmp_path, expected=PROVENANCE)
    assert loaded.output_size == 6
    for name, value in model.mm_projector.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_save_allows_omitting_provenance(tmp_path):
    save_projector_artifact(_Model(), tmp_path)
    assert read_projector_artifact(tmp_path).manifest.provenance == {}


def test_module_written_artifact_loads_into_a_sidecar_host_bit_identically(tmp_path):
    """The alignment->GRPO handoff: module ownership writes, sidecar reads."""
    module_host = _Model()  # mm_projector is a registered child (module ownership)
    expected = {key: value.clone() for key, value in module_host.mm_projector.state_dict().items()}
    save_projector_artifact(module_host, tmp_path, provenance=PROVENANCE, stage=1)

    class _SidecarHost(nn.Module):
        def __init__(self):
            super().__init__()
            config = {"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 4, "hidden_size": 8}
            self.config = SimpleNamespace(mm_projectors=[config])
            object.__setattr__(self, "mm_projector", MultimodalProjector.from_config([config], output_size=6))

    sidecar_host = _SidecarHost()
    load_projector_artifact(sidecar_host, tmp_path, expected=PROVENANCE)
    for name, value in sidecar_host.mm_projector.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_artifact_round_trip_is_dtensor_safe(tmp_path):
    """FSDP2-sharded (DTensor) registries export full tensors and warm-start from them."""
    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    try:
        model = _Model()
        for module in model.mm_projector.modules():
            for attr, parameter in list(module._parameters.items()):
                if parameter is not None:
                    module._parameters[attr] = nn.Parameter(
                        distribute_tensor(parameter.detach().clone(), mesh, [Shard(0)]),
                        requires_grad=parameter.requires_grad,
                    )
        expected = {key: value.full_tensor().clone() for key, value in model.mm_projector.state_dict().items()}
        save_projector_artifact(model, tmp_path / "artifact", provenance=PROVENANCE, stage=1)

        for parameter in model.mm_projector.parameters():
            parameter.data.to_local().zero_()
        load_projector_artifact(model, tmp_path / "artifact", expected=PROVENANCE)
        for name, value in model.mm_projector.state_dict().items():
            torch.testing.assert_close(value.full_tensor(), expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_read_fails_closed_on_checksum_mismatch(tmp_path):
    save_projector_artifact(_Model(), tmp_path, provenance=PROVENANCE)
    weights = load_file(tmp_path / PROJECTOR_FILENAME)
    save_file({k: v + 1 for k, v in weights.items()}, tmp_path / PROJECTOR_FILENAME)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_projector_artifact(tmp_path)


def test_read_requires_checksums_for_loaded_files(tmp_path):
    save_projector_artifact(_Model(), tmp_path, provenance=PROVENANCE, extra_state={"marker": torch.ones(6)})
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    del manifest["checksums"][PROJECTOR_FILENAME]
    del manifest["checksums"]["extra/marker.safetensors"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing checksums.*extra/marker.safetensors.*mm-projector.safetensors"):
        read_projector_artifact(tmp_path)


def test_read_fails_closed_on_bad_discriminator(tmp_path):
    save_projector_artifact(_Model(), tmp_path, provenance=PROVENANCE)
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "something-else"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported manifest format"):
        read_projector_artifact(tmp_path)


def test_read_fails_closed_without_a_manifest(tmp_path):
    with pytest.raises(ValueError, match="no mm-projector-manifest.json"):
        read_projector_artifact(tmp_path)


def test_schema_rejects_projector_config_name_mismatch():
    with pytest.raises(ValueError, match="must match"):
        ProjectorManifest(
            output_size=6,
            projector_configs=[{"name": "a", "kind": "mlp2x_gelu"}],
            projectors={"b": {"class": "Mlp2xGeluProjector", "parameters": 1, "num_tokens": None}},
            provenance={},
        ).validate()


def test_extra_state_round_trip(tmp_path):
    model = _Model()
    marker = torch.randn(6)
    save_projector_artifact(model, tmp_path, provenance=PROVENANCE, extra_state={"marker_delta": marker})
    artifact = read_projector_artifact(tmp_path)
    torch.testing.assert_close(artifact.extra_state["marker_delta"], marker)
    assert artifact.manifest.extra_state["marker_delta"]["shape"] == [6]


class _ModelWithExtra(nn.Module):
    """A host carrying an extra-family parameter (a learned marker) in-tree."""

    def __init__(self):
        super().__init__()
        config = {"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 4, "hidden_size": 8}
        self.config = SimpleNamespace(mm_projectors=[config])
        self.mm_projector = MultimodalProjector.from_config([config], output_size=6)
        self.marker_embed_delta = nn.Parameter(torch.zeros(2, 6))


def test_load_restores_extra_state_by_parameter_name(tmp_path):
    source = _ModelWithExtra()
    with torch.no_grad():
        source.marker_embed_delta.fill_(3.0)
    save_projector_artifact(
        source,
        tmp_path,
        provenance=PROVENANCE,
        extra_state={"marker_embed_delta": source.marker_embed_delta.detach()},
    )

    target = _ModelWithExtra()
    load_projector_artifact(target, tmp_path, expected=PROVENANCE, extra_patterns=("marker_embed_delta",))
    torch.testing.assert_close(target.marker_embed_delta, torch.full((2, 6), 3.0))


def test_load_fails_closed_when_the_artifact_lacks_the_extra_family(tmp_path):
    save_projector_artifact(_ModelWithExtra(), tmp_path, provenance=PROVENANCE)
    with pytest.raises(ValueError, match="extra family"):
        load_projector_artifact(
            _ModelWithExtra(), tmp_path, expected=PROVENANCE, extra_patterns=("marker_embed_delta",)
        )


def test_load_fails_closed_on_extra_state_the_model_lacks(tmp_path):
    save_projector_artifact(_Model(), tmp_path, provenance=PROVENANCE, extra_state={"ghost": torch.zeros(2)})
    with pytest.raises(ValueError, match="no counterpart"):
        load_projector_artifact(_Model(), tmp_path, expected=PROVENANCE)


def test_load_fails_closed_on_extra_state_shape_mismatch(tmp_path):
    save_projector_artifact(
        _Model(), tmp_path, provenance=PROVENANCE, extra_state={"marker_embed_delta": torch.zeros(9)}
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_projector_artifact(_ModelWithExtra(), tmp_path, expected=PROVENANCE)
