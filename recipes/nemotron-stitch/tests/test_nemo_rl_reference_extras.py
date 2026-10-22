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

"""Artifact extras share the donor KL anchor across ownership and resume paths.

The constructor tests replace only upstream's expensive decoder construction;
Stitch's real constructor, artifact IO and anchor repair execute unchanged.
The distributed test verifies unequal local shards with real CPU DTensors.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_stitch.nemo_rl import policy
from nemotron_stitch.projector import MultimodalProjector
from nemotron_stitch.projector.artifact import save_projector_artifact


class _ArtifactHost(nn.Module):
    """Keep registered parameter/buffer extras beside either projector ownership."""

    def __init__(self, ownership):
        """Use tiny tensors to expose snapshot ordering without a decoder load."""
        super().__init__()
        configs = [{"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 2}]
        self.config = SimpleNamespace(mm_projectors=configs)
        self.backbone = nn.Parameter(torch.ones(2))
        self.extra_parameter = nn.Parameter(torch.zeros(6), requires_grad=False)
        self.register_buffer("extra_buffer", torch.zeros(6))
        projector = MultimodalProjector.from_config(configs, output_size=2).requires_grad_(False)
        if ownership == "module":
            self.mm_projector = projector
        else:
            object.__setattr__(self, "mm_projector", projector)


@pytest.mark.parametrize("ownership", ["module", "sidecar"])
@pytest.mark.parametrize("resume", [False, True])
def test_worker_constructor_anchors_artifact_extras_to_donor(monkeypatch, tmp_path, ownership, resume):
    """Reference extras retain donor values even when resumed state differs."""
    pytest.importorskip("nemo_automodel", reason="actual Stitch policy constructor needs AutoModel")
    from nemo_rl.models.policy.workers.automodel_policy_worker import AutomodelPolicyWorkerImpl

    donor = _ArtifactHost(ownership)
    donor.extra_parameter.fill_(3)
    donor.extra_buffer.fill_(5)
    artifact = tmp_path / "donor"
    save_projector_artifact(
        donor,
        artifact,
        extra_state={"extra_parameter": donor.extra_parameter.detach(), "extra_buffer": donor.extra_buffer},
    )
    weights = tmp_path / "checkpoint" if resume else None
    if resume and ownership == "module":
        donor.extra_parameter.fill_(9)
        donor.extra_buffer.fill_(11)
        save_projector_artifact(
            donor,
            weights / policy.PROJECTOR_CHECKPOINT_DIRNAME,
            extra_state={"extra_parameter": donor.extra_parameter.detach(), "extra_buffer": donor.extra_buffer},
        )

    def initialize(worker, config, weights_path=None):
        """Model the upstream capture-before-resume boundary, not Stitch's repair."""
        worker.model = _ArtifactHost(ownership)
        worker.reference_model_state_dict = {name: tensor.clone() for name, tensor in worker.model.state_dict().items()}
        if weights_path:
            worker.model.backbone.data.fill_(7)
        worker.optimizer = torch.optim.AdamW([worker.model.backbone])
        worker.lora_enabled = False
        worker.rank = 1

    monkeypatch.setattr(AutomodelPolicyWorkerImpl, "__init__", initialize)
    monkeypatch.setattr(policy, "resolve_callback", lambda _: lambda: None)
    config = {
        "model_registry_callback": "consumer.register",
        "projector_artifact_path": str(artifact),
        "hf_config_overrides": {"mm_projector_trainable": False, "mm_projector_ownership": ownership},
        "dtensor_cfg": {"lora_cfg": {"enabled": False}},
    }
    cls = policy.EncoderDTensorPolicyWorkerV2.__ray_metadata__.modified_class
    worker = cls(config=config, weights_path=weights)
    assert torch.equal(worker.reference_model_state_dict["extra_parameter"], torch.full((6,), 3.0))
    assert torch.equal(worker.reference_model_state_dict["extra_buffer"], torch.full((6,), 5.0))
    assert torch.equal(worker.reference_model_state_dict["backbone"], torch.ones(2))
    assert torch.equal(worker.model.backbone, torch.full((2,), 7.0 if resume else 1.0))
    expected_parameter, expected_buffer = (9.0, 11.0) if resume and ownership == "module" else (3.0, 5.0)
    assert torch.equal(worker.model.extra_parameter, torch.full((6,), expected_parameter))
    assert torch.equal(worker.model.extra_buffer, torch.full((6,), expected_buffer))


def _check_extra_reference_shards(rank, rendezvous):
    """Ensure full donor anchors become the correct unequal rank-local shards."""
    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        anchors = {
            "extra_parameter": torch.arange(7.0),
            "extra_buffer": torch.arange(7.0) + 20,
            "short_buffer": torch.tensor([42.0]),
        }
        live = {name: distribute_tensor(tensor + 100, mesh, [Shard(0)]) for name, tensor in anchors.items()}
        reference = {name: torch.zeros_like(tensor.to_local()) for name, tensor in live.items()}
        assert policy.patch_reference_projector_state(reference, anchors, prefix="", shard_geometry=live) == 3
        for name, tensor in anchors.items():
            width = (tensor.numel() + 1) // 2
            expected = tensor[rank * width : (rank + 1) * width]
            assert torch.equal(reference[name], expected)
            assert torch.equal(live[name].to_local(), expected + 100)
    finally:
        dist.destroy_process_group()


def test_extra_reference_anchor_uses_real_two_rank_shard_geometry(tmp_path):
    """Unequal shards detect full-copy and wrong-rank anchor mistakes."""
    torch.multiprocessing.start_processes(
        _check_extra_reference_shards,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
