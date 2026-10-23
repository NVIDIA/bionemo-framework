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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_stitch.automodel.checkpoint import ProjectorStateDictAdapterMixin
from nemotron_stitch.contracts import MANIFEST_FILENAME
from nemotron_stitch.nemo_rl.policy import (
    PROJECTOR_STATE_PREFIX,
    add_projector_param_group,
    patch_reference_projector_state,
    relative_projector_drift,
    require_replicated_projector_policy_config,
    resolve_projector_trainability,
    restore_checkpoint_projector,
    save_checkpoint_projector,
    validate_initial_adapter_provenance,
)
from nemotron_stitch.nemo_rl.vllm_worker import (
    _merge_encoder_hf_overrides,
    _select_vllm_executor_backend,
    require_generation_topology,
)
from nemotron_stitch.projector import MultimodalProjector


def test_dtensor_refit_inventory_excludes_frozen_extra_parameters():
    pytest.importorskip("nemo_rl", reason="NeMo RL seam tests need the framework installed")
    pytest.importorskip("nemo_automodel", reason="the DTensor policy worker needs AutoModel")
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import dtensor_params_generator

    class _BaseAdapter:
        def __init__(self):
            self.config = SimpleNamespace(hf_checkpoint_prefix="")

        def convert_single_tensor_to_hf(self, name, tensor, **_kwargs):
            return [(name, tensor)]

    class _Adapter(ProjectorStateDictAdapterMixin, _BaseAdapter):
        PROJECTOR_STATE_PREFIXES = ("learned_start", "learned_end")

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Parameter(torch.ones(2, 2))
            self.learned_start = nn.Parameter(torch.ones(2), requires_grad=False)
            self.learned_end = nn.Parameter(torch.ones(2), requires_grad=False)
            self.state_dict_adapter = _Adapter()

    names = {name for name, _ in dtensor_params_generator(_Model(), torch.bfloat16)}
    assert names == {"backbone"}


def test_initial_adapter_provenance_resolves_the_upstream_layout(tmp_path):
    """The gate resolves the same restore_from layouts NeMo RL's loader does."""
    adapter_dir = tmp_path / "LATEST" / "model"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        '{"peft_type": "LORA", "r": 8, "lora_alpha": 16, "extra": "descriptive"}'
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

    # The weights directory or its 'model' subdirectory both resolve; locked
    # values match recursively and descriptive extras are allowed.
    validate_initial_adapter_provenance(adapter_dir, {"peft_type": "LORA", "r": 8, "lora_alpha": 16})
    validate_initial_adapter_provenance(tmp_path / "LATEST", {"peft_type": "LORA", "lora_alpha": 16})


def test_initial_adapter_provenance_fails_closed(tmp_path):
    adapter_dir = tmp_path / "model"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 8}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(ValueError, match="must not be empty"):
        validate_initial_adapter_provenance(adapter_dir, {})
    with pytest.raises(ValueError, match="provenance mismatch for provenance.r"):
        validate_initial_adapter_provenance(adapter_dir, {"r": 16})
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        validate_initial_adapter_provenance(tmp_path / "missing", {"r": 8})

    (adapter_dir / "adapter_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        validate_initial_adapter_provenance(adapter_dir, {"r": 8})


def test_policy_topology_rejects_the_combined_replicate_expert_mesh():
    # U-66: dense replication inside fused expert-parallel ranks is
    # inexpressible on the pinned stack (dp_size = world // (tp*cp*ep) leaves
    # no dense DP axis), and the combination is unqualified for the projector
    # contract. Each dimension alone stays admitted.
    require_replicated_projector_policy_config({"dtensor_cfg": {"dp_replicate_size": 4}})
    require_replicated_projector_policy_config({"dtensor_cfg": {"expert_parallel_size": 8}})
    with pytest.raises(NotImplementedError, match="U-66"):
        require_replicated_projector_policy_config({"dtensor_cfg": {"dp_replicate_size": 2, "expert_parallel_size": 8}})


def test_policy_and_generation_topology_contracts():
    require_replicated_projector_policy_config({"dtensor_cfg": {"tensor_parallel_size": 1}})
    require_replicated_projector_policy_config({"dtensor_cfg": {"expert_parallel_size": 8}})
    with pytest.raises(NotImplementedError, match="TP=1 and CP=1"):
        require_replicated_projector_policy_config({"dtensor_cfg": {"tensor_parallel_size": 2}})
    with pytest.raises(ValueError, match="cpu_offload"):
        require_replicated_projector_policy_config({"keep_policy_on_gpu": True, "dtensor_cfg": {"cpu_offload": True}})

    require_generation_topology(
        {
            "keep_vllm_on_gpu": True,
            "colocated": {"enabled": True},
            "vllm_cfg": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "async_engine": False,
            },
        }
    )
    require_generation_topology({"vllm_cfg": {"tensor_parallel_size": 4}})
    require_generation_topology(
        {
            "vllm_cfg": {
                "tensor_parallel_size": 8,
                "expert_parallel_size": 8,
            }
        }
    )
    require_generation_topology({"vllm_cfg": {"tensor_parallel_size": 1, "expert_parallel_size": 8}})
    for dimension in ("tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size"):
        with pytest.raises(ValueError, match="positive"):
            require_generation_topology({"vllm_cfg": {dimension: 0}})
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        require_generation_topology({"vllm_cfg": {"pipeline_parallel_size": 2}})

    # U-70: colocated external-DP generation must pin async_scheduling off.
    require_generation_topology(
        {
            "colocated": {"enabled": True},
            "vllm_cfg": {"tensor_parallel_size": 1, "expert_parallel_size": 8},
            "vllm_kwargs": {"async_scheduling": False},
        }
    )
    for vllm_kwargs in ({}, {"async_scheduling": True}):
        with pytest.raises(ValueError, match="U-70"):
            require_generation_topology(
                {
                    "colocated": {"enabled": True},
                    "vllm_cfg": {"tensor_parallel_size": 1, "expert_parallel_size": 8},
                    "vllm_kwargs": vllm_kwargs,
                }
            )
    # Non-colocated and single-width colocated layouts are out of U-70's scope.
    require_generation_topology(
        {
            "colocated": {"enabled": False},
            "vllm_cfg": {"tensor_parallel_size": 1, "expert_parallel_size": 8},
        }
    )
    require_generation_topology(
        {
            "colocated": {"enabled": True},
            "vllm_cfg": {"tensor_parallel_size": 8, "expert_parallel_size": 8},
        }
    )


def test_explicit_vllm_executor_backend_overrides_nemo_default():
    """Reference inference may retain its qualified local executor topology."""
    kwargs = {"distributed_executor_backend": "ray"}
    _select_vllm_executor_backend(kwargs, "mp")
    assert kwargs["distributed_executor_backend"] == "mp"

    with pytest.raises(ValueError, match="must be 'mp' or 'ray'"):
        _select_vllm_executor_backend(kwargs, "uni")


def test_encoder_hf_overrides_preserve_native_mxfp8_config():
    quantization = {"quant_method": "modelopt", "quant_algo": "MXFP8"}
    llm_kwargs = {"hf_overrides": {"quantization_config": quantization}}

    _merge_encoder_hf_overrides(
        llm_kwargs,
        {"mm_image_num_tokens": 49},
        "LlavaMBridgeNemotronProjected",
    )

    assert llm_kwargs["hf_overrides"] == {
        "quantization_config": quantization,
        "mm_image_num_tokens": 49,
        "architectures": ["LlavaMBridgeNemotronProjected"],
    }


def test_encoder_hf_overrides_translate_the_training_architecture():
    # The config carries the training-side host class; the vLLM engine must
    # name the plugin-registered rollout class. Translated, not a conflict.
    llm_kwargs: dict = {}
    _merge_encoder_hf_overrides(
        llm_kwargs,
        {"architectures": ["LlavaExampleNemotronForCausalLM"], "mm_image_num_tokens": 49},
        "LlavaExampleNemotronProjected",
    )
    assert llm_kwargs["hf_overrides"]["architectures"] == ["LlavaExampleNemotronProjected"]
    assert llm_kwargs["hf_overrides"]["mm_image_num_tokens"] == 49


def test_encoder_hf_overrides_reject_conflicts():
    with pytest.raises(ValueError, match="conflicting vLLM HF override"):
        _merge_encoder_hf_overrides(
            {"hf_overrides": {"mm_image_num_tokens": 49}},
            {"mm_image_num_tokens": 50},
            "LlavaExampleNemotronProjected",
        )


# ---------------------------------------------------------------------------
# Opt-in trainable projector (mm_projector_trainable: true)
# ---------------------------------------------------------------------------


def _policy_config(*, trainable=None, ownership=None, dp=1, ep=1, tp=1):
    overrides = {}
    if trainable is not None:
        overrides["mm_projector_trainable"] = trainable
    if ownership is not None:
        overrides["mm_projector_ownership"] = ownership
    return {
        "hf_config_overrides": overrides,
        "dtensor_cfg": {
            "tensor_parallel_size": tp,
            "context_parallel_size": 1,
            "expert_parallel_size": ep,
            "dp_replicate_size": dp,
        },
    }


def test_resolve_projector_trainability_reads_the_host_contract():
    frozen = resolve_projector_trainability(_policy_config(trainable=False, ownership="sidecar"))
    assert frozen == (False, "sidecar")
    trainable = resolve_projector_trainability(_policy_config(trainable=True, ownership="module"))
    assert trainable == (True, "module")


def test_resolve_projector_trainability_fails_closed_on_an_absent_key():
    # An absent key must not silently inherit either contract.
    with pytest.raises(ValueError, match="mm_projector_trainable must be set explicitly"):
        resolve_projector_trainability(_policy_config())


def test_resolve_projector_trainability_rejects_a_trainable_sidecar():
    with pytest.raises(ValueError, match="requires mm_projector_ownership: module"):
        resolve_projector_trainability(_policy_config(trainable=True, ownership="sidecar"))
    # Absent ownership defaults to sidecar, which is equally rejected.
    with pytest.raises(ValueError, match="requires mm_projector_ownership: module"):
        resolve_projector_trainability(_policy_config(trainable=True))


def test_resolve_projector_trainability_rejects_non_boolean_and_unknown_ownership():
    with pytest.raises(TypeError, match="must be a boolean"):
        resolve_projector_trainability(_policy_config(trainable="true", ownership="module"))
    with pytest.raises(ValueError, match="unknown mm_projector_ownership"):
        resolve_projector_trainability(_policy_config(trainable=False, ownership="hybrid"))


class _Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.mm_projector = _Projector()
        for parameter in self.mm_projector.parameters():
            parameter.requires_grad_(False)


def test_add_projector_param_group_inherits_group_zero_options():
    model = _Policy()
    optimizer = torch.optim.AdamW(
        (p for n, p in model.named_parameters() if p.requires_grad),
        lr=1.0e-6,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=10**10)],
        milestones=[],
    )
    add_projector_param_group(optimizer, model.mm_projector, scheduler=scheduler)
    assert all(p.requires_grad for p in model.mm_projector.parameters())
    assert len(optimizer.param_groups) == 2
    base, projector_group = optimizer.param_groups
    for key, value in base.items():
        if key != "params":
            assert projector_group[key] == value
    # The scheduler's bookkeeping key is mirrored, and stepping tolerates the
    # group that postdates scheduler construction.
    assert projector_group["initial_lr"] == base["initial_lr"]
    loss = (model.backbone(torch.randn(2, 4)) + model.mm_projector(torch.randn(2, 4))).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    assert model.mm_projector.fc.weight.grad is not None


def test_add_projector_param_group_applies_the_lr_override():
    model = _Policy()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1.0e-6)
    # A scheduler constructed first adds its bookkeeping key to group zero.
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=10**10)
    add_projector_param_group(optimizer, model.mm_projector, lr=1.0e-5, scheduler=scheduler)
    assert optimizer.param_groups[-1]["lr"] == 1.0e-5
    assert optimizer.param_groups[-1]["initial_lr"] == 1.0e-5


def _milestone_warmup_schedule(optimizer):
    """The U-65 production schedule: LinearLR warmup -> ConstantLR at step 3."""
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=3),
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
        ],
        milestones=[3],
    )


def test_add_projector_param_group_rejects_milestone_crossing_schedules():
    # SequentialLR.step() at the milestone invokes the sub-scheduler's closed
    # form over construction-frozen base_lrs: the step-3 strict-zip ValueError.
    model = _Policy()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1.0e-6)
    scheduler = _milestone_warmup_schedule(optimizer)
    with pytest.raises(NotImplementedError, match="U-65"):
        add_projector_param_group(optimizer, model.mm_projector, scheduler=scheduler)
    # Rejected before any mutation: the projector stays frozen and the
    # optimizer keeps its single group.
    assert all(not p.requires_grad for p in model.mm_projector.parameters())
    assert len(optimizer.param_groups) == 1


def test_milestone_schedules_break_at_the_milestone_with_a_late_group():
    # Why the admission check exists: on the pinned torch the same schedule
    # without the guard walks two steps — rescaling the projector LR from
    # 1e-5 to 6.7e-4 — and then raises the production ValueError at step 3.
    model = _Policy()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1.0e-6)
    scheduler = _milestone_warmup_schedule(optimizer)
    add_projector_param_group(optimizer, model.mm_projector, lr=1.0e-5)
    for _ in range(2):
        scheduler.step()
    assert optimizer.param_groups[-1]["lr"] == pytest.approx(6.7e-4)
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        scheduler.step()


def test_add_projector_param_group_rejects_lambda_passthrough_schedules():
    # NeMo RL's default no-scheduler config becomes a LambdaLR passthrough,
    # whose get_lr zips construction-frozen base_lrs: first-step failure.
    model = _Policy()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1.0e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    with pytest.raises(NotImplementedError, match="LambdaLR"):
        add_projector_param_group(optimizer, model.mm_projector, scheduler=scheduler)


def test_add_projector_param_group_rejects_rescaling_constant_schedules():
    # A ConstantLR factor calibrated for the base groups rescales the late
    # group too; only the factor-1.0 passthrough holds per-group LRs constant.
    model = _Policy()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1.0e-6)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5)
    with pytest.raises(NotImplementedError, match="ConstantLR"):
        add_projector_param_group(optimizer, model.mm_projector, scheduler=scheduler)


def test_add_projector_param_group_fails_closed_on_double_add():
    model = _Policy()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1.0e-6)
    add_projector_param_group(optimizer, model.mm_projector)
    with pytest.raises(RuntimeError, match="already in the optimizer"):
        add_projector_param_group(optimizer, model.mm_projector)


def test_add_projector_param_group_rejects_an_empty_registry():
    with pytest.raises(RuntimeError, match="no parameters"):
        add_projector_param_group(torch.optim.AdamW([torch.zeros(1)], lr=1e-6), nn.Module())


def test_patch_reference_projector_state_copies_the_warm_start():
    model = _Policy()
    reference = {
        "backbone.weight": torch.zeros(4, 4),
        PROJECTOR_STATE_PREFIX + "fc.weight": torch.zeros(4, 4),
        PROJECTOR_STATE_PREFIX + "fc.bias": torch.zeros(4),
    }
    patched = patch_reference_projector_state(reference, model.mm_projector.state_dict())
    assert patched == 2
    live = model.mm_projector.state_dict()
    assert torch.equal(reference[PROJECTOR_STATE_PREFIX + "fc.weight"], live["fc.weight"])
    assert torch.equal(reference["backbone.weight"], torch.zeros(4, 4))


def test_patch_reference_projector_state_fails_closed_on_a_missing_key():
    reference = {PROJECTOR_STATE_PREFIX + "other.weight": torch.zeros(1)}
    with pytest.raises(KeyError, match="missing from the KL reference snapshot"):
        patch_reference_projector_state(reference, {"fc.weight": torch.zeros(4, 4)})


def test_patch_reference_projector_state_requires_geometry_for_shard_snapshots():
    # An expert-parallel policy mesh gives the reference snapshot rank-local
    # shards; a full anchor cannot be copied without the live shard geometry.
    reference = {PROJECTOR_STATE_PREFIX + "fc.weight": torch.zeros(2, 4)}
    with pytest.raises(ValueError, match="shard_geometry"):
        patch_reference_projector_state(reference, {"fc.weight": torch.zeros(4, 4)})


def test_relative_projector_drift_measures_against_the_anchor():
    state = {"fc.weight": torch.ones(2, 2)}
    anchor = {"fc.weight": torch.ones(2, 2) * 2}
    drift = relative_projector_drift(state, anchor)
    # |x - 2x| / |2x| = 0.5 for the aligned perturbation.
    assert drift == {"fc.weight": pytest.approx(0.5)}
    with pytest.raises(KeyError, match="missing projector tensor"):
        relative_projector_drift(state, {})


# ---------------------------------------------------------------------------
# Trained-projector checkpoint persistence
# ---------------------------------------------------------------------------


class _CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = {"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 4, "hidden_size": 8}
        self.config = SimpleNamespace(mm_projectors=[config])
        self.mm_projector = MultimodalProjector.from_config([config], output_size=6)


def test_checkpoint_projector_round_trips_inside_the_weights_path(tmp_path):
    model = _CheckpointModel()
    before = {name: tensor.clone() for name, tensor in model.mm_projector.state_dict().items()}
    for parameter in model.mm_projector.parameters():
        parameter.data.add_(1.0)  # simulate training

    destination = save_checkpoint_projector(
        model, tmp_path / "policy" / "weights", provenance={"base_model": {"repo_id": "x"}}
    )
    assert destination == tmp_path / "policy" / "weights" / "mm_projector"
    assert (destination / MANIFEST_FILENAME).is_file()

    # A fresh model (a resumed worker's freshly-built host) restores bit-exact.
    resumed = _CheckpointModel()
    assert restore_checkpoint_projector(resumed, tmp_path / "policy" / "weights") is True
    restored = resumed.mm_projector.state_dict()
    for name, tensor in model.mm_projector.state_dict().items():
        assert torch.equal(restored[name], tensor)
        assert not torch.equal(restored[name], before[name])


def test_checkpoint_projector_restore_reports_a_missing_sidecar(tmp_path):
    # A frozen-stage checkpoint tree has no sidecar; False is the caller's
    # signal to continue from the warm-start artifact.
    model = _CheckpointModel()
    (tmp_path / "policy" / "weights").mkdir(parents=True)
    assert restore_checkpoint_projector(model, tmp_path / "policy" / "weights") is False
    assert restore_checkpoint_projector(model, tmp_path / "nowhere") is False


@pytest.mark.parametrize("ep,tp", [(3, 2), (2, 4), (8, 3)])
def test_fractional_generation_mesh(ep, tp):
    """Reject fractional external DP before generation workers rendezvous."""
    with pytest.raises(NotImplementedError, match="divisible"):
        require_generation_topology(
            {
                "colocated": {"enabled": True},
                "vllm_cfg": {"expert_parallel_size": ep, "tensor_parallel_size": tp},
                "vllm_kwargs": {"async_scheduling": False},
            }
        )


@pytest.mark.parametrize("trainable", [False, True])
def test_module_refit_guards(trainable):
    """Unsupported transports must exclude frozen module-owned state too."""
    pytest.importorskip("nemo_rl")
    pytest.importorskip("nemo_automodel")
    from nemotron_stitch.nemo_rl.policy import EncoderDTensorPolicyWorkerV2

    implementation = EncoderDTensorPolicyWorkerV2.__ray_metadata__.modified_class
    worker = SimpleNamespace(_projector_ownership="module", _projector_trainable=trainable)
    with pytest.raises(NotImplementedError, match="SGLang"):
        implementation.update_weights_to_sglang_colocated(worker)
    with pytest.raises(NotImplementedError, match="checkpoint-engine"):
        implementation._checkpoint_engine_params(worker)
