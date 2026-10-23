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

"""DTensor policy worker bridge for arbitrary encoder payloads.

Compact-PEFT warm start uses NeMo RL's ``dtensor_cfg.lora_cfg.restore_from``
path; this module keeps the package's fuller provenance gate over the
donor adapter. The consumer's architecture registration is config-driven
(``policy.model_registry_callback``) so the package holds no modality names.

The projector is frozen by default. ``policy.hf_config_overrides
.mm_projector_trainable: true`` opts into training it during GRPO: the projector
must then be module-owned (``mm_projector_ownership: module``), and the worker
re-enables gradients and places it in the policy optimizer as one param
group. The helpers below (``resolve_projector_trainability``, ``add_projector_param_group``
and kin) are framework-free and CPU-testable without NeMo RL installed.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import ConstantLR, SequentialLR

from nemotron_stitch.automodel.model import materialize_mm_projector
from nemotron_stitch.contracts import (
    MANIFEST_FILENAME,
    OWNERSHIP_MODES,
    OWNERSHIP_MODULE,
    OWNERSHIP_SIDECAR,
)
from nemotron_stitch.nemo_rl.transport import resolve_callback
from nemotron_stitch.projector.artifact import (
    load_projector_artifact,
    read_projector_artifact,
    save_projector_artifact,
)
from nemotron_stitch.provenance import require_provenance


def validate_initial_adapter_provenance(source: str | Path, expected_config: dict[str, Any]) -> None:
    """Check the donor adapter against the locked provenance before warm start.

    NeMo RL 4d969c93 loads ``dtensor_cfg.lora_cfg.restore_from`` inside policy
    setup through the model-family state-dict adapter, validating the donor's
    rank, scaling, and base model. The package's provenance lock can carry more
    than those three fields, and the upstream PEFT load is non-strict by the
    time this worker could inspect the result, so the check runs here — at
    construction, before any placement group is allocated.
    """
    if not expected_config:
        raise ValueError("initial_adapter_expected_config must not be empty")
    source = Path(source)
    adapter_dir = next(
        (candidate for candidate in (source, source / "model") if (candidate / "adapter_model.safetensors").is_file()),
        None,
    )
    if adapter_dir is None:
        raise FileNotFoundError(
            f"dtensor_cfg.lora_cfg.restore_from={str(source)!r}: no adapter_model.safetensors "
            "found there or in its 'model' subdirectory"
        )
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"SFT adapter config is missing: {config_path}")
    require_provenance(
        json.loads(config_path.read_text()),
        expected_config,
        context="SFT LoRA adapter",
    )


def require_replicated_projector_policy_config(config: dict[str, Any]) -> None:
    """Keep sidecar features local while upstream owns FSDP and expert sharding.

    Frozen projector replicas consume complete hidden vectors on each policy
    rank. Tensor/context parallel feature partitioning is not implemented;
    data and expert meshes leave this modality boundary local and unchanged.
    """
    dtensor = config.get("dtensor_cfg") or {}
    dimensions = {
        name: int(dtensor.get(name, 1))
        for name in (
            "tensor_parallel_size",
            "context_parallel_size",
            "expert_parallel_size",
            "dp_replicate_size",
        )
    }
    if any(value < 1 for value in dimensions.values()):
        raise ValueError(f"policy mesh dimensions must be positive: {dimensions}")
    if dimensions["tensor_parallel_size"] != 1 or dimensions["context_parallel_size"] != 1:
        raise NotImplementedError(f"encoder policy requires TP=1 and CP=1: {dimensions}")
    if dimensions["dp_replicate_size"] > 1 and dimensions["expert_parallel_size"] > 1:
        # U-66: NeMo RL b03da0f4's setup_distributed derives
        # dp_size = world_size // (tp*cp*ep) and requires dp_replicate |
        # dp_size, so a fused expert-parallel mesh leaves no dense DP axis to
        # replicate over and every rank dies in setup before this package's
        # projector machinery runs; on meshes with leftover DP ranks the
        # combined mesh has never been qualified for the projector contract
        # . Reject the combination at admission instead of deep in setup.
        # Delete when U-66 closes and the combined mesh is qualified.
        raise NotImplementedError(
            "dp_replicate_size > 1 with expert_parallel_size > 1 is not admitted: dense "
            "replication inside fused expert-parallel ranks is inexpressible on this stack "
            "(U-66), and the combined mesh is unqualified for the projector contract. "
            f"Mesh dimensions: {dimensions}"
        )
    if config.get("keep_policy_on_gpu") and dtensor.get("cpu_offload"):
        raise ValueError("keep_policy_on_gpu is incompatible with dtensor_cfg.cpu_offload")


#: State-dict FQN prefix of the module-owned projector registry on the host.
PROJECTOR_STATE_PREFIX = "mm_projector."

#: Per-checkpoint directory (inside NeMo RL's ``weights_path``) holding the
#: trained projector as the package's portable artifact. AutoModel's family
#: state-dict adapter excludes projector state from the model DCP save (U-47),
#: so the trained tensors need their own checksum-verified home.
PROJECTOR_CHECKPOINT_DIRNAME = "mm_projector"


def resolve_projector_trainability(config: dict[str, Any]) -> tuple[bool, str]:
    """Read the run's projector trainability and ownership from the policy config.

    Both keys live in ``policy.hf_config_overrides`` — the same mapping the host
    consumes at construction — so the worker and the model cannot disagree.
    The trainability key must be set explicitly: an absent key fails closed
    instead of inheriting either default, because the two contracts differ in
    what the optimizer trains.
    """
    overrides = config.get("hf_config_overrides") or {}
    trainable = overrides.get("mm_projector_trainable")
    if trainable is None:
        raise ValueError(
            "policy.hf_config_overrides.mm_projector_trainable must be set explicitly: "
            "false keeps the frozen-projector GRPO contract, true trains the projector"
        )
    if not isinstance(trainable, bool):
        raise TypeError(f"mm_projector_trainable must be a boolean, got {trainable!r}")
    ownership = str(overrides.get("mm_projector_ownership") or OWNERSHIP_SIDECAR)
    if ownership not in OWNERSHIP_MODES:
        raise ValueError(f"unknown mm_projector_ownership: {ownership!r} (expected one of {sorted(OWNERSHIP_MODES)})")
    if trainable and ownership != OWNERSHIP_MODULE:
        raise ValueError(
            "a trainable projector requires mm_projector_ownership: module. Sidecar "
            "parameters live outside the module tree, so torch DCP's "
            "get_optimizer_state_dict cannot map them into the optimizer state save "
            "(KeyError) and FSDP2 never manages their gradients. See the sidecar "
            "module-owned projector trainability contract."
        )
    return trainable, ownership


def require_schedule_safe_for_late_param_group(scheduler: Any) -> None:
    """Admit only LR schedules that hold a post-construction param group honest.

    NeMo RL b03da0f469cdb2882bda83c0f04317c6159ed266 constructs the policy
    optimizer and its config-driven scheduler inside the policy worker's
    ``__init__``, before any worker-extension body runs, so each scheduler
    freezes its bookkeeping (``base_lrs``, lambda lists) over the param groups
    that exist at construction. Torch 2.11 then breaks on a group added
    later: every closed-form path consumes the frozen ``base_lrs`` (a
    ``SequentialLR`` crossing a milestone calls the sub-scheduler's
    ``_update_lr(0)`` -> ``_get_closed_form_lr``, the strict-zip ``ValueError``
    of the U-65 production crash), and NeMo RL's default no-scheduler
    passthrough is a ``LambdaLR`` whose ``get_lr`` zips the same frozen lists
    and fails on the first step. Even the per-group schedules that survive
    stepping rescale every group by the factor calibrated for the base groups,
    so the demo warmup multiplied the projector LR from 1e-5 to 6.7e-4 on its
    way to the crash. The one schedule this stack holds honest is the
    LLaVA-qualified passthrough — ``ConstantLR(factor=1.0)``, directly or as
    the sole sub-scheduler of a milestone-free ``SequentialLR`` — verified on
    the pinned torch to keep per-group LRs constant across steps and a
    ``state_dict`` resume round-trip. Fail closed here, at construction,
    instead of at a milestone of a GPU run. Delete after adopting U-65.
    """
    if scheduler is None:
        return
    active = scheduler
    if isinstance(active, SequentialLR):
        # Torch 2.11 exposes no public milestone or sub-scheduler accessors;
        # the state_dict carries the milestones under a private key.
        milestones = list(active.state_dict()["_milestones"])
        if milestones:
            raise NotImplementedError(
                "a milestone-crossing LR schedule cannot accompany a late-added param group "
                f"on this stack (milestones={milestones}): SequentialLR.step() at the milestone "
                "invokes the sub-scheduler's closed form over construction-frozen base_lrs and "
                "raises ValueError (U-65). Use the qualified constant-only passthrough: "
                "scheduler: [{name: ConstantLR, kwargs: {factor: 1.0}}, {milestones: []}]."
            )
        subs = getattr(active, "_schedulers", None)
        active = subs[0] if subs else None
    if not (isinstance(active, ConstantLR) and float(active.factor) == 1.0):
        name = type(active).__name__ if active is not None else "<no active sub-scheduler>"
        raise NotImplementedError(
            f"the {name} schedule is not admitted for a late-added param group on this stack "
            "(U-65): torch 2.11 freezes scheduler bookkeeping over the groups present at "
            "construction, so closed-form paths strict-zip against stale base_lrs (NeMo RL's "
            "default LambdaLR passthrough fails on the first step), and per-group schedules "
            "rescale the late group by the factor calibrated for the base groups. Use the "
            "qualified constant-only passthrough: ConstantLR(factor=1.0)."
        )


def add_projector_param_group(
    optimizer: torch.optim.Optimizer,
    projector: torch.nn.Module,
    *,
    lr: float | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Re-enable projector gradients and append one param group to the optimizer.

    AutoModel 1814c6c9 applies PEFT's global freeze after FSDP2 wrapping
    (``infrastructure.py``, "Freeze parameters after checkpoint loading and
    parallelization"), and NeMo RL builds the optimizer over the surviving
    trainable parameters afterwards — so the projector misses both. FSDP2 saw
    ``requires_grad=True`` at wrap time (the host honors mm_projector_trainable
    at construction), so re-enabling the flag re-enters the state FSDP2 wrapped
    for; the freeze that unset it was itself a post-wrap flag flip. The new
    group inherits the optimizer's group-zero options verbatim, with an
    optional lr override; scheduler bookkeeping keys (``initial_lr``) are
    mirrored when the earlier groups carry them.

    When ``scheduler`` is supplied (the worker passes the scheduler NeMo RL
    built over this optimizer), it must be the qualified constant-only
    passthrough — see :func:`require_schedule_safe_for_late_param_group`
    (U-65). The check runs before any mutation, so a rejected schedule leaves
    the projector frozen and the optimizer untouched.
    """
    parameters = list(projector.parameters())
    if not parameters:
        raise RuntimeError("projector registry has no parameters")
    existing = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if any(id(parameter) in existing for parameter in parameters):
        raise RuntimeError("projector parameters are already in the optimizer")
    if scheduler is not None:
        require_schedule_safe_for_late_param_group(scheduler)
    for parameter in parameters:
        parameter.requires_grad_(True)
    options = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    if lr is not None:
        options["lr"] = float(lr)
    if "initial_lr" in options:
        options["initial_lr"] = options["lr"]
    group: dict[str, Any] = {"params": parameters, **options}
    optimizer.add_param_group(group)
    return group


def patch_reference_projector_state(
    reference_state_dict: dict[str, Any],
    projector_state: dict[str, torch.Tensor],
    *,
    prefix: str = PROJECTOR_STATE_PREFIX,
    shard_geometry: dict[str, Any] | None = None,
) -> int:
    """Copy the warm-started projector into the KL reference state snapshot.

    NeMo RL 4d969c93 captures ``setup_reference_model_state(self.model)`` — a
    CPU copy of the model state dict — before this worker loads the projector
    artifact, so module-owned projector entries hold construction-time random
    values. Without this patch the reference policy would score rollouts under
    embeddings it never had, and a KL penalty could not see projector drift at
    all. Sidecar-owned projectors never appear in the model state dict, so the
    caller skips this entirely on that path.

    Under an expert-parallel mesh the snapshot holds rank-local shards — NeMo
    RL copies it through ``to_local_if_dtensor`` — so a full warm-start tensor
    cannot be copied directly. ``shard_geometry`` supplies a live DTensor per
    projector tensor whose mesh and placements define this rank's shard of
    the anchor; it is required whenever the snapshot shapes disagree with the
    anchor's, and the mismatch otherwise fails closed.
    """
    patched = 0
    for name, tensor in projector_state.items():
        key = prefix + name
        target = reference_state_dict.get(key)
        if target is None:
            raise KeyError(f"module-owned projector state {key!r} is missing from the KL reference snapshot")
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"KL reference snapshot entry {key!r} is not a tensor")
        if tuple(tensor.shape) != tuple(target.shape):
            live = (shard_geometry or {}).get(name)
            if live is None or not hasattr(live, "placements"):
                raise ValueError(
                    f"the KL reference snapshot holds a rank-local projector shard for {key!r} "
                    "(expert-parallel policy mesh); shard_geometry must supply a live DTensor "
                    "per projector tensor"
                )
            from torch.distributed.tensor import distribute_tensor

            tensor = distribute_tensor(
                tensor.to(live.device),
                device_mesh=live.device_mesh,
                placements=live.placements,
            ).to_local()
        target.copy_(tensor.detach().cpu())
        patched += 1
    return patched


def relative_projector_drift(
    projector_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per-tensor relative L2 drift of the projector against a fixed state.

    The trainable-projector KL watch: NeMo RL reports the KL term, but that
    number hides *which* part of the policy moved. This attribute-free signal
    (no autograd graph, no device assumptions) lets the worker report drift
    per training phase, matching how the earlier projector+LoRA runs surfaced
    divergence spikes through the KL metric alone.
    """
    drift: dict[str, float] = {}
    for name, tensor in projector_state.items():
        reference = reference_state.get(name)
        if reference is None:
            raise KeyError(f"drift reference is missing projector tensor {name!r}")
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        # The live projector rides the training device while the anchor stays
        # on CPU; compare on the live device so the drift report never mixes
        # devices (the anchor itself is not mutated).
        reference = reference.to(device=tensor.device, dtype=tensor.dtype)
        denominator = reference.norm().clamp_min(1e-12)
        drift[name] = float(((tensor.detach() - reference).norm() / denominator).item())
    return drift


def save_checkpoint_projector(
    model: Any,
    weights_path: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Persist the trained projector beside a policy checkpoint.

    Writes the package's portable artifact into ``weights_path/mm_projector``
    (DTensor-safe full-tensor collection, rank-zero write). The projector's
    optimizer moments ride upstream's DCP optimizer save, keyed by FQN — the
    weights themselves cannot, because the model DCP save routes through the
    state-dict adapter that excludes projector state (U-47).
    """
    destination = Path(weights_path) / PROJECTOR_CHECKPOINT_DIRNAME
    return save_projector_artifact(model, destination, provenance=provenance)


def restore_checkpoint_projector(model: Any, weights_path: str | Path) -> bool:
    """Restore a checkpointed projector into the model, if the checkpoint has one.

    Key-exact and checksum-verified like the warm start, but without the
    warm-start provenance gate: a resumed run restores whatever that run's
    own checkpoints recorded. Returns False when the checkpoint carries no
    projector sidecar (a frozen-stage checkpoint resumed into the trainable
    variant) — the caller then continues the projector from the warm-start
    artifact, which is its correct state, since that projector never trained.
    """
    sidecar = Path(weights_path) / PROJECTOR_CHECKPOINT_DIRNAME
    if not (sidecar / MANIFEST_FILENAME).is_file():
        return False
    load_projector_artifact(model, sidecar)
    return True


try:
    import ray
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2Impl

    @ray.remote(runtime_env=get_runtime_env_for_policy_worker("encoder_dtensor_policy_worker_v2"))
    class EncoderDTensorPolicyWorkerV2(DTensorPolicyWorkerV2Impl):
        """Add projector warm-start provenance and encoder payloads to NeMo RL's policy worker.

        NeMo RL 4d969c93 owns the compact-PEFT warm start and the worker
        extension seam; this subclass adds the package's donor-adapter
        provenance gate, the projector warm start (frozen by default; see
        ``resolve_projector_trainability``), and external encoder payload
        handling.
        """

        def __init__(self, *args, **kwargs):
            positional = list(args)
            config = kwargs.get("config") or (positional[0] if positional else None)
            if config is None:
                raise TypeError("encoder policy worker requires a config mapping")
            registry_callback = config.get("model_registry_callback")
            if not registry_callback:
                raise ValueError("policy.model_registry_callback is required")
            resolve_callback(str(registry_callback))()
            projector_trainable, projector_ownership = resolve_projector_trainability(config)
            require_replicated_projector_policy_config(config)
            # The warm start itself is upstream's — NeMo RL loads
            # dtensor_cfg.lora_cfg.restore_from inside super().__init__() and
            # ignores it when resuming a checkpoint. Only the provenance gate
            # is ours, and it must run before that load.
            initial_adapter = (config.get("dtensor_cfg") or {}).get("lora_cfg", {}).get("restore_from")
            supplied_weights = kwargs.get("weights_path")
            if len(positional) > 1:
                supplied_weights = positional[1]
            initial_sft_adapter = bool(initial_adapter) and supplied_weights is None
            if initial_sft_adapter:
                validate_initial_adapter_provenance(
                    initial_adapter,
                    config.get("initial_adapter_expected_config") or {},
                )
            # Fail closed on the one resume path whose ordering this worker
            # cannot control: without a reference model (reference_policy_kl_
            # penalty == 0) NeMo RL loads the checkpoint inside
            # super().__init__() before this body can create the projector's
            # optimizer group, so the trainable projector's Adam moments
            # restore against an optimizer that has no group for them. The
            # deferred path (any KL penalty > 0) routes through load_checkpoint
            # below instead; reject the unsupported combination outright.
            init_reference_model = kwargs.get("init_reference_model")
            if init_reference_model is None and len(positional) > 4:
                init_reference_model = positional[4]
            if init_reference_model is None:
                init_reference_model = True
            if projector_trainable and supplied_weights is not None and not init_reference_model:
                raise ValueError(
                    "a trainable projector requires the deferred checkpoint load, i.e. a "
                    "nonzero loss_fn.reference_policy_kl_penalty; set mm_projector_trainable: "
                    "false for a KL-free GRPO resume"
                )
            self.keep_policy_on_gpu = bool(config.get("keep_policy_on_gpu", False))
            # Stash the projector contract before super().__init__(): the
            # deferred resume load inside it dispatches to this class's
            # load_checkpoint override, which needs all of these before the
            # body below runs.
            artifact = config.get("projector_artifact_path")
            if not artifact:
                raise ValueError("policy.projector_artifact_path is required")
            projector_group_lr = config.get("projector_lr")
            if projector_group_lr is not None:
                projector_group_lr = float(projector_group_lr)
            self._projector_trainable = projector_trainable
            self._projector_ownership = projector_ownership
            self._projector_artifact_path = Path(artifact)
            self._projector_expected_provenance = config.get("projector_expected_provenance")
            self._projector_group_lr = projector_group_lr
            self._projector_group_added = False
            self._projector_checkpoint_weights_path: Path | None = None
            self._projector_warm_provenance: dict[str, Any] = {}
            self._projector_warm_state = None
            super().__init__(*positional, **kwargs)

            expected_provenance = self._projector_expected_provenance
            anchor_state: dict[str, torch.Tensor] = {}
            if projector_ownership == OWNERSHIP_MODULE:
                # Module ownership: the host registered the projector before
                # AutoModel's meta materialization, so the host's
                # initialize_weights has already placed and reset it and the
                # base-checkpoint load skipped its keys (the state-dict
                # adapter excludes them). One decision point owns the
                # projector's live weights: a resume restores the checkpointed
                # projector when that run wrote one, and every other start —
                # fresh, or a frozen-stage tree resumed into this variant —
                # warm-starts from the SFT artifact, which is exactly the
                # untrained projector's state.
                restored_from_checkpoint = supplied_weights is not None and restore_checkpoint_projector(
                    self.model, supplied_weights
                )
                if restored_from_checkpoint:
                    # The KL reference and the drift report stay anchored to
                    # the warm-start artifact on every path — the same
                    # anchor-across-resumes policy NeMo RL applies to the
                    # language model (it defers the checkpoint load until
                    # after capturing the reference from base weights).
                    warm = read_projector_artifact(self._projector_artifact_path)
                    anchor_state = warm.state
                    self._projector_warm_provenance = dict(warm.manifest.provenance or {})
                else:
                    warm_manifest = load_projector_artifact(
                        self.model,
                        self._projector_artifact_path,
                        expected=dict(expected_provenance) if expected_provenance else None,
                    )
                    self._projector_warm_provenance = dict(warm_manifest.provenance or {})
                    anchor_state = self.model.mm_projector.state_dict()
                if supplied_weights is not None and self.rank == 0:
                    payload = {"projector_state": "checkpoint" if restored_from_checkpoint else "warm_start"}
                    if restored_from_checkpoint:
                        payload["weights_path"] = str(supplied_weights)
                    else:
                        payload["reason"] = "checkpoint has no trained-projector sidecar"
                    print("ENCODER_POLICY_PROJECTOR_RESUME " + json.dumps(payload, sort_keys=True), flush=True)
                if self.reference_model_state_dict is not None:
                    patch_reference_projector_state(
                        self.reference_model_state_dict,
                        anchor_state,
                        shard_geometry=self.model.mm_projector.state_dict(),
                    )
            else:
                # Sidecar ownership: the projector is held
                # outside the module tree, so it stays on meta after model
                # construction. Materialize it before the load and move it to
                # the model device.
                materialize_mm_projector(self.model)
                load_projector_artifact(
                    self.model,
                    self._projector_artifact_path,
                    expected=dict(expected_provenance) if expected_provenance else None,
                )
            projector_parameters = list(self.model.mm_projector.parameters())
            if not projector_parameters:
                raise RuntimeError("encoder policy has no projector parameters")
            self._ensure_projector_param_group()
            # Fail closed on contract violations instead of auditing softly:
            # the projector's trainability must match the configured contract
            # exactly, and nothing outside LoRA and the projector may train.
            observed_trainable = any(parameter.requires_grad for parameter in projector_parameters)
            if observed_trainable != projector_trainable:
                raise RuntimeError(
                    f"projector trainability {observed_trainable} does not match the "
                    f"configured mm_projector_trainable={projector_trainable}"
                )
            optimizer_ids = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
            projector_in_optimizer = sum(id(parameter) in optimizer_ids for parameter in projector_parameters)
            if projector_trainable and projector_in_optimizer != len(projector_parameters):
                raise RuntimeError("the trainable projector did not enter the GRPO optimizer")
            if not projector_trainable and projector_in_optimizer:
                raise RuntimeError("projector parameters entered the GRPO optimizer")
            unexpected_trainable = [
                name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and "lora_" not in name and not name.startswith(PROJECTOR_STATE_PREFIX)
            ]
            if unexpected_trainable:
                raise RuntimeError(
                    f"policy has trainable parameters outside LoRA and the projector: {unexpected_trainable[:4]}"
                )
            # KL watch: anchor the per-phase drift report to the warm-started
            # projector — the same state the patched KL reference scores —
            # not to the previous step. The anchor is a full-tensor CPU copy:
            # a module-owned projector rides FSDP2, so state-dict values can be
            # DTensors whose .cpu() clone would keep the local shard.
            if projector_trainable:
                self._projector_warm_state = {
                    name: (tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor).detach().cpu().clone()
                    for name, tensor in anchor_state.items()
                }
            trainable = [
                (name, parameter.numel())
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            if self.rank == 0:
                print(
                    "ENCODER_POLICY_AUDIT "
                    + json.dumps(
                        {
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "trainable_parameters": sum(size for _, size in trainable),
                            "lora_tensors": sum("lora_" in name for name, _ in trainable),
                            "projector_trainable_tensors": sum(
                                parameter.requires_grad for parameter in projector_parameters
                            ),
                            "projector_trainable": projector_trainable,
                            "projector_ownership": projector_ownership,
                            "projector_optimizer_lr": (
                                (projector_group_lr or self.optimizer.param_groups[-1]["lr"])
                                if projector_trainable
                                else None
                            ),
                            "initial_sft_adapter": initial_sft_adapter,
                            "keep_policy_on_gpu": self.keep_policy_on_gpu,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        def _ensure_projector_param_group(self) -> None:
            """Place the trainable projector in the optimizer, once.

            Idempotent because two paths need the group: the deferred resume
            load inside super().__init__ dispatches to this class's
            load_checkpoint override before the body runs, and a fresh start
            adds it from the body. The optimizer-state restore maps moments by
            FQN, so the group must exist before super().load_checkpoint.
            """
            if not self._projector_trainable or self._projector_group_added:
                return
            # NeMo RL always builds a scheduler over this optimizer (the
            # no-config default is a LambdaLR passthrough, which breaks at the
            # first step with a late group — U-65), so hand it to the admission
            # check rather than assuming None is safe.
            add_projector_param_group(
                self.optimizer,
                self.model.mm_projector,
                lr=self._projector_group_lr,
                scheduler=self.scheduler,
            )
            self._projector_group_added = True

        def _report_projector_drift(self, stage: str) -> None:
            """Log projector drift against the warm start once per training phase.

            The drift computation is collective at expert-parallel and
            HSDP meshes — relative_projector_drift full-tensors the sharded
            live projector — so every rank must run it; only rank zero
            prints. Gating the whole method on rank zero instead would leave
            rank zero issuing all-gathers the other ranks never enqueue, and
            the mesh hangs.

            The same collectives double as the replica-sync guarantee: with
            the identical artifact warm start and HSDP's cross-replica
            gradient all-reduce, every rank's projector is the same tensor,
            so every rank's drift is identical. The all-gather below fails
            the step closed on any divergence — a replica that trained on
            different weights than it rolls out with is exactly the silent
            wrongness this package refuses.
            """
            if self._projector_warm_state is None:
                return
            drift = relative_projector_drift(
                self.model.mm_projector.state_dict(),
                self._projector_warm_state,
            )
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                names = sorted(drift)
                local = torch.tensor([drift[name] for name in names], dtype=torch.float64, device="cuda")
                gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered, local)
                for peer_rank, peer in enumerate(gathered):
                    if (peer - local).abs().max().item() > 1e-6:
                        raise RuntimeError(
                            "trainable projector replicas diverged before the update: rank 0 drift "
                            f"{[drift[name] for name in names]} vs rank {peer_rank} drift "
                            f"{peer.tolist()}; the HSDP cross-replica gradient reduction is not "
                            "synchronizing the projector — fail closed rather than training "
                            "replicas on different weights"
                        )
            if self.rank != 0:
                return
            print(
                "ENCODER_PROJECTOR_DRIFT " + json.dumps({"stage": stage, "relative_l2": drift}, sort_keys=True),
                flush=True,
            )

        def _report_resident_memory(self, stage: str) -> None:
            if self.rank != 0:
                return
            gib = 1024**3
            print(
                "ENCODER_POLICY_MEMORY "
                + json.dumps(
                    {
                        "allocated_gib": round(torch.cuda.memory_allocated() / gib, 3),
                        "reserved_gib": round(torch.cuda.memory_reserved() / gib, 3),
                        "stage": stage,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        def prepare_for_lp_inference(self, *args, **kwargs) -> None:
            # U-9: NeMo RL e983576 always follows its offload lifecycle. This
            # opt-in GB300 path keeps the policy resident because policy + vLLM
            # fit together. Delete these overrides when upstream supports a
            # colocated residency policy. af0a11f added keep_train_buffers;
            # e983576's signature lacks it,
            # so accept and forward variadically like prepare_for_training.
            if not self.keep_policy_on_gpu:
                return super().prepare_for_lp_inference(*args, **kwargs)

            keep_train_buffers = bool(kwargs.get("keep_train_buffers", False))
            self.model = self.move_buffer_to_device(self.model, "cuda")
            self.model.eval()
            torch.randn(1, device="cuda")
            if self.optimizer is not None and self.offload_optimizer_for_logprob and not keep_train_buffers:
                self.move_optimizer_to_device("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            self._report_resident_memory("logprob")

        def prepare_for_training(self, *args, **kwargs) -> None:
            # Symmetric override for the same resident-policy lifecycle; all
            # other configurations delegate unchanged to NeMo RL. The drift
            # report runs on every path so the trainable-projector KL watch
            # does not depend on the residency policy.
            self._report_projector_drift("before_update")
            if not self.keep_policy_on_gpu:
                return super().prepare_for_training(*args, **kwargs)

            self.model = self.move_buffer_to_device(self.model, "cuda")
            self.model.train()
            if self.optimizer is not None and not self.cpu_offload:
                self.move_optimizer_to_device("cuda")
            torch.cuda.empty_cache()
            self._report_resident_memory("training")

        @torch.no_grad()
        def offload_after_refit(self) -> None:
            if not self.keep_policy_on_gpu:
                return super().offload_after_refit()

            # A GB300 can hold the BF16 policy beside the colocated vLLM engine.
            # Keep the expensive base parameters resident and move only optimizer
            # state while vLLM owns the compute phase.
            self.model = self.move_buffer_to_device(self.model, "cuda")
            self.model.eval()
            self.offload_before_refit()
            self._report_resident_memory("after_refit")

        def _refit_weight_generator(self):
            """Yield refit payload tensors without the module-owned projector.

            The rollout engine never owns projector weights: it receives
            already-projected soft tokens, so a projector tensor in the refit
            payload can only fail the generation-side load. NeMo RL
            b03da0f469cdb2882bda83c0f04317c6159ed266 skips only LoRA tensors in
            its refit enumeration, and a remote-code host without a family
            state-dict adapter has no projector exclusion that would filter
            these keys earlier. Delete after adopting U-61.
            """
            from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import dtensor_params_generator

            for name, tensor in dtensor_params_generator(self.model, self.dtype):
                if name.startswith(PROJECTOR_STATE_PREFIX):
                    continue
                yield name, tensor

        @torch.no_grad()
        def prepare_refit_info(self, *, refit_payload_mode: Any = "hf_export"):
            # U-61: NeMo RL b03da0f4 exposes no refit exclusion for co-trained
            # state outside LoRA, so a module-owned projector would enter the
            # refit manifest and fail the generation-side load. The manifest
            # and both refit streams below must agree on the same key set.
            # Delete this override when an upstream exclusion seam lands.
            info = super().prepare_refit_info(refit_payload_mode=refit_payload_mode)
            return {name: value for name, value in info.items() if not name.startswith(PROJECTOR_STATE_PREFIX)}

        @torch.no_grad()
        def stream_weights_via_ipc_zmq(
            self,
            buffer_size_bytes: int = 0,
            kv_scales: dict[str, float] | None = None,
        ) -> None:
            # U-61: colocated IPC refit must stream exactly the manifest's
            # keys, so the projector is excluded here too. The body mirrors
            # the pinned base method with only the generator swapped.
            if kv_scales is not None:
                raise NotImplementedError(
                    "FP8 kvcache is not currently supported for DTensor path, we will support it in the future."
                )

            self.maybe_init_zmq()
            # Manually move model to cuda for cpu offload case
            if self.cpu_offload:
                self.model = self.move_to_cuda(self.model)

            from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

            stream_weights_via_ipc_zmq_impl(
                params_generator=self._refit_weight_generator(),
                buffer_size_bytes=buffer_size_bytes,
                zmq_socket=self.zmq_socket,
                rank=self.rank,
                worker_name=str(self),
            )

        def _broadcast_weights_for_collective(
            self,
            kv_scales: dict[str, float] | None = None,
            *,
            buffer_size_bytes: int | None = None,
            num_buffers: int | None = None,
        ) -> None:
            # U-61: the non-colocated NCCL refit. The base watchdog wrapper
            # stays upstream's; this mirrors only the pinned broadcast body
            # with the iterator filtered like the IPC stream above.
            if kv_scales is not None:
                raise NotImplementedError(
                    "FP8 kvcache is not currently supported for DTensor path, we will support it in the future."
                )

            # Manually move model to cuda for cpu offload case
            if self.cpu_offload:
                print(
                    "[WARNING]: Unless you are lacking of memory, it is not recommended to enable cpu_offload when "
                    "using non-colocated generation since it will have an extra onload and offload at refit stage."
                )
                self.model = self.move_to_cuda(self.model)

            from nemo_rl.utils.packed_tensor import packed_broadcast_producer

            packed_broadcast_producer(
                iterator=self._refit_weight_generator(),
                group=self.model_update_group,
                src=0,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=buffer_size_bytes,
                num_buffers=num_buffers,
            )

            # Manually move model to cpu for cpu offload case
            # cpu offload needs model on CPU before model forward
            if self.cpu_offload:
                self.model = self.move_to_cpu(self.model)

        def update_weights_to_sglang_colocated(self, *args: Any, **kwargs: Any) -> None:
            # U-61: SGLang's colocated refit is a fourth enumeration site
            # upstream (dtensor_policy_worker_v2.py, dtensor_params_generator)
            # that this variant does not cover with the projector exclusion.
            # Fail closed rather than streaming tensors the rollout engine
            # cannot own. Delete this guard with the U-61 overrides.
            if self._projector_ownership == OWNERSHIP_MODULE:
                raise NotImplementedError(
                    "the module-owned projector is excluded from vLLM refit payloads (U-61); "
                    "the SGLang backend is not covered"
                )
            return super().update_weights_to_sglang_colocated(*args, **kwargs)

        def _checkpoint_engine_params(self):
            # U-61: the checkpoint-engine refit transport enumerates the same
            # payload without the projector exclusion. Unreachable under this
            # variant's vLLM transport; fail closed if selected anyway.
            # Delete this guard with the U-61 overrides.
            if self._projector_ownership == OWNERSHIP_MODULE:
                raise NotImplementedError(
                    "the module-owned projector is excluded from vLLM refit payloads (U-61); "
                    "the checkpoint-engine refit transport is not covered"
                )
            return super()._checkpoint_engine_params()

        def save_checkpoint(
            self,
            weights_path: str,
            optimizer_path: str | None = None,
            tokenizer_path: str | None = None,
            checkpointing_cfg: dict[str, Any] | None = None,
        ) -> None:
            """Save the policy checkpoint; the trained projector lands at finalize time.

            The projector's Adam moments ride upstream's DCP optimizer save
            (keyed by FQN); its weights cannot, because the model DCP save
            routes through the state-dict adapter that excludes projector
            state (U-47). The portable artifact is written beside the shards
            in finalize_async_save — after the async DCP writer completes and
            before the caller renames tmp_step_N, so the two never race, a
            failed write aborts the rename, and no checkpoint is published
            claiming a projector it does not carry.
            """
            super().save_checkpoint(
                weights_path,
                optimizer_path=optimizer_path,
                tokenizer_path=tokenizer_path,
                checkpointing_cfg=checkpointing_cfg,
            )
            if self._projector_trainable:
                self._projector_checkpoint_weights_path = Path(weights_path)

        def finalize_async_save(self) -> None:
            """Write the trained projector beside the shards once the DCP save lands.

            NeMo RL b03da0f4 stages DCP writes in the background (is_async
            defaults true) and defers the tmp_step_N -> step_N rename behind
            this method, on every save path. Writing the sidecar here — after
            the blocking finalize, before the rename — is the only placement
            that is both ordered against the DCP writer and safe for the
            collective full-tensor collection inside save_projector_artifact;
            writing it in save_checkpoint instead would leave a tmp_step_N
            holding an authoritative-looking projector with no shards when
            an async finalize fails (U-52's failure mode).
            """
            super().finalize_async_save()
            pending = self._projector_checkpoint_weights_path
            self._projector_checkpoint_weights_path = None
            if pending is not None:
                save_checkpoint_projector(
                    self.model,
                    pending,
                    provenance=self._projector_warm_provenance,
                )

        def load_checkpoint(self, weights_path: str, optimizer_path: str | None = None) -> None:
            """Ensure the projector's optimizer group exists before the state restore.

            NeMo RL maps optimizer moments by FQN during the restore, and the
            trainable projector's group postdates the optimizer's construction
            (AutoModel's PEFT freeze ran in between). This override also serves
            the deferred resume load inside super().__init__ — NeMo RL defers it
            until after capturing the KL reference — which dispatches here
            before the worker body runs. Projector weight restoration is owned
            by the body's single decision point, not this override, so a
            non-deferred resume (no reference model) restores identically.
            """
            self._ensure_projector_param_group()
            super().load_checkpoint(weights_path, optimizer_path)

except ImportError:
    # The helpers above are framework-free and remain importable without
    # NeMo RL; only the worker class requires the try-block imports.
    EncoderDTensorPolicyWorkerV2 = None
