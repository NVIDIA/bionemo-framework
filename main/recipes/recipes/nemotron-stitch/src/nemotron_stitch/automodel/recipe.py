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

"""ProjectorRecipeMixin: topology guards, projector materialization, artifact lifecycle.

DTensor-safe codec calls give both projector ownership modes the same
warm-start and export lifecycle.

The mixin carries the pieces that are framework seams rather than modality
behaviour.:class:`ProjectorFinetuneRecipe` adds the shared module-owned,
typed-optimizer wiring used by feature-backed applications; consumers with a
different optimizer surface can still compose the mixin directly.

The mixin is cooperative: it expects to sit left of an AutoModel recipe class
(``class ConsumerRecipe(ProjectorRecipeMixin, FinetuneRecipeForVLM)``) and
reads the AutoModel recipe surface — ``self.cfg``, ``self.dist_env``,
``self.model_parts``, ``self.pp_enabled``, ``self._get_cp_group_size()``.
Sidecar ownership is qualified for exactly one rank; module
ownership  is FSDP2-sharded and multi-rank.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

from nemotron_stitch.automodel.model import materialize_mm_projector
from nemotron_stitch.automodel.optim import ProjectorAdamWConfig
from nemotron_stitch.contracts import OWNERSHIP_MODULE, OWNERSHIP_SIDECAR
from nemotron_stitch.projector.artifact import load_projector_artifact, save_projector_artifact
from nemotron_stitch.projector.trainability import TrainabilityPolicy, collect_extra_state


class ProjectorSidecarState:
    """Non-``nn.Module`` stateful wrapper putting the sidecar projector under AutoModel's recipe-state tracker.

    A sidecar-held projector lives outside the module tree, so AutoModel's
    model/optimizer checkpointing never serializes it. ``BaseRecipe``
    (AutoModel 24b47e856263d313b942f0ed666c63fff83306b4) already tracks any
    assigned object with callable ``state_dict()``/``load_state_dict()`` and
    carries it through periodic and final checkpoints (``<attribute>.pt`` on
    the coordinator) and native resume. The wrapper must stay a
    plain object: an ``nn.Module`` would be
    routed into the model save path instead, and the attribute name must not
    contain ``val``/``eval``/``test``/``loss``, which the tracker skips.

    ``BaseRecipe.load_checkpoint`` runs inside ``setup()``, before the mixin
    can materialize the sidecar projector (it is on meta until then), so a
    resumed ``load_state_dict`` records the tensors and :meth:`bind` applies
    them once the projector exists.
    """

    def __init__(self) -> None:
        self._projector = None
        self._pending: dict[str, torch.Tensor] | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._projector is None:
            raise RuntimeError("projector sidecar state was saved before setup bound the projector")
        # CPU copies keep the checkpoint device-agnostic for resume; sidecar
        # ownership is one-rank, so the coordinator write sees full tensors.
        return {name: tensor.detach().cpu() for name, tensor in self._projector.state_dict().items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if self._projector is None:
            self._pending = dict(state)
            return
        self._projector.load_state_dict(state)

    def bind(self, projector) -> bool:
        """Bind the materialized projector; returns True if resumed state was applied."""
        if self._projector is not None:
            raise RuntimeError("projector sidecar state is already bound")
        self._projector = projector
        if self._pending is None:
            return False
        projector.load_state_dict(self._pending)
        self._pending = None
        return True


def require_one_rank(world_size: int, seam: str) -> None:
    if int(world_size) != 1:
        raise NotImplementedError(f"{seam} is qualified for exactly one rank")


def plain_mapping(value) -> dict:
    # omegaconf is a training-environment dependency, not a base one; degrade
    # gracefully so the package's CI env (no omegaconf) can exercise the
    # plain-mapping paths.
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        OmegaConf = None

    if OmegaConf is not None and OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True))
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value)


def resolve_resume_artifact(
    *,
    checkpoint_dir: str | Path,
    restore_from: str | None,
) -> Path | None:
    """Locate the portable sidecar corresponding to AutoModel native resume."""
    root = Path(checkpoint_dir)
    if restore_from and restore_from.upper() != "LATEST":
        checkpoint = Path(restore_from)
        if checkpoint.parent == Path("."):
            checkpoint = root / checkpoint
    else:
        checkpoint = root / "LATEST"
    artifact = checkpoint / "artifact"
    return artifact.resolve() if artifact.is_dir() else None


def resolve_initialization_artifact(
    *,
    checkpoint_dir: str | Path | None,
    restore_from: str | None,
    initial_artifact: str | Path | None,
    stage1_artifact: str | Path | None,
    stage: int,
    stage1_artifact_key: str = "stage1_artifact",
) -> Path | None:
    """Resolve resume, fresh warm-start, and stage-2 bridge precedence.

    ``checkpoint_dir=None`` disables native-resume discovery (a config without
    a checkpoint section cannot hold a resume artifact).
    """
    resumed = None
    if checkpoint_dir is not None:
        resumed = resolve_resume_artifact(
            checkpoint_dir=checkpoint_dir,
            restore_from=restore_from,
        )
    if resumed is not None:
        return resumed
    if initial_artifact:
        source = Path(initial_artifact)
        if not source.is_dir():
            raise FileNotFoundError(f"initialization artifact is absent: {source}")
        return source.resolve()
    if int(stage) == 2:
        if not stage1_artifact:
            raise FileNotFoundError(f"Stage 2 requires {stage1_artifact_key}")
        source = Path(stage1_artifact)
        if not source.is_dir():
            raise FileNotFoundError(f"Stage 2 requires a Stage 1 artifact: {source}")
        return source.resolve()
    return None


class ProjectorRecipeMixin:
    """AutoModel recipe mixin owning the topology guards and sidecar lifecycle.

    Owns the PP/CP fail-closed guards (both ownership modes), and in sidecar
    mode additionally the one-rank topology boundary, projector device
    placement (the projector is project state, not a registered submodule),
    the checkpoint-tracked resume state (``ProjectorSidecarState``), the
    initial sidecar load, and the portable sidecar export. Optimizer
    construction is the application's seam: AutoModel revisions can have
    different optimizer-config surfaces.

    ``projector.trainability_extra_patterns`` names the consumer's extra
    family (learned marker embeddings and kin — in-tree parameters, so
    AutoModel's own checkpointing covers them). The mixin transports that
    family through the portable sidecar: warm starts restore it and exports
    carry it, in both ownership modes.
    """

    #: Sidecar ownership (default) holds the projector outside the module tree
    #: and is qualified for exactly one rank; module ownership makes it an
    #: FSDP2-sharded registered submodule and supports multi-rank TP/EP via
    #: nemotron_stitch.automodel.parallel.
    projector_ownership: str = OWNERSHIP_SIDECAR

    # Framework-surface declarations: the AutoModel recipe base this mixin is
    # composed with provides these. Annotation-only — nothing is assigned, so
    # MRO resolution still reaches the base class.
    cfg: Any  # the recipe ConfigNode
    model_parts: list[Any]
    dist_env: Any  # world_size / device
    pp_enabled: bool
    _get_cp_group_size: Callable[[], int]

    def setup(self):
        if self.projector_ownership == OWNERSHIP_SIDECAR:
            # Attach before super().setup(): BaseRecipe.__setattr__ tracks the
            # wrapper, and the base setup's terminal load_checkpoint restores
            # any checkpointed sidecar state into it (recorded as pending
            # until the projector is materialized below). Reassigning the
            # attribute after tracking would trip the tracker's duplicate-key
            # guard, so bind() mutates the wrapper in place.
            self.projector_sidecar_state = ProjectorSidecarState()
        # super() is the AutoModel recipe base; the checker cannot see it from
        # the mixin, so the two cooperative calls below are cast.
        result = cast(Any, super()).setup()
        if self.projector_ownership == OWNERSHIP_SIDECAR:
            require_one_rank(self.dist_env.world_size, "multimodal AutoModel training")
        elif self.projector_ownership != OWNERSHIP_MODULE:
            raise ValueError(f"unknown projector ownership mode {self.projector_ownership!r}")
        # PP/CP guards live in setup so unsupported distributed modes fail
        # before any training step rather than halfway through a microbatch.
        # The projector registry carries a flat token-index contract with no
        # per-stage payload sharder.
        if self.pp_enabled:
            raise NotImplementedError(
                "external encoder payloads with pipeline parallelism need a microbatch payload sharder"
            )
        if self._get_cp_group_size() > 1:
            raise NotImplementedError(
                "external encoder payloads with context parallelism need scatter-before-shard support"
            )
        # U-15: AutoModel 24b47e8 can leave custom MoE buffers (e.g. Nemotron V3's
        # e_score_correction_bias) on CPU while parameters land on CUDA: the
        # single-rank path skips parallelization, and the PEFT path forces the
        # post-shard init load ("load-before-shard" is explicitly skipped for
        # PEFT). Both modes need the sweep, so it runs before the ownership
        # branches; move only buffers rather than traversing all parameters.
        # Delete when upstream places persistent buffers consistently.
        for buffer in self.model_parts[0].buffers():
            if buffer.device != self.dist_env.device:
                torch.utils.swap_tensors(buffer, buffer.to(self.dist_env.device))
        if self.projector_ownership == OWNERSHIP_MODULE:
            # FSDP2 owns placement and materialization for a registered
            # submodule; the mixin owns the DTensor-safe warm start. Resume
            # takes precedence over an explicit initialization artifact, which
            # takes precedence over the stage-2 bridge.
            projector_cfg = self.cfg.get("projector", {})
            checkpoint = self.cfg.get("checkpoint", {})
            # Native module-mode checkpoints already restore projector tensors
            # with the model. Do not let an explicit warm start overwrite them.
            source = None
            if checkpoint.get("restore_from"):
                logging.info("Native checkpoint resume restores module-owned projector state")
            else:
                source = resolve_initialization_artifact(
                    checkpoint_dir=checkpoint.get("checkpoint_dir"),
                    restore_from=None,
                    initial_artifact=projector_cfg.get("initial_artifact"),
                    stage1_artifact=projector_cfg.get("stage1_artifact"),
                    stage=int(self.cfg.model.get("mm_training_stage", 1)),
                )
            if source is not None:
                expected = projector_cfg.get("expected_provenance")
                manifest = load_projector_artifact(
                    self.model_parts[0],
                    source,
                    expected=plain_mapping(expected) if expected else None,
                    extra_patterns=tuple(projector_cfg.get("trainability_extra_patterns", ())),
                )
                logging.info("Loaded initial projector artifact from %s (stage %s)", source, manifest.stage)
            return result
        model = self.model_parts[0]
        if getattr(model, "mm_projector", None) is None:
            raise TypeError(f"multimodal model was not constructed: {type(model).__name__}")
        # The projector is owned as project state outside the module tree
        # , so AutoModel's device placement
        # never moves it; it stays on meta. Bring it to the model device and
        # initialize it (from meta) before the sidecar load and any forward or
        # optimizer use. Parameters are materialized in place, so an optimizer
        # already constructed over them stays valid.
        materialize_mm_projector(model)
        # Resume state beats a fresh warm start: native resume already
        # restored the projector through the tracked wrapper.
        resumed = self.projector_sidecar_state.bind(model.mm_projector)
        projector_cfg = self.cfg.get("projector", {})
        if resumed:
            logging.info("Restored sidecar projector state from the native checkpoint")
        else:
            checkpoint = self.cfg.get("checkpoint", {})
            source = resolve_initialization_artifact(
                checkpoint_dir=checkpoint.get("checkpoint_dir"),
                restore_from=checkpoint.get("restore_from"),
                initial_artifact=projector_cfg.get("initial_artifact"),
                stage1_artifact=projector_cfg.get("stage1_artifact"),
                stage=int(self.cfg.model.get("mm_training_stage", 1)),
            )
            if source is None:
                return result
            expected = projector_cfg.get("expected_provenance")
            manifest = load_projector_artifact(
                model,
                source,
                expected=plain_mapping(expected) if expected else None,
                extra_patterns=tuple(projector_cfg.get("trainability_extra_patterns", ())),
            )
            logging.info("Loaded initial projector sidecar from %s (stage %s)", source, manifest.stage)
        return result

    def run_train_validation_loop(self):
        result = cast(Any, super()).run_train_validation_loop()
        projector_cfg = self.cfg.get("projector", {})
        artifact_dir = projector_cfg.get("artifact_dir")
        if not artifact_dir:
            return result
        # AutoModel 24b47e856263d313b942f0ed666c63fff83306b4 checkpoints
        # PEFT/base state only and offers no rank-aware external-artifact
        # export callback, so the recipe exports it here. The codec is
        # DTensor-safe and rank-aware: module ownership collects full tensors
        # on every rank and rank zero writes; sidecar ownership stays
        # qualified for exactly one rank.
        if self.projector_ownership == OWNERSHIP_SIDECAR:
            require_one_rank(self.dist_env.world_size, "projector sidecar export")
        provenance = plain_mapping(projector_cfg.get("provenance", {}))
        # The extra family rides the sidecar whether or not this stage trains
        # it: a frozen marker trained in an earlier stage must still
        # round-trip. Patterns that match nothing are a config error, not an
        # empty export.
        extra_patterns = tuple(projector_cfg.get("trainability_extra_patterns", ()))
        extra_state = None
        if extra_patterns:
            extra_state = collect_extra_state(self.model_parts[0], TrainabilityPolicy(extra_patterns=extra_patterns))
            if not extra_state:
                raise ValueError(
                    f"projector.trainability_extra_patterns matched no parameters: {sorted(extra_patterns)}"
                )
        save_projector_artifact(
            self.model_parts[0],
            Path(artifact_dir),
            provenance=provenance,
            stage=int(self.cfg.model.get("mm_training_stage", 1)),
            extra_state=extra_state,
        )
        return result


# Both fixed-geometry consumers use AutoModel r0.6.0's typed optimizer config
# and previously carried the same recipe subclass. Keep the framework import
# lazy so cache preparation and CPU-only application imports do not require
# AutoModel.
TrainFinetuneRecipeForNextTokenPrediction: Any
try:
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
except ModuleNotFoundError as error:
    _AUTOMODEL_IMPORT_ERROR = error

    class TrainFinetuneRecipeForNextTokenPrediction:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "nemo_automodel is required for projector fine-tuning"
            ) from _AUTOMODEL_IMPORT_ERROR


def _application_callback(fqn: str) -> Callable[[], Any]:
    module_name, separator, attribute = fqn.rpartition(".")
    if not separator:
        raise ValueError(f"callback must be a fully qualified name: {fqn!r}")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError(f"configured callback is not callable: {fqn}")
    return callback


class ProjectorFinetuneRecipe(ProjectorRecipeMixin, TrainFinetuneRecipeForNextTokenPrediction):
    """Shared module-owned alignment/SFT recipe for feature-backed applications.

    The ``projector`` config names the application registration callback and
    explicitly states whether this stage trains the projector. Everything
    else is the package lifecycle in :class:`ProjectorRecipeMixin` plus
    AutoModel's upstream next-token recipe.
    """

    projector_ownership = OWNERSHIP_MODULE

    def setup(self):
        projector_cfg = self.cfg.get("projector", {})
        callback_fqn = projector_cfg.get("model_registry_callback")
        if not callback_fqn:
            raise ValueError("projector.model_registry_callback is required")
        _application_callback(str(callback_fqn))()

        if "train_projector" not in projector_cfg:
            raise ValueError("projector.train_projector must be set explicitly for each stage")
        optimizer_config = self.cfg.optimizer
        if not isinstance(optimizer_config, ProjectorAdamWConfig):
            raise TypeError(
                "point optimizer._target_ at "
                "nemotron_stitch.automodel.optim.ProjectorAdamWConfig "
                f"(got {type(optimizer_config).__name__})"
            )
        optimizer_config.policy = TrainabilityPolicy(
            projector_patterns=tuple(projector_cfg.get("trainability_projector_patterns", ("mm_projector",))),
            extra_patterns=tuple(projector_cfg.get("trainability_extra_patterns", ())),
            decoder_patterns=tuple(projector_cfg.get("trainability_decoder_patterns", ())),
        )
        optimizer_config.stage = int(self.cfg.model.get("mm_training_stage", 1))
        # ConfigNode has .get/__contains__ but no __getitem__; the guard above
        # makes .get total here.
        optimizer_config.train_projector = bool(projector_cfg.get("train_projector"))
        optimizer_config.train_extra = bool(projector_cfg.get("train_extra", False))
        optimizer_config.train_decoder = bool(projector_cfg.get("train_decoder", False))
        optimizer_config.manifest_path = projector_cfg.get("trainability_manifest")

        result = super().setup()
        registry = getattr(self.model_parts[0], "mm_projector", None)
        if registry is None or registry.projector_ownership != self.projector_ownership:
            raise TypeError(
                "the host was not constructed with module-owned projector wiring; check model.mm_projector_ownership"
            )
        return result


class ProjectorEvaluationRecipe(ProjectorFinetuneRecipe):
    """Evaluate a restored projector artifact without taking an optimizer step.

    AutoModel 0.6.1 exposes validation only as a private method inside its
    fine-tuning recipe. Keeping this thin lifecycle adapter in Stitch prevents
    every modality example from copying the same dependency-version-sensitive
    loop. The surrounding setup deliberately remains AutoModel's ordinary
    setup: it constructs the model, restores the configured projector, and
    materializes validation loaders exactly as training does.

    ``evaluation.output_path`` is optional. When present, rank zero writes a
    compact JSON summary suitable for qualification comparisons. All ranks
    still execute every validation batch because model parallel collectives
    require them to advance together.

    This class is a temporary compatibility seam for the pinned AutoModel
    revision. Remove it in favor of an upstream public evaluation entry point
    once one is available; the remaining requirement for an unused training
    loader and optimizer is tracked in ``docs/upstream-gaps.md``.
    """

    def run_train_validation_loop(self) -> dict[str, dict[str, Any]]:
        """Run each configured validation loader once and return plain metrics.

        This intentionally does not call the parent training loop: even a
        one-step scheduler would mutate the artifact before measuring it.
        Logger and checkpointer cleanup mirrors AutoModel's training lifecycle
        so evaluation is safe to invoke from its standard CLI.
        """
        results: dict[str, dict[str, Any]] = {}
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            if not self.val_dataloaders:
                raise ValueError("at least one validation dataloader is required")
            for name, dataloader in self.val_dataloaders.items():
                log_data = self._run_validation_epoch(dataloader)
                self.log_val_metrics(name, log_data, self.metric_logger_valid[name])
                results[name] = {
                    key: value.item() if isinstance(value, torch.Tensor) else value
                    for key, value in log_data.metrics.items()
                }
                results[name]["num_batches"] = len(dataloader)
        finally:
            self.metric_logger_train.close()
            for metric_logger in self.metric_logger_valid.values():
                metric_logger.close()
            self._finalize_and_close_checkpointer()
            graph_manager = getattr(self, "partial_cuda_graph_manager", None)
            if graph_manager is not None:
                graph_manager.close()
                self.partial_cuda_graph_manager = None
            self._partial_cuda_graph_capture_pending = False

        if self.dist_env.is_main:
            payload = {
                "elapsed_seconds": time.perf_counter() - started,
                "evaluation_only": True,
                "optimizer_steps": 0,
                "validation": results,
            }
            output_path = self.cfg.get("evaluation.output_path")
            if output_path:
                destination = Path(output_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"NEMOTRON_STITCH_EVALUATION_RESULT={json.dumps(payload, sort_keys=True)}", flush=True)
        return results
