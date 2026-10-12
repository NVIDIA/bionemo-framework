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

"""NeMo RL vLLM worker for the generic encoder modality.

The application's vLLM plugin registration and architecture name are config-driven (``mm_plugin_callback``,
``mm_architecture``) so the package holds no modality names.
"""

from __future__ import annotations

from typing import Any


class ResidentVllmLifecycle:
    """Retain vLLM only after its startup sleep and first complete wake (U-9)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ready = False

    @property
    def should_sleep(self) -> bool:
        return not self.enabled or not self.ready

    @property
    def should_wake(self) -> bool:
        return not self.enabled or not self.ready

    def mark_wake_complete(self, tags) -> None:
        if self.enabled and (tags is None or "kv_cache" in tags):
            self.ready = True


def require_generation_topology(config: dict[str, Any]) -> None:
    """Keep encoder constraints while NeMo RL owns expert/data parallel ranks.

    NeMo RL 4d969c93 configures external vLLM data-parallel actors when
    EP exceeds TP. Replicated projected features need no separate transport
    for that layout. Pipeline partitioning remains unsupported here.
    """
    vllm = config.get("vllm_cfg") or {}
    dimensions = {
        name: int(vllm.get(name, 1))
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size")
    }
    if any(value < 1 for value in dimensions.values()):
        raise ValueError(f"parallel dimensions must be positive: {dimensions}")
    if dimensions["pipeline_parallel_size"] != 1:
        raise NotImplementedError(f"encoder generation does not support vLLM pipeline parallelism: {dimensions}")
    if config.get("keep_vllm_on_gpu"):
        if config.get("colocated", {}).get("enabled") is not True:
            raise ValueError("keep_vllm_on_gpu requires colocated.enabled=true")
        if vllm.get("async_engine"):
            raise ValueError("keep_vllm_on_gpu is incompatible with vllm_cfg.async_engine")
    # U-70: vLLM 0.25.1 async scheduling stalls between request completion
    # and the dummy-batch DP collectives on colocated external-DP engines
    # (coordinate_batch_across_dp -> _synchronize_dp_ranks -> gloo
    # all_reduce, surfacing as an 1800 s gloo recv timeout). Require the
    # explicit synchronous pin so configs cannot inherit the vLLM default.
    # Delete when the pinned vLLM fixes the completion -> dummy-batch
    # collective stall at fused DP/EP>1.
    ep = dimensions["expert_parallel_size"]
    tp = dimensions["tensor_parallel_size"]
    if ep > 1 and ep % tp:
        raise NotImplementedError(
            f"encoder generation requires expert parallel size to be divisible by tensor parallel size: {dimensions}"
        )
    external_dp_width = ep // tp
    if external_dp_width > 1 and (config.get("colocated") or {}).get("enabled") is True:
        vllm_kwargs = config.get("vllm_kwargs") or {}
        if vllm_kwargs.get("async_scheduling") is not False:
            raise ValueError(
                "colocated generation with external DP width "
                f"{external_dp_width} must pin "
                "generation.vllm_kwargs.async_scheduling: false (U-70); got "
                f"{vllm_kwargs.get('async_scheduling')!r}"
            )


def _merge_encoder_hf_overrides(
    llm_kwargs: dict[str, Any],
    configured: dict[str, Any],
    architecture: str,
) -> None:
    """Restore encoder overrides without replacing upstream quantization.

    ``architectures`` is the one key with a cross-backend contract: the config
    carries the training-side class there (the AutoModel-registered host), and
    the vLLM engine must instead name the plugin-registered rollout class. It
    is translated, not merged. Every other key fails closed on conflict.
    """
    hf_overrides = llm_kwargs.setdefault("hf_overrides", {})
    if not isinstance(hf_overrides, dict):
        raise TypeError("vLLM hf_overrides must be a mapping")
    translated = {**configured, "architectures": [architecture]}
    for name, value in translated.items():
        existing = hf_overrides.get(name, value)
        if existing != value and name != "architectures":
            raise ValueError(f"conflicting vLLM HF override for {name}: {existing!r} != {value!r}")
        hf_overrides[name] = value


def _select_vllm_executor_backend(llm_kwargs: dict[str, Any], requested: str | None) -> None:
    """Apply an explicit vLLM executor after NeMo RL selects its default.

    U-38: NeMo RL 4d969c93268fda1687fed8ca38668da4087ce76b overwrites the
    public vLLM kwarg with Ray for multi-rank model parallelism. Some inference
    artifacts were qualified with vLLM's local multiprocessing executor, whose
    collective and kernel launch order can be numerically significant for long
    greedy trajectories. Delete this override when NeMo RL exposes an explicit
    executor selection.
    """
    if requested is None:
        return
    if requested not in {"mp", "ray"}:
        raise ValueError(f"generation.vllm_executor_backend must be 'mp' or 'ray', got {requested!r}")
    llm_kwargs["distributed_executor_backend"] = requested


try:
    import ray
    from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    @ray.remote(runtime_env={**get_nsight_config_if_pattern_matches("encoder_vllm_generation_worker")})
    class EncoderVllmGenerationWorker(VllmGenerationWorkerImpl):
        def __init__(self, config, *args, **kwargs):
            from nemotron_stitch.nemo_rl.transport import resolve_callback

            plugin_callback = config.get("mm_plugin_callback")
            if not plugin_callback:
                raise ValueError("generation.mm_plugin_callback is required")
            resolve_callback(str(plugin_callback))()
            require_generation_topology(config)
            # GRPO deliberately copies policy.hf_config_overrides over vLLM's
            # hf_overrides immediately before constructing the generation
            # worker.  The training-side architecture is an AutoModel class,
            # whereas vLLM needs the architecture registered by our plugin.
            # Own that backend translation here instead of making either
            # implementation pretend to be the other.
            architecture = config.get("mm_architecture")
            if not architecture:
                raise ValueError("generation.mm_architecture is required")
            self.mm_architecture = str(architecture)
            configured_hf_overrides = (config.get("vllm_kwargs") or {}).get("hf_overrides", {})
            if not isinstance(configured_hf_overrides, dict):
                raise TypeError("generation.vllm_kwargs.hf_overrides must be a mapping")
            if "quantization_config" in configured_hf_overrides:
                raise ValueError("encoder generation leaves quantization_config to NeMo RL")
            self.encoder_hf_overrides = dict(configured_hf_overrides)
            requested_executor = config.get("vllm_executor_backend")
            self.vllm_executor_backend = None if requested_executor is None else str(requested_executor)
            _select_vllm_executor_backend({}, self.vllm_executor_backend)
            self.keep_vllm_on_gpu = bool(config.get("keep_vllm_on_gpu", False))
            self._resident_lifecycle = ResidentVllmLifecycle(self.keep_vllm_on_gpu)

            super().__init__(config, *args, **kwargs)

        def _create_engine(self, llm_kwargs):
            # U-18: NeMo RL 4d969c93268fda1687fed8ca38668da4087ce76b accepts
            # generation.vllm_kwargs.hf_overrides, but GRPO replaces them with
            # the training policy's architecture. Apply only the rollout
            # architecture at engine construction while preserving upstream's
            # quantization config. Delete this after GRPO merges user overrides.
            _merge_encoder_hf_overrides(
                llm_kwargs,
                self.encoder_hf_overrides,
                self.mm_architecture,
            )
            _select_vllm_executor_backend(llm_kwargs, self.vllm_executor_backend)
            return super()._create_engine(llm_kwargs)

        def sleep(self):
            # U-9: NeMo RL e983576 unconditionally sleeps colocated vLLM after
            # every phase and exposes no lifecycle policy callback. Delete this
            # override when upstream supports a colocated residency policy.
            if self._resident_lifecycle.should_sleep:
                return super().sleep()

            # The GB300 has enough HBM for the BF16 training policy and vLLM
            # engine together. Invalidate caches after a refit without moving
            # ~59 GiB of inference weights through host memory every step.
            if self.llm is None:
                raise RuntimeError("cannot retain an uninitialized vLLM engine")
            self.llm.llm_engine.reset_prefix_cache()
            self.llm.reset_mm_cache()
            return True

        def wake_up(self, **kwargs):
            # Pair the sleep override while preserving the mandatory first wake,
            # when vLLM materializes weights and KV allocations.
            if self.llm is None:
                raise RuntimeError("cannot wake an uninitialized vLLM engine")
            if self._resident_lifecycle.should_wake:
                result = super().wake_up(**kwargs)
                tags = kwargs.get("tags")
                self._resident_lifecycle.mark_wake_complete(tags)
                return result
            return True

except ImportError:
    EncoderVllmGenerationWorker = None
