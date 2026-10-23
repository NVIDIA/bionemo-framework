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

"""The multimodal host contract: input mixin, ownership modes, host factory.

One home for the three pieces a host model needs:
``MultimodalInputMixin`` carries the flat forward-kwarg contract and the
cooperative embedding-boundary forward bridge; ``_init_mm_projector``
constructs the projector registry in either ownership mode; and
``build_multimodal_host`` decorates a dynamically resolved HF causal-LM class
into a registered architecture without copying any of the base model's
constructor, forward body, generation, or checkpoint code.

The host model class
provides ``self.config`` and a text embedder — ``self.language_model`` whose
``get_input_embeddings()`` is used, falling back to the host itself so a plain
``*ForCausalLM`` host also works.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import cache
from typing import Any, cast

import torch

from nemotron_stitch.contracts import (
    MM_FEATURES_PREFIX,
    MM_PLACEHOLDER_TOKEN_IDS_KEY,
    MM_TOKEN_INDICES_PREFIX,
    OWNERSHIP_MODES,
    OWNERSHIP_MODULE,
    OWNERSHIP_SIDECAR,
)
from nemotron_stitch.lm_config import resolve_lm_hidden_size
from nemotron_stitch.projector.multimodal import MultimodalProjector


@cache
def _projector_state_adapter_cls(base_cls: type, excluded_prefixes: tuple[str, ...]) -> type:
    from nemotron_stitch.automodel.checkpoint import ProjectorStateDictAdapterMixin

    return type(
        f"ProjectorAware{base_cls.__name__}",
        (ProjectorStateDictAdapterMixin, base_cls),
        {"PROJECTOR_STATE_PREFIXES": excluded_prefixes},
    )


def _exclude_project_state_from_base_checkpoint(
    model: Any,
    config: Any,
    additional_parameter_prefixes: tuple[str, ...],
) -> None:
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is None:
        return
    # AutoModel 1814c6c93a66b9d59d254960ef6a99a64249b671 has
    # allow_checkpoint_key_subset in Checkpointer.load_model, but its public
    # base-model load never forwards it. Reuse the family adapter it does call
    # and omit freshly initialized project-owned parameters from that load.
    # On a pin containing ac44d92f, first replace this load behavior with the
    # public skip_task_head_prefixes_for_base_model config; retain composition
    # only if consolidated export/refit still needs the exclusion.
    excluded_prefixes = ("mm_projector.", *additional_parameter_prefixes)
    adapter_cls = _projector_state_adapter_cls(type(adapter), excluded_prefixes)
    wrapped = cast(Any, adapter_cls).__new__(adapter_cls)
    wrapped.__dict__.update(vars(adapter))
    wrapped.config = config
    wrapped.set_include_projector_state(False)
    model.state_dict_adapter = wrapped


class MultimodalInputMixin:
    """Flat per-projector feature fields as the model's forward contract.

    Both AutoModel's ``filter_forward_kwargs`` and NeMo RL's
    ``_accepted_forward_kwargs`` pass the batch through unchanged when the
    model's forward declares ``**kwargs``, so
    ``mm_features__<name>`` fields reach the model with no upstream change.
    """

    mm_projector: MultimodalProjector
    # Provided by the HF host class at decoration time (build_multimodal_host).
    # Annotation-only: nothing is assigned, so MRO still reaches the host.
    config: Any
    get_input_embeddings: Callable[..., Any]

    def _init_mm_projector(self, config: Any, *, ownership: str = OWNERSHIP_SIDECAR) -> None:
        if ownership not in OWNERSHIP_MODES:
            raise ValueError(
                f"unknown projector ownership mode: {ownership!r} (expected one of {sorted(OWNERSHIP_MODES)})"
            )
        projector_configs = getattr(config, "mm_projectors", None)
        if not projector_configs:
            raise ValueError("multimodal model config must define at least one entry in mm_projectors")
        # One resolver for both the training and serving paths (P7): the two
        # must not be able to disagree about the width of the same model.
        output_size = resolve_lm_hidden_size(config)
        projector = MultimodalProjector.from_config(
            projector_configs, output_size=output_size, projector_ownership=ownership
        )
        placeholder_ids = getattr(config, MM_PLACEHOLDER_TOKEN_IDS_KEY, None)
        if not isinstance(placeholder_ids, dict) or not placeholder_ids:
            raise ValueError(
                f"multimodal model config must define {MM_PLACEHOLDER_TOKEN_IDS_KEY} "
                "as a mapping of projector name to placeholder token id"
            )
        if set(placeholder_ids) != set(projector.projectors):
            raise ValueError(
                f"{MM_PLACEHOLDER_TOKEN_IDS_KEY} keys must equal the registered projector names: "
                f"{sorted(placeholder_ids)} != {sorted(projector.projectors)}"
            )
        token_ids = [int(token_id) for token_id in placeholder_ids.values()]
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("placeholder token ids must be distinct across projectors")
        if not bool(getattr(config, "mm_projector_trainable", True)):
            # Construction-time freeze for runtimes with no trainability policy
            # (the NeMo RL GRPO policy worker asserts a frozen projector).
            # Training stages keep the default (True); the recipe's
            # trainability policy owns requires_grad there.
            projector.requires_grad_(False)
        if ownership == OWNERSHIP_MODULE:
            # Module ownership: an ordinary
            # registered submodule, so FSDP2/DTensor wrapping, the host's
            # initialize_weights extension, and the trainability policy all
            # see it. Placement is owned by that wrapping.
            self.mm_projector = projector
            return
        # Sidecar ownership: held outside the module tree, so
        # framework state-dict conversion, weight initialization, and PEFT
        # freezing never see it. Serialization, provenance, and device
        # placement are owned by the artifact codec and the recipe/policy
        # worker. Not covered by FSDP2/DTensor wrapping — fail closed above
        # one rank.
        object.__setattr__(self, "mm_projector", projector)

    def _extract_mm_kwargs(self, kwargs: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Pop ``mm_features__*`` and ``mm_token_indices__*`` fields out of kwargs."""
        features = {
            key[len(MM_FEATURES_PREFIX) :]: kwargs.pop(key)
            for key in list(kwargs)
            if key.startswith(MM_FEATURES_PREFIX)
        }
        token_indices = {
            key[len(MM_TOKEN_INDICES_PREFIX) :]: kwargs.pop(key)
            for key in list(kwargs)
            if key.startswith(MM_TOKEN_INDICES_PREFIX)
        }
        return features, token_indices

    def _prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        features: dict[str, torch.Tensor],
        token_indices: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids are required for multimodal soft-token scatter")
        if inputs_embeds is not None:
            raise ValueError("multimodal features and caller-supplied inputs_embeds are mutually exclusive")
        placeholder_token_ids = getattr(self.config, MM_PLACEHOLDER_TOKEN_IDS_KEY, None)
        if placeholder_token_ids is None:
            raise ValueError(f"model config must define {MM_PLACEHOLDER_TOKEN_IDS_KEY}")
        embedder = getattr(self, "language_model", self)
        inputs_embeds = embedder.get_input_embeddings()(input_ids)
        return self.mm_projector(
            input_ids,
            inputs_embeds,
            features,
            placeholder_token_ids=placeholder_token_ids,
            token_indices_by_projector=token_indices or None,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args: Any,
        inputs_embeds: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> Any:
        """Cooperative embedding-boundary bridge over the base forward.

        Pops the ``mm_*`` kwargs, scatters soft tokens into the text embeds
        when features are present, and delegates to ``super().forward`` — the
        base model's forward body, loss, and checkpoint mapping stay upstream.
        A consumer with a genuinely different boundary (marker embeddings, a
        TP gradient boundary) overrides this method; that is domain code, not
        a failure of this bridge.

        NeMo AutoModel ``7d36972d`` checks for an explicit ``logits_to_keep``
        parameter before enabling its public fused-linear-CE path, but ignores
        this bridge's ``**kwargs`` support. Expose and forward the parameter so
        decorated causal-LM hosts retain that upstream capability. Delete this
        explicit bridge when U-26 is adopted.
        """
        # super() is the HF host, validated at decoration time by
        # _validate_forward_surface; the checker cannot see it from the mixin.
        features, token_indices = self._extract_mm_kwargs(kwargs)
        if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0:
            kwargs["logits_to_keep"] = logits_to_keep
        if not features:
            if token_indices:
                raise ValueError("explicit mm_token_indices__* require matching mm_features__*")
            return cast(Any, super()).forward(input_ids, *args, inputs_embeds=inputs_embeds, **kwargs)
        inputs_embeds = self._prepare_inputs_embeds(input_ids, inputs_embeds, features, token_indices)
        return cast(Any, super()).forward(None, *args, inputs_embeds=inputs_embeds, **kwargs)


def _validate_forward_surface(base_cls: type) -> None:
    """Fail closed unless the base forward accepts ``inputs_embeds`` and ``**kwargs``."""
    forward = getattr(base_cls, "forward", None)
    if forward is None:
        raise TypeError(f"{base_cls.__name__} has no forward to decorate")
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError) as error:
        raise TypeError(f"cannot inspect {base_cls.__name__}.forward: {error}") from error
    if "inputs_embeds" not in parameters:
        raise TypeError(f"{base_cls.__name__}.forward does not accept inputs_embeds")
    if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        raise TypeError(f"{base_cls.__name__}.forward does not accept **kwargs")


# Config keys the host consumes from either the config object or constructor
# kwargs (AutoModel 24b47e8's custom-model path passes non-config kwargs to
# the constructor; the NeMo RL policy worker sets config overrides instead).
_MM_CONFIG_KEYS = (
    "mm_projectors",
    MM_PLACEHOLDER_TOKEN_IDS_KEY,
    "mm_projector_ownership",
    "mm_projector_trainable",
)


def build_multimodal_host(
    base_cls: type,
    *,
    projector_ownership: str | None = None,
    architecture: str | None = None,
    additional_parameter_prefixes: tuple[str, ...] = (),
) -> type:
    """Decorate an HF causal-LM class with the multimodal input contract.

    The returned subclass adds ``MultimodalInputMixin`` (including its
    cooperative forward), constructs the projector registry from the model
    config, and nothing else: every base parameter FQN, generation method, and
    checkpoint mapping is preserved by inheritance, so a Hub-resolved
    remote-code class can be bridged without copying any of its
    implementation. The package's config keys are accepted either as config
    attributes or as constructor kwargs (both framework paths occur), with a
    conflict failing closed. Registration of the returned class (AutoModel's
    registry, Transformers' auto-mapping) is a separate, per-worker-process
    step — see ``automodel/registry.py``.

    ``projector_ownership=None`` (the default) defers the mode to the model
    config key ``mm_projector_ownership`` (default ``"sidecar"``), so one
    declarative call serves both wirings and stages differ by one config key.
    ``additional_parameter_prefixes`` names consumer-owned parameters that,
    like the module-owned projector, are initialized by the decorated model
    and absent from its base checkpoint. These are ordinary state-dict tensor
    names, not PyTorch module ``_extra_state``.
    The host always exposes ``initialize_weights``: under module ownership it
    resets the registered projector after AutoModel materializes meta tensors
    (AutoModel calls it post-materialization when present), delegating to the
    base's own ``initialize_weights`` first when one exists.
    """
    if projector_ownership is not None and projector_ownership not in OWNERSHIP_MODES:
        raise ValueError(
            f"unknown projector ownership mode: {projector_ownership!r} (expected one of {sorted(OWNERSHIP_MODES)})"
        )
    if isinstance(additional_parameter_prefixes, str) or any(
        not isinstance(prefix, str) or not prefix for prefix in additional_parameter_prefixes
    ):
        raise ValueError("additional_parameter_prefixes must contain non-empty strings")
    additional_parameter_prefixes = tuple(additional_parameter_prefixes)
    _validate_forward_surface(base_cls)
    base_initialize_weights = getattr(base_cls, "initialize_weights", None)
    if base_initialize_weights is not None and getattr(base_initialize_weights, "__module__", "").startswith(
        "transformers."
    ):
        # Transformers' generic PreTrainedModel.initialize_weights re-runs the
        # base's _init_weights, which is not DTensor-safe for all remote-code
        # models (AutoModel 24b47e8 skips initialize_weights for NemotronH by
        # architecture name — a check a decorated architecture name does not
        # match). The checkpoint load fills every base tensor; only the
        # projector needs reset. A consumer/AutoModel-native override still
        # delegates. Initializing the added projector remains the out-of-tree
        # model's responsibility.
        base_initialize_weights = None
    base_signature = inspect.signature(base_cls.__init__)
    base_accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in base_signature.parameters.values()
    )
    base_params = set(base_signature.parameters) - {"self"}

    class MultimodalHost(MultimodalInputMixin, base_cls):
        def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
            for key in _MM_CONFIG_KEYS:
                if key in kwargs:
                    value = kwargs.pop(key)
                    existing = getattr(config, key, None)
                    if existing is not None and existing != value:
                        raise ValueError(f"conflicting {key} between the model config and constructor")
                    setattr(config, key, value)
            if not base_accepts_kwargs:
                # AutoModel's custom-model path forwards loading-level kwargs
                # (trust_remote_code, local_files_only, revision) to the model
                # constructor; its own _filter_kwargs_for_init applies the same
                # signature filter we mirror here for a no-**kwargs base.
                kwargs = {key: value for key, value in kwargs.items() if key in base_params}
            super().__init__(config, *args, **kwargs)
            ownership = projector_ownership or getattr(config, "mm_projector_ownership", OWNERSHIP_SIDECAR)
            self._init_mm_projector(config, ownership=ownership)
            if ownership == OWNERSHIP_MODULE or additional_parameter_prefixes:
                _exclude_project_state_from_base_checkpoint(self, config, additional_parameter_prefixes)

        def initialize_weights(self, *args: Any, **kwargs: Any) -> Any:
            result = None
            if base_initialize_weights is not None:
                result = base_initialize_weights(self, *args, **kwargs)
            if self.mm_projector.projector_ownership == OWNERSHIP_MODULE:
                self.mm_projector.reset_parameters()
            return result

    if architecture is not None:
        MultimodalHost.__name__ = architecture
        MultimodalHost.__qualname__ = architecture
    return MultimodalHost


def build_cp1_packed_multimodal_host(
    base_cls: type,
    *,
    architecture: str,
    attn_backend: str = "flash",
    projector_ownership: str | None = None,
    additional_parameter_prefixes: tuple[str, ...] = (),
) -> type:
    """Decorate a host and route indexed packs through AutoModel's cp=1 hook.

    AutoModel's NEAT batches carry document ids in ``attention_mask`` while
    recurrent model families consume the same ids as ``_packed_seq_ids``.
    This adapter moves that public batch representation to the model-facing
    field, scopes AutoModel's varlen SDPA hook to attention modules, and leaves
    ordinary un-packed forwards unchanged. The helper is intentionally in
    Stitch so applications only select a supported packing mode; they do not
    duplicate framework wiring.
    """
    from nemo_automodel.components.distributed.blockdiag_cp import (
        attach_cp1_packed_varlen_hooks,
        configure_cp_varlen,
        cp1_packed_varlen_backend,
        disable_cp1_packed_varlen,
        enable_cp1_packed_varlen,
    )
    from nemo_automodel.components.models.common.packing import is_indexed_packed_mask

    if not architecture:
        raise ValueError("architecture must be non-empty")
    if not attn_backend:
        raise ValueError("attn_backend must be non-empty")
    host_cls = build_multimodal_host(
        base_cls,
        projector_ownership=projector_ownership,
        architecture=architecture,
        additional_parameter_prefixes=additional_parameter_prefixes,
    )

    class Cp1PackedMultimodalHost(host_cls):
        def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
            configure_cp_varlen(attn_backend=attn_backend)
            super().__init__(config, *args, **kwargs)
            attach_cp1_packed_varlen_hooks(self)

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            *args: Any,
            attention_mask: torch.Tensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs: Any,
        ) -> Any:
            disable_cp1_packed_varlen()
            packed_seq_ids = kwargs.get("_packed_seq_ids")
            if is_indexed_packed_mask(attention_mask):
                packed_seq_ids = attention_mask
            if is_indexed_packed_mask(packed_seq_ids):
                configured_backend = cp1_packed_varlen_backend()
                if configured_backend != attn_backend:
                    raise RuntimeError(
                        "indexed packed forward requires "
                        f"{attn_backend!r}, configured backend is {configured_backend!r}"
                    )
                enable_cp1_packed_varlen(packed_seq_ids, configured_backend)
                kwargs["_packed_seq_ids"] = packed_seq_ids
                attention_mask = None
            return super().forward(
                input_ids,
                *args,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

    Cp1PackedMultimodalHost.__name__ = architecture
    Cp1PackedMultimodalHost.__qualname__ = architecture
    return Cp1PackedMultimodalHost


def materialize_mm_projector(model, device=None, dtype=None) -> MultimodalProjector:
    """Place the sidecar-held projector registry on the model device and initialize it.

    The registry is held outside the module tree (sidecar ownership), so
    AutoModel's meta construction and device placement never touch it:
    its parameters remain on ``meta`` after ``from_pretrained``.
    Materialize them on the model device (defaulting to the first base
    parameter) before any forward or sidecar load.
    """
    projector = getattr(model, "mm_projector", None)
    if projector is None:
        raise TypeError("multimodal model has no sidecar-held mm_projector")
    if device is None or dtype is None:
        base_parameter = next(model.parameters())
        if device is None:
            device = base_parameter.device
        if dtype is None:
            dtype = base_parameter.dtype
    projector.to_empty(device=device)
    projector.to(dtype=dtype)
    projector.reset_parameters()
    return projector
