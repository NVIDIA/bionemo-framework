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

"""build_mm_plugin: one factory for out-of-tree projected-token vLLM modalities.

``mode="projected"`` serves pre-projected ``[T, H_lm]`` soft tokens straight
through to ``embed_multimodal`` — vLLM never loads the modality encoder.
``mode="encode"``  builds the processing
layer — data parser, processor, prompt replacement, dummy inputs — around
consumer payload callbacks and registers it against the consumer's own
composite model class, whose ``embed_multimodal`` is where the consumer's
``encode_fn`` equivalent lives: the model side of an
encode-mode modality is a vLLM-native composite (state-caching hybrid
protocols, LoRA mappings, checkpoint weight routing) that no factory should
generate.

The factory is parameterized by the modality name, architecture name, base model/processor
classes, and the sentinel config attributes. All vLLM imports stay inside the
returned ``register`` (projected) or inside the factory call (encode — the
consumer imports the built classes by name, so they must exist before
``register()`` runs) so the package imports framework-free.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SentinelAttrs:
    """HF-config attribute names carrying the modality's sentinel token strings."""

    start: str
    placeholder: str
    end: str


#: Projected-mode payload geometries. ``fixed`` (default): every request's
#: payload has exactly the configured token count. ``variable``: each
#: request's payload sizes itself, up to the configured count as maximum.
PROJECTED_GEOMETRIES = frozenset({"fixed", "variable"})


def load_token_embedding_overrides(
    artifact_path: str | Path,
    declarations: Mapping[int | str, str],
    *,
    hidden_size: int,
    tokenizer: Any | None = None,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Load explicitly declared token embeddings from artifact EXTRA state.

    String tokens are resolved through ``tokenizer`` and must encode to one
    token. Integer tokens are checked against its vocabulary when a tokenizer
    is supplied. Only declared EXTRA keys are interpreted as embeddings.
    """
    import torch

    from nemotron_stitch.projector.artifact import read_projector_artifact

    artifact = read_projector_artifact(artifact_path)
    resolved: dict[int, str] = {}
    vectors = []
    for token, extra_key in declarations.items():
        if not isinstance(extra_key, str) or not extra_key:
            raise ValueError("token embedding override EXTRA keys must be non-empty strings")
        if isinstance(token, bool) or not isinstance(token, int | str):
            raise TypeError(f"token embedding override keys must be token IDs or strings, got {token!r}")
        if isinstance(token, str):
            if not token:
                raise ValueError("token embedding override strings must be non-empty")
            if tokenizer is None:
                raise ValueError(f"a tokenizer is required to resolve token embedding override {token!r}")
            token_id = tokenizer.convert_tokens_to_ids(token)
            unknown_id = getattr(tokenizer, "unk_token_id", None)
            unknown_token = getattr(tokenizer, "unk_token", None)
            if not isinstance(token_id, int) or token_id < 0 or (token_id == unknown_id and token != unknown_token):
                raise ValueError(f"token embedding override {token!r} is not in the tokenizer vocabulary")
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if list(encoded) != [token_id]:
                raise ValueError(
                    f"token embedding override {token!r} must resolve to exactly one token, got {list(encoded)}"
                )
        else:
            token_id = token
            if token_id < 0:
                raise ValueError(f"token embedding override ID must be non-negative, got {token_id}")
            if tokenizer is not None and token_id >= len(tokenizer):
                raise ValueError(
                    f"token embedding override ID {token_id} is outside tokenizer vocabulary size {len(tokenizer)}"
                )
        if token_id in resolved:
            raise ValueError(
                f"duplicate token embedding override for token ID {token_id}: {resolved[token_id]!r} and {extra_key!r}"
            )
        tensor = artifact.extra_state.get(extra_key)
        if tensor is None:
            raise ValueError(f"projector artifact has no declared EXTRA tensor {extra_key!r}")
        if tuple(tensor.shape) != (hidden_size,):
            raise ValueError(
                f"token embedding override {extra_key!r} must have shape ({hidden_size},), got {tuple(tensor.shape)}"
            )
        resolved[token_id] = extra_key
        vectors.append(tensor)
    if not vectors:
        return (), torch.empty(0, hidden_size)
    return tuple(resolved), torch.stack(vectors)


def apply_token_embedding_overrides(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    token_ids: tuple[int, ...],
    vectors: torch.Tensor,
    *,
    mode: str = "replace",
) -> torch.Tensor:
    """Replace or add declared vectors after ordinary/MM embedding merge."""
    if mode not in ("replace", "add"):
        raise ValueError(f"unknown token embedding override mode: {mode!r}")
    result = inputs_embeds
    for index, token_id in enumerate(token_ids):
        keep = (input_ids != token_id).unsqueeze(-1)
        result = result.where(keep, vectors[index] if mode == "replace" else result + vectors[index])
    return result


def check_projected_token_count(modality: str, count: int, configured: int, *, geometry: str) -> None:
    """Validate a projected payload's token count against the configured geometry.

    ``fixed``: the count must equal the configured count exactly — a payload
    that differs is a transport bug, not a smaller input. ``variable``: the
    payload sizes the request, bounded below by one and above by the
    configured count, which remains the maximum the scheduler profiled for
    and dummy inputs are built with.
    """
    if geometry == "fixed":
        if count != configured:
            raise ValueError(f"projected {modality} payload has {count} tokens, expected {configured}")
        return
    if geometry == "variable":
        if not 1 <= count <= configured:
            raise ValueError(
                f"projected {modality} payload has {count} tokens, outside the configured maximum {configured}"
            )
        return
    raise ValueError(f"unknown projected geometry: {geometry!r}")


def _import(fqn: str) -> Any:
    module_name, separator, attribute = fqn.replace(":", ".").rpartition(".")
    if not separator:
        raise ValueError(f"expected a fully qualified 'module:Class' name, got {fqn!r}")
    return getattr(importlib.import_module(module_name), attribute)


def build_mm_plugin(
    *,
    modality: str,
    architecture: str,
    placeholder_text: str,
    mode: str = "projected",
    # projected mode
    base_model_cls: str | None = None,
    base_processing_info_cls: str | None = None,
    base_processor_cls: str | None = None,
    base_dummy_inputs_cls: str | None = None,
    sentinels: SentinelAttrs | None = None,
    num_tokens_attr: str | None = None,
    geometry: str = "fixed",
    lora_declarations_from: str | None = None,
    token_embedding_overrides: Mapping[int | str, str] | None = None,
    token_embedding_override_mode: str = "replace",
    embedding_override_artifact_attr: str = "mm_projector_artifact_path",
    # encode mode
    encode_model_cls: str | None = None,
    encode_normalize_item_fn: Callable[[Any], Any] | None = None,
    encode_geometry_fn: Callable[[Any], int] | None = None,
    encode_pack_items_fn: Callable[[Any], dict[str, Any]] | None = None,
    encode_field_configs_fn: Callable[[Any], dict[str, Any]] | None = None,
    encode_dummy_item_fn: Callable[[Any], Any] | None = None,
    encode_dummy_text: str | None = None,
    encode_dummy_fallback_text: str = "What is the answer?",
    encode_supported_mm_limits: dict[str, int | None] | None = None,
    encode_max_tokens_per_item: int | None = None,
    encode_sentinel_start: str | None = None,
    encode_sentinel_placeholder: str | None = None,
    encode_sentinel_end: str | None = None,
) -> Callable[[], None] | EncodePlugin:
    """Return an idempotent ``register()`` wiring a custom-embedding modality into vLLM.

    ``mode="projected"`` returns the register callable itself; ``mode="encode"``
    returns the built processing layer plus its registrar (``EncodePlugin``),
    because the consumer imports the encode-mode classes by name before
    ``register()`` runs.

    Projected payloads default to ``geometry="fixed"``: every request carries
    exactly the configured ``num_tokens_attr`` token count. With
    ``geometry="variable"`` each request's payload sizes itself (a ragged
    collator's per-request counts) up to the configured count as the maximum;
    the maximum remains the profiling budget and dummy-input size.

    ``token_embedding_overrides`` explicitly maps a token ID or single-token
    string to an EXTRA-state tensor name in the projector artifact. The
    artifact path is read from ``embedding_override_artifact_attr`` on the
    vLLM HF config. Overrides are loaded once and applied after vLLM's existing
    projected-feature scatter. ``token_embedding_override_mode="replace"``
    replaces token embeddings; ``"add"`` adds learned deltas to them.
    """
    token_embedding_overrides = dict(token_embedding_overrides or {})
    if token_embedding_override_mode not in ("replace", "add"):
        raise ValueError(f"unknown token embedding override mode: {token_embedding_override_mode!r}")
    if mode == "encode":
        if token_embedding_overrides or token_embedding_override_mode != "replace":
            raise ValueError("token_embedding_overrides are supported only in mode='projected'")
        return _build_encode_plugin(
            modality=modality,
            architecture=architecture,
            placeholder_text=placeholder_text,
            model_cls=encode_model_cls,
            normalize_item_fn=encode_normalize_item_fn,
            geometry_fn=encode_geometry_fn,
            pack_items_fn=encode_pack_items_fn,
            field_configs_fn=encode_field_configs_fn,
            dummy_item_fn=encode_dummy_item_fn,
            dummy_text=encode_dummy_text,
            dummy_fallback_text=encode_dummy_fallback_text,
            supported_mm_limits=encode_supported_mm_limits,
            max_tokens_per_item=encode_max_tokens_per_item,
            sentinel_start=encode_sentinel_start,
            sentinel_placeholder=encode_sentinel_placeholder,
            sentinel_end=encode_sentinel_end,
        )
    if mode != "projected":
        raise ValueError(f"unknown build_mm_plugin mode: {mode!r}")
    for name, value in (
        ("base_model_cls", base_model_cls),
        ("base_processing_info_cls", base_processing_info_cls),
        ("base_processor_cls", base_processor_cls),
        ("base_dummy_inputs_cls", base_dummy_inputs_cls),
        ("sentinels", sentinels),
        ("num_tokens_attr", num_tokens_attr),
    ):
        if value is None:
            raise ValueError(f"mode='projected' requires {name}")
    if geometry not in PROJECTED_GEOMETRIES:
        raise ValueError(f"unknown projected geometry: {geometry!r} (expected one of {sorted(PROJECTED_GEOMETRIES)})")
    if token_embedding_overrides and not embedding_override_artifact_attr:
        raise ValueError("embedding_override_artifact_attr must be non-empty")
    # The loop above is the presence proof; the Optional annotations exist only
    # because the two modes share one public signature.
    base_model_cls = cast(str, base_model_cls)
    base_processing_info_cls = cast(str, base_processing_info_cls)
    base_processor_cls = cast(str, base_processor_cls)
    base_dummy_inputs_cls = cast(str, base_dummy_inputs_cls)
    sentinels = cast(SentinelAttrs, sentinels)
    num_tokens_attr = cast(str, num_tokens_attr)
    embeds_field = f"{modality}_embeds"

    def register() -> None:
        if getattr(register, "_done", False):
            return

        import torch
        from transformers import BatchFeature
        from vllm import ModelRegistry
        from vllm.config.multimodal import BaseDummyOptions
        from vllm.inputs import MultiModalDataDict
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.inputs import MultiModalFieldConfig
        from vllm.multimodal.parse import EmbeddingItems, MultiModalDataParser
        from vllm.multimodal.processing.processor import PromptReplacement, PromptUpdateDetails

        base_model = _import(base_model_cls)
        base_info = _import(base_processing_info_cls)
        base_processor = _import(base_processor_cls)
        base_dummy = _import(base_dummy_inputs_cls)

        # Projected mode over a text-only base: when the base
        # model has no multimodal surface of its own, the plugin adds
        # SupportsMultiModal and implements only embed_multimodal, and the
        # processing layer is tokenizer-based rather than an upstream
        # multimodal processor. CT's projected mode (multimodal base) is
        # unchanged by construction: every text-base branch is selected by the
        # absence of the base's own multimodal methods.
        text_base = not hasattr(base_model, "embed_multimodal")

        bases: tuple[type, ...] = (base_model,)
        if text_base:
            from vllm.model_executor.models.interfaces import SupportsMultiModal

            bases = (base_model, SupportsMultiModal)
        if lora_declarations_from is not None:
            # The base model wrapper may not declare LoRA support even though
            # its registered inner language model exposes the standard linear
            # targets; copy the declarations from that inner class.
            from vllm.model_executor.models.interfaces import SupportsLoRA

            bases = (*bases, SupportsLoRA)

        class ModalityEmbeddingItems(EmbeddingItems):
            def __init__(self, data, expected_hidden_size):
                super().__init__(data, modality, expected_hidden_size)

        class ModalityDataParser(MultiModalDataParser):
            # vLLM documents these protected processor methods as its model
            # integration surface; keep the implementation local.
            def _parse_modality_data(self, data):
                if data is None:
                    return None
                return ModalityEmbeddingItems(data, self.expected_hidden_size)

            def _get_subparsers(self):
                return {**super()._get_subparsers(), modality: self._parse_modality_data}

        class ModalityProcessingInfo(base_info):
            def get_modality_hidden_size(self) -> int:
                # One resolver shared with the training path (P7): plain,
                # text_config, or llm_config spellings, fail closed on
                # missing or conflicting values.
                from nemotron_stitch.lm_config import resolve_lm_hidden_size

                return resolve_lm_hidden_size(self.get_hf_config())

            def get_supported_mm_limits(self):
                result = {} if text_base else dict(super().get_supported_mm_limits())
                return {**result, modality: None}

            def get_data_parser(self):
                return ModalityDataParser(expected_hidden_size=self.get_modality_hidden_size())

            def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
                result = {} if text_base else dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts))
                if mm_counts.get(modality, 0):
                    value = getattr(self.get_hf_config(), num_tokens_attr, None)
                    if value is None:
                        raise ValueError(f"vLLM config must define {num_tokens_attr}")
                    result[modality] = int(value)
                return result or None

        def _text_base_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
            mm_data = dict(mm_data)
            items = list(mm_data.pop(modality, []) or [])
            out = self.info.ctx.call_hf_processor(
                self.info.get_tokenizer(),
                dict(text=prompt, **mm_data),
                dict(**mm_kwargs, **tok_kwargs),
            )
            if items:
                tensors = [item if torch.is_tensor(item) else torch.as_tensor(item) for item in items]
                if all(tuple(t.shape) == tuple(tensors[0].shape) for t in tensors):
                    out.update({embeds_field: torch.stack(tensors)})
                else:
                    out.update({embeds_field: tensors})
            return out

        class ModalityProcessor(base_processor):
            # vLLM's public registry selects a processor class, while field mapping
            # and prompt replacement are currently available only through these
            # protected methods. Keep both overrides small and fail on missing config.
            def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs):
                fields = {} if text_base else dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
                if embeds_field in hf_inputs:
                    fields[embeds_field] = MultiModalFieldConfig.batched(modality)
                return fields

            def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
                updates = (
                    []
                    if text_base
                    else list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
                )
                if modality in mm_items:
                    items = mm_items.get_items(modality, ModalityEmbeddingItems)

                    def replacement(index: int):
                        count = items.get_feature_size(index)
                        config = self.info.get_hf_config()
                        expected = getattr(config, num_tokens_attr, None)
                        if expected is None:
                            raise ValueError(f"vLLM config must define {num_tokens_attr}")
                        check_projected_token_count(modality, count, int(expected), geometry=geometry)
                        start = getattr(config, sentinels.start, None)
                        placeholder = getattr(config, sentinels.placeholder, None)
                        end = getattr(config, sentinels.end, None)
                        if not all(isinstance(value, str) and value for value in (start, placeholder, end)):
                            raise ValueError(
                                f"vLLM config must define {sentinels.start}/{sentinels.placeholder}/{sentinels.end}"
                            )
                        text = start + placeholder * count + end
                        return PromptUpdateDetails.select_text(text, embed_text=placeholder)

                    updates.append(
                        PromptReplacement(
                            modality=modality,
                            target=placeholder_text,
                            replacement=replacement,
                        )
                    )
                return updates

            if text_base:
                # No upstream multimodal processor exists for a text-only base:
                # tokenize with the tokenizer and merge the projected payloads
                # into the BatchFeature so the field config routes them to
                # embed_multimodal. Same protected-hook contract as the
                # encode-mode processor.
                _call_hf_processor = _text_base_call_hf_processor

        class ModalityDummyInputsBuilder(base_dummy):
            def get_dummy_text(self, mm_counts):
                base_text = "" if text_base else super().get_dummy_text(mm_counts)
                return base_text + placeholder_text * mm_counts.get(modality, 0)

            def get_dummy_mm_data(
                self,
                seq_len: int,
                mm_counts,
                mm_options: dict[str, BaseDummyOptions],
            ) -> MultiModalDataDict:
                result = {} if text_base else dict(super().get_dummy_mm_data(seq_len, mm_counts, mm_options))
                count = mm_counts.get(modality, 0)
                if count:
                    hidden = self.info.get_modality_hidden_size()
                    value = getattr(self.info.get_hf_config(), num_tokens_attr, None)
                    if value is None:
                        raise ValueError(f"vLLM config must define {num_tokens_attr}")
                    tokens = int(value)
                    result[modality] = [torch.zeros(tokens, hidden) for _ in range(count)]
                return result

        @MULTIMODAL_REGISTRY.register_processor(
            ModalityProcessor,
            info=ModalityProcessingInfo,
            dummy_inputs=ModalityDummyInputsBuilder,
        )
        # pyrefly: ignore[invalid-inheritance] — the base set is dynamic
        # (text-base and LoRA conditionals), so it is not statically checkable.
        class ModalityModel(*bases):  # noqa: B903 — built inside register()
            if token_embedding_overrides:

                def __init__(self, *, vllm_config, prefix=""):
                    super().__init__(vllm_config=vllm_config, prefix=prefix)
                    config = vllm_config.model_config.hf_config
                    artifact_path = getattr(config, embedding_override_artifact_attr, None)
                    if not isinstance(artifact_path, str) or not artifact_path:
                        raise ValueError(
                            f"vLLM config must define {embedding_override_artifact_attr} "
                            "when token_embedding_overrides are configured"
                        )
                    from vllm.tokenizers.registry import cached_tokenizer_from_config

                    tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
                    if tokenizer is None:
                        raise ValueError("token embedding overrides require vLLM tokenizer initialization")
                    from nemotron_stitch.lm_config import resolve_lm_hidden_size

                    hidden_size = resolve_lm_hidden_size(config)
                    token_ids, vectors = load_token_embedding_overrides(
                        artifact_path,
                        token_embedding_overrides,
                        hidden_size=hidden_size,
                        tokenizer=tokenizer,
                    )
                    placeholder = getattr(config, sentinels.placeholder, None)
                    if isinstance(placeholder, str):
                        placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
                        if placeholder_id in token_ids:
                            raise ValueError("token embedding overrides cannot target the projected placeholder token")
                    parameter = next(self.parameters(), None)
                    device = None if parameter is None else parameter.device
                    dtype = vllm_config.model_config.dtype
                    self._stitch_token_embedding_override_ids = token_ids
                    self.register_buffer(
                        "_stitch_token_embedding_override_vectors",
                        vectors.to(device=device, dtype=dtype),
                        persistent=False,
                    )

            if text_base:
                # The text-only base's one-argument embed_input_ids shadows the
                # SupportsMultiModal default (MRO); re-expose the Protocol's,
                # which merges the projected embeddings by placeholder mask.
                # vLLM documents embed_input_ids overrides for additional merge
                # logic; apply the declared frozen token overrides there.
                def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
                    inputs_embeds = SupportsMultiModal.embed_input_ids(
                        self, input_ids, multimodal_embeddings, is_multimodal=is_multimodal
                    )
                    if not token_embedding_overrides:
                        return inputs_embeds
                    return apply_token_embedding_overrides(
                        input_ids,
                        inputs_embeds,
                        self._stitch_token_embedding_override_ids,
                        self._stitch_token_embedding_override_vectors,
                        mode=token_embedding_override_mode,
                    )

            elif token_embedding_overrides:
                # Same documented vLLM model-extension point as the text-base
                # branch above.
                def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
                    inputs_embeds = super().embed_input_ids(
                        input_ids, multimodal_embeddings, is_multimodal=is_multimodal
                    )
                    return apply_token_embedding_overrides(
                        input_ids,
                        inputs_embeds,
                        self._stitch_token_embedding_override_ids,
                        self._stitch_token_embedding_override_vectors,
                        mode=token_embedding_override_mode,
                    )

            if not text_base and hasattr(base_model, "get_mrope_input_positions"):
                # vLLM 0.25.1's Qwen MRoPE helper rejects embedding-only
                # modalities other than its native image/video grids. External
                # projected tokens use ordinary sequence positions, so exclude
                # only this modality and preserve every native feature. Delete
                # when vLLM accepts custom embedding modalities in MRoPE.
                def get_mrope_input_positions(self, input_tokens, mm_features):
                    native_features = [feature for feature in mm_features if feature.modality != modality]
                    return super().get_mrope_input_positions(input_tokens, native_features)

            @classmethod
            def get_placeholder_str(cls, modality_: str, i: int):
                if modality_.startswith(modality):
                    return placeholder_text
                if text_base:
                    return None
                return super().get_placeholder_str(modality_, i)

            def embed_multimodal(self, **kwargs):
                payloads = kwargs.pop(embeds_field, None)
                base_embeddings = None if text_base else super().embed_multimodal(**kwargs)
                embeddings = () if base_embeddings is None else tuple(base_embeddings)
                if payloads is None:
                    return embeddings
                if torch.is_tensor(payloads):
                    if payloads.ndim == 2:
                        projected = (payloads,)
                    elif payloads.ndim == 3:
                        projected = tuple(payloads)
                    else:
                        raise ValueError(f"{embeds_field} must be [T,H] or [N,T,H], got {tuple(payloads.shape)}")
                else:
                    projected = tuple(payloads)
                return embeddings + projected

        if lora_declarations_from is not None:
            lora_source = _import(lora_declarations_from)
            ModalityModel.packed_modules_mapping = lora_source.packed_modules_mapping
            ModalityModel.is_non_gated_moe = lora_source.is_non_gated_moe

        ModalityModel.__name__ = architecture
        ModalityModel.__qualname__ = architecture
        ModelRegistry.register_model(architecture, ModalityModel)
        setattr(register, "_done", True)

    return register


# ---------------------------------------------------------------------------
# mode="encode"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodePlugin:
    """The encode-mode processing layer plus its registrar.

    Built eagerly by ``build_mm_plugin(mode=\"encode\")`` so the consumer can
    import the classes by name before ``register()`` runs (vLLM is imported at
    factory-call time; call it from a vLLM-context module). ``register`` is
    re-entrant: vLLM tolerates repeat model/processor registration within a
    worker process.
    """

    processor_items: type
    data_parser: type
    processing_info: type
    processor: type
    dummy_inputs: type
    register: Callable[[], None]


def split_flat_payload(payload: Any, sizes: Any) -> list[Any]:
    """Normalize vLLM's three hand-off shapes to a list of per-item tensors.

    vLLM may pass (a) a list of per-item tensors (after ``flat_from_sizes``
    slicing), (b) one concatenated tensor plus per-item sizes, or (c) a single
    tensor with no sizes (single-item case).
    """
    import torch

    if isinstance(payload, torch.Tensor):
        if sizes is not None and torch.is_tensor(sizes) and sizes.numel() > 1:
            return list(payload.split([int(value) for value in sizes.flatten().tolist()], dim=0))
        return [payload]
    return list(payload)


def _build_encode_plugin(
    *,
    modality: str,
    architecture: str,
    placeholder_text: str,
    model_cls: str | None,
    normalize_item_fn: Callable[[Any], Any] | None,
    geometry_fn: Callable[[Any], int] | None,
    pack_items_fn: Callable[[Any], dict[str, Any]] | None,
    field_configs_fn: Callable[[Any], dict[str, Any]] | None,
    dummy_item_fn: Callable[[Any], Any] | None,
    dummy_text: str | None,
    dummy_fallback_text: str,
    supported_mm_limits: dict[str, int | None] | None,
    max_tokens_per_item: int | None,
    sentinel_start: str | None,
    sentinel_placeholder: str | None,
    sentinel_end: str | None,
) -> EncodePlugin:
    for name, value in (
        ("encode_model_cls", model_cls),
        ("encode_normalize_item_fn", normalize_item_fn),
        ("encode_geometry_fn", geometry_fn),
        ("encode_pack_items_fn", pack_items_fn),
        ("encode_field_configs_fn", field_configs_fn),
        ("encode_dummy_item_fn", dummy_item_fn),
        ("encode_dummy_text", dummy_text),
        ("encode_supported_mm_limits", supported_mm_limits),
        ("encode_max_tokens_per_item", max_tokens_per_item),
        ("encode_sentinel_start", sentinel_start),
        ("encode_sentinel_placeholder", sentinel_placeholder),
        ("encode_sentinel_end", sentinel_end),
    ):
        if value is None:
            raise ValueError(f"mode='encode' requires {name}")
    # The loop above is the presence proof; the Optional annotations exist only
    # because the two modes share one public signature.
    model_cls = cast(str, model_cls)
    normalize_item_fn = cast(Callable[[Any], Any], normalize_item_fn)
    geometry_fn = cast(Callable[[Any], int], geometry_fn)
    pack_items_fn = cast(Callable[[Any], dict[str, Any]], pack_items_fn)
    field_configs_fn = cast(Callable[[Any], dict[str, Any]], field_configs_fn)
    dummy_item_fn = cast(Callable[[Any], Any], dummy_item_fn)
    dummy_text = cast(str, dummy_text)
    supported_mm_limits = cast(dict[str, int | None], supported_mm_limits)
    max_tokens_per_item = cast(int, max_tokens_per_item)
    sentinel_start = cast(str, sentinel_start)
    sentinel_placeholder = cast(str, sentinel_placeholder)
    sentinel_end = cast(str, sentinel_end)

    from vllm import ModelRegistry
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.parse import MultiModalDataParser, ProcessorBatchItems
    from vllm.multimodal.processing import (
        BaseDummyInputsBuilder,
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
        PromptUpdateDetails,
    )

    class ModalityProcessorItems(ProcessorBatchItems):
        """Wraps a list of modality payload items."""

        def __init__(self, data):
            super().__init__(data, modality)

        def get_processor_data(self):
            # Override the base class's auto-pluralization (which would pluralize
            # the key): the un-pluralized modality name matches the modality
            # registration and the field-config registrations.
            return {modality: self.get_all()}

    class ModalityDataParser(MultiModalDataParser):
        # vLLM documents subclassing its parser/processor methods as the model
        # integration path; add the out-of-tree modality there.
        def _parse_modality_payload(self, data):
            if data is None:
                return None
            if not isinstance(data, list | tuple):
                data = [data]
            return ModalityProcessorItems([normalize_item_fn(item) for item in data])

        def _get_subparsers(self):
            return {**super()._get_subparsers(), modality: self._parse_modality_payload}

    class ModalityProcessingInfo(BaseProcessingInfo):
        def get_supported_mm_limits(self):
            return dict(supported_mm_limits)

        def get_data_parser(self):
            return ModalityDataParser(
                expected_hidden_size=self._get_expected_hidden_size(),
            )

        def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
            # A static cap short-circuits vLLM's dummy-profiling path, which
            # would otherwise run a dummy processor.apply through per-item
            # batching paths the packed encoder inputs do not satisfy.
            if mm_counts.get(modality, 0) > 0:
                return {modality: max_tokens_per_item}
            return None

    class ModalityDummyInputsBuilder(BaseDummyInputsBuilder):
        def get_dummy_text(self, mm_counts):
            if mm_counts.get(modality, 0) > 0:
                return dummy_text
            return dummy_fallback_text

        def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
            count = mm_counts.get(modality, 0)
            if count == 0:
                return {}
            item = dummy_item_fn(self.info.get_hf_config())
            return {modality: [item for _ in range(count)]}

    class ModalityProcessor(BaseMultiModalProcessor):
        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            return field_configs_fn(hf_inputs)

        def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
            items = mm_items.get_items(modality, ModalityProcessorItems)

            def replacement(item_idx: int):
                count = int(geometry_fn(items.get(item_idx)))
                if count <= 0:
                    raise ValueError(f"{modality} item {item_idx} has no soft tokens")
                full = sentinel_start + sentinel_placeholder * count + sentinel_end
                # Only the placeholder positions consume soft-token embeddings;
                # start/end are vocabulary-only boundary markers. select_text
                # keeps vLLM's feature-count assertion aligned with the
                # encoder's soft-token count.
                return PromptUpdateDetails.select_text(seq=full, embed_text=sentinel_placeholder)

            return [
                PromptReplacement(
                    modality=modality,
                    target=placeholder_text,
                    replacement=replacement,
                )
            ]

        def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
            # Strip the modality payload before the upstream HF processor call
            # (a text tokenizer does not accept it), tokenize, then merge the
            # packed modality tensors into the BatchFeature so the field
            # configs route them to embed_multimodal.
            mm_data = dict(mm_data)
            items = list(mm_data.pop(modality, []) or [])
            hf_processor = self.info.get_tokenizer()
            out = self.info.ctx.call_hf_processor(
                hf_processor,
                dict(text=prompt, **mm_data),
                dict(**mm_kwargs, **tok_kwargs),
            )
            if items:
                out.update(pack_items_fn(items))
            return out

    def register() -> None:
        """Re-entrant; called per vLLM worker process at startup."""
        model = _import(model_cls)
        ModelRegistry.register_model(architecture, model)
        MULTIMODAL_REGISTRY.register_processor(
            ModalityProcessor,
            info=ModalityProcessingInfo,
            dummy_inputs=ModalityDummyInputsBuilder,
        )(model)

    return EncodePlugin(
        processor_items=ModalityProcessorItems,
        data_parser=ModalityDataParser,
        processing_info=ModalityProcessingInfo,
        processor=ModalityProcessor,
        dummy_inputs=ModalityDummyInputsBuilder,
        register=register,
    )
