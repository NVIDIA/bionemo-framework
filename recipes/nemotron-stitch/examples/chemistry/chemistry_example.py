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

"""Route cached KERMT molecule, atom, and undirected-bond features to Stitch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import ProcessorMixin
else:
    try:
        from transformers import ProcessorMixin
    except ImportError:
        ProcessorMixin = object

from nemotron_stitch.automodel.data import FeatureCollator
from nemotron_stitch.nemo_rl.data import EncoderRLDataset

ARCHITECTURE = "ChemistryExampleNemotronForCausalLM"
# name, reserved token ID, native feature width
BRANCHES = (("chem_molecule", 33, 3200), ("chem_atom", 32, 1600), ("chem_bond", 34, 3200))


@dataclass(frozen=True)
class ModelSpec:
    """Native backend classes to decorate with molecule soft tokens."""

    architecture: str
    automodel_cls: str
    vllm_model_cls: str
    processing_info_cls: str
    processor_cls: str
    dummy_inputs_cls: str


MODELS = (
    ModelSpec(
        architecture=ARCHITECTURE,
        automodel_cls="nemo_automodel.components.models.nemotron_v3.model.NemotronHForCausalLM",
        vllm_model_cls="vllm.model_executor.models.nemotron_h.NemotronHForCausalLM",
        processing_info_cls="vllm.multimodal.processing.BaseProcessingInfo",
        processor_cls="vllm.multimodal.processing.BaseMultiModalProcessor",
        dummy_inputs_cls="vllm.multimodal.processing.BaseDummyInputsBuilder",
    ),
)


@cache
def build_host_cls(spec: ModelSpec) -> type:
    """Keep one registered class identity per native model description."""
    from nemotron_stitch.automodel.model import build_multimodal_host

    module, _, name = spec.automodel_cls.rpartition(".")
    return build_multimodal_host(getattr(import_module(module), name), architecture=spec.architecture)


def register_models() -> None:
    """Register each host; Stitch owns projection and scatter for all branches."""
    from nemotron_stitch.automodel.registry import register_models as register_model

    for spec in MODELS:
        register_model(spec.architecture, build_host_cls(spec))


def split_features(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restore native widths from the padded graph cache.

    Adjacent low→high/high→low bond rows form one undirected feature. Their
    order is the encoder contract; averaging them would change the model.
    """
    packed = np.asarray(row["features"], dtype=np.float32)
    molecules, atoms, bonds = row["molecule_count"], row["atom_count"], row["directed_bond_count"]
    if not isinstance(molecules, int) or molecules < 0 or atoms < molecules:
        raise ValueError("KERMT requires nonnegative molecule_count and at least one atom per molecule")
    if not isinstance(atoms, int) or not isinstance(bonds, int) or atoms < 0 or bonds < 0 or bonds % 2:
        raise ValueError("KERMT requires nonnegative atom_count and nonnegative paired directed_bond_count")
    if row["bond_representation"] != "undirected_concat":
        raise ValueError("this recipe requires bond_representation='undirected_concat'")
    if packed.shape != (molecules + atoms + bonds, 3200):
        raise ValueError("KERMT cache shape does not match the manifest counts")
    return (
        np.ascontiguousarray(packed[:molecules]),
        np.ascontiguousarray(packed[molecules : molecules + atoms, :1600]),
        np.ascontiguousarray(packed[molecules + atoms :, :1600]).reshape(bonds // 2, 3200),
    )


def format_molecule(block: dict[str, Any]) -> str:
    """Render one cached molecule without changing native content blocks."""
    lines = ["MOLECULE", "<SPECIAL_33>", "ATOMS"]
    lines.extend(f"{i} <SPECIAL_32>" for i in range(block["atom_count"]))
    lines.append("BONDS")
    lines.extend(f"{low} {high} <SPECIAL_34>" for low, high in block["bond_endpoints"])
    if not block["bond_endpoints"]:
        lines.append("NONE")
    return "<SPECIAL_30>" + "\n".join(lines) + "<SPECIAL_31>"


class KERMTProcessor(ProcessorMixin):
    """Compose the added encoder with the checkpoint's existing processor."""

    def __init__(self, base_processor: Any, molecule_processor: Callable[[Any], dict[str, Any]] | None = None) -> None:
        self.base_processor = base_processor
        self.molecule_processor = molecule_processor
        self.tokenizer = getattr(base_processor, "tokenizer", base_processor)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Save native processor files; loaders reconstruct this wrapper from YAML."""
        return self.base_processor.save_pretrained(*args, **kwargs)

    def __call__(self, text: Any = None, *, molecules: Any = None, **kwargs: Any) -> Any:
        inputs = self.base_processor(text=text, **kwargs)
        if molecules is not None:
            if self.molecule_processor is None:
                raise ValueError("provide molecule_processor to encode uncached inputs")
            features = self.molecule_processor(molecules)
            if inputs.keys() & features.keys():
                raise ValueError("encoder features must not overwrite native processor fields")
            inputs.update(features)
        return inputs

    def apply_chat_template(self, conversation: Any, *args: Any, **kwargs: Any) -> Any:
        # Transformers 5.12.1 has no composable custom-content handler; U-82
        # tracks native rendering/payload dispatch so this traversal can be deleted.
        def format_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            result = []
            for message in messages:
                content = message["content"]
                if isinstance(content, list):
                    content = [
                        {"type": "text", "text": format_molecule(block)} if block.get("type") == "molecule" else block
                        for block in content
                    ]
                    if not hasattr(self.base_processor, "tokenizer") and all(
                        block.get("type") == "text" for block in content
                    ):
                        content = "".join(block["text"] for block in content)
                result.append({**message, "content": content})
            return result

        messages = (
            [format_chat(chat) for chat in conversation]
            if conversation and isinstance(conversation[0], list | tuple)
            else format_chat(conversation)
        )
        return self.base_processor.apply_chat_template(messages, *args, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        return self.base_processor.model_input_names

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        base = object.__getattribute__(self, "base_processor")
        return getattr(base if hasattr(base, name) else object.__getattribute__(self, "tokenizer"), name)


@cache
def load_processor(path: str, trust_remote_code: bool = True) -> KERMTProcessor:
    from transformers import AutoProcessor

    return KERMTProcessor(base_processor=AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code))


class ChemistryCollator:
    """Attach the three graph branches using the shared feature collator."""

    def __init__(
        self, tokenizer: Any, *, max_length: int = 16384, chat_template_kwargs: dict[str, Any] | None = None
    ) -> None:
        self.collators = [
            FeatureCollator(
                tokenizer,
                projector_name=name,
                placeholder_token_id=token_id,
                max_length=max_length,
                chat_template_kwargs=chat_template_kwargs,
                truncate=False,
            )
            for name, token_id, _ in BRANCHES
        ]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [split_features(row) for row in batch]
        result: dict[str, torch.Tensor] = {}
        for index, collator in enumerate(self.collators):
            result.update(
                collator([{**row, "features": arrays[index]} for row, arrays in zip(batch, features, strict=True)])
            )
        return result


@cache
def register_vllm() -> None:
    """Each authored graph slot becomes one projected vLLM item."""
    from nemotron_stitch.vllm.plugin import SentinelAttrs, build_mm_plugin

    for spec in MODELS:
        plugin = build_mm_plugin(
            architecture=spec.architecture.removesuffix("ForCausalLM") + "Projected",
            base_model_cls=spec.vllm_model_cls,
            base_processing_info_cls=spec.processing_info_cls,
            base_processor_cls=spec.processor_cls,
            base_dummy_inputs_cls=spec.dummy_inputs_cls,
            modality="chemistry",
            placeholder_text="<chemistry>",
            mode="projected",
            sentinels=SentinelAttrs(
                start="mm_encoder_start", placeholder="mm_encoder_placeholder", end="mm_encoder_end"
            ),
            num_tokens_attr="mm_encoder_max_tokens",
            geometry="variable",
        )
        cast(Callable[[], None], plugin)()


class ChemistryRLDataset(EncoderRLDataset):
    """Keep the package dataset lifecycle; select the three-branch processor."""

    def set_processor(self) -> None:
        self.processor = grpo_processor


@cache
def _feature_cache(root: str) -> Any:
    from nemotron_stitch.features.cache import open_feature_cache

    return open_feature_cache(root)


def grpo_processor(
    datum_dict: dict[str, Any],
    task_data_spec: Any,
    tokenizer: Any,
    max_seq_length: int,
    idx: int,
) -> dict[str, Any]:
    """Route the three graph branches through the shared span transport."""
    from nemotron_stitch.nemo_rl.data import projected_span_processor
    from nemotron_stitch.prompt import PlaceholderSpec

    kwargs = datum_dict["encoder_loader_kwargs"]
    packed = _feature_cache(kwargs["cache_root"]).get(datum_dict["feature_key"])
    arrays = split_features({**datum_dict, "features": packed})
    return projected_span_processor(
        datum_dict,
        task_data_spec,
        KERMTProcessor(base_processor=tokenizer),
        max_seq_length,
        idx,
        features={name: torch.from_numpy(array) for (name, _, _), array in zip(BRANCHES, arrays, strict=True)},
        specs=tuple(PlaceholderSpec(name, "", f"<SPECIAL_{token}>", "") for name, token, _ in BRANCHES),
        modality="chemistry",
    )
