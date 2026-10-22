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

"""Route cached DNA features to native models through Stitch soft-token hosts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from transformers import ProcessorMixin
else:
    try:
        from transformers import ProcessorMixin
    except ImportError:
        ProcessorMixin = object

from nemotron_stitch.nemo_rl.data import EncoderRLDataset

ARCHITECTURE = "GeneticsExampleNemotronForCausalLM"
POOLING_WINDOW = 64


@dataclass(frozen=True)
class ModelSpec:
    """Native backend classes to decorate; no model-family dispatch."""

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
    ModelSpec(
        architecture="GeneticsExampleSuperForCausalLM",
        automodel_cls="nemo_automodel.components.models.nemotron_omni.model.NemotronOmniForConditionalGeneration",
        vllm_model_cls="vllm.model_executor.models.nano_nemotron_vl.NemotronH_Nano_VL_V2",
        processing_info_cls="vllm.model_executor.models.nano_nemotron_vl.NanoNemotronVLProcessingInfo",
        processor_cls="vllm.model_executor.models.nano_nemotron_vl.NanoNemotronVLMultiModalProcessor",
        dummy_inputs_cls="vllm.model_executor.models.nano_nemotron_vl.NanoNemotronVLDummyInputsBuilder",
    ),
)


@cache
def build_host_cls(spec: ModelSpec) -> type:
    """Keep one registered class identity per native model description."""
    from nemotron_stitch.automodel.model import build_multimodal_host

    module, _, name = spec.automodel_cls.rpartition(".")
    return build_multimodal_host(getattr(import_module(module), name), architecture=spec.architecture)


def register_models() -> None:
    """Register each host; Stitch owns DNA projection and scatter."""
    from nemotron_stitch.automodel.registry import register_models as register_model

    for spec in MODELS:
        register_model(spec.architecture, build_host_cls(spec))


@cache
def register_vllm() -> None:
    """Register projected DNA in each native generation model."""
    from nemotron_stitch.vllm.plugin import SentinelAttrs, build_mm_plugin

    for spec in MODELS:
        plugin = build_mm_plugin(
            architecture=spec.architecture.removesuffix("ForCausalLM") + "Projected",
            base_model_cls=spec.vllm_model_cls,
            base_processing_info_cls=spec.processing_info_cls,
            base_processor_cls=spec.processor_cls,
            base_dummy_inputs_cls=spec.dummy_inputs_cls,
            modality="dna",
            placeholder_text="<dna>",
            mode="projected",
            sentinels=SentinelAttrs(
                start="mm_encoder_start", placeholder="mm_encoder_placeholder", end="mm_encoder_end"
            ),
            num_tokens_attr="mm_encoder_max_tokens",
            geometry="variable",
        )
        cast(Callable[[], None], plugin)()


def format_dna(block: dict[str, Any], *, pooling_window: int = POOLING_WINDOW) -> str:
    """Derive the Evo2 span from the sequence and configured mean-pooling window."""
    count = (len(block["sequence"]) + pooling_window - 1) // pooling_window
    return "<SPECIAL_30>" + "<SPECIAL_32>" * count + "<SPECIAL_31>"


class Evo2Processor(ProcessorMixin):
    """Compose the added encoder with the checkpoint's existing processor."""

    def __init__(self, base_processor: Any, sequence_processor: Callable[[Any], dict[str, Any]] | None = None) -> None:
        self.base_processor = base_processor
        self.sequence_processor = sequence_processor
        self.tokenizer = getattr(base_processor, "tokenizer", base_processor)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Save native processor files; loaders reconstruct this wrapper from YAML."""
        return self.base_processor.save_pretrained(*args, **kwargs)

    def __call__(self, text: Any = None, *, sequences: Any = None, **kwargs: Any) -> Any:
        inputs = self.base_processor(text=text, **kwargs)
        if sequences is not None:
            if self.sequence_processor is None:
                raise ValueError("provide sequence_processor to encode uncached inputs")
            features = self.sequence_processor(sequences)
            if inputs.keys() & features.keys():
                raise ValueError("encoder features must not overwrite native processor fields")
            inputs.update(features)
        return inputs

    def apply_chat_template(
        self, conversation: Any, *args: Any, pooling_window: int | str = POOLING_WINDOW, **kwargs: Any
    ) -> Any:
        # Transformers 5.12.1 has no composable custom-content handler; U-82
        # tracks native rendering/payload dispatch so this traversal can be deleted.
        def format_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            result = []
            for message in messages:
                content = message["content"]
                if isinstance(content, list):
                    content = [
                        {"type": "text", "text": format_dna(block, pooling_window=int(pooling_window))}
                        if block.get("type") == "dna"
                        else block
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
def load_processor(path: str, trust_remote_code: bool = True) -> Evo2Processor:
    from transformers import AutoProcessor

    return Evo2Processor(base_processor=AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code))


class GeneticsRLDataset(EncoderRLDataset):
    """Use the package dataset with support for multiple prepared DNA spans."""

    def set_processor(self) -> None:
        self.processor = grpo_processor


def grpo_processor(
    datum_dict: dict[str, Any], task_data_spec: Any, tokenizer: Any, max_seq_length: int, idx: int
) -> dict[str, Any]:
    """Keep each sequence's span as one variable-length rollout item."""
    from nemotron_stitch.nemo_rl.data import projected_span_processor
    from nemotron_stitch.nemo_rl.transport import load_cached_policy_features
    from nemotron_stitch.prompt import PlaceholderSpec

    features = load_cached_policy_features(
        datum_dict, "dna", cache_root=datum_dict["encoder_loader_kwargs"]["cache_root"]
    )
    return projected_span_processor(
        datum_dict,
        task_data_spec,
        Evo2Processor(base_processor=tokenizer),
        max_seq_length,
        idx,
        features={"dna": features},
        specs=(PlaceholderSpec("dna", "<SPECIAL_30>", "<SPECIAL_32>", "<SPECIAL_31>"),),
        modality="dna",
    )
