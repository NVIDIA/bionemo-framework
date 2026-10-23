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

"""Qwen3.5/Qwen3.6 bindings for the otherwise model-neutral LLaVA example."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import cast

from nemotron_stitch.automodel.model import build_multimodal_host
from nemotron_stitch.automodel.registry import register_models as _register_model
from nemotron_stitch.vllm.plugin import SentinelAttrs, build_mm_plugin

TRAINING_ARCHITECTURE = "LlavaExampleQwen3_5ForConditionalGeneration"
ROLLOUT_ARCHITECTURE = "LlavaExampleQwen3_5Projected"
MOE_TRAINING_ARCHITECTURE = "LlavaExampleQwen3_5MoeForConditionalGeneration"
MOE_ROLLOUT_ARCHITECTURE = "LlavaExampleQwen3_5MoeProjected"
PROJECTOR_NAME = "clip_image"


@cache
def build_host_cls() -> type:
    """Decorate AutoModel's native Qwen3.5 VLM class with the projector host."""
    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

    return build_multimodal_host(Qwen3_5ForConditionalGeneration, architecture=TRAINING_ARCHITECTURE)


def register_models() -> None:
    _register_model(TRAINING_ARCHITECTURE, build_host_cls())


@cache
def build_moe_host_cls() -> type:
    """Decorate AutoModel's native Qwen3.5-MoE VLM used by Qwen3.6."""
    from nemo_automodel.components.models.qwen3_5_moe.model import (
        Qwen3_5MoeForConditionalGeneration,
    )

    return build_multimodal_host(
        Qwen3_5MoeForConditionalGeneration,
        architecture=MOE_TRAINING_ARCHITECTURE,
    )


def register_moe_models() -> None:
    _register_model(MOE_TRAINING_ARCHITECTURE, build_moe_host_cls())


def _vllm_registration(architecture: str, model: str, processing_info: str) -> Callable[[], None]:
    return cast(
        Callable[[], None],
        build_mm_plugin(
            modality=PROJECTOR_NAME,
            architecture=architecture,
            placeholder_text=f"<{PROJECTOR_NAME}>",
            mode="projected",
            base_model_cls=f"vllm.model_executor.models.qwen3_5:{model}",
            base_processing_info_cls=f"vllm.model_executor.models.qwen3_5:{processing_info}",
            base_processor_cls="vllm.model_executor.models.qwen3_vl:Qwen3VLMultiModalProcessor",
            base_dummy_inputs_cls="vllm.model_executor.models.qwen3_vl:Qwen3VLDummyInputsBuilder",
            sentinels=SentinelAttrs(
                start="mm_encoder_start",
                placeholder="mm_encoder_placeholder",
                end="mm_encoder_end",
            ),
            num_tokens_attr="mm_encoder_max_tokens",
        ),
    )


register_vllm = _vllm_registration(
    ROLLOUT_ARCHITECTURE,
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ProcessingInfo",
)
register_moe_vllm = _vllm_registration(
    MOE_ROLLOUT_ARCHITECTURE,
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeProcessingInfo",
)
