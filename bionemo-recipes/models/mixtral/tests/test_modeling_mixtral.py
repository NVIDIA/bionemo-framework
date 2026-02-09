# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the Mixtral MoE model.

This file provides comprehensive tests for the Mixtral model including:
- Common tests from the test library (meta device init, golden values, conversion, FP8)
- Mixtral-specific tests
"""

import os
from typing import Callable, Dict, List, Literal, Type

import pytest
import torch
import transformers
from torch import nn
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from collator import DataCollatorWithFlattening
from convert import convert_mixtral_hf_to_te, convert_mixtral_te_to_hf
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM
from tests.common import BaseModelTest, TestTolerances


class TestMixtralModel(BaseModelTest):
    """Model tester for Mixtral.

    This class provides Mixtral-specific configuration for the common test suite.
    """

    def get_model_class(self) -> Type[PreTrainedModel]:
        """Return the Mixtral TE model class."""
        return NVMixtralForCausalLM

    def get_config_class(self) -> Type[PretrainedConfig]:
        """Return the Mixtral config class."""
        return NVMixtralConfig

    def get_upstream_model_id(self) -> str:
        """Return the upstream HuggingFace model ID."""
        return "NeuralNovel/Mini-Mixtral-v0.2"

    def get_upstream_model_revision(self) -> str:
        """Return the specific revision for the upstream model."""
        return "2fb530d"

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the Mixtral tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.get_upstream_model_id())
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_upstream_model_class(self) -> Type[PreTrainedModel]:
        """Return the upstream HuggingFace model class."""
        return transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM

    def create_test_config(self, **kwargs) -> PretrainedConfig:
        # Limit the number of hidden layers to 2 for faster tests.
        return super().create_test_config(num_hidden_layers=2, **kwargs)

    def get_layer_path(self, model: PreTrainedModel) -> List[nn.Module]:
        """Return the list of transformer layers."""
        return list(model.model.layers)  # type: ignore

    def get_reference_model(
        self, dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "flash_attention_2"
    ) -> PreTrainedModel:
        """Return the reference HuggingFace model."""
        if os.environ.get("CI") == "true":
            pytest.skip("Skipping Mixtral reference model test in CI, requires Mini-Mixtral download ~25GB")
        return super().get_reference_model(dtype=dtype, attn_implementation=attn_implementation)

    def get_test_input_data(
        self, format: Literal["bshd", "thd"] = "bshd", pad_to_multiple_of: int | None = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare test input data (text sequences)."""
        tokenizer = self.get_tokenizer()
        test_texts = [
            "Unless required by applicable law or agreed to in writing, software distributed under the License.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            mlm=False,
        )

        if format == "thd":
            data_collator = DataCollatorWithFlattening(
                collator=data_collator,
                pad_sequences_to_be_divisible_by=pad_to_multiple_of,
                separator_id=-100,
            )

        batch = data_collator([tokenizer(text) for text in test_texts])
        return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def get_hf_to_te_converter(self) -> Callable:
        """Return the HF to TE conversion function."""
        return convert_mixtral_hf_to_te

    def get_te_to_hf_converter(self) -> Callable:
        """Return the TE to HF conversion function."""
        return convert_mixtral_te_to_hf

    def get_tolerances(self) -> TestTolerances:
        """Return Mixtral-specific test tolerances."""
        return TestTolerances(
            golden_value_loss_atol=5e-3,
            golden_value_loss_rtol=0.01,
            golden_value_logits_atol=1.5,
            golden_value_logits_rtol=0.01,
            cp_loss_atol=0.5,
            cp_loss_rtol=0.25,
        )

    @pytest.mark.skip(
        reason="MoE routing is batch-dependent: padding tokens in BSHD affect softmax normalization "
        "and top-k expert selection, so BSHD and THD produce fundamentally different routing decisions."
    )
    def test_golden_values_thd(self, te_attn_backend):
        """Skip: BSHD vs THD comparison is not meaningful for MoE models."""
