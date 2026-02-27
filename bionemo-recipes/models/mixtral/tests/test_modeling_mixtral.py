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
from modeling_mixtral_te import HFInferenceParams, NVMixtralConfig, NVMixtralForCausalLM
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
        # TE only supports right-padding for BSHD inputs, left-padding (Mixtral default) causes issues with RoPE and
        # attention calculations.
        tokenizer.padding_side = "right"
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

    def get_reference_model_no_weights(self, **kwargs) -> PreTrainedModel:
        # Limit the number of hidden layers to 2 for faster tests.
        return super().get_reference_model_no_weights(num_hidden_layers=2, **kwargs)

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

    # ==================== Mixtral-Specific KV-Cache Tests ====================

    def _create_inference_params(self, config, batch_size=1, max_seq_len=256, num_beams=1):
        """Create HFInferenceParams for the given config."""
        past_key_values = HFInferenceParams(
            max_batch_size=batch_size * num_beams,
            max_sequence_length=max_seq_len,
            num_heads_kv=config.num_key_value_heads,
            head_dim_k=config.hidden_size // config.num_attention_heads,
            dtype=torch.bfloat16,
            qkv_format="thd",
            max_ctx_len=max_seq_len,
        )
        for layer_number in range(1, config.num_hidden_layers + 1):
            past_key_values.allocate_memory(layer_number)
        return past_key_values

    def test_generate_with_cache(self):
        """Test single-prompt generation with KV-cache (THD format)."""
        config = self.create_test_config(attn_input_format="thd", self_attn_mask_type="padding_causal")
        model = self.get_model_class()(config).to("cuda").to(torch.bfloat16)
        model.eval()

        tokenizer = self.get_tokenizer()
        prompt = "The quick brown fox jumps over"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        past_key_values = self._create_inference_params(config, batch_size=1)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=16, use_cache=True, past_key_values=past_key_values)

        # Verify generation produced new tokens
        assert output_ids.shape[1] > inputs["input_ids"].shape[1]

    def test_generate_with_cache_batched(self):
        """Test batched generation with KV-cache (left-padded BSHD converted to THD)."""
        config = self.create_test_config(attn_input_format="thd", self_attn_mask_type="padding_causal")
        model = self.get_model_class()(config).to("cuda").to(torch.bfloat16)
        model.eval()

        tokenizer = self.get_tokenizer()
        prompts = (
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        past_key_values = self._create_inference_params(config, batch_size=2)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=16, use_cache=True, past_key_values=past_key_values)

        # Verify generation produced new tokens for both sequences
        assert output_ids.shape[0] == 2
        assert output_ids.shape[1] > inputs["input_ids"].shape[1]

    def test_generate_with_cache_beam_search(self):
        """Test batched generation with KV-cache and beam search."""
        config = self.create_test_config(attn_input_format="thd", self_attn_mask_type="padding_causal")
        model = self.get_model_class()(config).to("cuda").to(torch.bfloat16)
        model.eval()

        tokenizer = self.get_tokenizer()
        prompts = (
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        num_beams = 2
        past_key_values = self._create_inference_params(config, batch_size=2, num_beams=num_beams)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                use_cache=True,
                past_key_values=past_key_values,
                num_beams=num_beams,
                do_sample=True,
            )

        # Verify generation produced new tokens for both sequences
        assert output_ids.shape[0] == 2
        assert output_ids.shape[1] > inputs["input_ids"].shape[1]

    # ==================== Standalone Mixtral Generation Tests ====================

    def test_te_mixtral_model_generate_with_cache_beam_search(self):
        """Test Mixtral generation with KV-cache and beam search using real model weights."""
        import gc

        model_hf = self.get_reference_model()
        model_te = convert_mixtral_hf_to_te(model_hf, attn_input_format="thd", self_attn_mask_type="padding_causal")
        del model_hf
        gc.collect()

        model_te.to("cuda")
        model_te.eval()

        tokenizer = self.get_tokenizer()

        prompts = (
            'Licensed under the Apache License, Version 2.0 (the "License");'
            " you may not use this file except in compliance with the License."
            " You may obtain a copy of the License at",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore",
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        num_beams = 2
        config = model_te.config
        past_key_values = self._create_inference_params(config, batch_size=2, num_beams=num_beams)

        with torch.no_grad():
            output_ids = model_te.generate(
                **inputs,
                max_new_tokens=16,
                use_cache=True,
                past_key_values=past_key_values,
                num_beams=num_beams,
                do_sample=False,
            )

        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        assert "http://www.apache.org/licenses/LICENSE-2.0" in generated_text[0]
        assert "et dolore magna aliqua" in generated_text[1]
