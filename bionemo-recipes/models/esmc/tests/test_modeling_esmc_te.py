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

"""Tests for the ESMC TransformerEngine model.

This file provides tests extending the common BaseModelTest for ESMC, including:
- Forward/backward smoke tests
- Golden value tests against the EvolutionaryScale reference model
- FP8 tests
- Meta device initialization tests
- Conversion roundtrip tests
"""

import gc
from pathlib import Path
from typing import Callable, Dict, List, Literal, Type

import torch
from torch import nn
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PretrainedConfig, PreTrainedModel

from collator import DataCollatorWithFlattening
from convert import convert_esmc_te_to_ref, convert_esmc_to_te
from modeling_esmc_te import NVEsmcConfig, NVEsmcForMaskedLM
from tests.common import BaseModelTest, TestTolerances


TOKENIZER_DIR = str(Path(__file__).resolve().parent.parent / "esmc_fast_tokenizer")


class TestEsmcModel(BaseModelTest):
    """Model tester for ESMC.

    ESMC uses the EvolutionaryScale library (not standard HF), so we override
    several methods to handle custom model loading and conversion.
    """

    def get_model_class(self) -> Type[PreTrainedModel]:
        return NVEsmcForMaskedLM

    def get_config_class(self) -> Type[PretrainedConfig]:
        return NVEsmcConfig

    def get_upstream_model_id(self) -> str:
        return "EvolutionaryScale/esmc-300m-2024-12"

    def get_upstream_model_revision(self) -> str:
        return "main"

    def get_upstream_model_class(self) -> Type[PreTrainedModel]:
        # ESMC doesn't have a standard HF model class; we skip HF-specific tests.
        return PreTrainedModel  # Placeholder, not used directly

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    def get_layer_path(self, model: PreTrainedModel) -> List[nn.Module]:
        return list(model.esmc.layers)

    def create_test_config(self, **kwargs) -> PretrainedConfig:
        """Create test config for ESMC - use full architecture params but limit layers for speed."""
        num_hidden_layers = kwargs.pop("num_hidden_layers", 2)
        # Shim: the base test class passes `dtype=` which works on transformers v5, but the esm
        # package pins transformers<4.53 where PretrainedConfig only accepts `torch_dtype=`.
        # This can be dropped when esm updates its transformers version constraint.
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return NVEsmcConfig(
            vocab_size=64,
            hidden_size=960,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=15,
            intermediate_size=2560,
            **kwargs,
        )

    def get_test_input_data(
        self,
        format: Literal["bshd", "thd"] = "bshd",
        pad_to_multiple_of: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare test input data with protein sequences."""
        tokenizer = self.get_tokenizer()

        # Short protein sequences for testing
        sequences = [
            "MKTVRQERLKSIVRILERSKEPV",
            "KALTARQQEVFDLIRDHISQTGMPPTRA",
            "MFKVYGYDSNIHKCV",
        ]

        # Tokenize
        tokenized = [tokenizer(seq) for seq in sequences]

        # Use data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of if format == "bshd" else None,
        )

        if format == "thd":
            data_collator = DataCollatorWithFlattening(
                collator=data_collator,
                pad_sequences_to_be_divisible_by=pad_to_multiple_of,
            )

        batch = data_collator(tokenized)

        # Move to device
        return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def get_hf_to_te_converter(self) -> Callable:
        """Return the ESMC ref -> TE conversion function.

        Wraps convert_esmc_to_te to accept a model and return a model.
        """

        def _converter(model_ref, **kwargs):
            """Convert a reference ESMC model to TE format."""
            ref_state_dict = model_ref.state_dict()
            config = NVEsmcConfig(
                vocab_size=64,
                hidden_size=model_ref.embed.weight.shape[1],
                num_hidden_layers=len(model_ref.transformer.blocks),
                num_attention_heads=model_ref.transformer.blocks[0].attn.n_heads,
                intermediate_size=model_ref.transformer.blocks[0].ffn[1].weight.shape[0] // 2,
                **kwargs,
            )
            return convert_esmc_to_te(ref_state_dict, config)

        return _converter

    def get_te_to_hf_converter(self) -> Callable:
        """Return the TE -> ESMC ref conversion function."""
        return convert_esmc_te_to_ref

    def get_tolerances(self) -> TestTolerances:
        """Return ESMC-specific tolerances.

        With full d_model QK LayerNorm (matching the reference model exactly), the TE model
        closely reproduces reference outputs. These tolerances are comparable to LLaMA3.
        """
        return TestTolerances(
            golden_value_loss_atol=5e-3,
            golden_value_loss_rtol=0.01,
            golden_value_logits_atol=1.5,
            golden_value_logits_rtol=0.01,
        )

    # ==================== Override methods for non-HF reference model ====================

    def get_reference_model(self, dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        """Load the EvolutionaryScale ESMC reference model."""
        from esm.models.esmc import ESMC
        from esm.utils.constants.models import ESMC_300M

        model = ESMC.from_pretrained(ESMC_300M, device=torch.device("cuda"))
        model.to(dtype)
        model.eval()
        return model

    def get_reference_model_no_weights(self):
        """Create a reference ESMC model with random weights for conversion tests."""
        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer

        model = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=EsmSequenceTokenizer(),
            use_flash_attn=False,
        )
        return model

    def get_converted_te_model_checkpoint(self) -> Path:
        """Load ESMC, convert to TE, and save checkpoint.

        We override this to handle the non-HF model loading and to work on CPU
        for memory efficiency.
        """
        ref_model = self.get_reference_model(dtype=torch.bfloat16)
        ref_state_dict = {k: v.cpu() for k, v in ref_model.state_dict().items()}

        del ref_model
        gc.collect()
        torch.cuda.empty_cache()

        config = NVEsmcConfig(
            vocab_size=64,
            hidden_size=960,
            num_hidden_layers=30,
            num_attention_heads=15,
            intermediate_size=2560,
            torch_dtype="bfloat16",
        )

        model_te = convert_esmc_to_te(ref_state_dict, config)
        model_te.to("cpu")

        checkpoint_path: Path = self._tmp_dir / "converted_te_model"
        model_te.save_pretrained(checkpoint_path)

        del model_te
        gc.collect()

        return checkpoint_path

    def get_converted_te_model(self, **kwargs) -> PreTrainedModel:
        """Get the converted TE model.

        Shim: the base class passes `dtype=` which works on transformers v5, but the esm
        package pins transformers<4.53 where `from_pretrained` only accepts `torch_dtype=`.
        This can be dropped when esm updates its transformers version constraint.
        """
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return super().get_converted_te_model(**kwargs)

    # ==================== Override tests that don't apply to ESMC ====================

    def test_convert_hf_to_te(self):
        """Test conversion from ESMC ref to TE format."""
        model_ref = self.get_reference_model_no_weights()
        converter = self.get_hf_to_te_converter()
        model_te = converter(model_ref)

        assert model_te is not None
        assert isinstance(model_te, NVEsmcForMaskedLM)

    def test_convert_te_to_hf(self):
        """Test conversion from TE to ESMC ref format."""
        model_ref = self.get_reference_model_no_weights()
        converter = self.get_hf_to_te_converter()
        model_te = converter(model_ref)

        ref_state_dict = convert_esmc_te_to_ref(model_te)
        assert ref_state_dict is not None
        assert "embed.weight" in ref_state_dict

    def test_convert_te_to_hf_roundtrip(self):
        """Test roundtrip conversion ESMC ref -> TE -> ESMC ref.

        With full d_model QK LayerNorm, all weights should roundtrip exactly.
        The only non-exact weights are output projection and fc2 (due to residue
        scaling absorption/removal via float division/multiplication).
        """
        model_ref = self.get_reference_model_no_weights()
        original_state_dict = {k: v.clone() for k, v in model_ref.state_dict().items()}

        # Forward: ref -> TE
        converter = self.get_hf_to_te_converter()
        model_te = converter(model_ref)

        # Reverse: TE -> ref
        converted_state_dict = convert_esmc_te_to_ref(model_te)

        # Compare - all weights should roundtrip with high precision
        for key in original_state_dict:
            if key not in converted_state_dict:
                continue
            original = original_state_dict[key]
            converted = converted_state_dict[key]

            torch.testing.assert_close(
                original.float(),
                converted.float(),
                atol=1e-5,
                rtol=1e-5,
                msg=f"Roundtrip mismatch for {key}",
            )

    def test_convert_config(self):
        """Test that ESMC config can be created properly."""
        config = NVEsmcConfig(
            vocab_size=64,
            hidden_size=960,
            num_hidden_layers=30,
            num_attention_heads=15,
            intermediate_size=2560,
        )
        assert config.hidden_size == 960
        assert config.num_hidden_layers == 30
        assert config.num_attention_heads == 15

    def test_golden_values(self):
        """Test that TE model produces matching outputs compared to ESMC reference model.

        Overrides the base class because the ESMC ref model has a non-HF API:
        it takes (sequence_tokens, sequence_id) and returns .sequence_logits.
        """
        tokenizer = self.get_tokenizer()
        sequences = ["MKTVRQERLKSIVRILERSKEPV", "KALTARQQEVFDLIRDHISQTGMPPTRA"]
        encodings = tokenizer(sequences, return_tensors="pt", padding=True)
        input_ids = encodings["input_ids"].to("cuda")
        attention_mask = encodings["attention_mask"].to("cuda")

        # Run reference model
        ref_model = self.get_reference_model(dtype=torch.bfloat16)
        ref_model.eval()
        with torch.no_grad():
            sequence_id = input_ids != tokenizer.pad_token_id
            ref_output = ref_model(sequence_tokens=input_ids, sequence_id=sequence_id)
        ref_logits = ref_output.sequence_logits.detach().clone()

        del ref_model, ref_output
        gc.collect()
        torch.cuda.empty_cache()

        # Run TE model
        model_te = self.get_converted_te_model(dtype=torch.bfloat16)
        model_te.eval()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        with torch.no_grad():
            te_output = model_te(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        del model_te
        gc.collect()
        torch.cuda.empty_cache()

        # Verify outputs are finite
        mask = attention_mask.bool()
        assert torch.isfinite(te_output.logits[mask]).all(), "TE model produced non-finite logits"
        assert torch.isfinite(te_output.loss), "TE model produced non-finite loss"
        assert torch.isfinite(ref_logits[mask]).all(), "Reference model produced non-finite logits"

        # Compare logits
        tolerances = self.get_tolerances()
        torch.testing.assert_close(
            te_output.logits[mask],
            ref_logits[mask],
            atol=tolerances.golden_value_logits_atol,
            rtol=tolerances.golden_value_logits_rtol,
        )
