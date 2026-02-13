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

"""Tests for the ESMC HuggingFace-compatible tokenizer.

Verifies that the custom PreTrainedTokenizerFast produces identical token IDs
to the EvolutionaryScale EsmSequenceTokenizer, while using standard HF
model_input_names (input_ids, attention_mask) for compatibility with
DataCollatorForLanguageModeling.
"""

from pathlib import Path

import pytest
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizerFast


TOKENIZER_DIR = str(Path(__file__).resolve().parent.parent / "esmc_fast_tokenizer")


@pytest.fixture()
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


@pytest.fixture()
def ref_tokenizer():
    from esm.tokenization import EsmSequenceTokenizer

    return EsmSequenceTokenizer()


class TestEsmcTokenizer:
    """Tests comparing the HF tokenizer against the ESM reference tokenizer."""

    def test_loads_as_pretrained_tokenizer_fast(self, hf_tokenizer):
        assert isinstance(hf_tokenizer, PreTrainedTokenizerFast)

    def test_model_input_names(self, hf_tokenizer):
        assert hf_tokenizer.model_input_names == ["input_ids", "attention_mask"]

    def test_vocab_size(self, hf_tokenizer, ref_tokenizer):
        assert hf_tokenizer.vocab_size == len(ref_tokenizer.vocab)

    def test_special_token_ids_match(self, hf_tokenizer, ref_tokenizer):
        assert hf_tokenizer.pad_token_id == ref_tokenizer.pad_token_id
        assert hf_tokenizer.cls_token_id == ref_tokenizer.cls_token_id
        assert hf_tokenizer.eos_token_id == ref_tokenizer.eos_token_id
        assert hf_tokenizer.unk_token_id == ref_tokenizer.unk_token_id
        assert hf_tokenizer.mask_token_id == ref_tokenizer.mask_token_id

    def test_chain_break_token(self, hf_tokenizer, ref_tokenizer):
        """Verify the chain break token | maps to the same ID."""
        hf_id = hf_tokenizer.convert_tokens_to_ids("|")
        ref_id = ref_tokenizer.convert_tokens_to_ids("|")
        assert hf_id == ref_id == 31

    def test_single_sequence_tokenization(self, hf_tokenizer, ref_tokenizer):
        seq = "MKTVRQERLKSIVRILERSKEPV"
        hf_out = hf_tokenizer(seq)
        ref_out = ref_tokenizer(seq)
        assert hf_out["input_ids"] == ref_out["input_ids"]

    def test_batch_tokenization_with_padding(self, hf_tokenizer, ref_tokenizer):
        sequences = [
            "MKTVRQERLKSIVRILERSKEPV",
            "KALTARQQEVFDLIRDHISQTGMPPTRA",
            "MFKVYGYDSNIHKCV",
        ]
        hf_out = hf_tokenizer(sequences, padding=True)
        ref_out = ref_tokenizer(sequences, padding=True)
        assert hf_out["input_ids"] == ref_out["input_ids"]
        assert hf_out["attention_mask"] == ref_out["attention_mask"]

    def test_all_amino_acids(self, hf_tokenizer, ref_tokenizer):
        """Verify all standard amino acid characters produce matching IDs."""
        amino_acids = "LAGVSERTIDPKQNFYMHWCXBUZO"  # pragma: allowlist secret
        for aa in amino_acids:
            hf_id = hf_tokenizer.convert_tokens_to_ids(aa)
            ref_id = ref_tokenizer.convert_tokens_to_ids(aa)
            assert hf_id == ref_id, f"Mismatch for amino acid {aa}: HF={hf_id}, ref={ref_id}"

    def test_data_collator_compatibility(self, hf_tokenizer):
        """Verify DataCollatorForLanguageModeling works without errors."""
        sequences = ["MKTVRQERLK", "KALTARQQEV", "MFKVYGYD"]
        tokenized = [hf_tokenizer(seq) for seq in sequences]

        collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)
        batch = collator(tokenized)

        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 3
