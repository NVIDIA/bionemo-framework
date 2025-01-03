# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import ftfy
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from bionemo.evo2.utils.config import StipedHyena2PreprocessingConfig


class StripedHyena2Tokenizer:
    """Tokenizer for StripedHyena2."""

    def __init__(self, params: StipedHyena2PreprocessingConfig | None = None):
        """Initialize the StripedHyena2Tokenizer."""
        # Pass all NeMo2/Megatron-compliant parameters associated with config.StipedHyena2PreprocessingConfig.
        self.params: StipedHyena2PreprocessingConfig = params if params is not None else StipedHyena2PreprocessingConfig()
        self.tokenizer: TokenizerSpec = get_nmt_tokenizer(
            library=self.params.tokenizer_type.lower(),
            vocab_file=str(self.params.vocab_file) if self.params.vocab_file is not None else None,
            merges_file=str(self.params.merges_file) if self.params.merges_file is not None else None,
            model_name=self.params.pretrained_tokenizer_model,
            tokenizer_model=self.params.pretrained_tokenizer_model,
            special_tokens=self.params.special_tokens,
            use_fast=self.params.fast_hf_tokenizer,
        )

    def tokenize(
        self,
        text: str | list[str],
        use_ftfy: bool = False,
        enforce_sample_length: None | int = None,
        append_eod: bool = False,
        drop_empty_sequences: bool = False,
    ):
        """Tokenize the input text data for StripedHyena2."""
        if isinstance(text, str):
            text = [text]
        # Tokenize a document or batch of strings.
        doc_ids = []
        for l, t in enumerate(text):
            if use_ftfy:
                t = ftfy.fix_text(t)
            # Tokenize the string.
            text_ids: list = self.tokenizer.text_to_ids(t)
            if drop_empty_sequences and len(text_ids) == 0:
                continue
            # Append EOD token if appropriate.
            eod_length = int(append_eod and l == len(text) - 1)
            token_length = len(text_ids) + eod_length
            text_ids += [self.tokenizer.eod] * eod_length
            if enforce_sample_length is not None:
                # Pad shorter sequences and except excessive sequences.
                if token_length > enforce_sample_length:
                    raise ValueError(
                        "Detected input text with a length greater than the maximum "
                        f"possible sample length of {enforce_sample_length}.)"
                    )
                else:
                    text_ids += [self.tokenizer.pad] * (enforce_sample_length - token_length)
            # Append to document.
            doc_ids.append(text_ids)
        return doc_ids
