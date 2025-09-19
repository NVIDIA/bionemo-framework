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

import json
from pathlib import Path
from nemo.lightning import io
from nemo.collections.llm.gpt.model.llama import HFLlamaExporter
from nemo.utils import logging

from typing import TYPE_CHECKING
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

if TYPE_CHECKING:
    from transformers import LlamaForCausalLM


try:
    from bionemo.evo2.models.gpt import Evo2GPTModel
except ImportError:
    raise ImportError("BioNeMo Evo2 package is required. Please install BioNeMo.")


@io.model_exporter(Evo2GPTModel, "hf")
class HFEvo2GPTExporter(HFLlamaExporter):
    """Minimal Evo2 exporter that only customizes tokenizer handling."""
    
    def apply(self, output_path: Path) -> Path:
        """Apply conversion with Evo2-specific tokenizer handling."""
        # Do the standard conversion
        result = super().apply(output_path)
        
                # Handle ByteLevelTokenizer conversion
        self._convert_bytelevel_tokenizer(output_path)
        
        return output_path
    
    def _convert_bytelevel_tokenizer(self, output_path: Path):
        """Convert ByteLevelTokenizer to HuggingFace compatible format."""
        from nemo.collections.common.tokenizers.bytelevel_tokenizers import ByteLevelTokenizer
        
        nemo_tokenizer = self.tokenizer
        
        if isinstance(nemo_tokenizer, ByteLevelTokenizer):
            logging.info("Converting ByteLevelTokenizer to HuggingFace format...")
            
            # Create a byte-level tokenizer using HuggingFace tokenizers library
            vocab = {}
            # Map bytes (0-255) to their values
            for i in range(256):
                vocab[chr(i)] = i
            
            # Add special tokens
            for token, token_id in nemo_tokenizer.special_token_to_id.items():
                if token not in [nemo_tokenizer.pad_id, nemo_tokenizer.bos_id, nemo_tokenizer.eos_id]:
                    vocab[str(token)] = token_id
            
            # Create the tokenizer
            tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
            
            # Set up byte-level pre-tokenization
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            
            # Create PreTrainedTokenizerFast wrapper
            special_tokens = {}
            if nemo_tokenizer.bos_id is not None:
                special_tokens["bos_token"] = str(nemo_tokenizer.bos_id)
            if nemo_tokenizer.eos_id is not None:
                special_tokens["eos_token"] = str(nemo_tokenizer.eos_id)
            if nemo_tokenizer.pad_id is not None:
                special_tokens["pad_token"] = str(nemo_tokenizer.pad_id)
            
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                model_input_names=["input_ids", "attention_mask"],
                **special_tokens
            )
            
            hf_tokenizer.save_pretrained(output_path)
            logging.info(f"ByteLevelTokenizer converted and saved to {output_path}")
            
        else:
            # Fall back to standard tokenizer handling
            try:
                self.tokenizer.tokenizer.save_pretrained(output_path)
            except Exception:
                logging.warning("Failed to save tokenizer")