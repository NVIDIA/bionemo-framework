# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
import os
from typing import ClassVar, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class AsciiOrdTokenizer(PreTrainedTokenizer):
    """A tokenizer that maps characters directly to their ASCII/Unicode ord() values.

    Modes:
    - 256: Strictly follows standard extended ASCII (0-255).
    - 512: Allows for an extended vocabulary (0-511).
    """

    model_input_names: ClassVar[list[str]] = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_size_limit: int = 256,
        unk_token: str = "[UNK]",
        bos_token: str | None = None,  # ASCII Start of Text
        eos_token: str = "\x00",  # ASCII Null
        pad_token: str = "\x01",  # ASCII Start of Heading
        sep_token: str = "\x03",  # ASCII End of Text
        **kwargs,
    ):
        """Temporary tokenizer for Evo2.

        Args:
            vocab_size_limit: Either 256 or 512.
            unk_token: Token to use for characters outside the vocab range.
            bos_token: Token to use for beginning of sequence.
            eos_token: Token to use for end of sequence.
            pad_token: Token to use for padding.
            sep_token: Token to use for separation.
            **kwargs: Additional keyword arguments.
        """
        self.vocab_size_limit = vocab_size_limit

        # We manually construct the vocab to ensure ord(c) == id
        # Characters 0 through limit-1 map to themselves.
        self.char_to_id = {chr(i): i for i in range(vocab_size_limit)}
        self.id_to_char = {i: chr(i) for i in range(vocab_size_limit)}

        # Add specific handling for the unknown token if it's not a single char
        # or if we want to map unknown chars to a specific ID (like the max ID)
        self.unk_token_id_val = vocab_size_limit - 1

        # If the UNK token is a string literal like "[UNK]", we map it to the last available ID
        if unk_token not in self.char_to_id:
            # We override the last slot for UNK if it's a special string
            self.char_to_id[unk_token] = self.unk_token_id_val
            self.id_to_char[self.unk_token_id_val] = unk_token

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            sep_token=sep_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size."""
        return self.vocab_size_limit

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary."""
        return dict(self.char_to_id)

    def _tokenize(self, text: str) -> List[str]:
        """Converts a string into a list of characters."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Maps a character to its ord() value."""
        # If the token is one of our special single-char tokens (like \x00), return ord
        if len(token) == 1:
            val = ord(token)
            if val < self.vocab_size_limit:
                return val
            return self.unk_token_id_val

        # Fallback for special string tokens (like "[UNK]")
        return self.char_to_id.get(token, self.unk_token_id_val)

    def _convert_id_to_token(self, index: int) -> str:
        """Maps an ID back to its character."""
        if 0 <= index < self.vocab_size_limit:
            return self.id_to_char.get(index, self.unk_token)
        return self.unk_token

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens.

        Returns:
            [BOS] + token_ids_0 + [EOS]
        """
        bos = [self.bos_token_id] if self.bos_token_id is not None else []
        eos = [self.eos_token_id] if self.eos_token_id is not None else []

        if token_ids_1 is None:
            return bos + token_ids_0

        return bos + token_ids_0 + eos + token_ids_1

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the vocabulary to a JSON file.

        Required for tokenizer.save_pretrained() to work.

        Args:
            save_directory: Directory to save the vocabulary to.
            filename_prefix: Prefix to add to the filename.

        Returns:
            Tuple of the vocabulary file path.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")

        with open(vocab_file, "w", encoding="utf-8") as f:
            safe_vocab = dict(self.char_to_id)
            json.dump(safe_vocab, f, ensure_ascii=False)

        return (vocab_file,)


# ==========================================
# DEMO / USAGE SCENARIO
# ==========================================
if __name__ == "__main__":
    print("--- Testing 256 Limit Tokenizer ---")

    # 1. Initialize for the efficient model
    tokenizer_256 = AsciiOrdTokenizer(vocab_size_limit=256)

    # Check IDs of special tokens to ensure they aren't None
    print(f"BOS ID: {tokenizer_256.bos_token_id}")
    print(f"EOS ID: {tokenizer_256.eos_token_id}")
    print(f"PAD ID: {tokenizer_256.pad_token_id}")

    text = "Hello World!"

    # encode() calls build_inputs_with_special_tokens which we now implemented
    encoded = tokenizer_256.encode(text, add_special_tokens=True)

    print(f"Input: '{text}'")
    print(f"Tokens: {encoded}")
    # Expect: [2, 72, ..., 0]

    decoded = tokenizer_256.decode(encoded)
    print(f"Decoded: '{decoded}'")

    print("\n--- Testing 512 Limit Tokenizer ---")

    # 2. Initialize for the larger model
    tokenizer_512 = AsciiOrdTokenizer(vocab_size_limit=512)

    # Test with a character outside normal range
    # 'Ĭ' is 300
    special_char_text = "A" + chr(300) + "B"
    encoded_512 = tokenizer_512.encode(special_char_text, add_special_tokens=False)

    print("Input: A + chr(300) + B")
    print(f"Tokens: {encoded_512}")
    # Expect: [65, 300, 66]
