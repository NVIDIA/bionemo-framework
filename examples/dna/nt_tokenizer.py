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


from typing import Optional

from bionemo.data.dataloader.collate import TokenizerAdapterSpec


DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_MASK_TOKEN = "<MASK>"
DEFAULT_UNK_TOKEN = "<UNK>"
DEFAULT_BEGIN_TOKEN = "<CLS>"
DEFAULT_END_TOKEN = "<SEP>"


class NtTokenizer(TokenizerAdapterSpec):
    def __init__(
        self,
        begin_token: Optional[str] = DEFAULT_BEGIN_TOKEN,
        end_token: Optional[str] = DEFAULT_END_TOKEN,
        pad_token: Optional[str] = DEFAULT_PAD_TOKEN,
        unk_token: Optional[str] = DEFAULT_UNK_TOKEN,
        mask_token: Optional[str] = DEFAULT_MASK_TOKEN,
    ):
        """Initializes the k-mer Tokenizer
        Args:
            begin_token (str): Token to use at start of each sequence
            end_token (str): Token to use at end of each sequence
            pad_token (str): Token to use when padding batches of sequences
            unk_token (str): Token to use for tokens which are not in the vocabulary
            mask_token (str): Token to use when masking pieces of the sequence
        Examples:
            >>> tokenizer = NtTokenizer()
            >>> s = 'ACGTCG'
            >>> tokenizer.text_to_tokens(s)
        """
        # First describe the extra tokens, and the integers that represent them, in a dictionary.
        self.begin_token = begin_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        # The IDs have to be < len(self.vocab)
        self.vocab = {
            pad_token: 0,  # pad_token
            unk_token: 1,  # unk_token
            begin_token: 2,  # begin_token
            end_token: 3,  # end_token
            mask_token: 4,  # mask_token
        }
        self.special_tokens = set(self.vocab.keys())

        # This is our nucleotide alphabet, 'N' being an ambiguous base, and lowercase being softmasked bases.
        for base in "NACGTnacgt":
            self.vocab[base] = len(self.vocab)

        self.decode_vocab = self.reverse_vocab_lookup()

    def reverse_vocab_lookup(self):
        """
        Updates the id_to_vocab index based on the current vocab
        """
        return {id_: token for token, id_ in self.vocab.items()}

    # TODO these are from TokenizerSpec in NeMo
    def text_to_tokens(self, text):
        return list(text)

    def tokens_to_text(self, tokens):
        return "".join(tokens)

    def tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def ids_to_tokens(self, ids):
        return [self.decode_vocab[_id] for _id in ids]

    # NOTE Shouldnt these be provided by the baseclass?
    def text_to_ids(self, text):
        return self.tokens_to_ids(self.text_to_tokens(text))

    def ids_to_text(self, ids):
        return self.tokens_to_text(self.ids_to_tokens(ids))

    # These are from the TokenizerAdapterSpec for Dataloaders
    def get_bos_id(self):
        """Gets ID for beginning of sentence token

        Returns:
            int: ID for beginning of sentence token

        """
        return self.vocab[self.begin_token]

    def get_bos_token(self):
        """Gets  beginning of sentence token

        Returns:
            str: beginning of sentence token

        """
        return self.begin_token

    def get_eos_id(self):
        """Gets ID for beginning of sentence token

        Returns:
            int: ID for beginning of sentence token

        """
        return self.vocab[self.end_token]

    def get_eos_token(self):
        """Gets end of sentence token

        Returns:
            str: end of sentence token

        """
        return self.end_token

    def get_pad_id(self):
        """Gets ID for pad token

        Returns:
            int: ID for pad token

        """
        return self.vocab[self.pad_token]

    def get_pad_token(self):
        """Gets pad token

        Returns:
            str: pad token

        """
        return self.pad_token

    def get_mask_token(self):
        """Gets mask token

        Returns:
            str: mask token

        """
        return self.mask_token

    def get_mask_id(self):
        """Gets the mask id

        Returns:
            int: the mask id

        """
        return self.vocab[self.mask_token]

    def is_special_token(self, token: str):
        """Determines if a token is a special token

        Args:
            token (str): Token to test

        Returns:
            bool: True if `token` is a special token. False otherwise.
        """
        return token in self.special_tokens

    def vocab_list(self):
        """A list representation of the vocabulary.

        Returns:
            List[str]: Contains all of the tokens in the vocabulary

        """
        return list(self.vocab.keys())


test_sequences = ["ATGAATAGATAGATAGAGATATAGA", "naTGATGGggaCccCACACGAN"]
tokenizer = NtTokenizer()
for text in test_sequences:
    print(tokenizer.text_to_tokens(text))
    print(tokenizer.text_to_ids(text))
    ids = tokenizer.text_to_ids(text)
    _text = tokenizer.ids_to_text(ids)
    print(text)
    print(_text)
    print()
# Show this works with a DataLoader
