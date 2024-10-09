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


import os
from itertools import product
from typing import Iterable, List, Optional, Union

from nemo.collections.common.tokenizers import TokenizerSpec


DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_MASK_TOKEN = "<MASK>"
DEFAULT_UNK_TOKEN = "<UNK>"
DEFAULT_BEGIN_TOKEN = "<CLS>"
DEFAULT_END_TOKEN = "<SEP>"

__all__ = ["KmerTokenizer"]


class KmerTokenizer(TokenizerSpec):
    def __init__(
        self,
        k: int,
        begin_token: Optional[str] = DEFAULT_BEGIN_TOKEN,
        end_token: Optional[str] = DEFAULT_END_TOKEN,
        pad_token: Optional[str] = DEFAULT_PAD_TOKEN,
        unk_token: Optional[str] = DEFAULT_UNK_TOKEN,
        mask_token: Optional[str] = DEFAULT_MASK_TOKEN,
    ):
        """Initializes the k-mer Tokenizer

        Args:
            k (int): Length of k-mers to use
            begin_token (str): Token to use at start of each sequence
            end_token (str): Token to use at end of each sequence
            pad_token (str): Token to use when padding batches of sequences
            unk_token (str): Token to use for tokens which are not in the vocabulary
            mask_token (str): Token to use when masking pieces of the sequence

        Examples:
            >>> tokenizer = KmerTokenizer(k=3)
            >>> s = 'ACGTCG'
            >>> tokenizer.text_to_tokens(s)
            ['ACG', 'CGT', 'GTC', 'TCG']

        """
        self.k = k

        self.vocab = {
            pad_token: 0,  # pad_token
            unk_token: 1,  # unk_token
            begin_token: 2,  # begin_token
            end_token: 3,  # end_token
            mask_token: 4,  # mask_token
        }

        self._update_index()

        self.begin_token = begin_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

    @staticmethod
    def _infer_vocab_file(model_file, vocab_file):
        if vocab_file is None:
            vocab_file = os.path.splitext(model_file)[0] + ".vocab"
        return vocab_file

    def save_vocab(self, model_file, vocab_file=None):
        """Saves information about the vocab.

        Args:
            model_file (str): file-containing the tokenizer model/k
            vocab_file (Optional[str]): file-path containing the
                vocabulary. If not supplied, it will be inferred
                from `model_file`

        """

        vocab_file = KmerTokenizer._infer_vocab_file(model_file, vocab_file)

        with open(model_file, "w") as f:
            f.write(str(self.k))

        with open(vocab_file, "w") as f:
            for i in range(len(self.vocab)):
                f.write(self.decode_vocab[i] + "\n")

    @staticmethod
    def from_vocab_file(model_file, vocab_file=None):
        """Instantiates a Tokenizer from a vocab

        Args:
            model_file (str): file-containing the tokenizer model/k
            vocab_file (Optional[str]): file-path containing the
                vocabulary. If not supplied, it will be inferred
                from `model_file`

        Returns:
            KmerTokenizer: Tokenizer with the given vocabulary
        """

        vocab_file = KmerTokenizer._infer_vocab_file(model_file, vocab_file)

        with open(model_file) as f:
            k = int(f.read().strip())

        with open(vocab_file) as f:
            ids_to_text = {id: line.strip() for id, line in enumerate(f.readlines())}

        tokenizer = KmerTokenizer(
            k=k,
            pad_token=ids_to_text[0],
            unk_token=ids_to_text[1],
            begin_token=ids_to_text[2],
            end_token=ids_to_text[3],
            mask_token=ids_to_text[4],
        )

        vocab = (ids_to_text[id] for id in range(5, len(ids_to_text)))

        tokenizer.build_vocab(vocab)

        return tokenizer

    def _update_index(self):
        """
        Updates the id_to_vocab index based on the current vocab
        """
        self.decode_vocab = {id_: token for token, id_ in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocab being used."""
        return len(self.vocab)

    def text_to_tokens(self, text: str) -> List[str]:
        """Converts a string to tokens

        Args:
            text (str): A string to convert to k-mer based tokens

        Returns:
            (List[str]): A list containing k-mer based tokens

        """
        tokens = []
        for i in range(len(text) - self.k + 1):
            token = text[i : i + self.k]
            tokens.append(token)
        return tokens

    def tokens_to_text(self, tokens):
        raise NotImplementedError("Non-ambiguous mapping from tokens to text does not" "exist.")

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indexes/ids

        Args:
            tokens (List[str]):  Containing tokens
        Returns:
            (List[int]): Containing ID's for each token

        """
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert Ids to tokens

        Args:
            ids (List[int]): Containg ids for each token

        Returns:
            (List[str]): Containing tokens
        """
        tokens = []
        for id_ in ids:
            token = self.decode_vocab.get(id_)
            if token is None:
                raise ValueError(f"Do not recognize ID: {id_}")
            tokens.append(token)
        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Converts text to ids

        Args:
            text (str): String containing text to convert

        Returns:
            (List[int]): Id's corresponding to the tokenization
            of the text
        """
        tokens = self.text_to_tokens(text)
        return self.tokens_to_ids(tokens)

    def ids_to_text(self, ids):
        raise NotImplementedError("Non-ambiguous mapping from IDs to text does not" "exist.")

    def build_vocab(self, strings: Union[str, Iterable[str]]):
        """Builds the vocabulary of the tokenizer from strings

        Args:
            strings: (Union[str, Iterable[str]]): Strings to
                build the vocabulary with. If a string is supplied,
                then the vocabulary is built from the single string.
                Otherwise, the vocabulary is progressively built
                from all of the strings in `strings`.

        """

        if isinstance(strings, str):
            strings = [strings]

        for string in strings:
            tokens = self.text_to_tokens(string)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.decode_vocab[self.vocab[token]] = token

        return self

    def build_vocab_from_k(self, alphabet: List[str] = None):
        """Builds the complete k-mer vocabulary on an alphabet

        E.g.,

        the 3-mer vocabulary on alphabet=['A', 'C'] is ['AAA', 'AAC', 'ACA',
        'ACC', 'CAA', 'CAC', 'CCA', 'CCC']


        Args:
            alphabet (List[str], optional): Alphabet to use to generate k-mers.
            Defaults to ['A', 'C', 'G', 'T'].
        """

        if alphabet is None:
            alphabet = ["A", "C", "G", "T"]
        for kmer in product(alphabet, repeat=self.k):
            self.build_vocab("".join(kmer))
