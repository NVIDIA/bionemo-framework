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


from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from nemo.collections.common.tokenizers import TokenizerSpec

from bionemo.data.dataloader.collate import (
    BertCollate,
    BertMasking,
    TokenizerAdapterSpec,
)


__all__ = [
    "KmerTokenizerAdapter",
    "KmerBertCollate",
    "SpanMasking",
    "LengthTruncator",
    "DeterministicLengthTruncator",
]


class KmerTokenizerAdapter(TokenizerAdapterSpec):
    """
    A Tokenizer Adapter for BioNeMo's KmerTokenizer

    Args:
        tokenizer (TokenizerSpec): tokenizer to adapt

    """

    def __init__(self, tokenizer: TokenizerSpec):
        super().__init__(tokenizer)
        tk = self.tokenizer
        self._special_tokens = {
            tk.pad_token,
            tk.unk_token,
            tk.begin_token,
            tk.end_token,
            tk.mask_token,
        }
        self._vocab_list = [self.decode_vocab[i] for i in range(len(self.vocab))]

    def get_bos_id(self):
        """Gets ID for beginning of sentence token

        Returns:
            int: ID for beginning of sentence token

        """
        return self.vocab[self.tokenizer.begin_token]

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
        return self.vocab[self.tokenizer.end_token]

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
        return self.vocab[self.tokenizer.pad_token]

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
        return self.vocab[self.mask_token]

    def is_special_token(self, token):
        """Determines if a token is a special token

        Args:
            token (str): Token to test

        Returns:
            bool: True if `token` is a special token. False otherwise.
        """
        return token in self._special_tokens

    def vocab_list(self):
        """A list representation of the vocabulary.

        Returns:
            List[str]: Contains all of the tokens in the vocabulary

        """
        return self._vocab_list


@dataclass
class SpanMasking:
    """A strategy for masking sequences that masks a minimal neighborhood
    of length `span_length` around each item it masks.
    This strategy originates from DNABERT.

    This masks tokens by determining `seed_probabilty * len(sequence)`
    indices (or seeds) where masks should be located, then it extends
    the masks in the neighborhood(s) around the seed(s) such that each
    string of masks is at least `span_length` long.


    E.g.,
    For the following sequence:
    ```
    ['A', 'C', 'G', 'T', 'A']
    ```

    For seed_probability=0.2, span_length=3,
    the following would be a valid masking:
    ['A', '<MASK>', '<MASK>', '<MASK>', 'T', 'A']

    For seed probability=0.4, span_length=2,
    the masking may look like:
    ['<MASK>', '<MASK>', 'G', '<MASK>', '<MASK>']

    Arguments:
        tokenizer (TokenizerAdapterSpec): Tokenizer to use for determining
        mask token.
        seed_probability (float): Value between 0 and 1 (inclusive).
            Determines the probability of positions in the sequences
            being chosen as seeds for masking
        span_length (int): Value greater than 0. Determines the length
            of spans that are masked around masking seeds.

    """

    tokenizer: TokenizerAdapterSpec
    seed_probability: float
    span_length: int

    def extending_seed_masking(self, seq):
        """Adds masked spans to `seq`.

        Arguments:
            seq (List[str]): Sequence to be masked.

        Returns:
            List[str]: A masked version of the sequence.

        """
        mask_indices = self.pick_mask_indices(seq)
        new_seq = np.array(seq, dtype=np.object_)
        loss_mask = np.zeros_like(new_seq, dtype=int)
        for idx in mask_indices:
            start = max(0, idx - (self.span_length - 1) // 2)
            end = min(idx + self.span_length // 2, len(seq) - 1)
            new_seq[start : end + 1] = self.tokenizer.get_mask_token()
            loss_mask[start : end + 1] = 1
        return new_seq.tolist(), loss_mask.tolist()

    def pick_mask_indices(self, seq):
        """Picks indices to use for span seeds in `seq`.
        Arguments:
            seq (List[str]): sequence to be masked.

        Returns:
            List[int] indices to mask in the sequence.

        """
        return np.random.choice(len(seq), size=int(self.seed_probability * len(seq)))

    def __call__(self, seqs):
        """Masks spans in each sequence in `seqs`.

        Arguments:
            seqs (List[List[str]]): List of sequences to be randomly masked.

        Returns:
            List[List[str]]: Masked versions of all sequences.

        """
        masked_tokens = [self.extending_seed_masking(seq) for seq in seqs]
        return [tokens[0] for tokens in masked_tokens], [tokens[1] for tokens in masked_tokens]


@dataclass
class LengthTruncatorABC:
    subsequence_probability: float = 0.5
    """
    Randomly subsamples the length of each sentence.

    Arguments:
        subsequence_probability (float): Probability that a sequence's
            length is reduced

    """

    @staticmethod
    def sample_length(sentence):
        """Determines the length of the subsequence to slice.

        Arguments:
            sentence (str): Sentence whose subsample length is to be determined.

        Returns:
            int: length of subsequence to slice.
        """
        raise NotImplementedError("Method not implemented.")

    @staticmethod
    def sample_probability(sentence):
        """Determines the random cutoff variable for subsampling.

        It is possible this probability depends on the sentence, but not
        necessarily.

        Arguments:
            sentence (str): Sentence whose subsample length is to be determined.

        Returns:
            float: between 0 and 1.
        """
        raise NotImplementedError("Method not implemented.")

    def __call__(self, sentences):
        """
        Arguments:
            sentences (List[str]): sentences to individually subsample.

        Returns:
            List[str]: contains the subsampled versions of `sentences`.

        """
        return [self.sample_sentence(sentence) for sentence in sentences]

    def get_sentence(self, entry):
        """
        Override this method if your dataset returns dict or something other
        than a string.
        """
        return entry

    def sample_sentence(self, entry):
        """Subsamples an individual sentence

        Arguments:
            entry (str): sentence to subsample.

        Returns
            str: subsampled version of the sentence.
        """
        sentence = self.get_sentence(entry)
        if self.sample_probability(sentence) < self.subsequence_probability:
            sentence = sentence[: self.sample_length(sentence)]
        return sentence


class LengthTruncator(LengthTruncatorABC):
    @staticmethod
    def sample_length(sentence):
        """Determines the length of the subsequence to slice.

        Arguments:
            sentence (str): Sentence whose subsample length is to be determined.

        Returns:
            int: length of subsequence to slice.
        """
        return np.random.choice(len(sentence)) + 1

    @staticmethod
    def sample_probability(sentence):
        """Determines the random cutoff variable for subsampling.

        sentence: Unused.

        Returns:
            float: between 0 and 1.
        """
        return np.random.random()


class DeterministicLengthTruncator(LengthTruncatorABC):
    probability_scale: int = 10**8

    @staticmethod
    def _hash(sentence):
        """Returns a hash for the sentence

        Arguments:
            sentence (str): Sentence to hash

        Returns:
            int: Hash of the string
        """
        return hash(sentence)

    def sample_length(self, sentence):
        """Determines the length of the subsequence to slice.

        Arguments:
            sentence (str): Sentence whose subsample length is to be determined.

        Returns:
            int: length of subsequence to slice.
        """
        return (self._hash(sentence) % len(sentence)) + 1

    def sample_probability(self, sentence):
        """Determines the random cutoff variable for subsampling.

        Returns:
            float: between 0 and 1.
        """
        hash_ = self._hash(sentence)
        const_ = self.probability_scale
        return hash_ % const_ / const_


class KmerBertCollate(BertCollate):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        seq_length: int,
        pad_size_divisible_by_8: bool,
        modify_percent: float = 0.15,
        perturb_token_percent: float = 0.1,
        identity_token_percent: float = 0.1,
        mask_token_percent: float = 0.8,
        masking_strategy: Optional[Callable] = None,
        transform_sentences: Optional[Callable] = None,
        dynamic_padding: bool = False,
    ):
        """
        A BertCollate function for k-mer based tokenization. This collate function
        specifically prepares input strings for consumption by a BERT model with
        a k-mer tokenization scheme. E.g., 'ACGTA' -> 'ACG', 'CGT', 'GTA', for 3-mers,
        though it is generalizable to any `k` with the use of `KmerTokenizer`.

        Arguments:
            tokenizer (TokenizerSpec): The desired tokenizer for collation
            seq_length (int): Final length of all sequences
            pad_size_divisible_by_8 (bool): Makes pad size divisible by 8.
                Needed for NeMo.
            modify_percent (float): The percentage of total tokens to modify (including identity) and include in the loss mask.
                Default is from BERT/ESM paper.
            perturb_token_percent (float): Of the total tokens being modified,
                percentage of tokens to perturb. Perturbation changes the
                tokens randomly to some other non-special token typically 0.1.
            identity_token_percent (float): Of the total tokens being modified, percentage of tokens to keep the same in the loss, typically 0.1.
            mask_token_percent (float): Of the total tokens being modified, percentage of tokens to mask typically 0.8.
            masking_strategy (Optional[Callable]): A callable that takes a
                List[List[str]] and returns
                Tuple[List[List[str]], List[List[float]]], where each
                list has the same shape as the original.
                Used to modify the unadulterated token strings for pre-training
                purposes.
            transform_sentences (Optional[Callable]): A callable that takes
                List[str] and returns List[str]. This allows sentences to be
                modified prior to tokenization.

        Examples:
            >>> import random
            >>> from bionemo.tokenizer import KmerTokenizer
            >>> from bionemo.data.dataloader import KmerBertCollate
            >>> s = 'ACGTCG'
            >>> tokenizer = KmerTokenizer(k=3)
            >>> tokenizer.build_vocab(s)
            >>> collate = KmerBertCollate(tokenizer, seq_length=7,
            ...     pad_size_divisible_by_8=False, modify_percent=0.5)
            >>> collate_fn = collate.collate_fn
            >>> random.seed(905)
            >>> collate_fn([s])
            {'text': tensor([[2, 5, 4, 7, 4, 8]]),
             'types': tensor([[0, 0, 0, 0, 0, 0]]),
             'is_random': tensor([0, 1, 2]),
             'loss_mask': tensor([[0, 0, 1, 0, 1, 1]]),
             'labels': tensor([[2, 5, 6, 7, 8, 3]]),
             'padding_mask': tensor([[1, 1, 1, 1, 1, 1]]),
             'batch': ['ACGTCG']}
        """
        tokenizer = KmerTokenizerAdapter(tokenizer)
        if masking_strategy is None:
            masking_strategy = BertMasking(
                tokenizer=tokenizer,
                modify_percent=modify_percent,
                mask_token_percent=mask_token_percent,
                identity_token_percent=identity_token_percent,
                perturb_token_percent=perturb_token_percent,
            )
        if transform_sentences is None:

            def transform_sentences(x):
                return x

        self.transform_sentences = transform_sentences
        super().__init__(
            tokenizer=tokenizer,
            seq_length=seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            masking_strategy=masking_strategy,
            dynamic_padding=dynamic_padding,
        )

    def tokenize(self, sentences):
        """
        Tokenizes the sentences

        Arguments:
            sentences (List[str]): Sentences to tokenize.

        Returns:
            List[List[str]]: Token sequences for each entry in `sentences`.

        """
        sentences = self.transform_sentences(sentences)
        tokens = [self.tokenizer.text_to_tokens(s) for s in sentences]
        return tokens
