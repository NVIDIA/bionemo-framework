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


"""
This file implements a general collate function for NeMo BERT model.
The functionality include utilities for BERT pre-training, which
generally relies on modifying random entries in the input:
by masking (so the token is ignored) and perturbing (so the
real token is replaced by a random alternative).

In order to generalize this functionality for any tokenizer,
which tend to have heterogeneous interfaces, we need to ensure
each tokenizer uses a consistent set of methods to identify tokens
and special tokens. For this reason, we introduce the `TokenizerAdapterSpec`
which is an interface that can be implemented to ensure a given tokenizer
has the methods required by the BERT collate function.

A generic interface for random samples used by the collate function is also
introduced.
"""

import copy
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from nemo.collections.common.tokenizers import TokenizerSpec
from pytest import approx


__all__ = [
    "TokenSampler",
    "TokenizerAdapterSpec",
    "SentencePieceTokenizerAdapter",
    "BertCollate",
    "BertMasking",
]


class TokenSampler:
    """Group of sampling methods for sampling from a list of tokens"""

    def sample_token_id(self, tokenizer: TokenizerSpec):
        """Samples a token id from the tokens vocabulary

        Args:
            tokenizer (TokenizerSpec): tokenizer to sample from

        Returns
            int: ID sampled from the tokenizers vocab

        """
        return random.randint(0, len(tokenizer.vocab) - 1)

    def sample_indices(self, num_tokens: int, num_samples: int):
        """Samples a specifieid number of indices from a vocabulary size

        Args:
            num_tokens (int): Total number of tokens to sample from
            num_samples (int): Total number of indices to sample

        Returns:
            List[int]: Sampled indices of length `num_samples`
        """
        return random.sample(range(num_tokens), num_samples)


class TokenizerAdapterSpec:
    """Interface for adapting tokenizers for DataLoaders.

    Tokenizers can be fairly homogenous, so an Adapter is used
    to create consistent interface for code that depends on various
    tokenizers.
    """

    def __init__(self, tokenizer: TokenizerSpec):
        """

        Args:
            tokenizer (TokenizerSpec): Tokenizer to adapt


        """
        self.tokenizer = tokenizer

    def __getattr__(self, attr: str):
        """Gets attributes from the object.

        This allows the adapter to get attributes of original tokenizer,
        such as `tokenizer.text_to_tokens()`, which allows the adapter
        to also be treated by client code as if it is the original tokenizer.

        Args:
            attr (str): Name of attribute to get.

        Returns:
            Any: Value of the attribute

        """
        if attr not in self.__dict__:
            return getattr(self.tokenizer, attr)
        return super().__getattr__(attr)

    def get_bos_id(self):
        """Gets ID for beginning of sentence token

        Returns:
            int: ID for beginning of sentence token

        """
        raise NotImplementedError()

    def get_bos_token(self):
        """Gets  beginning of sentence token

        Returns:
            str: beginning of sentence token

        """
        raise NotImplementedError()

    def get_eos_id(self):
        """Gets ID for beginning of sentence token

        Returns:
            int: ID for beginning of sentence token

        """
        raise NotImplementedError()

    def get_eos_token(self):
        """Gets end of sentence token

        Returns:
            str: end of sentence token

        """
        raise NotImplementedError()

    def get_pad_id(self):
        """Gets ID for pad token

        Returns:
            int: ID for pad token

        """
        raise NotImplementedError()

    def get_pad_token(self):
        """Gets pad token

        Returns:
            str: pad token

        """
        raise NotImplementedError()

    def get_mask_token(self):
        """Gets mask token

        Returns:
            str: mask token

        """
        raise NotImplementedError()

    def get_mask_id(self):
        """Gets the mask id

        Returns:
            int: the mask id

        """
        raise NotImplementedError()

    def is_special_token(self, token: str):
        """Determines if a token is a special token

        Args:
            token (str): Token to test

        Returns:
            bool: True if `token` is a special token. False otherwise.
        """
        raise NotImplementedError()

    def vocab_list(self):
        """A list representation of the vocabulary.

        Returns:
            List[str]: Contains all of the tokens in the vocabulary

        """
        raise NotImplementedError()


class SentencePieceTokenizerAdapter(TokenizerAdapterSpec):
    """
    A Tokenizer Adapter for NeMo's SentencePieceTokenizer
    """

    def get_bos_id(self):
        """Gets ID for beginning of sentence token

        Returns:
            int: ID for beginning of sentence token

        """
        return self.bos_id

    def get_bos_token(self):
        """Gets  beginning of sentence token

        Returns:
            str: beginning of sentence token

        """
        return self.vocab[self.get_bos_id()]

    def get_eos_id(self):
        """Gets ID for beginning of sentence token

        Returns:
            int: ID for beginning of sentence token

        """
        return self.eos_id

    def get_eos_token(self):
        """Gets end of sentence token

        Returns:
            str: end of sentence token

        """
        return self.vocab[self.get_eos_id()]

    def get_pad_id(self):
        """Gets ID for pad token

        Returns:
            int: ID for pad token

        """
        return self.pad_id

    def get_pad_token(self):
        """Gets pad token

        Returns:
            str: pad token

        """
        return self.vocab[self.get_pad_id()]

    def get_mask_token(self):
        """Gets mask token

        Returns:
            str: mask token

        """
        return "<mask>"

    def get_mask_id(self):
        """Gets the mask id

        Returns:
            int: the mask id

        """
        return self.tokens_to_ids(self.get_mask_token())[0]

    def is_special_token(self, token):
        """Determines if a token is a special token

        Args:
            token (str): Token to test

        Returns:
            bool: True if `token` is a special token. False otherwise.
        """
        return token.startswith("<")

    def vocab_list(self):
        """A list representation of the vocabulary.

        Returns:
            List[str]: Contains all of the tokens in the vocabulary

        """
        return self.tokenizer.vocab


@dataclass
class BertMasking:
    """
    Produces a callable that can be used to mask/perturb sequences of tokens.

    Arguments:
        tokenizer (TokenizerAdapterSpec): Tokenizer to use for determining mask
            token.
        modify_percent (float): The fraction of total tokens to modify
        perturb_token_percent (float): Of the total tokens being modified,
            fraction of tokens to perturb. Perturbation changes the
            tokens randomly to some other non-special token.
        mask_token_percent (float): Of the total tokens being modified,
            fraction of tokens make the [MASK].
        identity_token_percent (float): Of the total tokens being modified,
            fraction of tokens to leave unchanged.
        sampler (TokenSampler): Sampler to use for determining indices
            and tokens to use for modifying the token strings.

    """

    tokenizer: TokenizerAdapterSpec
    # The following defaults come from the BERT/ESM publications.
    modify_percent: float = 0.15
    mask_token_percent: float = 0.8
    perturb_token_percent: float = 0.1
    identity_token_percent: float = 0.1
    sampler: TokenSampler = TokenSampler()

    def _perturb_token(self, token, indices):
        """Replace the token with some other token from the vocab

        Args:
        token: a list of tokens
        indices: Indices to perturb / corrupt

        Returns:
        token: modified tokens
        """
        new_tokens = copy.copy(token)
        for idx, t in enumerate(token):
            if idx in indices:
                randid = self.sampler.sample_token_id(self.tokenizer)
                # Make sure random token is same as original token.
                # Also make sure the token isn't one of the special tokens.
                # Special tokens are <unk>, <s>, <\s>, <pad>, <mask>
                sampled_token = self.tokenizer.vocab_list()[randid]
                while t == sampled_token or self.tokenizer.is_special_token(sampled_token):
                    randid = self.sampler.sample_token_id(self.tokenizer)
                    sampled_token = self.tokenizer.vocab_list()[randid]
                # replace token with a random token from vocab
                new_tokens[idx] = sampled_token

        return new_tokens

    def _mask_token(self, token, indices):
        """Replace tokens at indices with MASK token

        Args:
        token: list of tokens
        indices: Indices to replaces with MASK token

        Return:
        token: modified tokens
        """
        mask_token = self.tokenizer.get_mask_token()
        new_tokens = copy.copy(token)
        for idx in indices:
            new_tokens[idx] = mask_token
        return new_tokens

    def __call__(
        self,
        tokens,
    ):
        """Modify tokens to add masking and perturbations

        Args:
            tokens (List[List[str]]): the entries of the ineer lsits contain
                tokens to modify

        Returns:
            modified_tokens (List[List[str]]): Tokens with perturb_percent
                amount of corruption
            token_masks (List[List[float]]): Indicated if the token masks

        """
        if self.modify_percent > 1.0 or self.modify_percent < 0:
            raise ValueError("Modification ratio must be between 0 and 1")
        if self.perturb_token_percent > 1.0 or self.perturb_token_percent < 0:
            raise ValueError("Perturbation ratio must be between 0 and 1")
        if self.mask_token_percent > 1.0 or self.mask_token_percent < 0:
            raise ValueError("Mask token ratio must be between 0 and 1")
        if self.identity_token_percent > 1.0 or self.identity_token_percent < 0:
            raise ValueError("Identity token ratio must be between 0 and 1")
        if self.mask_token_percent + self.identity_token_percent + self.perturb_token_percent != approx(1.0):
            raise ValueError("Mask token, identity token, and perturb token ratios must sum to 1")

        modified_tokens = []
        token_masks = []
        for token in tokens:
            token_len = len(token)
            mod_tokens = copy.deepcopy(token)  # make a copy of tokens that we modify below

            # Use binomial probability of masks so that we can handle short sequences probabilistically.
            mask = np.random.binomial(1, self.modify_percent, token_len).astype(bool)
            if self.modify_percent > 0:
                n_masked = mask.sum()
                if n_masked == 0:
                    # If we have no tokens to mask, we need to mask at least one if there is a non-zero request for mask.
                    mask[np.random.randint(0, token_len)] = True

            # Step 3: Assign categories to each mask type
            category_probs = np.random.rand(token_len)
            mask_mask = (category_probs < self.mask_token_percent) & mask
            perturb_mask = (
                (category_probs >= self.mask_token_percent)
                & (category_probs < (self.mask_token_percent + self.perturb_token_percent))
            ) & mask
            # The following category is implicit, but this is what we would use if we were doing this.
            # identity_mask = (category_probs >= (self.mask_token_percent + self.perturb_token_percent)) & mask

            # There is an implicit third category of tokens, num_identities, which are the tokens that are not perturbed or masked.
            #  ie we cannot assume that the number mask_mask | perturb_mask == mask, since some will also be identities.

            mask_indices = np.where(mask_mask)[0]
            perturb_indices = np.where(perturb_mask)[0]
            # Each call below will mutate the sequence after the next line our tokens will be perturbed at the sampled indices.
            if len(perturb_indices) > 0:
                mod_tokens = self._perturb_token(mod_tokens, perturb_indices)
            if len(mask_indices) > 0:
                mod_tokens = self._mask_token(mod_tokens, mask_indices)

            # Note, the third category, num_identites, is not perturbed or masked, so that token isn't
            #  modified, but it is still included in the loss mask since it is a member of mask.
            modified_tokens.append(mod_tokens)

            token_masks.append(mask.astype(int).tolist())  # convert the mask to a list for output.

        return modified_tokens, token_masks


class BertCollate:
    def __init__(
        self,
        tokenizer: TokenizerAdapterSpec,
        seq_length: int,
        pad_size_divisible_by_8: bool,
        masking_strategy: Optional[Callable] = None,
        dynamic_padding: bool = True,
    ):
        """

        Arguments:
            tokenizer (TokenizerAdapterSpec): The desired tokenizer
                for collation.
            seq_length (int): Final length of all sequences
            pad_size_divisible_by_8 (bool): If True, the sequence length is
                automatically increased to be divisible by 8. Needed for NeMo.
            masking_strategy: A callable that takes a List[List[str]] and
                returns Tuple[List[List[str]], List[List[float]]], where each
                list has the same shape as the original.
                Used to modify the unadulterated token strings for pre-training
                purposes.
            dynamic_padding: If True, enables dynamic batch padding, where
                each batch is padded to the maximum sequence length within that batch.
                By default True.
        """
        self.sampler = TokenSampler()
        self._tokenizer = tokenizer
        self.seq_length = seq_length
        if masking_strategy is None:
            masking_strategy = BertMasking(
                self.tokenizer,
            )
        self.masking_strategy = masking_strategy
        self.dynamic_padding = dynamic_padding
        # workaround for CUDA alignment bug
        self.pad_size_divisible_by_8 = pad_size_divisible_by_8

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, other):
        self._tokenizer = self.adapter_cls(other)

    def _check_seq_len(self, tokens: List[List[str]], mask: List[List[int]]):
        """Warn user and shorten sequence if the tokens are too long,
        otherwise return original

        Args:
            tokens (List[List[str]]): List of token sequences
            mask (List[List[int]]): List of mask sequences

        Returns:
            tokens (List[List[str]]): List of token sequences (shortened,
                if necessary)
            mask (List[List[int]]): List of mask sequences (shortened,
                if necessary)
        """

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self.seq_length:
            tokens_short = [ts[: self.seq_length] for ts in tokens]
            mask_short = [ms[: self.seq_length] for ms in mask]
            return (tokens_short, mask_short)
        return (tokens, mask)

    def tokenize_batch(self, batch: List[str]):
        """Tokenizes a batch of strings

        Args:
            batch (List[str]): Batch of input strings

        Returns:
            List[List[str]]: Batch of tokens

        """
        tokens = self.tokenize(batch)
        tokens = [[self.tokenizer.get_bos_token()] + ts + [self.tokenizer.get_eos_token()] for ts in tokens]
        return tokens

    def _prepare_tokens_and_masks(self, tokens: List[List[str]], mask_data: bool = True):
        """Prepare tokens and masks for encoder or decoder from batch of input
        strings

        Args:
            tokens (List[List[[str]]): Batch of input tokens
            mask_data (bool, optional): Mask decoder tokens. Defaults to False.

        Returns:
            dict: token output
        """
        if mask_data:
            tokens, mask = self.masking_strategy(tokens)
        else:
            # 1/True = Active, 0/False = Inactive
            mask = [[True] * len(ts) for ts in tokens]
        # Verify sequence length
        tokens, mask = self._check_seq_len(tokens, mask)

        token_output = {"tokens": tokens, "mask": mask}

        return token_output

    def _pad_seqs(self, seqs, masks, pad_token):
        # TODO: switch to torch.nn.utils.rnn.pad_sequence
        if self.dynamic_padding:
            pad_length = max([len(seq) for seq in seqs])
        else:
            pad_length = self.seq_length
        if self.pad_size_divisible_by_8:
            pad_length = int(math.ceil(pad_length / 8) * 8)

        padding_mask = [[1] * len(seq) + [0] * (pad_length - len(seq)) for seq in seqs]
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        # 1/True = Active, 0/False = Inactive
        masks = [mask + ([0] * (pad_length - len(mask))) for mask in masks]
        return padded, masks, padding_mask

    def collate_fn(self, batch: List[str], label_pad: int = -1) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Collate function for NeMo BERT.
        The function prepares the dictionary of tensors expected by the BERT language model.
        - It tokenizes the input strings based on the provided tokenizer `self._tokenize`,
        - randomly masks a subset of tokens in each sequence, and
        - pads the sequences to the maximum length in the given batch, using the pad index from `self.tokenizer.get_pad_id()`.

        Args:
            batch: A list of Strings to tokenize.

        Returns:
            collate_output: A dictionary with the following key-value pairs:
                {
                    'text' (torch.Tensor[int]): padded and modified sequences of token IDs,
                    'types' (torch.Tensor[int]): possible types within tokens, generally used to refer to tokens from two segments
                        in the same input string.  In this case set all to 0, as each string corresponds to exactly one segment.
                    'is_random' (torch.Tensor[int]): order of the sequences in the batch, len(is_random) == len(text)
                    'loss_mask' (List[List[float]]): same shape as `text`. 1 indicates the token has been modified,
                                                     otherwise 0,
                    'labels' (List[List[int]]): same shape as `text`. Contains the padded but non-modified versions of
                                                the token IDs.
                    'padding_mask' (List[List[float]]): same shape as `text`. 1 indicates the corresponding position
                                                        is not a pad token. 0 indicates the position is a pad token.
                    'batch' (List[str]): contains the unmodified strings that were tokenized.
                }
        """

        # Dimensions required by NeMo: [batch, sequence + padding]
        # Encoder
        tokens = self.tokenize_batch(batch)
        encoder_dict = self._prepare_tokens_and_masks(tokens, mask_data=True)
        encoder_tokens = encoder_dict["tokens"]
        loss_mask = encoder_dict["mask"]
        enc_token_ids = [self.tokenizer.tokens_to_ids(t) for t in encoder_tokens]
        enc_token_ids, loss_mask, padding_mask = self._pad_seqs(enc_token_ids, loss_mask, self.tokenizer.get_pad_id())

        label_dict = self._prepare_tokens_and_masks(tokens, mask_data=False)
        label_tokens = label_dict["tokens"]
        label_mask = label_dict["mask"]
        label_ids = [self.tokenizer.tokens_to_ids(t) for t in label_tokens]
        label_ids, _, _ = self._pad_seqs(label_ids, label_mask, self.tokenizer.get_pad_id())

        loss_mask = torch.tensor(loss_mask, dtype=torch.int64)
        enc_token_ids = torch.tensor(enc_token_ids, dtype=torch.int64)
        padding_mask = torch.tensor(padding_mask, dtype=torch.int64)
        label_ids = torch.tensor(label_ids, dtype=torch.int64)
        types = torch.zeros_like(enc_token_ids).long()  # expected by training & validation methods
        sentence_order = torch.arange(len(enc_token_ids)).long()  # expected by training & validation methods

        collate_output = {
            "text": enc_token_ids,
            "types": types,
            "is_random": sentence_order,
            "loss_mask": loss_mask,
            "labels": label_ids,
            "padding_mask": padding_mask,
            "batch": batch,
        }

        return collate_output

    def tokenize(self, sents1, mask=False):
        # TODO this function needs cleanup
        tokens = [self.tokenizer.text_to_tokens(s) for s in sents1]
        return tokens
