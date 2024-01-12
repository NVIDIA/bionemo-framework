# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import List, TypedDict

from nemo.collections.common.tokenizers import TokenizerSpec

from bionemo.data.dataloader.collate import (
    BertCollate,
    BertMasking,
    SentencePieceTokenizerAdapter,
)


__all__ = ['ProteinBertCollate']


class ProteinBertCollate(BertCollate):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        seq_length: int,
        pad_size_divisible_by_8: bool,
        modify_percent: float = 0.1,
        perturb_percent: float = 0.5,
        dynamic_padding: bool = True,
    ):
        """
        A collate function for Protein sequences.

        Arguments:
            tokenizer (TokenizerSpec): The desired tokenizer for collation
            seq_length (int): Final length of all sequences
            pad_size_divisible_by_8 (bool): Makes pad size divisible by 8.
                Needed for NeMo.
            modify_percent (float): The percentage of total tokens to modify
            perturb_percent (float): Of the total tokens being modified,
                percentage of tokens to perturb. Perturbation changes the
                tokens randomly to some other non-special token.
            dynamic_padding: If True, enables dynamic batch padding, where
                each batch is padded to the maximum sequence length within that batch.
                By default True.
        """
        tokenizer = SentencePieceTokenizerAdapter(tokenizer)
        masking = BertMasking(
            tokenizer=tokenizer,
            modify_percent=modify_percent,
            perturb_percent=perturb_percent,
        )
        super().__init__(
            tokenizer=tokenizer,
            seq_length=seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            masking_strategy=masking,
            dynamic_padding=dynamic_padding,
        )


class BatchItem(TypedDict):
    sequence: str
    sequence_id: str


class ESM2BertCollate(ProteinBertCollate):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        seq_length: int,
        pad_size_divisible_by_8: bool,
        modify_percent: float = 0.1,
        perturb_percent: float = 0.5,
        dynamic_padding: bool = True,
    ):
        """Extends the parent class by allowing collate_fns that operate on dictionaries rather
        that lists of strings.

        Arguments:
            tokenizer (TokenizerSpec): The desired tokenizer for collation
            seq_length (int): Final length of all sequences
            pad_size_divisible_by_8 (bool): Makes pad size divisible by 8.
                Needed for NeMo.
            modify_percent (float): The percentage of total tokens to modify
            perturb_percent (float): Of the total tokens being modified,
                percentage of tokens to perturb. Perturbation changes the
                tokens randomly to some other non-special token.
            dynamic_padding: If True, enables dynamic batch padding, where
                each batch is padded to the maximum sequence length within that batch.
                By default True.
        """
        super().__init__(
            tokenizer=tokenizer,
            seq_length=seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            modify_percent=modify_percent,
            perturb_percent=perturb_percent,
            dynamic_padding=dynamic_padding,
        )

    def collate_fn(self, batch: List[BatchItem], label_pad: int = -1):
        '''Modifies the underlying collate_fn to handle a dictionary as input instead of a list of sequences.'''
        return super().collate_fn([x['sequence'] for x in batch], label_pad=label_pad)
