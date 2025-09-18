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

"""Data collator for THD input format tests.

This should eventually get moved to a separate package, or possibly upstreamed into `transformers`.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator, PreTrainedTokenizerBase


class MLMDataCollatorWithFlattening:
    """Combines a DataCollatorForLanguageModeling and a DataCollatorWithFlattening.

    This data collator enables efficient training on variable-length sequences by:
    1. First flattening multiple sequences into a single packed tensor (no padding)
    2. Then applying MLM masking to the flattened sequence
    3. Providing Flash Attention metadata (cu_seq_lens) for sequence boundary awareness.
        Note. cu_seq_lens stands for cumulative sequence lengths.

    The result is a THD-format batch optimized for Flash Attention with sequence packing,
    eliminating the need for traditional attention masks while maintaining proper sequence
    boundaries during attention computation.

    Attributes:
        mlm_collator (DataCollatorForLanguageModeling): Handles MLM token masking.
        flattening_collator (DataCollatorWithFlattening): Handles sequence packing and
            Flash Attention metadata generation.

    Example:
        >>> from transformers import AutoTokenizer, DataCollatorForLanguageModeling
        >>> from transformers.data.data_collator import DataCollatorWithFlattening
        >>>
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        >>>
        >>> # Input: Variable-length protein sequences
        >>> sequences = [
        ...     {"input_ids": [0, 5, 6, 7, 2]},      # CLS + amino acids + EOS (5 tokens)
        ...     {"input_ids": [0, 8, 9, 10, 11, 2]}, # CLS + amino acids + EOS (6 tokens)
        ...     {"input_ids": [0, 12, 13, 2]},       # CLS + amino acids + EOS (4 tokens)
        ... ]
        >>>
        >>> # Create the collator
        >>> collator = MLMDataCollatorWithFlattening(
        ...     tokenizer=tokenizer,
        ...     mlm_probability=0.15,
        ... )
        >>>
        >>> # Process batch
        >>> batch = collator(sequences)
        >>>
        >>> # Output: Flattened and masked sequences
        >>> print(batch['input_ids'])
        >>> # tensor([[ 0,  5,  6,  7,  2,  0,  8,  9, 10, 11,  2,  0, 12, 16,  2]])
        >>> #                                                      ↑ masked token
        >>>
        >>> print(batch['labels'])
        >>> # tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 13, -100]])
        >>> #                                                                                        ↑ original token
        >>>
        >>> print(batch['cu_seq_lens_q'])
        >>> # tensor([ 0,  5, 11, 15], dtype=torch.int32)  # Sequence boundaries: [0:5], [5:11], [11:15]
        >>>
        >>> # Ready for Flash Attention without attention masks!
    """

    def __init__(
        self,
        # DataCollatorForLanguageModeling
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = True,
        mlm_probability: float | None = 0.15,
        mask_replace_prob: float = 0.8,
        random_replace_prob: float = 0.1,
        pad_to_multiple_of: int | None = None,
        label_pad: int = -100,
        tf_experimental_compile: bool = False,
        return_tensors: str = "pt",
        seed: int | None = None,
    ):
        """Initialize the MLMDataCollatorWithFlattening."""
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            mask_replace_prob=mask_replace_prob,
            random_replace_prob=random_replace_prob,
            pad_to_multiple_of=pad_to_multiple_of,
            tf_experimental_compile=tf_experimental_compile,
            return_tensors=return_tensors,
            seed=seed,
        )
        self.return_tensors = return_tensors
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad = label_pad

    def __call__(self, features, return_tensors=None):
        """Process a batch of variable-length sequences for Flash Attention with MLM.

        This method performs a two-step process:
        1. Flattens multiple sequences into a single packed tensor with Flash Attention metadata
        2. Applies MLM masking to the flattened sequence while preserving special tokens

        Args:
            features (List[Dict[str, List[int]]]): List of tokenized sequences, each containing
                'input_ids' and optionally 'attention_mask'. Example:
                [
                    {"input_ids": [0, 5, 6, 7, 2]},      # Protein sequence 1
                    {"input_ids": [0, 8, 9, 10, 11, 2]}, # Protein sequence 2
                    {"input_ids": [0, 12, 13, 2]}        # Protein sequence 3
                ]
            return_tensors (str, optional): Format for returned tensors ('pt' for PyTorch).
                Defaults to None (uses collator default).

        Returns:
            Dict[str, torch.Tensor]: Batch dictionary containing:
                - input_ids (torch.Tensor): Flattened and MLM-masked token sequences.
                  Shape: [1, total_tokens] where total_tokens = sum of all sequence lengths.
                - labels (torch.Tensor): MLM labels with -100 for non-masked tokens and
                  original token IDs for masked positions. Same shape as input_ids.
                - position_ids (torch.Tensor): Position indices that reset at sequence boundaries.
                  Shape: [1, total_tokens].
                - cu_seq_lens_q (torch.IntTensor): Cumulative sequence lengths for queries.
                  Shape: [num_sequences + 1]. Example: [0, 5, 11, 15].
                - cu_seq_lens_k (torch.IntTensor): Cumulative sequence lengths for keys.
                  Same as cu_seq_lens_q for self-attention.
                - max_length_q (int): Maximum sequence length in the batch.
                - max_length_k (int): Same as max_length_q for self-attention.

        Example:
            >>> # Input features
            >>> features = [
            ...     {"input_ids": [0, 5, 6, 7, 2]},      # 5 tokens
            ...     {"input_ids": [0, 8, 9, 10, 11, 2]}, # 6 tokens
            ...     {"input_ids": [0, 12, 13, 2]}        # 4 tokens
            ... ]
            >>>
            >>> batch = collator(features)
            >>>
            >>> # Output shapes and values
            >>> batch['input_ids'].shape          # torch.Size([1, 15])
            >>> batch['labels'].shape             # torch.Size([1, 15])
            >>> batch['cu_seq_lens_q']            # tensor([0, 5, 11, 15], dtype=torch.int32)
            >>>
            >>> # Flash Attention can now process this without attention masks!

        Note:
            The output is in THD (Total, Height, Depth) format with batch_size=1 and
            sequence_length=total_tokens, optimized for Flash Attention's variable-length
            sequence processing capabilities.
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        if return_tensors != "pt":
            raise NotImplementedError(f'return_tensors must be "pt", {return_tensors=} not implemented')

        batch = _pt_flatten_collate(features)

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mlm_collator.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        if self.pad_to_multiple_of is not None:
            batch = _pt_pad_to_multiple_of(
                batch,
                self.pad_to_multiple_of,
                token_pad=self.mlm_collator.tokenizer.pad_token_id,
                label_pad=self.label_pad,
            )

        return batch


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """Data collator for sequence packing with flash attentions cu_seqlens-style attention.

    Inspired by transformers.data.data_collator.DataCollatorWithFlattening.
    """

    pad_to_multiple_of: int | None = None
    token_pad: int = 1
    label_pad: int = -100

    def __call__(self, features: list[dict[str, list[int]]], return_tensors: str | None = None) -> dict[str, Any]:
        """Collate a batch of variable-length sequences for Flash Attention with MLM.

        Args:
            features: List of tokenized sequences, each containing 'input_ids' and optionally 'labels'.
            return_tensors: Currently only "pt" is supported.

        Returns:
            Dict[str, torch.Tensor]: Batch dictionary containing:
                - input_ids (torch.Tensor): Flattened and MLM-masked token sequences.
                  Shape: [1, total_tokens] where total_tokens = sum of all sequence lengths.
                - labels (torch.Tensor): MLM labels with -100 for non-masked tokens and
                  original token IDs for masked positions. Same shape as input_ids.
                - cu_seq_lens_q (torch.IntTensor): Cumulative sequence lengths for queries.
                  Shape: [num_sequences + 1]. Example: [0, 5, 11, 15].
                - cu_seq_lens_k (torch.IntTensor): Cumulative sequence lengths for keys.
                  Same as cu_seq_lens_q for self-attention.
                - max_length_q (int): Maximum sequence length in the batch.
                - max_length_k (int): Same as max_length_q for self-attention.
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        if return_tensors != "pt":
            raise NotImplementedError(f'return_tensors must be "pt", {return_tensors=} not implemented')

        batch = _pt_flatten_collate(features)
        if self.pad_to_multiple_of is not None:
            batch = _pt_pad_to_multiple_of(batch, self.pad_to_multiple_of, self.token_pad, self.label_pad)
        return batch


def _pt_flatten_collate(features: list[dict[str, list[int]]]):
    is_labels_provided = "labels" in features[0]
    sample_lengths = [len(sample["input_ids"]) for sample in features]

    batch = {}
    batch["max_length_q"] = batch["max_length_k"] = max(sample_lengths)
    batch["input_ids"] = torch.tensor(
        [[token for sample in features for token in sample["input_ids"]]], dtype=torch.int64
    )
    if is_labels_provided:
        batch["labels"] = torch.tensor(
            [[label for sample in features for label in sample["labels"]]], dtype=torch.int64
        )
    cu_seq_lens = torch.zeros(len(features) + 1, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(torch.tensor(sample_lengths), dim=0, dtype=torch.int32)
    batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.tensor(
            [[attention_mask for sample in features for attention_mask in sample["attention_mask"]]], dtype=torch.int64
        )
    return batch


def _pt_pad_to_multiple_of(batch: dict[str, Any], pad_to_multiple_of: int, token_pad: int, label_pad: int):
    """Pad a batch to a multiple of pad_to_multiple_of.

    Appends a mock sequence to the end of the batch with the given token_pad and label_pad to make the total number of
    tokens divisible by pad_to_multiple_of.

    Args:
        batch: Input batch, possibly containing labels and/or cu_seq_lens / max_length keys.
        pad_to_multiple_of: Multiple to pad to.
        token_pad: Token to pad with.
        label_pad: Label to pad with.

    Returns:
        Batch dictionary with padded input_ids, labels, cu_seq_lens_q, cu_seq_lens_k, max_length_q, and max_length_k.
    """
    # Number of tokens we need to pad to make the total number of tokens divisible by pad_to_multiple_of
    remainder = -batch["input_ids"].numel() % pad_to_multiple_of

    if remainder == 0:
        return batch

    batch["input_ids"] = torch.cat(
        [batch["input_ids"], torch.full((1, remainder), token_pad, dtype=batch["input_ids"].dtype)], dim=1
    )

    if "labels" in batch:
        batch["labels"] = torch.cat(
            [batch["labels"], torch.full((1, remainder), label_pad, dtype=batch["labels"].dtype)], dim=1
        )

    if "cu_seq_lens_q" in batch:
        batch["cu_seq_lens_q"] = torch.cat(
            [
                batch["cu_seq_lens_q"],
                torch.tensor([batch["cu_seq_lens_q"][-1] + remainder], dtype=batch["cu_seq_lens_q"].dtype),
            ],
            dim=0,
        )
        batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

    if "max_length_q" in batch:
        batch["max_length_q"] = max(batch["max_length_q"], remainder)
        batch["max_length_k"] = batch["max_length_q"]

    if "attention_mask" in batch:
        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"], torch.zeros((1, remainder), dtype=batch["attention_mask"].dtype)], dim=1
        )

    return batch
