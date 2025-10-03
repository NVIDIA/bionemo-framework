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

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator, PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


class MLMDataCollatorWithFlattening:
    """Data collator that combines MLM masking with sequence packing for Flash Attention.

    This data collator enables efficient training on variable-length sequences by:
    1. First flattening multiple sequences into a single packed tensor (no padding between sequences)
    2. Then applying MLM masking to the flattened sequence
    3. Providing Flash Attention metadata (cu_seq_lens) for sequence boundary awareness
    4. Optionally padding the total sequence length to be divisible by a specified number

    The result is a THD-format batch optimized for Flash Attention with sequence packing,
    eliminating the need for traditional attention masks while maintaining proper sequence
    boundaries during attention computation.

    **Padding to Multiple**: When `pad_to_multiple_of` is specified, the collator ensures
    that the total number of tokens across all sequences is divisible by the given number.
    This is accomplished by appending a mock sequence to the end of the packed batch with
    padding tokens and corresponding labels set to -100. This feature is useful for
    optimizing memory alignment and computational efficiency on specific hardware.

    **Tensor Support**: Currently only supports PyTorch tensors (return_tensors="pt").
    Other tensor formats are not implemented.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for masking tokens.
        mlm (bool): Whether to use masked language modeling. Defaults to True.
        mlm_probability (float | None): Probability of masking tokens. Defaults to 0.15.
        mask_replace_prob (float): Probability of replacing masked tokens with [MASK]. Defaults to 0.8.
        random_replace_prob (float): Probability of replacing masked tokens with random tokens. Defaults to 0.1.
        return_tensors (str): Format for returned tensors. Only "pt" (PyTorch) is supported. Defaults to "pt".
        seed (int | None): Random seed for reproducible masking. Defaults to None.
        pad_to_multiple_of (int | None): If set, pads the total sequence length to be divisible
            by this number by adding a mock sequence at the end. Defaults to None.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        >>>
        >>> # Input: Variable-length protein sequences
        >>> sequences = [
        ...     {"input_ids": [0, 5, 6, 7, 2]},      # CLS + amino acids + EOS (5 tokens)
        ...     {"input_ids": [0, 8, 9, 10, 11, 2]}, # CLS + amino acids + EOS (6 tokens)
        ...     {"input_ids": [0, 12, 13, 2]},       # CLS + amino acids + EOS (4 tokens)
        ... ]  # Total: 15 tokens
        >>>
        >>> # Create collator with padding to multiple of 8
        >>> collator = MLMDataCollatorWithFlattening(
        ...     tokenizer=tokenizer,
        ...     mlm_probability=0.15,
        ...     pad_to_multiple_of=8,  # Pad 15 -> 16 tokens
        ...     seed=42
        ... )
        >>>
        >>> # Process batch
        >>> batch = collator(sequences)
        >>>
        >>> # Output: Flattened, masked, and padded sequences
        >>> print(batch['input_ids'].shape)    # torch.Size([1, 16])
        >>> print(batch['labels'].shape)       # torch.Size([1, 16])
        >>> print(batch['cu_seq_lens_q'])      # tensor([0, 5, 11, 15, 16], dtype=torch.int32)
        >>>                                    # Note: Extra entry for mock padding sequence
        >>> # Ready for Flash Attention without traditional attention masks!

    Note:
        The output is in THD (Total, Height, Depth) format with batch_size=1 and
        sequence_length=total_tokens, optimized for Flash Attention's variable-length
        sequence processing capabilities.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = True,
        mlm_probability: float | None = 0.15,
        mask_replace_prob: float = 0.8,
        random_replace_prob: float = 0.1,
        return_tensors: str = "pt",
        seed: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_position_ids: bool = False,
        bshd_equivalent: bool = False,
        bshd_pad_to_multiple_of: int | None = None,
    ):
        """Initialize the MLMDataCollatorWithFlattening.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for masking tokens.
            mlm (bool): Whether to use masked language modeling. Defaults to True.
            mlm_probability (float | None): Probability of masking tokens. Defaults to 0.15.
            mask_replace_prob (float): Probability of replacing masked tokens with [MASK]. Defaults to 0.8.
            random_replace_prob (float): Probability of replacing masked tokens with random tokens. Defaults to 0.1.
            return_tensors (str): Format for returned tensors. Only "pt" (PyTorch) is supported. Defaults to "pt".
            seed (int | None): Random seed for reproducible masking. Defaults to None.
            pad_to_multiple_of (int | None): If set, pads the total sequence length to be divisible
                by this number by adding a mock sequence at the end. Defaults to None.
            return_position_ids (bool): Whether to return position ids. Defaults to False.
            bshd_equivalent (bool): Whether to return a batch exactly reproduces the random masking of the BSHD
                collator, at the expense of additional computation time. Defaults to False.
            bshd_pad_to_multiple_of (int | None): For the bshd_equivalent mode, mimics padding that would be done by the
                BSHD collator. Defaults to None.
        """
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            mask_replace_prob=mask_replace_prob,
            random_replace_prob=random_replace_prob,
            return_tensors=return_tensors,
            seed=seed,
            pad_to_multiple_of=bshd_pad_to_multiple_of,
        )
        self.return_tensors = return_tensors
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_position_ids = return_position_ids
        self.bshd_equivalent = bshd_equivalent
        self.bshd_pad_to_multiple_of = bshd_pad_to_multiple_of

        if bshd_pad_to_multiple_of is not None and not bshd_equivalent:
            raise ValueError("bshd_pad_to_multiple_of can only be used when bshd_equivalent is True")

    def __call__(self, features, return_tensors=None):
        """Process a batch of variable-length sequences for Flash Attention with MLM.

        This method performs the following steps:
        1. Flattens multiple sequences into a single packed tensor with Flash Attention metadata
        2. Applies MLM masking to the flattened sequence while preserving special tokens
        3. Optionally pads to a multiple of a specified number for hardware optimization

        Args:
            features (List[Dict[str, List[int]]]): List of tokenized sequences, each containing
                'input_ids' and optionally 'attention_mask'. Example:
                [
                    {"input_ids": [0, 5, 6, 7, 2]},      # Protein sequence 1
                    {"input_ids": [0, 8, 9, 10, 11, 2]}, # Protein sequence 2
                    {"input_ids": [0, 12, 13, 2]}        # Protein sequence 3
                ]
            return_tensors (str, optional): Format for returned tensors. Only "pt" (PyTorch)
                is supported. Defaults to None (uses collator default).

        Returns:
            Dict[str, torch.Tensor]: Batch dictionary containing:
                - input_ids (torch.Tensor): Flattened and MLM-masked token sequences.
                  Shape: [1, total_tokens] where total_tokens = sum of all sequence lengths
                  (plus padding if pad_to_multiple_of is specified).
                - labels (torch.Tensor): MLM labels with -100 for non-masked tokens and
                  original token IDs for masked positions. Same shape as input_ids.
                - cu_seq_lens_q (torch.IntTensor): Cumulative sequence lengths for queries.
                  Shape: [num_sequences + 1] or [num_sequences + 2] if padding is added.
                  Example: [0, 5, 11, 15] or [0, 5, 11, 15, 16] with padding.
                - cu_seq_lens_k (torch.IntTensor): Cumulative sequence lengths for keys.
                  Same as cu_seq_lens_q for self-attention.
                - max_length_q (int): Maximum sequence length in the batch.
                - max_length_k (int): Same as max_length_q for self-attention.
                - attention_mask (torch.Tensor): Attention mask with 1s for actual tokens
                  and 0s for padding tokens (if any).

        Raises:
            NotImplementedError: If return_tensors is not "pt".

        Example:
            >>> # Input features
            >>> features = [
            ...     {"input_ids": [0, 5, 6, 7, 2]},      # 5 tokens
            ...     {"input_ids": [0, 8, 9, 10, 11, 2]}, # 6 tokens
            ... ]
            >>>
            >>> batch = collator(features)
            >>>
            >>> # Output shapes and values
            >>> batch['input_ids'].shape          # torch.Size([1, 11]) or larger if padded
            >>> batch['labels'].shape             # torch.Size([1, 11]) or larger if padded
            >>> batch['cu_seq_lens_q']            # tensor([0, 5, 11], dtype=torch.int32) or larger

        Note:
            The output is in THD (Total, Height, Depth) format with batch_size=1 and
            sequence_length=total_tokens, optimized for Flash Attention's variable-length
            sequence processing capabilities. When pad_to_multiple_of is used, an additional
            mock sequence is appended to reach the desired total length.
        """
        if self.bshd_equivalent:
            return self.bshd_compatible_call(features, return_tensors=return_tensors)

        if return_tensors is None:
            return_tensors = self.return_tensors

        if return_tensors != "pt":
            raise NotImplementedError(f'return_tensors must be "pt", {return_tensors=} not implemented')

        batch = _pt_flatten_collate(features, return_position_ids=self.return_position_ids)

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mlm_collator.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        if self.pad_to_multiple_of is not None:
            batch = self._pad_batch_to_multiple_of(batch)

        return batch

    def bshd_compatible_call(self, features, return_tensors=None):
        """Mask tokens in a way that's identical to the BSHD collator.

        This ensures the randomized masking outputs of the THD collator are identical to the BSHD collator.
        """
        # Perform the masking with the BSHD collator.
        bshd_batch = self.mlm_collator(features, return_tensors=return_tensors)

        # Create the flattened batch to get the cu_seq_lens_q and cu_seq_lens_k values.
        packed_batch = _pt_flatten_collate(features, return_position_ids=self.return_position_ids)

        # Get the masked input_ids and labels from the BSHD batch.
        masked_input_ids = bshd_batch["input_ids"][bshd_batch["attention_mask"].bool()].unsqueeze(0)
        masked_labels = bshd_batch["labels"][bshd_batch["attention_mask"].bool()].unsqueeze(0)

        # Update the packed batch with the masked input_ids and labels.
        packed_batch["input_ids"] = masked_input_ids
        packed_batch["labels"] = masked_labels

        if self.pad_to_multiple_of is not None:
            packed_batch = self._pad_batch_to_multiple_of(packed_batch)

        return packed_batch

    def _pad_batch_to_multiple_of(self, batch):
        """Add a mock sequence to make the total number of tokens divisible by pad_to_multiple_of."""
        # Ensure token_pad is an integer, defaulting to 1 if pad_token_id is None or invalid
        pad_token_id = self.mlm_collator.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            logger.warning(f"tokenizer.pad_token_id is not an integer, using 1 instead: {pad_token_id}")
            pad_token_id = 1

        return _pt_pad_to_multiple_of(
            batch,
            self.pad_to_multiple_of,
            token_pad=pad_token_id,
            label_pad=-100,
        )


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """Data collator for sequence packing with flash attentions cu_seqlens-style attention.

    Inspired by transformers.data.data_collator.DataCollatorWithFlattening, but skips adding a separator_id in the
    output labels, since this overwrites the first token in MLM masking.

    Optionally returns position_ids, which are not needed for Flash Attention, but can be needed for context
    parallelism.
    """

    pad_to_multiple_of: int | None = None
    token_pad: int = 1
    label_pad: int = -100
    return_tensors: str = "pt"
    return_position_ids: bool = False

    def __call__(self, features: list[dict[str, list[int]]], return_tensors: str | None = None) -> dict[str, Any]:
        """Collate a batch of variable-length sequences for Flash Attention with sequence packing.

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
                - attention_mask (torch.Tensor): Attention mask with 1s for non-padding tokens and 0s for padding tokens.
        """
        if not features:
            raise ValueError("features must be a non-empty list")

        if return_tensors is None:
            return_tensors = self.return_tensors

        if return_tensors != "pt":
            raise NotImplementedError(f'return_tensors must be "pt", {return_tensors=} not implemented')

        batch = _pt_flatten_collate(features, return_position_ids=self.return_position_ids)
        if self.pad_to_multiple_of is not None:
            batch = _pt_pad_to_multiple_of(batch, self.pad_to_multiple_of, self.token_pad, self.label_pad)
        return batch


def _pt_flatten_collate(features: list[dict[str, list[int]]], return_position_ids: bool = False):
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
            [[v for sample in features for v in sample["attention_mask"]]], dtype=torch.int64
        )
    if return_position_ids:
        batch["position_ids"] = torch.hstack(
            [torch.arange(sample_len, dtype=torch.int64) for sample_len in sample_lengths]
        ).unsqueeze(0)

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

    if "position_ids" in batch:
        batch["position_ids"] = torch.cat(
            [batch["position_ids"], torch.arange(remainder, dtype=batch["position_ids"].dtype).unsqueeze(0)], dim=1
        )

    return batch
