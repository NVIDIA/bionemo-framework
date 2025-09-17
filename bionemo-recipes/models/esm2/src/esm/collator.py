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

import numpy as np
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
        ...     return_flash_attn_kwargs=True,
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
        tf_experimental_compile: bool = False,
        return_tensors: str = "pt",
        seed: int | None = None,
        # DataCollatorWithFlattening
        return_flash_attn_kwargs=True,
        return_seq_idx=False,
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
        self.flattening_collator = DataCollatorWithFlattening(
            return_flash_attn_kwargs=return_flash_attn_kwargs,
            return_seq_idx=return_seq_idx,
            return_tensors=return_tensors,
        )
        self.return_tensors = return_tensors

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

        batch = self.flattening_collator(features, return_tensors)

        special_tokens_mask = batch.pop("special_tokens_mask", None)

        if return_tensors == "pt":
            batch["input_ids"], batch["labels"] = self.mlm_collator.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        elif return_tensors == "np":
            batch["input_ids"], batch["labels"] = self.mlm_collator.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            raise ValueError(f'return_tensors must be one of ("pt", "np"), {return_tensors=} not suported')

        return batch


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """Data collator used for padding free approach.

    Modified from transformers.data.data_collator.DataCollatorWithFlattening to not use a separator_id.

    Does the following:

    - concatenates the entire mini batch into single long sequence of shape [1, total_tokens]
    - no padding will be added, returns `input_ids`, `labels` and `position_ids` by default
    - optionally returns the kwargs contained in FlashAttentionKwargs
    - optionally returns seq_idx indicating which sequence each token belongs to

    <Tip warning={true}>

    Using `DataCollatorWithFlattening` will flatten the entire mini batch into single long sequence.
    Make sure your attention computation is able to handle it!

    </Tip>
    """

    def __init__(
        self,
        *args,
        return_flash_attn_kwargs=True,
        return_seq_idx=False,
        **kwargs,
    ):
        """Initialize the DataCollatorWithFlattening.

        Args:
            *args: Arguments for the parent class.
            return_flash_attn_kwargs (bool): Whether to return FlashAttention kwargs.
            return_seq_idx (bool): Whether to return sequence indices.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        self.return_flash_attn_kwargs = return_flash_attn_kwargs
        self.return_seq_idx = return_seq_idx
        self._int_64_keys = {"labels", "position_ids", "input_ids"}
        self._batch_dim_keys = {"labels", "position_ids", "input_ids", "seq_idx"}
        self._py_int_keys = {"max_length_q", "max_length_k"}

    def __call__(self, features, return_tensors=None):
        """Process a batch of variable-length sequences for Flash Attention with MLM.

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
            The output is in THD (Tokens, Height, Depth) format with batch_size=1 and
            sequence_length=total_tokens, optimized for Flash Attention's variable-length
            sequence processing capabilities.
        """
        if return_tensors is None:
            return_tensors = self.return_tensors
        is_labels_provided = "labels" in features[0]
        batch = {"input_ids": [], "labels": []}
        if self.return_seq_idx:
            batch.update({"seq_idx": []})
        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0
        for seq_idx, sample in enumerate(features):
            input_ids = sample["input_ids"]
            batch["input_ids"] += input_ids
            if is_labels_provided:
                batch["labels"] += sample["labels"]
            if self.return_seq_idx:
                batch["seq_idx"] += [seq_idx for _ in range(len(input_ids))]
            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))

        if self.return_flash_attn_kwargs:
            batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
            batch["max_length_q"] = batch["max_length_k"] = max_length

        # FlashAttentionKwargs and seq_idx are expected to be int32s.
        if return_tensors == "pt":
            import torch

            data_cls = torch.tensor
            dtype_64 = torch.int64
            dtype_32 = torch.int32
        elif return_tensors == "np":
            data_cls = np.array
            dtype_64 = np.int64
            dtype_32 = np.int32
        else:
            raise ValueError(f'return_tensors must be one of ("pt", "np"), {return_tensors=} not suported')

        for k, v in batch.items():
            v_ = v  # Avoid modifying the original loop variable v
            if k in self._batch_dim_keys:
                v_ = [v]
            # Flash attention max_len_{q,k} are python ints
            if k not in self._py_int_keys:
                batch[k] = data_cls(v_, dtype=dtype_64 if k in self._int_64_keys else dtype_32)

        return batch
