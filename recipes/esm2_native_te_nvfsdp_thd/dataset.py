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

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorWithFlattening


# Create the dataset -- here, we just use a simple parquet file with some raw protein sequences
# stored in the repo itself to avoid external dependencies.
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


def infinite_dataloader(dataloader, sampler):
    """Create an infinite iterator that automatically restarts at the end of each epoch."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)  # Update epoch for proper shuffling
        for batch in dataloader:
            yield batch
        epoch += 1  # Increment epoch counter after completing one full pass


def create_dataloader(data_dir, batch_size, max_length=1024):
    """Create a dataloader for the dataset.

    Args:
        data_dir: The directory containing the dataset.
        batch_size: The batch size.
        max_length: The maximum length of the protein sequences.

    Returns:
        A dataloader that just infinitely loops over the dataset.
        The number of batches in the dataloader.
    """
    # We copy this parquet file to the container to avoid external dependencies, modify if you're
    # using a local dataset. If you're reading this and scaling up the dataset to a larger size,
    # look into `set_transform` and other streaming options from the `datasets` library.
    data_path = Path(data_dir) / "train.parquet"
    dataset = load_dataset("parquet", data_files=data_path.as_posix(), split="train")

    # Perfect. The sequences have variable length.
    def tokenize_function(examples):
        """Tokenize the protein sequences."""
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = MLMDataCollatorWithFlattening(
        DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
        ),
        DataCollatorWithFlattening(
            return_flash_attn_kwargs=True,
        ),
    )

    # Create dataloader with distributed sampler
    train_sampler = DistributedSampler(tokenized_dataset)
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Create the infinite iterator
    train_iterator = infinite_dataloader(train_dataloader, train_sampler)

    return train_iterator, len(train_dataloader)


@dataclass
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
        ...     mlm_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
        ...     flattening_collator=DataCollatorWithFlattening(return_flash_attn_kwargs=True)
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

    mlm_collator: DataCollatorForLanguageModeling
    flattening_collator: DataCollatorWithFlattening

    def __call__(self, features, return_tensors=None, separator_id=None):
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
            separator_id (int, optional): Token ID used for sequence separation in labels.
                Defaults to None (uses -100, which is ignored by loss functions).

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
        batch = self.flattening_collator(features, return_tensors, separator_id)

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mlm_collator.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        return batch
