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

import logging

import copy
from typing import Any
import torch
import datasets
import datasets.distributed
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import MLMDataCollatorWithFlattening, TokenPackingDataset
from distributed_config import DistributedConfig

from utils import get_batch_on_this_cp_rank
logger = logging.getLogger(__name__)

# TODO(@jomitchell): Create an enum for the batch keys, which we can then use later.

BATCH_KEYS_DTYPE = {
    'max_length_q': torch.int32,  # regular int
    'max_length_k': torch.int32,  # regular int
    'input_ids': torch.int64,
    'cu_seq_lens_q': torch.int32,
    'cu_seq_lens_k': torch.int32,
    # 'attention_mask': torch.int64,  # Missing!
    'labels': torch.int64,
}

def create_tokenized_dataset(
    distributed_config: DistributedConfig,
    tokenizer_name: str,
    load_dataset_kwargs: dict,
    max_seq_length: int = 1024,
    buffer_size: int = 10_000,
    use_lazy_tokenization: bool = True,
):
    """Create a tokenized dataset."""
    logger.info(f"Loading dataset with kwargs: {load_dataset_kwargs}")
    dataset = datasets.load_dataset(**load_dataset_kwargs)
    logger.info(f"Loaded dataset: {dataset}")

    if isinstance(dataset, datasets.IterableDataset):
        dataset = datasets.distributed.split_dataset_by_node(
            dataset,
            rank=distributed_config.rank,
            world_size=distributed_config.world_size,
        )
        dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        """Tokenize the protein sequences."""
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=max_seq_length,
        )

    if isinstance(dataset, datasets.Dataset) and use_lazy_tokenization:
        # Using dataset.map on a non-streaming dataset will automatically perform and cache the transform, which can
        # trigger an expensive tokenization.
        tokenized_dataset = dataset.with_transform(tokenize_function)

    else:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

    return tokenized_dataset, tokenizer


def create_bshd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int,
    num_workers: int,
    max_seq_length: int = 1024,
    seed: int = 42,
    buffer_size: int = 10_000,
    use_lazy_tokenization: bool = True,
    use_stateful_dataloader: bool = False,
    mlm_probability: float = 0.15,
):
    """Create a dataloader for the dataset.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name: The name of the tokenizer to pull from the HuggingFace Hub.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset` for the train dataset.
        micro_batch_size: The batch size (number of sequences) per device.
        num_workers: The number of workers to use for the dataloader.
        max_seq_length: The maximum length of the protein sequences.
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size to use for the distributed sampler.
        use_lazy_tokenization: Whether to use datasets.set_transform for tokenization if the dataset is a
            non-streaming datasets.Dataset. Defaults to True.
        use_stateful_dataloader: Whether to use the StatefulDataLoader to enable checkpointing the dataloader state.
        mlm_probability: The probability of masking tokens for MLM (default 0.15). Set to 0 for no masking.
        **kwargs: Unused, here to enable kwargs to match the signature of create_thd_dataloader.

    Returns:
        A dataloader that can be used for training.
    """
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        buffer_size=buffer_size,
        use_lazy_tokenization=use_lazy_tokenization,
    )

    if isinstance(tokenized_dataset, datasets.IterableDataset):
        sampler = None
    else:
        sampler = DistributedSampler(
            tokenized_dataset,
            rank=distributed_config.rank,
            num_replicas=distributed_config.world_size,
            seed=seed,
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=max_seq_length,
        seed=seed,
    )

    # TODO(BIONEMO-3246) - remove the pin_memory=False once StatefulDataLoader supports pin_memory again.
    dataloader_class = StatefulDataLoader if use_stateful_dataloader else DataLoader
    train_dataloader = dataloader_class(
        tokenized_dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True if not use_stateful_dataloader else False,
        persistent_workers=num_workers > 0,  # DELETEME
    )

    return train_dataloader, tokenized_dataset if sampler is None else sampler


def create_thd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int | None = None,
    token_micro_batch_size: int | None = None,
    num_workers: int = 1,
    max_seq_length: int = 1024,
    seed: int = 42,
    buffer_size: int = 10_000,
    use_stateful_dataloader: bool = False,
    mlm_probability: float = 0.15,
    pad_sequences_to_be_divisible_by: int | None = None,
):
    """Create a dataloader that packs up to the maximum number of tokens per batch.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name: The name of the tokenizer to pull from the HuggingFace Hub.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset` for the train dataset.
        micro_batch_size: The batch size (number of sequences) per device. This will set the token_micro_batch_size to
            micro_batch_size * max_seq_length. Defaults to None.
        token_micro_batch_size: The maximum number of tokens per batch. If None, the micro_batch_size * max_seq_length
            will be used. Defaults to None.
        num_workers: The number of workers to use for the dataloader. For iterable datasets, this should be 1.
        max_seq_length: The maximum length of the protein sequences.
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size to use for the distributed sampler.
        use_stateful_dataloader: Whether to use the StatefulDataLoader to enable checkpointing the dataloader state.
        mlm_probability: The probability of masking tokens for MLM (default 0.15). Set to 0 for no masking.
        pad_sequences_to_be_divisible_by: If provided, sequences will be padded to be divisible by this value.
            This is useful for context parallelism. Defaults to None.

    Returns:
        A dataloader that can be used for training.
    """
    # If cp_rank not 0 return none.
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        buffer_size=buffer_size,
    )

    assert isinstance(tokenized_dataset, datasets.IterableDataset), "THD token packing requires a streaming dataset."
    if token_micro_batch_size is None:
        assert micro_batch_size is not None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        token_micro_batch_size = micro_batch_size * max_seq_length
    else:
        assert micro_batch_size is None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        assert token_micro_batch_size >= max_seq_length, "token_micro_batch_size must be greater than max_seq_length."

    # For THD, we pad out to the maximum number of tokens per batch for consistent array shapes.
    data_collator = MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=token_micro_batch_size if pad_sequences_to_be_divisible_by is None else None,
        seed=seed,
        pad_sequences_to_be_divisible_by=pad_sequences_to_be_divisible_by,
    )

    # TODO(BIONEMO-3246) - remove the pin_memory=False once StatefulDataLoader supports pin_memory again.
    dataloader_class = StatefulDataLoader if use_stateful_dataloader else DataLoader
    train_dataloader = dataloader_class(
        TokenPackingDataset(tokenized_dataset, max_tokens_per_batch=token_micro_batch_size),
        batch_size=None,  # The TokenPackingDataset will handle the batching.
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True if not use_stateful_dataloader else False,
        persistent_workers=num_workers > 0,  # TODO: DELETEME
    )
    return train_dataloader, tokenized_dataset

class CPAwareDataloader:
    """A dataloader that is aware of context parallelism."""

    def __init__(self, dataloader: StatefulDataLoader,
                    dist_config: DistributedConfig,
                    cp_group: torch.distributed.ProcessGroup,
                    cp_rank: int,
                    max_seq_length: int = 1024,
                    dtype: torch.dtype = torch.float32,
                    ):
        self.dataloader = dataloader
        self.dist_config = dist_config
        self.cp_rank = cp_rank
        self.cp_group = cp_group
        self.num_cp_ranks = cp_group.size()
        self.max_len = max_seq_length
        self.dtype = dtype
        self.sentinel_value = 1e8 # TODO(@jomitchell): Make this a configurable parameter. Not even sure if 1e8 makes sense lawl.
        self._iterator = None

    def __iter__(self):
        """Make the dataloader iterable."""
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self):
        batch = self.__get_data_scatter()
        input_ids_padded, labels_padded = get_batch_on_this_cp_rank(
            cu_seqlens_padded=batch["cu_seq_lens_q"],
            input_ids_padded=batch["input_ids"],
            labels_padded=batch["labels"],
            cp_group=self.cp_group,
            qvk_format="thd",
        )
        batch["input_ids"] = input_ids_padded
        batch["labels"] = labels_padded
        
        return batch

    def __get_data_broadcast(self):
        pass

    def __get_data_scatter(self):
        # Create device for current rank (use local_rank = GPU index on this node)
        device = torch.device(f"cuda:{self.dist_config.local_rank}")

        if self.cp_rank == 0: # Is this global rank 0? IDK.
            # Get data once, then make copies for each rank.
            if self._iterator is None:
                self._iterator = iter(self.dataloader)
            batch = next(self._iterator)

            # Convert everything to tensors and move to GPU
            # Create scatter list for each key
            combined_batch = {key: [] for key in BATCH_KEYS_DTYPE}
            tensor_sizes = {}  # Track actual tensor sizes
            tensor_shapes = {}  # Track original shapes for reconstruction

            for key, value in batch.items():
                if key in BATCH_KEYS_DTYPE:
                    # Convert to tensor if not already
                    if isinstance(value, torch.Tensor):
                        # Store original shape before flattening
                        tensor_shapes[key] = value.shape
                        # Flatten multi-dimensional tensors for scatter (scatter only works with 1D tensors)
                        tensor_value = value.to(device).flatten()
                    else:
                        # Convert scalar (int/float) to tensor
                        tensor_value = torch.tensor([value], device=device, dtype=BATCH_KEYS_DTYPE[key])
                        tensor_shapes[key] = tensor_value.shape

                    tensor_sizes[key] = tensor_value.numel()  # Store total number of elements

                    # Create copies for each CP rank
                    for _ in range(self.num_cp_ranks):
                        combined_batch[key].append(tensor_value.clone())
        else:
            combined_batch = None
            tensor_sizes = {}
            tensor_shapes = {}

        # Get the global rank of cp_rank=0 in this CP group (the source for broadcasts)
        cp_group_ranks = torch.distributed.get_process_group_ranks(self.cp_group)
        src_global_rank = cp_group_ranks[0]  # First rank in the CP group is cp_rank=0
        
        # Broadcast tensor sizes to all ranks
        size_tensor = torch.zeros(len(BATCH_KEYS_DTYPE), dtype=torch.int64, device=device)
        if self.cp_rank == 0:
            for i, key in enumerate(BATCH_KEYS_DTYPE.keys()):
                size_tensor[i] = tensor_sizes.get(key, 1)
        torch.distributed.broadcast(size_tensor, src=src_global_rank, group=self.cp_group)
        
        # Broadcast tensor shapes (max 4 dimensions should be enough)
        # Format: [ndim, dim0, dim1, dim2, dim3] for each key
        shape_tensor = torch.zeros(len(BATCH_KEYS_DTYPE) * 5, dtype=torch.int64, device=device)
        if self.cp_rank == 0:
            for i, key in enumerate(BATCH_KEYS_DTYPE.keys()):
                shape = tensor_shapes.get(key, (1,))
                shape_tensor[i * 5] = len(shape)  # number of dimensions
                for j, dim in enumerate(shape[:4]):  # max 4 dims
                    shape_tensor[i * 5 + 1 + j] = dim
        torch.distributed.broadcast(shape_tensor, src=src_global_rank, group=self.cp_group)

        # Reconstruct sizes and shapes on non-zero ranks
        if self.cp_rank != 0:
            for i, key in enumerate(BATCH_KEYS_DTYPE.keys()):
                tensor_sizes[key] = size_tensor[i].item()
                ndim = shape_tensor[i * 5].item()
                shape = tuple(shape_tensor[i * 5 + 1: i * 5 + 1 + ndim].tolist())
                tensor_shapes[key] = shape

        # Create batch buffers with correct dtypes and sizes
        batch_buffer = {}
        for key, dtype in BATCH_KEYS_DTYPE.items():
            size = tensor_sizes.get(key, 1)
            batch_buffer[key] = torch.zeros(size, device=device, dtype=dtype)

        # Scatter all values (now all tensors) across CP ranks
        for key in BATCH_KEYS_DTYPE:
            scatter_list = combined_batch[key] if combined_batch is not None else None
            torch.distributed.scatter(
                batch_buffer[key],      # Output: where received tensor is stored
                scatter_list=scatter_list,  # Input: only used on source rank
                src=src_global_rank,     # Source rank (global rank of cp_rank=0 in this group)
                group=self.cp_group,
                async_op=False  # Turn on Async later.
            )

        # Reconstruct the dictionary batch from the batch buffer with original shapes
        batch = {}
        for key in BATCH_KEYS_DTYPE:
            original_shape = tensor_shapes[key]
            if original_shape == (1,):  # Scalar values (max_length_q, max_length_k)
                batch[key] = batch_buffer[key].item()
            else:
                # Reshape back to original shape
                batch[key] = batch_buffer[key].view(original_shape)

        return batch

