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

from typing import Optional
import torch
import datasets
import datasets.distributed
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import MLMDataCollatorWithFlattening, TokenPackingDataset
from distributed_config import DistributedConfig

logger = logging.getLogger(__name__)


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
    )
    return train_dataloader, tokenized_dataset


class CPAwareDataloader:
    """A dataloader that is aware of context parallelism."""
    def __init__(self, dataloader: StatefulDataLoader,
                    cp_group: torch.distributed.ProcessGroup,
                    cp_rank: int,
                    ):
        """Initialize the CPAwareDataloader."""
        self.dataloader = dataloader
        self.cp_rank = cp_rank
        self.cp_group = cp_group
        self.num_cp_ranks = cp_group.size()
        self._iterator = None

    def __iter__(self):
        """Make the dataloader iterable."""
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self):
        """Get the batch from the dataloader for the current CP rank."""
        batch = self._send_data_to_cp_ranks()
        batch['pad_between_seqs'] = True
        return batch

    def _send_data_to_cp_ranks(self):
        """
        This function will get the batch from the dataloader on CP rank 0, and then determine 
        the shards for all the different CP group members.
        combined_batch = [<cp_rank_0_shard>, <cp_rank_1_shard>, ..., <cp_rank_n_shard>]
        Then it will scatter the shards to the different CP group members.
        The shards are then combined into a single batch and returned to the caller
        for the current CP rank.

        Scalability:
            Rank 0's work grows linearly with CP size, but the other ranks do not need to store all the shards so they do not 
            grow linearly with CP size.

        Args:
            None

        Returns:
            batch: The batch for the current CP rank.

        """
        if self.cp_rank == 0:
            # Get data once, then make copies for each rank.
            if self._iterator is None:
                self._iterator = iter(self.dataloader)
            batch = next(self._iterator)

            # Now let's do the sharding on the batch for the different CP group members.
            combined_batch = []
            for cp_rank in range(self.num_cp_ranks):
                input_ids_sharded, labels_sharded = get_batch_on_this_cp_rank(
                    cu_seqlens_padded=batch["cu_seq_lens_q_padded"],
                    input_ids_padded=batch["input_ids"],
                    labels_padded=batch["labels"],
                    cp_group=self.cp_group,
                    qvk_format="thd",
                    cp_rank=cp_rank,
                )
                batch_shard = dict(batch)
                batch_shard["input_ids"] = input_ids_sharded
                batch_shard["labels"] = labels_sharded
                combined_batch.append(batch_shard)
        else:
            combined_batch = None

        scatter_object_output_list = [None]
        # Note: This does not provide an async_op handle. Thus its blocking.
        torch.distributed.scatter_object_list(
            scatter_object_output_list=scatter_object_output_list,
            scatter_object_input_list=combined_batch,
            group=self.cp_group,
            group_src=0,
        )
        torch.distributed.barrier(group=self.cp_group)  # TODO(@jomitchell): Might not need this since its sync.
        return scatter_object_output_list[0]



# TODO(@jomitchell): Once this gets merged: https://github.com/NVIDIA/TransformerEngine/pull/2387 
# we can replace this with the one in TransformerEngine.
def get_batch_on_this_cp_rank(
    cu_seqlens_padded: torch.Tensor,
    input_ids_padded: torch.Tensor,
    labels_padded: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup = None,
    qvk_format: str = "thd",
    cp_rank: Optional[int] = None,
):
    """Slice batch input along sequence dimension into multiple chunks for THD format.
    This function is inteded for use in self attention. It will not work for cross attention because
    it does not handle the case where the sequence length of the query and key are different.
    Which are parallelized across GPUs in a context parallel group.
    This version works with variable-length sequences using cumulative sequence lengths.

    Args:
        cp_rank: Optional manual CP rank index. When provided, the function shards tensors as if it
            were executing on that rank without querying `torch.distributed.get_rank`.
    """
    if qvk_format not in ["thd", "bshd", "sbhd"]:
        raise ValueError(f"Unsupported qvk_format: {qvk_format}!")
    if qvk_format == "thd":
        # Get context parallel size and rank
        cp_size = torch.distributed.get_world_size(group=cp_group)
        if cp_size > 1:
            if cp_rank is None:
                cp_rank = torch.distributed.get_rank(group=cp_group)
            elif not (0 <= cp_rank < cp_size):
                raise ValueError(f"cp_rank must be in [0, {cp_size}), but received {cp_rank}.")

            # Calculate the chunk sizes for each sequence
            total_slices_of_any_sequence = 2 * cp_size
            slice_sizes = (
                cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
            ) // total_slices_of_any_sequence

            # Process each tensor directly instead of using keys_to_change loop
            def process_tensor(val):
                if val is None:
                    return val
                # Determine which dimension is the sequence dimension
                # Ensure cu_seqlens_padded[-1] is a Python int, not a 0-dim tensor
                if isinstance(cu_seqlens_padded[-1], torch.Tensor):
                    seq_len_val = cu_seqlens_padded[-1].item()
                else:
                    seq_len_val = cu_seqlens_padded[-1]

                # Handle 1D tensors (like position_ids that don't have batch dimension)
                if val.ndim == 1:
                    if val.shape[0] == seq_len_val:
                        current_seq_dim = 0
                    else:
                        raise ValueError(
                            "1D tensor shape doesn't match expected sequence length. Make sure the"
                            " inputs are in THD format and padded correctly."
                        )
                elif val.ndim >= 2:
                    if val.shape[1] == seq_len_val:
                        current_seq_dim = 1
                    elif val.shape[0] == seq_len_val:
                        current_seq_dim = 0
                    else:
                        raise ValueError(
                            "Make sure the inputs are in THD format and padded correctly."
                        )
                else:
                    raise ValueError("Tensor must be at least 1D")

                # On this particular rank, for each sequence, get two slices, one from the beginning
                # and one from the end.
                cp_rank_slices = []
                for slice_size, seq_start in zip(slice_sizes, cu_seqlens_padded[:-1]):
                    # 1st segment
                    cp_rank_slices.append(
                        torch.arange(
                            seq_start + (cp_rank * slice_size),
                            seq_start + ((cp_rank + 1) * slice_size),
                            device=val.device,
                        )
                    )

                    # 2nd segment
                    cp_rank_slices.append(
                        torch.arange(
                            seq_start + ((total_slices_of_any_sequence - cp_rank - 1) * slice_size),
                            seq_start + ((total_slices_of_any_sequence - cp_rank) * slice_size),
                            device=val.device,
                        )
                    )

                return val.index_select(current_seq_dim, torch.cat(cp_rank_slices))

            # Process each tensor directly
            input_ids_padded = process_tensor(input_ids_padded)
            labels_padded = process_tensor(labels_padded)
    else:
        raise ValueError(f"Support not implemented yet for qvk_format: {qvk_format}!")

    return input_ids_padded, labels_padded
