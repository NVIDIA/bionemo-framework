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

from typing import Optional, Tuple

import torch


def pad_thd_sequences_for_cp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
    divisibility_factor: int,
    padding_token_id: int = 0,
    padding_label_id: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences to be divisible by the divisibility factor.

    Args:
        input_ids: Tensor of shape (1, N) or (N,) containing concatenated sequences
        labels: Tensor of shape (1, N) or (N,) containing labels for each token
        cu_seqlens: Tensor of shape (M,) containing cumulative sequence lengths
        divisibility_factor: Each sequence length must be divisible by this factor
        padding_token_id: Token ID to use for padding (default: 0)
        padding_label_id: Label ID to use for padding (default: -100)

    Returns:
        Tuple of:
        - input_ids_padded: Padded input_ids tensor
        - labels_padded: Padded labels tensor
        - cu_seqlens_padded: Cumulative sequence lengths accounting for padding
    """
    # Flatten input_ids and labels if needed
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if labels.dim() == 2:
        labels = labels.squeeze(0)

    # Compute the sequence lengths from cu_seqlens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    # List: amount of padding needed for each sequence (make length a multiple of divisibility_factor)
    padding_amounts = [
        ((l.item() + divisibility_factor - 1) // divisibility_factor) * divisibility_factor - l.item() for l in seqlens
    ]

    # Extract sequences and labels for each batch item
    batch_sequences = [input_ids[start.item() : end.item()] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
    batch_labels = [labels[start.item() : end.item()] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]

    # Pad sequences and labels to required length
    input_ids_padded = torch.cat(
        [
            (torch.cat([seq, torch.full((pad,), padding_token_id, dtype=seq.dtype)]) if pad > 0 else seq)
            for seq, pad in zip(batch_sequences, padding_amounts)
        ]
    )
    labels_padded = torch.cat(
        [
            (torch.cat([seq, torch.full((pad,), padding_label_id, dtype=seq.dtype)]) if pad > 0 else seq)
            for seq, pad in zip(batch_labels, padding_amounts)
        ]
    )

    # Compute cumulative padded sequence lengths, starting from 0
    padded_lengths = seqlens + torch.tensor(padding_amounts, dtype=seqlens.dtype)
    cu_seqlens_padded = torch.cumsum(torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype), padded_lengths]), dim=0)

    return input_ids_padded, labels_padded, cu_seqlens_padded


def get_dummy_data_thd(cp_size: int):
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                1, 1, 1, 1, 1, 1, 1,  # 7 tokens
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # 11 tokens
                3, 3, 3, 3, 3  # 5 tokens
            ])
    labels = torch.tensor([
        10, 11, 12, 13, 14, 15, 16,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        5, 6, 7, 8, 9
    ])
    cu_seqlens_q = torch.tensor([0, 7, 18, 23])
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = \
                pad_thd_sequences_for_cp(
                    input_ids.unsqueeze(0),
                    labels.unsqueeze(0),
                    cu_seqlens_q,
                    divisibility_factor,
                    padding_token_id=pid,
                    padding_label_id=label_pad
                )
    expected_input_ids = torch.tensor([
                1, 1, 1, 1, 1, 1, 1, pid,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, pid,
                3, 3, 3, 3, 3, pid, pid, pid
            ])

    expected_labels = torch.tensor([
        10, 11, 12, 13, 14, 15, 16, label_pad,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, label_pad,
        5, 6, 7, 8, 9, label_pad, label_pad, label_pad
    ])

    expected_cu_seqlens_padded = torch.tensor([0, 8, 20, 28])

    torch.equal(input_ids_padded, expected_input_ids)
    torch.equal(labels_padded, expected_labels)
    torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded)

    # Now we have our data ready to go.
    # IMPORTANT: get_batch_on_this_cp_rank only needs batch dim for the *_padded keys
    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_dp0_nopadding():
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                1, 2, 3, 4, 5, 6, 7, 8,  # 8 tokens
            ])
    labels = torch.tensor([
        10, 20, 30, 40, 50, 60, 70, 80,
    ])
    cu_seqlens_q = torch.tensor([0, 8])
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_dp1_nopadding():
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                9, 10, 11, 12, 13, 14, 15, 16,  # 8 tokens
            ])
    labels = torch.tensor([
        90, 100, 110, 120, 130, 140, 150, 160,
    ])
    cu_seqlens_q = torch.tensor([0, 8])
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch

class DummyDataloader:
    def __init__(self, cp_size: int):
        self.data = get_dummy_data_thd(cp_size=cp_size)


    def __iter__(self):
        return self

    def __next__(self):
        return self.data

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
