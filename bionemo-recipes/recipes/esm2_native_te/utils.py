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
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp

import torch


def get_dummy_data_thd_with_padding_dp0(cp_size: int):
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                1, 2, 3, 5, 6
            ])
    labels = torch.tensor([
        10, 20, 30, 50, 60,
    ])
    cu_seqlens_q = torch.tensor([0, 3, 5])
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


def get_dummy_data_thd_with_padding_dp1(cp_size: int):
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                9, 10, 11, 13, 14, 15,
            ])
    labels = torch.tensor([
        90, 100, 110, 130, 140, 150,
    ])
    cu_seqlens_q = torch.tensor([0, 3, 6])
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
