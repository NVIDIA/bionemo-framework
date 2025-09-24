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

import torch
import torch.distributed

def get_batch_on_this_cp_rank(
    cu_seqlens_padded: torch.Tensor,
    input_ids_padded: torch.Tensor,
    labels_padded: torch.Tensor,
    position_ids_padded: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup = None,
    qvk_format: str = "thd",
):
    """Slice batch input along sequence dimension into multiple chunks for THD format.

    This function is inteded for use in self attention. It will not work for cross attention because
    it does not handle the case where the sequence length of the query and key are different.

    Which are parallelized across GPUs in a context parallel group.
    This version works with variable-length sequences using cumulative sequence lengths.
    """
    if qvk_format not in ["thd", "bshd", "sbhd"]:
        raise ValueError(f"Unsupported qvk_format: {qvk_format}!")
    cp_size = torch.distributed.get_world_size(group=cp_group)
    cp_rank = torch.distributed.get_rank(group=cp_group)
    if qvk_format == "thd":
        # Get context parallel size and rank
        if cp_size > 1:

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
            position_ids_padded = process_tensor(position_ids_padded)
    elif qvk_format == "bshd":
        def process_tensor(val):
            if val is None:
                return val
            
            # Dynamically determine sequence dimension based on format
            # For bshd format: batch, sequence, heads, dim
            seq_dim = 1
            
            # Validate tensor has enough dimensions
            if val.ndim < 2:
                raise ValueError(
                    f"Tensor must have at least 2 dimensions for bshd format, got {val.ndim}"
                )
            
            # Validate sequence dimension is divisible by 2*cp_size
            if val.shape[seq_dim] % (2 * cp_size) != 0:
                raise ValueError(
                    f"Sequence dimension (dim {seq_dim}) with size {val.shape[seq_dim]} "
                    f"must be divisible by 2*cp_size={2*cp_size}"
                )
            
            # Reshape tensor to separate chunks
            try:
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to reshape tensor from shape {list(val.shape)} "
                    f"to chunk-separated shape. Error: {e}"
                )
            
            # Create index tensor on the same device as input to avoid CPU-GPU sync
            index = torch.tensor(
                [cp_rank, (2 * cp_size - cp_rank - 1)], 
                device=val.device,
                dtype=torch.long
            )
            
            # Select the chunks for this rank
            val = val.index_select(seq_dim, index)
            
            # Reshape back to original format with reduced sequence dimension
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
            return val
        
        if cp_size > 1:
            input_ids_padded = process_tensor(input_ids_padded)
            labels_padded = process_tensor(labels_padded)
            position_ids_padded = process_tensor(position_ids_padded)
    
    elif qvk_format == "sbhd":
        def process_tensor(val):
            if val is None:
                return val
            
            # Dynamically determine sequence dimension based on format
            # For sbhd format: sequence, batch, heads, dim
            seq_dim = 0
            
            # Validate tensor has enough dimensions
            if val.ndim < 2:
                raise ValueError(
                    f"Tensor must have at least 2 dimensions for sbhd format, got {val.ndim}"
                )
            
            # Validate sequence dimension is divisible by 2*cp_size
            if val.shape[seq_dim] % (2 * cp_size) != 0:
                raise ValueError(
                    f"Sequence dimension (dim {seq_dim}) with size {val.shape[seq_dim]} "
                    f"must be divisible by 2*cp_size={2*cp_size}"
                )
            
            # Reshape tensor to separate chunks
            try:
                val = val.view(
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to reshape tensor from shape {list(val.shape)} "
                    f"to chunk-separated shape. Error: {e}"
                )
            
            # Create index tensor on the same device as input to avoid CPU-GPU sync
            index = torch.tensor(
                [cp_rank, (2 * cp_size - cp_rank - 1)], 
                device=val.device,
                dtype=torch.long
            )
            
            # Select the chunks for this rank (dim 0 for sbhd after reshape)
            val = val.index_select(0, index)
            
            # Reshape back to original format with reduced sequence dimension
            val = val.view(-1, *val.shape[2:])
            return val
        
        if cp_size > 1:
            input_ids_padded = process_tensor(input_ids_padded)
            labels_padded = process_tensor(labels_padded)
            if position_ids_padded is not None:
                position_ids_padded = process_tensor(position_ids_padded)
            else:
                position_ids_padded = None
        
    else:
        raise ValueError(f"Support not implemented yet for qvk_format: {qvk_format}!")

    return input_ids_padded, labels_padded, position_ids_padded


def generate_positional_ids_for_cp(
    cu_seqlens: torch.Tensor,
    divisibility_factor: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Generate positional IDs for sequences padded to be divisible by divisibility_factor.

    Args:
        cu_seqlens: Tensor of shape (M,) containing cumulative sequence lengths
        divisibility_factor: Each sequence length must be divisible by this factor
        dtype: Data type for the generated positional IDs (default: torch.long)

    Returns:
        Generated positional_ids tensor where each sequence starts from 0 and continues through padding
    """
    # Compute the sequence lengths from cu_seqlens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    # List: amount of padding needed for each sequence
    padding_amounts = [
        ((l.item() + divisibility_factor - 1) // divisibility_factor) * divisibility_factor
        - l.item()
        for l in seqlens
    ]

    # Generate positional IDs for each padded sequence (each starts from 0)
    padded_lengths = seqlens + torch.tensor(padding_amounts, dtype=seqlens.dtype)
    positional_ids = torch.cat(
        [torch.arange(0, int(length), dtype=dtype) for length in padded_lengths]
    )

    return positional_ids

# def generate_positional_ids_for_bshd(input_ids: torch.Tensor, divisibility_factor: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
#     """Generate positional IDs for sequences padded to be divisible by divisibility_factor.

#     Args:
#         input_ids: Tensor of shape (M,) containing input IDs
#         divisibility_factor: Each sequence length must be divisible by this factor
#         dtype: Data type for the generated positional IDs (default: torch.long)

#     Returns:
#         Generated positional_ids tensor where each sequence starts from 0 and continues through padding
#     """
#     return generate_positional_ids_for_cp(input_ids.shape[1], divisibility_factor, dtype)