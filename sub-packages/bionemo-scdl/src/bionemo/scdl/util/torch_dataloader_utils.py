# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def collate_sparse_matrix_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    """Collate function to create a batch out of sparse tensors.

    This is necessary to collate sparse matrices of various lengths.

    Args:
        batch: A list of Tensors to collate into a batch.

    Returns:
        The tensors collated into a CSR (Compressed Sparse Row) Format.
    """
    batch_rows = torch.cumsum(
        torch.tensor([0] + [sparse_representation.shape[1] for sparse_representation in batch]), dim=0
    )
    batch_cols = torch.cat([sparse_representation[1] for sparse_representation in batch]).to(torch.int32)
    batch_values = torch.cat([sparse_representation[0] for sparse_representation in batch])
    if len(batch_cols) == 0:
        max_pointer = 0
    else:
        max_pointer = int(batch_cols.max().item() + 1)
    batch_sparse_tensor = torch.sparse_csr_tensor(batch_rows, batch_cols, batch_values, size=(len(batch), max_pointer))
    return batch_sparse_tensor
