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


import numpy as np


def sample_or_truncate_plus_pad(
    gene_ids: np.array,
    max_length: int,
    pad_token_id: int,
    sample: bool = True,
) -> np.array:
    """Truncate and pad samples.

    Args:
        gene_ids (np.ndarray): Array of gene IDs.
        max_length (int): Maximum length of the samples.
        pad_token_id (int): ID of the padding token.
        sample (bool, optional): Whether to sample or truncate the samples. Defaults to True.

    Returns:
        np.array: Tuple containing the truncated or padded gene IDs.
    """
    if len(gene_ids) == max_length:
        return gene_ids

    if len(gene_ids) > max_length:  # - sample or truncate
        if sample:
            indices = np.random.permutation(len(gene_ids))[:max_length]
            return gene_ids[indices]
        else:
            return gene_ids[:max_length]
    else:  # - pad
        pad_tokens = np.full((max_length - len(gene_ids)), pad_token_id, dtype=np.int32)
        gene_ids = np.concatenate([gene_ids, pad_tokens])
        return gene_ids
