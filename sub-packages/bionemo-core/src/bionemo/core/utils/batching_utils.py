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


import math
from typing import List, Optional, Sequence, Tuple, Union

import torch


__all__: Sequence[str] = ("pad_token_ids",)


def pad_token_ids(
    token_ids: Union[List[int], List[torch.Tensor]],
    padding_value: int = 0,
    padding_len: Optional[int] = None,
    pad_size_divisible_by: int = 1,
    **convert_to_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads token ids with padding value, and return the padded tokens and the corresponding mask.

    Args:
        token_ids: List of token ids or tensors
        padding_value: Value to pad with. Defaults to 0.
        padding_len: Max length of the padded token ids. Defaults to None.
        pad_size_divisible_by: Pad the length of the token ids to be divisible by this number. Defaults to 1.
        **convert_to_kwargs: Passed directly to tensor.to(**kwargs) if provided

    Returns:
        Tuple[List[int], List[int]]: Padded token ids and mask
    """
    lengths = torch.tensor([len(s) for s in token_ids])
    if padding_len is None:
        padding_len = lengths.max()

    # make padding divisible by pad_size_divisible_by
    if pad_size_divisible_by > 1:
        padding_len = int(math.ceil(padding_len / pad_size_divisible_by) * pad_size_divisible_by)

    # build mask
    mask = torch.arange(padding_len)[None, :] < lengths[:, None]

    # make sure all sequences are pytorch tensors
    token_ids = [torch.tensor(s) if not torch.is_tensor(s) else s for s in token_ids]
    # pad sequences
    masked_token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=padding_value)

    # convert to desired device
    if len(convert_to_kwargs):
        mask = mask.to(**convert_to_kwargs)
        masked_token_ids = masked_token_ids.to(**convert_to_kwargs)

    # Further pad the sequences to the fixed maximum length, if necessary
    if masked_token_ids.size(1) < padding_len:
        padding_size = padding_len - masked_token_ids.size(1)
        masked_token_ids = torch.nn.functional.pad(masked_token_ids, [0, padding_size], value=padding_value)

    return masked_token_ids, mask
