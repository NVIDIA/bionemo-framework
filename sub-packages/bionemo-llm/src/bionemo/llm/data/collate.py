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


from typing import Sequence, TypeVar

import torch

from bionemo.llm.data import types


_T = TypeVar("_T", bound=dict[str, torch.Tensor])


def padding_collate_fn(
    batch: Sequence[_T],
    padding_values: dict[str, int],
    min_length: int | None = None,
    max_length: int | None = None,
) -> _T:
    """Collate function with padding.

    Args:
        batch: List of samples, each of which is a dictionary of tensors.
        padding_values: A dictionary of padding values for each tensor key.
        min_length: Minimum length of the output batch; tensors will be padded to this length. If not
            provided, no extra padding beyond the max_length will be added.
        max_length: Maximum length of the sequence. If not provided, tensors will be padded to the
            longest sequence in the batch.

    Returns:
        A collated batch with the same dictionary input structure.
    """
    for entry in batch:
        if entry.keys() != padding_values.keys():
            raise ValueError("All keys in inputs must match provided padding_values.")

    def _pad(tensors, padding_value):
        if max_length is not None:
            tensors = [t[:max_length] for t in tensors]
        batched_tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        if min_length is None:
            return batched_tensors
        return torch.nn.functional.pad(batched_tensors, (0, min_length - batched_tensors.size(1)), value=padding_value)

    return {k: _pad([s[k] for s in batch], padding_values[k]) for k in batch[0].keys()}  # type: ignore[return-value]


def bert_padding_collate_fn(
    batch: Sequence[types.BertSample],
    padding_value: int,
    min_length: int | None = None,
    max_length: int | None = None,
) -> types.BertSample:
    """Padding collate function for BERT dataloaders.

    Args:
        batch (list): List of samples.
        padding_value (int, optional): The tokenizer's pad token ID.
        min_length: Minimum length of the output batch; tensors will be padded to this length. If not
            provided, no extra padding beyond the max_length will be added.
        max_length: Maximum length of the sequence. If not provided, tensors will be padded to the
            longest sequence in the batch.
    """
    padding_values = {
        "text": padding_value,
        "types": 0,
        "attention_mask": False,
        "labels": -1,
        "loss_mask": False,
        "is_random": 0,
    }
    return padding_collate_fn(
        batch=batch,  # type: ignore[assignment]
        padding_values=padding_values,
        min_length=min_length,
        max_length=max_length,
    )
