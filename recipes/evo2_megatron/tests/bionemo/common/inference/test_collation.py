# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch

from bionemo.common.inference.collation import _pad_sequence, batch_collator


@pytest.mark.parametrize("seq_dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_collate_variable_lengths(seq_dim, dtype):
    batch_dim = (seq_dim + 1) % 3
    shape = [2, 2, 2]
    shape[seq_dim] = 3
    short = torch.ones(shape, dtype=dtype)
    shape[seq_dim] = 5
    long = torch.full(shape, 2, dtype=dtype)
    result = batch_collator([short, long], batch_dim=batch_dim, seq_dim=seq_dim)
    expected_short = torch.zeros(shape, dtype=dtype)
    expected_short.narrow(seq_dim, 0, 3).fill_(1)
    torch.testing.assert_close(result, torch.cat([expected_short, long], dim=batch_dim))
    assert result.dtype == dtype
    torch.testing.assert_close(short, torch.ones_like(short))


def test_collate_nested_and_vectors():
    batches = [
        {"tokens": torch.tensor([[1, 2]]), "ids": torch.tensor([7])},
        {"tokens": torch.tensor([[3, 4, 5]]), "ids": torch.tensor([8])},
    ]
    result = batch_collator(batches)
    torch.testing.assert_close(result["tokens"], torch.tensor([[1, 2, 0], [3, 4, 5]]))
    torch.testing.assert_close(result["ids"], torch.tensor([7, 8]))


@pytest.mark.parametrize("seq_dim", [0, 1, 2])
def test_pad_sequence(seq_dim):
    tensor = torch.ones(2, 2, 2, requires_grad=True)
    result = _pad_sequence(tensor, 4, seq_dim)
    torch.testing.assert_close(result.narrow(seq_dim, 0, 2), tensor)
    torch.testing.assert_close(result.narrow(seq_dim, 2, 2), torch.zeros(2, 2, 2))
    result.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
    assert _pad_sequence(tensor, 2, seq_dim) is tensor
    with pytest.raises(ValueError, match="Target length"):
        _pad_sequence(tensor, 1, seq_dim)
