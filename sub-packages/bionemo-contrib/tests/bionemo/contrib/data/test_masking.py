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

from bionemo.contrib.data.esm2 import tokenizer
from bionemo.contrib.data.masking import BertMaskConfig, apply_bert_pretraining_mask


def test_apply_bert_pretraining_mask():
    # fmt: off
    tokenized_sequence = torch.tensor(
        [20, 15, 11,  7, 10, 16,  9, 10,  4, 15,  8, 12,  7, 10, 12,  4,  9,
         10,  8, 15,  9, 14,  7,  8,  6,  5, 16,  4,  5,  9,  9,  4,  8,  7,
          8, 10, 16,  7, 12,  7, 16, 13, 12,  5, 19,  4, 10,  8,  4,  6, 19,
         17, 12,  7,  5, 11, 14, 10,  6, 19,  7,  4,  5,  6,  6])
    # fmt: on

    random_seed = 123

    # Apply the function
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(tokenized_sequence, random_seed)

    # Check the unmasked tokens are unchanged.
    assert torch.allclose(masked_sequence[~loss_mask], tokenized_sequence[~loss_mask])

    # Make sure the output labels are correct.
    assert torch.allclose(labels[loss_mask], tokenized_sequence[loss_mask])

    values, _ = torch.mode(masked_sequence[loss_mask])
    assert values.item() == tokenizer.SpecialToken.MASK


def test_apply_bert_pretraining_mask_no_mask_token():
    # fmt: off
    tokenized_sequence = torch.tensor(
        [20, 15, 11,  7, 10, 16,  9, 10,  4, 15,  8, 12,  7, 10, 12,  4,  9,
         10,  8, 15,  9, 14,  7,  8,  6,  5, 16,  4,  5,  9,  9,  4,  8,  7,
          8, 10, 16,  7, 12,  7, 16, 13, 12,  5, 19,  4, 10,  8,  4,  6, 19,
         17, 12,  7,  5, 11, 14, 10,  6, 19,  7,  4,  5,  6,  6])
    # fmt: on

    random_seed = 123

    # Apply the function
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(
        tokenized_sequence, random_seed, mask_config=BertMaskConfig(mask_token_prob=0.0)
    )

    # Check the unmasked tokens are unchanged.
    assert torch.allclose(masked_sequence[~loss_mask], tokenized_sequence[~loss_mask])

    # Make sure the output labels are correct.
    assert torch.allclose(labels[loss_mask], tokenized_sequence[loss_mask])

    # Make sure no mask tokens are in the output sequence
    assert torch.all(masked_sequence != tokenizer.SpecialToken.MASK)


def test_apply_bert_pretraining_mask_changing_mask_prob():
    # fmt: off
    tokenized_sequence = torch.tensor(
        [20, 15, 11,  7, 10, 16,  9, 10,  4, 15,  8, 12,  7, 10, 12,  4,  9,
         10,  8, 15,  9, 14,  7,  8,  6,  5, 16,  4,  5,  9,  9,  4,  8,  7,
          8, 10, 16,  7, 12,  7, 16, 13, 12,  5, 19,  4, 10,  8,  4,  6, 19,
         17, 12,  7,  5, 11, 14, 10,  6, 19,  7,  4,  5,  6,  6])
    # fmt: on

    random_seed = 123

    # Apply the function
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(
        tokenized_sequence, random_seed, mask_config=BertMaskConfig(mask_prob=0.0)
    )

    # All mask values should be False.
    assert torch.all(~loss_mask)
