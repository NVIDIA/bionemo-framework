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


import pytest
import torch

from bionemo.amplify.data.tokenizer import get_tokenizer


@pytest.fixture
def tokenizer():
    return get_tokenizer()


def test_tokenize_protein1(tokenizer):
    our_tokens = tokenizer.encode(
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", add_special_tokens=False
    )

    # fmt: off
    amplify_tokens = torch.tensor(
        [22, 17, 13, 9, 12, 18, 11, 12, 6, 17, 10, 14, 9, 12, 14, 6, 11, 12,
        10, 17, 11, 16, 9, 10, 8, 7, 18, 6, 7, 11, 11, 6, 10, 9, 10, 12, 18,
        9, 14, 9, 18, 15, 14, 7, 21, 6, 12, 10, 6, 8, 21, 19, 14, 9, 7, 13, 
        16, 12, 8, 21, 9, 6, 7, 8, 8])
    # fmt: on

    torch.testing.assert_close(torch.tensor(our_tokens), amplify_tokens)


def test_tokenize_protein2(tokenizer):
    our_tokens = tokenizer.encode(
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE", add_special_tokens=False
    )

    # fmt: off
    amplify_tokens = torch.tensor(
        [17, 7, 6, 13, 7, 12, 18, 18, 11, 9, 20, 15, 6, 14, 12, 15, 23, 14, 10, 
         18, 13, 8, 22, 16, 16, 13, 12, 7, 11, 14, 7, 18, 12, 6, 8, 20, 12, 10, 
         16, 19, 7, 7, 11, 11, 23, 6, 17, 7, 6, 7, 12, 17, 8, 9, 14, 11, 14, 9, 
         10, 8, 7, 10, 12, 8, 14, 12, 6, 6, 18, 11, 11])
    # fmt: on

    torch.testing.assert_close(torch.tensor(our_tokens), amplify_tokens)


def test_tokenize_protein2_with_mask(tokenizer):
    our_tokens = tokenizer.encode(
        "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE", add_special_tokens=False
    )

    # fmt: off
    amplify_tokens = torch.tensor(
        [17, 7, 6, 13, 7, 12, 18, 18, 11, 9, 20, 15, 6, 14, 12, 15, 2, 14, 10, 18, 
         13, 8, 22, 16, 16, 13, 12, 7, 11, 14, 7, 18, 12, 6, 8, 20, 12, 10, 16, 19, 
         7, 7, 11, 11, 23, 6, 17, 7, 6, 7, 12, 17, 8, 9, 14, 11, 14, 9, 10, 8, 7, 
         10, 12, 8, 14, 12, 6, 6, 18, 11, 11])
    # fmt: on

    torch.testing.assert_close(torch.tensor(our_tokens), amplify_tokens)


def test_tokenize_protein3(tokenizer):
    our_tokens = tokenizer.encode("KA<mask>I SQ", add_special_tokens=False)
    amplify_tokens = torch.tensor([17, 7, 2, 14, 1, 10, 18])
    torch.testing.assert_close(torch.tensor(our_tokens), amplify_tokens)


def test_tokenize_non_standard_tokens(tokenizer):
    our_tokens = tokenizer.encode("".join(["<pad>", "<eos>", "<unk>", "<mask>"]), add_special_tokens=False)
    amplify_tokens = torch.tensor([0, 4, 1, 2])
    torch.testing.assert_close(torch.tensor(our_tokens), amplify_tokens)


def test_tokenize_with_invalid_token(tokenizer):
    assert tokenizer.encode("<x>", add_special_tokens=False) == [1, 1, 1]


def test_tokenize_with_empty_string(tokenizer):
    assert tokenizer.encode("", add_special_tokens=True) == [3, 4]
