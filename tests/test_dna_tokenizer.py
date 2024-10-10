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


import tempfile
from unittest import TestCase

import pytest

from bionemo.tokenizer import KmerTokenizer


example_strings = [
    "ACGTCAGAC",
    "CGNNAC",
    "DOG",
    "CATDOG",
]

expected_3mer_tokens = [
    [
        "ACG",
        "CGT",
        "GTC",
        "TCA",
        "CAG",
        "AGA",
        "GAC",
    ],
    ["CGN", "GNN", "NNA", "NAC"],
    [
        "DOG",
    ],
    [
        "CAT",
        "ATD",
        "TDO",
        "DOG",
    ],
]

expected_3mer_ids = [
    [5, 6, 7, 8, 9, 10, 11],
    [12, 13, 14, 15],
    [16],
    [17, 18, 19, 16],
]

expected_5mer_tokens = [
    [
        "ACGTC",
        "CGTCA",
        "GTCAG",
        "TCAGA",
        "CAGAC",
    ],
    [
        "CGNNA",
        "GNNAC",
    ],
    [],
    [
        "CATDO",
        "ATDOG",
    ],
]

expected_5mer_ids = [
    [5, 6, 7, 8, 9],
    [10, 11],
    [],
    [12, 13],
]

expected_3mer_vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
    "<SEP>": 3,
    "<MASK>": 4,
    "ACG": 5,
    "CGT": 6,
    "GTC": 7,
    "TCA": 8,
    "CAG": 9,
    "AGA": 10,
    "GAC": 11,
    "CGN": 12,
    "GNN": 13,
    "NNA": 14,
    "NAC": 15,
    "DOG": 16,
    "CAT": 17,
    "ATD": 18,
    "TDO": 19,
}

expected_3mer_decode_vocab = {
    0: "<PAD>",
    1: "<UNK>",
    2: "<CLS>",
    3: "<SEP>",
    4: "<MASK>",
    5: "ACG",
    6: "CGT",
    7: "GTC",
    8: "TCA",
    9: "CAG",
    10: "AGA",
    11: "GAC",
    12: "CGN",
    13: "GNN",
    14: "NNA",
    15: "NAC",
    16: "DOG",
    17: "CAT",
    18: "ATD",
    19: "TDO",
}

single_string_example = "ACGTA"

single_string_3mer_vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
    "<SEP>": 3,
    "<MASK>": 4,
    "ACG": 5,
    "CGT": 6,
    "GTA": 7,
}

single_string_3mer_vocab_saved = [
    "<PAD>\n",
    "<UNK>\n",
    "<CLS>\n",
    "<SEP>\n",
    "<MASK>\n",
    "ACG\n",
    "CGT\n",
    "GTA\n",
]


@pytest.mark.parametrize("test_input,expected", zip(example_strings, expected_3mer_tokens))
def test_3mer_text_to_tokens(test_input, expected):
    tokenizer = KmerTokenizer(3)
    tokens = tokenizer.text_to_tokens(test_input)
    assert tokens == expected


@pytest.mark.parametrize("test_input,expected", zip(example_strings, expected_5mer_tokens))
def test_5mer_text_to_tokens(test_input, expected):
    tokenizer = KmerTokenizer(5)
    tokens = tokenizer.text_to_tokens(test_input)
    assert tokens == expected


def test_3mer_build_vocab_list():
    tokenizer = KmerTokenizer(3)
    tokenizer.build_vocab(example_strings)
    tc = TestCase()
    tc.assertDictEqual(tokenizer.vocab, expected_3mer_vocab)
    tc.assertDictEqual(tokenizer.decode_vocab, expected_3mer_decode_vocab)


@pytest.mark.parametrize("test_input,expected", zip(example_strings, expected_3mer_ids))
def test_3mer_text_to_ids(test_input, expected):
    tokenizer = KmerTokenizer(3)
    tokenizer.build_vocab(example_strings)
    tokens = tokenizer.text_to_ids(test_input)
    assert tokens == expected


@pytest.mark.parametrize("test_input,expected", zip(example_strings, expected_5mer_ids))
def test_5mer_text_to_ids(test_input, expected):
    tokenizer = KmerTokenizer(5)
    tokenizer.build_vocab(example_strings)
    tokens = tokenizer.text_to_ids(test_input)
    assert tokens == expected


def test_3mer_text_to_ids_with_unk():
    tokenizer = KmerTokenizer(3)
    tokenizer.build_vocab(example_strings)
    text = "CATDOGANDFISHDOG"
    expected_tokens = [
        17,
        18,
        19,
        16,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        16,
    ]
    tokens = tokenizer.text_to_ids(text)
    assert tokens == expected_tokens


def test_save_vocab():
    model_file = tempfile.NamedTemporaryFile(suffix=".model")
    vocab_file = tempfile.NamedTemporaryFile(suffix=".vocab")

    tokenizer = KmerTokenizer(3)
    tokenizer.build_vocab(single_string_example)
    tokenizer.save_vocab(model_file.name, vocab_file.name)

    observed_vocab = open(vocab_file.name).readlines()

    assert observed_vocab == single_string_3mer_vocab_saved

    expected_k = [
        "3",
    ]
    observed_k = open(model_file.name).readlines()
    assert observed_k == expected_k

    model_file.close()
    vocab_file.close()


def test_load_vocab():
    model_file = tempfile.NamedTemporaryFile(suffix=".model")
    vocab_file = tempfile.NamedTemporaryFile(suffix=".vocab")

    with open(model_file.name, "w") as f:
        f.write("3")

    with open(vocab_file.name, "w") as f:
        for word in single_string_3mer_vocab_saved:
            f.write(word)

    tokenizer = KmerTokenizer.from_vocab_file(
        model_file.name,
        vocab_file.name,
    )

    tc = TestCase()

    tc.assertDictEqual(tokenizer.vocab, single_string_3mer_vocab)
    tc.assertDictEqual(tokenizer.decode_vocab, {id: token for token, id in single_string_3mer_vocab.items()})
