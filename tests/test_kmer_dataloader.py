# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from unittest import TestCase

import numpy.testing as npt
import pytest

from bionemo.data import (
    BertMasking,
    DeterministicLengthTruncator,
    KmerBertCollate,
    KmerTokenizerAdapter,
    LengthTruncator,
    SpanMasking,
)
from bionemo.tokenizer import KmerTokenizer


tc = TestCase()

build_sequences = ['ACGTAG']

example_sequences = [
    'ACGTA',
    'AGA',
    'NNNN',
]


def _setup_tokenizer():
    return KmerTokenizer(3).build_vocab(build_sequences)


def _setup_mock_sampler():
    return MockSampler(
        indices_to_sample=[
            [0, 1, 3],
            [2],
            [3, 1],
        ],
        tokens_to_sample=[1, 7, 5],
    )


expected_token_ids = [
    [7, 4, 6, 4, 3, 0, 0, 0],
    [2, 1, 4, 0, 0, 0, 0, 0],
    [2, 4, 1, 5, 0, 0, 0, 0],
]

expected_types = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

expected_is_random = [0, 1, 2]

expected_loss_mask = [
    [1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
]

expected_labels = [
    [2, 5, 6, 7, 3, 0, 0, 0],
    [2, 1, 3, 0, 0, 0, 0, 0],
    [2, 1, 1, 3, 0, 0, 0, 0],
]

expected_padding_mask = [
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
]

expected_text = ['ACGTA', 'AGA', 'NNNN']


class MockCallable(object):
    """
    Yields from a given list any time the object is called.
    Examples:
        >>> foo = MockCallable(['bar', 'pi', 'hoge'])
        >>> foo()
        'bar'
        >>> foo()
        'pi'
        >>> foo()
        'hoge'
    """

    def __init__(self, index):
        self.gen = self.index_generator(index)

    def __call__(self, *args, **kwargs):
        return next(self.gen)

    @staticmethod
    def index_generator(index):
        for item in index:
            yield item


class MockSampler(object):
    """
    This is inted to mock `nemo_chem.data._dataloader.TokenSampler by
    yielding indices from predetermined lists.
    Examples:
    >>> sampler = MockSampler(['foo', 'bar'], [7, 2, 4])
    >>> sampler.sample_indices()
    'foo'
    >>> sampler.sample_indices()
    'bar'
    >>> sampler.sample_token_id()
    7
    """

    def __init__(self, indices_to_sample, tokens_to_sample):
        self.sample_indices = MockCallable(indices_to_sample)
        self.sample_token_id = MockCallable(tokens_to_sample)


def _setup_dataloader():
    tokenizer = KmerTokenizerAdapter(_setup_tokenizer())
    dataloader = KmerBertCollate(
        tokenizer=tokenizer,
        seq_length=8,
        pad_size_divisible_by_8=True,
        masking_strategy=BertMasking(
            tokenizer=tokenizer,
            perturb_percent=0.5,
            modify_percent=0.5,
            sampler=_setup_mock_sampler(),
        ),
    )
    return dataloader


keys = ['text', 'types', 'is_random', 'loss_mask', 'labels', 'padding_mask', 'batch']
expectations = [
    expected_token_ids,
    expected_types,
    expected_is_random,
    expected_loss_mask,
    expected_labels,
    expected_padding_mask,
    expected_text,
]


@pytest.mark.parametrize("key,expected", zip(keys, expectations))
def test_collate_fn(key, expected):
    dataloader = _setup_dataloader()

    collated_output = dataloader.collate_fn(example_sequences)

    npt.assert_array_equal(collated_output[key], expected)


def test_span_masking_extension():
    masking_strategy = SpanMasking(
        tokenizer=KmerTokenizerAdapter(_setup_tokenizer()), seed_probability=0, span_length=2
    )
    seqs = [
        [
            'A',
            'C',
            'G',
            'T',
            'A',
            'A',
            'A',
            'E',
            'T',
        ],
        ['TGIF', 'LMNOP', 'C', 'G', 'E', 'AA', 'B', 'E', 'F'],
    ]
    exp = [
        ['<MASK>', '<MASK>', 'G', 'T', '<MASK>', '<MASK>', 'A', 'E', '<MASK>'],
        ['<MASK>', '<MASK>', '<MASK>', '<MASK>', 'E', 'AA', 'B', 'E', 'F'],
    ]
    exp_loss_mask = [
        [1, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
    ]
    masking_strategy.pick_mask_indices = MockCallable(
        [
            [0, 4, 8],
            [0, 1, 2],
        ]
    )

    obs, obs_loss_mask = masking_strategy(seqs)
    npt.assert_array_equal(exp, obs)
    npt.assert_array_equal(exp_loss_mask, obs_loss_mask)


def test_length_sampler():
    length_sampler = LengthTruncator()

    seqs = [
        'GATTACA',
        'A quick brown fox jumped over the lazy dog',
        'Quick Brown Fox',
    ]

    exp = [
        'GATT',
        'A quick brown fox jumped over the lazy dog',
        'Quick B',
    ]

    # just make sure nothing is broken
    length_sampler(seqs)

    # now add in some mocked values to test the sampling logic
    length_sampler.sample_length = MockCallable([4, 7])
    length_sampler.sample_probability = MockCallable(
        [0.1, 0.6, 0.4999],
    )

    obs = length_sampler(seqs)
    tc.assertListEqual(exp, obs)


def test_deterministic_length_sampler():
    length_sampler = DeterministicLengthTruncator()
    hash_number = 99911934912
    seqs = length_sampler._hash = MockCallable([hash_number] * 6)

    seqs = ['ACGTA', 'ACGTA']

    exp = ['ACG', 'ACG']

    tc.assertAlmostEqual(length_sampler.sample_probability(seqs[0]), 0.11934912)
    assert length_sampler.sample_length(seqs[0]) == 3

    obs = length_sampler(seqs)
    tc.assertListEqual(exp, obs)
