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

import glob
import math
from contextlib import contextmanager
from unittest.mock import patch

import numpy.testing as npt
import pytest
import torch
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

from bionemo.data import ProteinBertCollate, SentencePieceTokenizerAdapter


@contextmanager
def passthrough_patch(func):
    """
    Applies a mock patch to a method that is a member of a class. The patched
    method behaves exactly the same as the original method.

    This is useful for accessing call behavior, such as `call_count` etc.
    """
    tmp = func
    cls = func.__self__.__class__
    module = cls.__module__
    name = f'{module}.{cls.__qualname__}.{func.__name__}'
    with patch(name) as p:
        p.side_effect = lambda x: tmp(x)
        yield p


protein_sequences = [
    "MNINNKKISKVVLLNSLGL",
    "MRIIFILFFCSLFLLSSC",
]


def get_tokenizer_model_paths():
    file_list = glob.glob('/tokenizers/protein/*/vocab/protein_sequence_sentencepiece.model')
    file_list = [str(x) for x in file_list]
    return file_list


def get_tokenizer(model_path):
    tokenizer = SentencePieceTokenizerAdapter(
        SentencePieceTokenizer(model_path=model_path, special_tokens=None, legacy=False)
    )
    return tokenizer


@pytest.mark.parametrize(
    "model_path",
    get_tokenizer_model_paths(),
)
def test_longer_sequence_than_seq_len_collate(model_path):
    tokenizer = get_tokenizer(model_path)
    collate = ProteinBertCollate(tokenizer=tokenizer, seq_length=6, pad_size_divisible_by_8=True)
    example = ["MNINNKKISKVVLLNSLGL"]
    collated_output = collate.collate_fn(example)
    expected = [[1, 22, 17, 14, 17, 17, 3, 3]]  # 3 is pad, 1 is bos, 2 is eos
    npt.assert_array_equal(expected, collated_output["labels"])


@pytest.mark.parametrize(
    "test_input,n_calls,model_path",
    [
        (protein_sequences, 2, get_tokenizer_model_paths()[0]),
        (["MRIIFIL"], 1, get_tokenizer_model_paths()[1]),
    ],
)
def test_number_of_tokenization_calls(test_input, n_calls, model_path):
    tokenizer = get_tokenizer(model_path)
    collate = ProteinBertCollate(
        tokenizer=tokenizer,
        seq_length=6,
        pad_size_divisible_by_8=False,
    )
    with passthrough_patch(collate.tokenizer.text_to_tokens) as p:
        collate.collate_fn(test_input)
        obs = p.call_count
    assert obs == n_calls


@pytest.mark.parametrize(
    "model_path",
    get_tokenizer_model_paths(),
)
def test_dataloader_collate(model_path):
    tokenizer = get_tokenizer(model_path)
    dataloader = ProteinBertCollate(tokenizer=tokenizer, seq_length=50, pad_size_divisible_by_8=True)
    collated_output = dataloader.collate_fn(protein_sequences)

    # calculate expected output
    tokens = [tokenizer.text_to_tokens(seq) for seq in protein_sequences]
    token_ids = tokenizer.tokens_to_ids(tokens)
    token_ids = [[tokenizer.bos_id] + token_id + [tokenizer.eos_id] for token_id in token_ids]
    # add padding
    pad_len = 0
    padding_mask = []
    padded_tokens = []
    for token_id in token_ids:
        pad_len = max(pad_len, len(token_id))
    pad_len = int(math.ceil(pad_len / 8) * 8)
    for token_id in token_ids:
        token_len = len(token_id)
        padded_tokens.append(token_id + [tokenizer.pad_id] * (pad_len - token_len))
        padding_mask.append([1] * token_len + [0] * (pad_len - token_len))

    # find out how many tokens are masked + perturbed
    expected_modification = 0.1
    expected_perturbation = 0.5  # rest of the 50% should be mask tokens)

    # get number of modifications
    num_mods_total = []
    mask_token_id = tokenizer.get_mask_id()
    for idx, tok in enumerate(collated_output["text"].numpy()):
        num_masks = 0
        num_perts = 0
        org_tok = padded_tokens[idx]
        expected_mods = int(expected_modification * len(org_tok))
        expected_pert = int(expected_perturbation * expected_mods)
        expected_masks = int(expected_mods - expected_pert)
        for idx2, token in enumerate(tok):
            if token != org_tok[idx2]:
                if token == mask_token_id:
                    num_masks = num_masks + 1
                else:
                    num_perts = num_perts + 1
        num_mods = num_perts + num_masks
        num_mods_total.append(num_mods)
        assert num_perts == expected_pert, "Expected number of perturbations don't match actual perturbations"
        assert num_masks == expected_masks, "Expected number of masks don't match actual masks"
        assert num_mods == expected_mods, "Expected number of modifications don't match actual modifications"

    for idx, loss_mask in enumerate(collated_output["loss_mask"].numpy()):
        num_ones = 0
        for m in loss_mask:
            if m == 1:
                num_ones = num_ones + 1
        assert num_ones == num_mods_total[idx], "Loss mask not accurate"
    assert torch.equal(collated_output["padding_mask"], torch.tensor(padding_mask)) is True, "Padding mask mismatch"
    assert torch.equal(collated_output["labels"], torch.tensor(padded_tokens)) is True, "Labels mismatch"
