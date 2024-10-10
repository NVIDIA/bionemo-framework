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


import glob
import math
import os
from contextlib import contextmanager
from typing import Dict
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
import torch
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from scipy.stats import binomtest

from bionemo.data import ProteinBertCollate, SentencePieceTokenizerAdapter


@contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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
    name = f"{module}.{cls.__qualname__}.{func.__name__}"
    with patch(name) as p:
        p.side_effect = lambda x: tmp(x)
        yield p


protein_sequences = [
    "MNINNKKISKVVLLNSLGL",
    "MRIIFILFFCSLFLLSSC",
]


def get_tokenizer_model_paths():
    tokenizer_dir = os.path.join(
        os.environ["BIONEMO_HOME"], "tokenizers/protein/*/vocab/protein_sequence_sentencepiece.model"
    )
    file_list = glob.glob(tokenizer_dir)
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
    # find out how many tokens are masked + perturbed
    expected_modification = 0.2
    expected_perturbation = 0.2
    expected_mask = 0.6
    expected_identity = 0.2
    with temp_seed(42):
        dataloader = ProteinBertCollate(
            tokenizer=tokenizer,
            seq_length=50,
            pad_size_divisible_by_8=True,
            modify_percent=expected_modification,
            perturb_token_percent=expected_perturbation,
            mask_token_percent=expected_mask,
            identity_token_percent=expected_identity,
        )
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

        # get number of modifications
        num_mods_total = []
        mask_token_id = tokenizer.get_mask_id()
        text = collated_output["text"].numpy()
        loss_mask = collated_output["loss_mask"].numpy()
        for idx, (toks, is_for_losses) in enumerate(zip(text, loss_mask)):
            num_masks = 0
            num_perts = 0
            num_identity = 0  # for tracking identity tokens included in the loss, not all identity tokens
            org_tok = padded_tokens[idx]
            expected_mods = int(expected_modification * len(org_tok))
            expected_pert = int(expected_perturbation * expected_mods)
            expected_masks = int(expected_mask * expected_mods)
            expected_identity = int(expected_mods - expected_pert - expected_masks)
            for idx2, (token, is_for_loss) in enumerate(zip(toks, is_for_losses)):
                if token != org_tok[idx2]:
                    assert is_for_loss == 1, "Loss mask not accurate"
                    if token == mask_token_id:
                        num_masks += 1
                    else:
                        num_perts += 1
                elif is_for_loss == 1:
                    num_identity += 1
            num_mods = num_perts + num_masks + num_identity  # Include identity token that are part of the mask here
            num_mods_total.append(num_mods)
            assert num_perts == pytest.approx(
                expected_pert, abs=math.ceil(0.5 * num_mods)
            ), "Expected number of perturbations don't match actual perturbations"
            assert num_masks == pytest.approx(
                expected_masks, abs=math.ceil(0.5 * num_mods)
            ), "Expected number of masks don't match actual masks"
            assert num_mods == pytest.approx(
                expected_mods, abs=math.ceil(0.5 * num_mods)
            ), "Expected number of modifications don't match actual modifications"
            assert num_identity == pytest.approx(
                expected_identity, abs=math.ceil(0.5 * num_mods)
            ), "Expected number of identity tokens don't match actual identity tokens"

        for idx, loss_mask in enumerate(collated_output["loss_mask"].numpy()):
            num_ones = 0
            for m in loss_mask:
                if m == 1:
                    num_ones = num_ones + 1
            assert num_ones == num_mods_total[idx], "Loss mask not accurate"
        assert (
            torch.equal(collated_output["padding_mask"], torch.tensor(padding_mask)) is True
        ), "Padding mask mismatch"
        assert torch.equal(collated_output["labels"], torch.tensor(padded_tokens)) is True, "Labels mismatch"


@pytest.mark.parametrize(
    "model_path",
    get_tokenizer_model_paths(),
)
def test_collated_probabilistic_convergence(model_path):
    """
    Test that on a long sequence with a high number of modifications, the number of modifications
        convergences to the expected value with tighter bounds than other tests.
    """
    tokenizer = get_tokenizer(model_path)
    # Chose out how many tokens are masked + perturbed + identity. Pick different values for everything so we can test that the proper setting is being
    #  used for each category in our tests on a very long sequence.
    expected_modification = 0.4  # 40% of tokens are modified
    expected_perturbation = 0.1  # 10% of modifications are perturbations
    expected_mask = 0.6  # 60% of modifications are masks
    expected_identity = 0.3  # 30% of modifications are identity tokens

    p_threshold_correct = (
        0.01  # Tests fail only 1/100 times by chance. Lower this value if this is a problem, maybe 0.001 or 0.005.
    )
    p_threshold_incorrect = (
        0.001  # We want a larger margin for incorrect assignment of probabilities to observations to test our power.
    )

    # 50000 tokens, 20000 modifications. The smallest fraction we test is 10%, so that would involve 2000 modifications, which should be enough to test.
    sequence_length = 50000
    protein_sequences = ["NVIDIAAIML" * int(sequence_length // 10)]  # NVIDIAAIML... 50000 tokens total

    with temp_seed(42):
        dataloader = ProteinBertCollate(
            tokenizer=tokenizer,
            seq_length=sequence_length,
            pad_size_divisible_by_8=True,
            modify_percent=expected_modification,
            perturb_token_percent=expected_perturbation,
            mask_token_percent=expected_mask,
            identity_token_percent=expected_identity,
        )
        collated_output: Dict[str, torch.Tensor] = dataloader.collate_fn(protein_sequences)
        assert set(collated_output.keys()).issuperset(
            {"text", "labels", "types", "padding_mask", "loss_mask"}
        ), "Incorrect keys in collated output"
        assert len(collated_output["text"]) == 1, "Incorrect number of sequences in collated output"
        assert len(collated_output["labels"]) == 1, "Incorrect number of labels in collated output"
        assert len(collated_output["loss_mask"]) == 1, "Incorrect number of types in collated output"
        assert len(collated_output["padding_mask"]) == 1, "Incorrect number of padding masks in collated output"
        assert torch.all(collated_output["padding_mask"] == 1), "Padding mask should be all 1s for this test"
        n_modified_tokens = torch.sum(collated_output["loss_mask"]).item()

        test_mods_correct = binomtest(
            n_modified_tokens, sequence_length, p=expected_modification, alternative="two-sided"
        )
        # Check that none of our other values could have been used instead of the correct one for the modification rate
        assert (
            test_mods_correct.pvalue > p_threshold_correct
        ), "Number of modified tokens does not converge to expected value"
        for setting_name, setting_value in [
            ("expected_mask", expected_mask),
            ("expected_identity", expected_identity),
            ("expected_perturbation", expected_perturbation),
        ]:
            test_mods_incorrect = binomtest(
                n_modified_tokens, sequence_length, p=setting_value, alternative="two-sided"
            )
            assert (
                test_mods_incorrect.pvalue < p_threshold_incorrect
            ), f"Risk that {setting_name} setting is being used instead of !"

        # Check that none of our other values could have been used instead of the correct one for the rate of identity tokens
        alphabet_size = 20  # 20 amino acids
        chance_of_identity = 1 / alphabet_size

        n_identity_tokens = torch.sum(
            (collated_output["text"] == collated_output["labels"]) * collated_output["loss_mask"]
        ).item()
        # A token is identity either by being identity by chance if it's a perturbation token, or by being selected for identity
        #   this test has corrections in place that assumes we do not filter out a chance identity match when asking for a random perturbation token.
        expected_identity_corrected = expected_modification * (
            expected_identity + chance_of_identity * expected_perturbation
        )
        test_identity_correct = binomtest(
            n_identity_tokens, sequence_length, p=expected_identity_corrected, alternative="two-sided"
        )
        assert (
            test_identity_correct.pvalue > p_threshold_correct
        ), "Number of identity loss tokens does not converge to expected value"
        for setting_name, setting_value in [
            ("expected_mask", expected_mask),
            ("expected_perturbation", expected_perturbation),
            ("expected_modification", expected_modification),
        ]:
            expected_bad_corrected = expected_modification * (
                setting_value + chance_of_identity * expected_perturbation
            )
            expected_bad_raw = setting_value * expected_modification
            test_mods_incorrect_corrected = binomtest(
                n_identity_tokens, sequence_length, p=expected_bad_corrected, alternative="two-sided"
            )
            test_mods_incorrect_raw = binomtest(
                n_identity_tokens, sequence_length, p=expected_bad_raw, alternative="two-sided"
            )
            assert (
                test_mods_incorrect_corrected.pvalue < p_threshold_incorrect
            ), f"Risk that {setting_name} setting is being used instead of identity_token!"
            assert (
                test_mods_incorrect_raw.pvalue < p_threshold_incorrect
            ), f"Risk that {setting_name} setting is being used instead of identity_token!"

        # Check that perturbed tokens are being properly assigned
        n_perturbed_tokens = torch.sum(
            (
                (collated_output["text"] != collated_output["labels"])
                & (collated_output["text"] != tokenizer.get_mask_id())
            )
            * collated_output["loss_mask"]
        ).item()
        expected_perturb_corrected = (
            expected_perturbation * expected_modification
            - chance_of_identity * expected_perturbation * expected_modification
        )
        test_perturb_correct = binomtest(
            n_perturbed_tokens, sequence_length, p=expected_perturb_corrected, alternative="two-sided"
        )
        assert (
            test_perturb_correct.pvalue > p_threshold_correct
        ), "Number of perturbed loss tokens does not converge to expected value"
        for setting_name, setting_value in [
            ("expected_mask", expected_mask),
            ("expected_identity", expected_identity),
            ("expected_modification", expected_modification),
        ]:
            expected_bad_corrected = (
                setting_value * expected_modification
                - chance_of_identity * expected_perturbation * expected_modification
            )
            expected_bad_raw = setting_value * expected_modification
            test_mods_incorrect_corrected = binomtest(
                n_perturbed_tokens, sequence_length, p=expected_bad_corrected, alternative="two-sided"
            )
            test_mods_incorrect_raw = binomtest(
                n_perturbed_tokens, sequence_length, p=expected_bad_raw, alternative="two-sided"
            )
            assert (
                test_mods_incorrect_corrected.pvalue < p_threshold_incorrect
            ), f"Risk that {setting_name} setting is being used instead of perturb_token!"
            assert (
                test_mods_incorrect_raw.pvalue < p_threshold_incorrect
            ), f"Risk that {setting_name} setting is being used instead of perturb_token!"

        # Check that masked tokens are being properly assigned
        n_masked_tokens = torch.sum(
            (collated_output["text"] == tokenizer.get_mask_id()) * collated_output["loss_mask"]
        ).item()
        test_masked_correct = binomtest(
            n_masked_tokens, sequence_length, p=expected_mask * expected_modification, alternative="two-sided"
        )
        assert (
            test_masked_correct.pvalue > p_threshold_correct
        ), "Number of masked loss tokens does not converge to expected value"
        for setting_name, setting_value in [
            ("expected_perturbation", expected_perturbation),
            ("expected_identity", expected_identity),
            ("expected_modification", expected_modification),
        ]:
            test_mods_incorrect = binomtest(
                n_masked_tokens, sequence_length, p=setting_value * expected_modification, alternative="two-sided"
            )
            assert (
                test_mods_incorrect.pvalue < p_threshold_incorrect
            ), f"Risk that {setting_name} setting is being used instead of mask_token!"

        # Check that tokens outside of the loss mask are either identity or all the same non-identity value
        non_loss_tokens = collated_output["text"][collated_output["loss_mask"] == 0]
        non_loss_labels = collated_output["labels"][collated_output["loss_mask"] == 0]

        # Make sure non-loss tokens are sane. They should eitehr be identity, or some random non-identity value
        all_others_identity = torch.all(
            non_loss_tokens == non_loss_labels
        )  # Sometimes all non-loss tokens are the identity value
        all_others_same = torch.all(
            non_loss_labels[0] == non_loss_labels
        )  # Sometimes, all non-loss tokens are the same value like -1, or -100
        all_tokens_not_identity = torch.all(non_loss_tokens != non_loss_labels)
        assert all_others_identity or (
            all_others_same and all_tokens_not_identity
        ), "Non-loss tokens are not being handled correctly. Expect either identity or some random non-identity value"


@pytest.mark.parametrize(
    "model_path",
    get_tokenizer_model_paths(),
)
def test_no_change_support(model_path):
    """
    Test that on a long sequence with a high number of modifications, the number of modifications
        convergences to the expected value with tighter bounds than other tests.
    """
    tokenizer = get_tokenizer(model_path)
    # Chose out how many tokens are masked + perturbed + identity. Pick different values for everything so we can test that the proper setting is being
    #  used for each category in our tests on a very long sequence.
    expected_modification = 0.0  # 40% of tokens are modified
    expected_perturbation = 0.1  # 10% of modifications are perturbations
    expected_mask = 0.6  # 60% of modifications are masks
    expected_identity = 0.3  # 30% of modifications are identity tokens

    # 50000 tokens, 20000 modifications. The smallest fraction we test is 10%, so that would involve 2000 modifications, which should be enough to test.
    sequence_length = 50000
    protein_sequences = ["NVIDIAAIML" * int(sequence_length // 10)]  # NVIDIAAIML... 50000 tokens total

    with temp_seed(42):
        dataloader = ProteinBertCollate(
            tokenizer=tokenizer,
            seq_length=sequence_length,
            pad_size_divisible_by_8=True,
            modify_percent=expected_modification,
            perturb_token_percent=expected_perturbation,
            mask_token_percent=expected_mask,
            identity_token_percent=expected_identity,
        )
        collated_output: Dict[str, torch.Tensor] = dataloader.collate_fn(protein_sequences)
        assert set(collated_output.keys()).issuperset(
            {"text", "labels", "types", "padding_mask", "loss_mask"}
        ), "Incorrect keys in collated output"
        assert len(collated_output["text"]) == 1, "Incorrect number of sequences in collated output"
        assert len(collated_output["labels"]) == 1, "Incorrect number of labels in collated output"
        assert len(collated_output["loss_mask"]) == 1, "Incorrect number of types in collated output"
        assert len(collated_output["padding_mask"]) == 1, "Incorrect number of padding masks in collated output"
        assert torch.all(collated_output["padding_mask"] == 1), "Padding mask should be all 1s for this test"
        n_modified_tokens = torch.sum(collated_output["loss_mask"]).item()
        assert n_modified_tokens == 0, "No tokens should be modified"
        assert torch.all(collated_output["text"] == collated_output["labels"]), "No tokens should be modified"
        # Make sure non-loss tokens are sane. They should eitehr be identity, or some random non-identity value
        all_others_identity = torch.all(
            collated_output["text"] == collated_output["labels"]
        )  # Sometimes all non-loss tokens are the identity value
        all_others_same = torch.all(
            collated_output["labels"][0] == collated_output["labels"]
        )  # Sometimes, all non-loss tokens are the same value like -1, or -100
        all_tokens_not_identity = torch.all(collated_output["text"] != collated_output["labels"])
        assert all_others_identity or (
            all_others_same and all_tokens_not_identity
        ), "Non-loss tokens are not being handled correctly. Expect either identity or some random non-identity value"


@pytest.mark.parametrize(
    "model_path",
    get_tokenizer_model_paths(),
)
def test_small_fraction_at_least_one_change(model_path):
    """
    Test that on a long sequence with a high number of modifications, the number of modifications
        convergences to the expected value with tighter bounds than other tests.
    """
    tokenizer = get_tokenizer(model_path)
    # Chose out how many tokens are masked + perturbed + identity. Pick different values for everything so we can test that the proper setting is being
    #  used for each category in our tests on a very long sequence.
    expected_modification = 1e-5  # Tiny non-zero fraction of masked tokens
    expected_perturbation = 0.1  # 10% of modifications are perturbations
    expected_mask = 0.6  # 60% of modifications are masks
    expected_identity = 0.3  # 30% of modifications are identity tokens

    # 50000 tokens, 20000 modifications. The smallest fraction we test is 10%, so that would involve 2000 modifications, which should be enough to test.
    sequence_length = 8
    protein_sequences = ["NVIDIAAI" * int(sequence_length // 8)]  # NVIDIAAIML... 8 tokens total

    with temp_seed(42):
        dataloader = ProteinBertCollate(
            tokenizer=tokenizer,
            seq_length=sequence_length,
            pad_size_divisible_by_8=True,
            modify_percent=expected_modification,
            perturb_token_percent=expected_perturbation,
            mask_token_percent=expected_mask,
            identity_token_percent=expected_identity,
        )
        collated_output: Dict[str, torch.Tensor] = dataloader.collate_fn(protein_sequences)
        assert set(collated_output.keys()).issuperset(
            {"text", "labels", "types", "padding_mask", "loss_mask"}
        ), "Incorrect keys in collated output"
        assert len(collated_output["text"]) == 1, "Incorrect number of sequences in collated output"
        assert len(collated_output["labels"]) == 1, "Incorrect number of labels in collated output"
        assert len(collated_output["loss_mask"]) == 1, "Incorrect number of types in collated output"
        assert len(collated_output["padding_mask"]) == 1, "Incorrect number of padding masks in collated output"
        assert torch.all(collated_output["padding_mask"] == 1), "Padding mask should be all 1s for this test"
        n_modified_tokens = torch.sum(collated_output["loss_mask"]).item()
        assert n_modified_tokens > 0, "At least one tokens should be modified"
