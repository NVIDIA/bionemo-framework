# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
from typing import Tuple

import pytest
import pytorch_lightning as pl
import pytorch_lightning as plt
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict

from bionemo.data.molecule import MoleculeEnumeration
from bionemo.model.molecule.megamolbart.megamolbart_model import MegaMolBARTModel
from bionemo.model.utils import pad_preds_after_first_eos, setup_trainer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    list_to_tensor,
    load_expected_training_results,
    save_expected_training_results,
    teardown_apex_megatron_cuda,
)


logger = logging.getLogger(__name__)

_SMIS = [
    "c1cc2ccccc2cc1",
    "COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC",
]

_BEAM_SIZE = 5
_BEAM_ALPHA = 0


CORRECT_RESULTS_DIR = "examples/tests/expected_results"
CORRECT_RESULTS = "megamolbart_inference_greedy_beam_search_preds.json"

UPDATE_EXPECTED_RESULTS = os.environ.get("UPDATE_EXPECTED_RESULTS", False)
COMPARE_EXPECTED_RESULTS = os.environ.get("COMPARE_EXPECTED_RESULTS", False)


def _adjust_config_for_test(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):
        cfg.exp_manager.resume_if_exists = False
        cfg.model.micro_batch_size = len(_SMIS)
        cfg.model.global_batch_size = len(_SMIS)
        cfg.model.data.encoder_augment = False
        cfg.model.data.decoder_independent_augment = False
        cfg.model.data.canonicalize_target_smile = True
        cfg.model.data.canonicalize_decoder_output = False
        cfg.model.data.canonicalize_encoder_input = False
        cfg.model.data.encoder_mask = False
        cfg.model.data.decoder_mask = False
        cfg.precision = 32
        cfg.seed = 42
    return cfg


@pytest.fixture(scope="module")
def megamolbart_model_trainer(model_cfg: DictConfig) -> Tuple[MegaMolBARTModel, plt.Trainer]:
    # TODO to remove the first reset in the future - test imp should ensire teardown after model is used
    pl.seed_everything(model_cfg.seed)
    model_cfg = _adjust_config_for_test(model_cfg)
    trainer = setup_trainer(model_cfg)
    model = MegaMolBARTModel(model_cfg.model, trainer)
    model.freeze()
    model.eval()
    yield model, trainer
    teardown_apex_megatron_cuda()


@pytest.fixture(scope="module")
def model_cfg(config_path_for_tests) -> DictConfig:
    cfg = load_model_config(config_name="megamolbart_test", config_path=config_path_for_tests)
    cfg = _adjust_config_for_test(cfg)
    return cfg


@pytest.mark.xfail(reason="Issue #195")
@pytest.mark.needs_gpu
def test_megamolbart_greedy_beam_search(megamolbart_model_trainer, model_cfg):
    """
    USAGE:
    The first part of this test examines greedy and beam search predictions generated on the fly.
    It is executed by python tests/test_beam_search.py

    The second part of this test compares saved results with generated results ensuring
    identical setup of the model and input. To run this comparison:
    1. generate results running: UPDATE_EXPECTED_RESULTS=True python tests/test_beam_search.py
    2. Compare predictions after codebase changes by running:
                                                    COMPARE_EXPECTED_RESULTS=True python tests/test_beam_search.py
    IMPORTANT: Make sure that steps 1 and 2 are executed using the same GPUs. Otherwise, the test from the step 2
    is very likely to not pass
    """
    model, trainer = megamolbart_model_trainer
    collate_fn = MoleculeEnumeration(
        tokenizer=model.tokenizer, seq_length=model._cfg.seq_length, pad_size_divisible_by_8=True, **model._cfg.data
    ).collate_fn
    batch = collate_fn(_SMIS)
    tokens_enc, _, _, _, enc_mask, _ = model.process_global_batch(batch)
    _NUM_TOKENS_TO_GENERATE = model._cfg.max_position_embeddings

    if not UPDATE_EXPECTED_RESULTS and COMPARE_EXPECTED_RESULTS:
        outputs = load_expected_training_results(
            results_comparison_dir=CORRECT_RESULTS_DIR, correct_results=CORRECT_RESULTS
        )
        weights = outputs["weights"]

        # Convert weights from list to tensor.
        for key, lst in weights.items():
            if isinstance(lst, list):
                weights[key] = list_to_tensor(lst).cuda()

        model.load_state_dict(weights)
        for key in weights.keys():
            assert torch.equal(model.state_dict()[key], weights[key])

        # Convert output batch from list to tensor.
        expected_batch = outputs["batch"]
        for key, lst in expected_batch.items():
            if isinstance(lst, list):
                if isinstance(lst[0], str):
                    expected_batch[key] = lst
                else:
                    expected_batch[key] = list_to_tensor(lst)

        for key in batch.keys():
            if key == "target_smiles":
                assert batch[key] == expected_batch[key]
            else:
                assert torch.equal(batch[key], expected_batch[key])

    pl.seed_everything(model_cfg.seed)
    assert not model.training
    # this test requires warmup - otherwise there are some logits discrepancies later on
    _ = model.decode(tokens_enc.clone(), enc_mask.clone(), 10)

    preds_ori, logits = model.decode(tokens_enc, enc_mask, _NUM_TOKENS_TO_GENERATE)
    # Make sure that greedy decoded preds are properly padded for equality comparison with beam search
    preds, greedy_pad_mask = pad_preds_after_first_eos(
        preds_ori.clone(), eos_id=model.tokenizer.eos_id, pad_id=model.tokenizer.pad_id
    )
    sampling_method = "beam-search"
    preds_beam1, logits_beam1 = model.decode(
        tokens_enc.clone(),
        enc_mask.clone(),
        _NUM_TOKENS_TO_GENERATE,
        sampling_method=sampling_method,
        sampling_kwargs={"beam_size": 1, "beam_alpha": 0, "keep_only_best_tokens": True},
    )
    _, beam1_pad_mask = pad_preds_after_first_eos(
        preds_beam1.clone(), eos_id=model.tokenizer.eos_id, pad_id=model.tokenizer.pad_id
    )

    preds_beam, logits_beam, scores_beam = model.decode(
        tokens_enc.clone(),
        enc_mask.clone(),
        _NUM_TOKENS_TO_GENERATE,
        sampling_method=sampling_method,
        sampling_kwargs={"beam_size": _BEAM_SIZE, "beam_alpha": _BEAM_ALPHA, "return_scores": True},
    )

    preds_beam_best, logits_beam_best, scores_beam_best = model.decode(
        tokens_enc.clone(),
        enc_mask.clone(),
        _NUM_TOKENS_TO_GENERATE,
        sampling_method=sampling_method,
        sampling_kwargs={
            "beam_size": _BEAM_SIZE,
            "beam_alpha": _BEAM_ALPHA,
            "return_scores": True,
            "keep_only_best_tokens": True,
        },
    )

    preds = preds.cpu().detach()
    logits = logits.cpu().detach()
    greedy_pad_mask = greedy_pad_mask.cpu().detach()
    preds_beam1 = preds_beam1.cpu().detach()
    logits_beam1 = logits_beam1.cpu().detach()
    beam1_pad_mask = beam1_pad_mask.cpu().detach()
    preds_beam = preds_beam.cpu().detach()
    logits_beam = logits_beam.cpu().detach()
    scores_beam = scores_beam.cpu().detach()
    preds_beam_best = preds_beam_best.cpu().detach()
    logits_beam_best = logits_beam_best.cpu().detach()
    scores_beam_best = scores_beam_best.cpu().detach()

    assert torch.equal(preds, preds_beam1)
    torch.testing.assert_close(logits * (~greedy_pad_mask[:, 1:]), logits_beam1 * (~beam1_pad_mask[:, 1:]))

    assert [int(x) for x in preds.shape] == [len(_SMIS), _NUM_TOKENS_TO_GENERATE + 1]
    assert [int(x) for x in logits.shape] == [len(_SMIS), _NUM_TOKENS_TO_GENERATE]

    assert preds.shape == preds_beam_best.shape and logits.shape == logits_beam_best.shape
    torch.testing.assert_close(scores_beam.max(dim=1, keepdim=True)[0], scores_beam_best)

    assert torch.all((scores_beam[:, :-1] - scores_beam[:, 1:]) >= 0)
    # num_smi_to_generate + 1 accounts for BOS token at the beginning of the decoding if no decoded tokens are provided
    assert (
        [int(x) for x in preds_beam.shape] == [len(_SMIS), _BEAM_SIZE, _NUM_TOKENS_TO_GENERATE + 1]
        and [int(x) for x in logits_beam.shape] == [len(_SMIS), _BEAM_SIZE, _NUM_TOKENS_TO_GENERATE]
        and [int(x) for x in scores_beam.shape] == [len(_SMIS), _BEAM_SIZE]
    )

    if UPDATE_EXPECTED_RESULTS:
        weights = model.state_dict()
        logger.warning(f"Updating expected results in {CORRECT_RESULTS_DIR}/{CORRECT_RESULTS}")

        # Convert weights from tensors to list so we can save them in JSON.
        for key, tensor in weights.items():
            if isinstance(tensor, torch.Tensor):
                weights[key] = tensor.tolist()

        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = tensor.tolist()

        outputs = {
            "seed": model_cfg.seed,
            "smiles": _SMIS,
            "num_tokens_to_generate": _NUM_TOKENS_TO_GENERATE,
            "beam_size": _BEAM_SIZE,
            "beam_alpha": _BEAM_ALPHA,
            "greedy": {"predictions": preds.tolist(), "logits": logits.tolist()},
            "beam": {
                "predictions": preds_beam.tolist(),
                "logits": logits_beam.tolist(),
                "scores": scores_beam.tolist(),
            },
            "weights": weights,
            "batch": batch,
        }
        save_expected_training_results(
            results_comparison_dir=CORRECT_RESULTS_DIR,
            correct_results=CORRECT_RESULTS,
            expected_results=outputs,
        )

    if not UPDATE_EXPECTED_RESULTS and COMPARE_EXPECTED_RESULTS:
        assert [
            k in ["greedy", "beam", "seed", "smiles", "num_tokens_to_generate", "beam_size", "beam_alpha"]
            for k in outputs.keys()
        ]
        assert all(
            outputs[k] == val
            for k, val in zip(
                ["seed", "smiles", "num_tokens_to_generate", "beam_size", "beam_alpha"],
                [model_cfg.seed, _SMIS, _NUM_TOKENS_TO_GENERATE, _BEAM_SIZE, _BEAM_ALPHA],
            )
        ), "Setup of the test does not match setup of the expected results"
        # Convert from list to tensor in order to compare.
        assert torch.equal(list_to_tensor(outputs["greedy"]["predictions"]), preds)
        assert torch.equal(list_to_tensor(outputs["greedy"]["logits"]), logits)
        assert torch.equal(list_to_tensor(outputs["beam"]["predictions"]), preds_beam)
        assert torch.equal(list_to_tensor(outputs["beam"]["logits"]), logits_beam)

    model.unfreeze()


if __name__ == "__main__":
    # To get into a pdb and set an external breakpoint, run:
    # python -m pdb tests/test_beam_search.py
    # then in the session, you can now set breakpoints, eg:
    #  b /usr/local/lib/python3.10/dist-packages/nemo/collections/nlp/models/language_modeling/megatron_lm_encoder_decoder_model.py:1309
    pytest.main([__file__, "-s"])
