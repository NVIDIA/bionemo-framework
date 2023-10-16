import logging
import os

import pytest
import pytorch_lightning as pl
import torch
from omegaconf.omegaconf import OmegaConf, open_dict

from bionemo.data.molecule import MoleculeEnumeration
from bionemo.model.molecule.megamolbart.megamolbart_model import MegaMolBARTModel
from bionemo.model.utils import initialize_model_parallel, setup_trainer
from bionemo.utils.tests import (
    clean_directory,
    list_to_tensor,
    load_expected_training_results,
    save_expected_training_results,
)
from tests.test_megamolbart_inference import get_cfg


logger = logging.getLogger(__name__)

_SMIS = [
    'c1cc2ccccc2cc1',
    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
    'CC(=O)C(=O)N1CCC([C@H]2CCCCN2C(=O)c2ccc3c(n2)CCN(C(=O)OC(C)(C)C)C3)CC1',
]
_BEAM_SIZE = 5
_BEAM_ALPHA = 0

CONFIG_NAME = 'megamolbart_test'
PREPEND_DIR = "../examples/tests/"
CONFIG_PATH = os.path.join(PREPEND_DIR, 'conf')
PREPEND_CONFIG_DIR = '../examples/molecule/megamolbart/conf'

CORRECT_RESULTS_DIR = 'examples/tests/expected_results'
CORRECT_RESULTS = 'megamolbart_inference_greedy_beam_search_preds.json'

UPDATE_EXPECTED_RESULTS = os.environ.get('UPDATE_EXPECTED_RESULTS', False)
COMPARE_EXPECTED_RESULTS = os.environ.get('COMPARE_EXPECTED_RESULTS', False)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _adjust_config_for_test(cfg: OmegaConf) -> OmegaConf:
    with open_dict(cfg):
        cfg.exp_manager.resume_if_exists = False
        cfg.model.micro_batch_size = len(_SMIS)
        cfg.model.global_batch_size = len(_SMIS)
        cfg.model.data.encoder_augment = False
        cfg.model.data.decoder_augment = False
        cfg.model.data.encoder_mask = False
        cfg.model.data.decoder_mask = False
        cfg.trainer.deterministic = True
        cfg.precision = 32
        cfg.seed = 42
    return cfg


@pytest.mark.needs_gpu
@pytest.mark.xfail(reason="FIXME: Currently broken")
def test_megamolbart_greedy_beam_search():
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
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name=CONFIG_NAME, config_path=CONFIG_PATH)
    cfg = _adjust_config_for_test(cfg)
    clean_directory(cfg.exp_manager.exp_dir)

    pl.seed_everything(cfg.seed)
    trainer = setup_trainer(cfg)
    model = MegaMolBARTModel(cfg.model, trainer)
    initialize_model_parallel(model)

    collate_fn = MoleculeEnumeration(
        tokenizer=model.tokenizer, seq_length=model._cfg.seq_length, pad_size_divisible_by_8=True, **model._cfg.data
    ).collate_fn
    batch = collate_fn(_SMIS)
    tokens_enc, _, _, _, enc_mask, _ = model.process_global_batch(batch)
    _NUM_TOKENS_TO_GENERATE = cfg.model.max_position_embeddings

    if not UPDATE_EXPECTED_RESULTS and COMPARE_EXPECTED_RESULTS:
        outputs = load_expected_training_results(
            results_comparison_dir=CORRECT_RESULTS_DIR, correct_results=CORRECT_RESULTS
        )
        weights = outputs['weights']

        # Convert weights from list to tensor.
        for key, lst in weights.items():
            if isinstance(lst, list):
                weights[key] = list_to_tensor(lst).cuda()

        model.load_state_dict(weights)
        for key in weights.keys():
            assert torch.equal(model.state_dict()[key], weights[key])

        # Convert output batch from list to tensor.
        expected_batch = outputs['batch']
        for key, lst in expected_batch.items():
            if isinstance(lst, list):
                if isinstance(lst[0], str):
                    expected_batch[key] = lst
                else:
                    expected_batch[key] = list_to_tensor(lst)

        for key in batch.keys():
            if key == 'target_smiles':
                assert batch[key] == expected_batch[key]
            else:
                assert torch.equal(batch[key], expected_batch[key])

    # this test requires warmup - otherwise there are some logits discrepancies later on
    model.freeze()
    model.eval()

    pl.seed_everything(cfg.seed)
    _ = model.decode(tokens_enc, enc_mask, 10)

    preds, logits = model.decode(tokens_enc, enc_mask, _NUM_TOKENS_TO_GENERATE)

    sampling_method = 'beam-search'
    preds_beam1, logits_beam1 = model.decode(
        tokens_enc,
        enc_mask,
        _NUM_TOKENS_TO_GENERATE,
        sampling_method=sampling_method,
        sampling_kwargs={'beam_size': 1, 'beam_alpha': 0},
    )

    preds_beam, logits_beam, scores_beam = model.decode(
        tokens_enc,
        enc_mask,
        _NUM_TOKENS_TO_GENERATE,
        sampling_method=sampling_method,
        sampling_kwargs={'beam_size': _BEAM_SIZE, 'beam_alpha': _BEAM_ALPHA, 'return_scores': True},
    )

    preds_beam_best, logits_beam_best, scores_beam_best = model.decode(
        tokens_enc,
        enc_mask,
        _NUM_TOKENS_TO_GENERATE,
        sampling_method=sampling_method,
        sampling_kwargs={
            'beam_size': _BEAM_SIZE,
            'beam_alpha': _BEAM_ALPHA,
            'return_scores': True,
            'keep_only_best_tokens': True,
        },
    )

    preds = preds.cpu().detach()
    logits = logits.cpu().detach()
    preds_beam1 = preds_beam1.cpu().detach()
    logits_beam1 = logits_beam1.cpu().detach()
    preds_beam = preds_beam.cpu().detach()
    logits_beam = logits_beam.cpu().detach()
    scores_beam = scores_beam.cpu().detach()
    preds_beam_best = preds_beam_best.cpu().detach()
    logits_beam_best = logits_beam_best.cpu().detach()
    scores_beam_best = scores_beam_best.cpu().detach()

    assert torch.equal(preds, preds_beam1)
    assert torch.equal(logits, logits_beam1)

    assert [int(x) for x in preds.shape] == [len(_SMIS), _NUM_TOKENS_TO_GENERATE + 1]
    assert [int(x) for x in logits.shape] == [len(_SMIS), _NUM_TOKENS_TO_GENERATE]

    assert preds.shape == preds_beam_best.shape and logits.shape == logits_beam_best.shape
    assert torch.equal(scores_beam_best.max(dim=1, keepdim=True)[0], scores_beam_best)

    assert torch.all((scores_beam[:, :-1] - scores_beam[:, 1:]) >= 0)
    # num_smi_to_generate + 1 accounts for BOS token at the beginning of the decoding if no decoded tokens are provided
    assert (
        [int(x) for x in preds_beam.shape] == [len(_SMIS), _BEAM_SIZE, _NUM_TOKENS_TO_GENERATE + 1]
        and [int(x) for x in logits_beam.shape] == [len(_SMIS), _BEAM_SIZE, _NUM_TOKENS_TO_GENERATE]
        and [int(x) for x in scores_beam.shape] == [len(_SMIS), _BEAM_SIZE]
    )

    if UPDATE_EXPECTED_RESULTS:
        weights = model.state_dict()
        logger.warning(f'Updating expected results in {CORRECT_RESULTS_DIR}/{CORRECT_RESULTS}')

        # Convert weights from tensors to list so we can save them in JSON.
        for key, tensor in weights.items():
            if isinstance(tensor, torch.Tensor):
                weights[key] = tensor.tolist()

        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch[key] = tensor.tolist()

        outputs = {
            'seed': cfg.seed,
            'smiles': _SMIS,
            'num_tokens_to_generate': _NUM_TOKENS_TO_GENERATE,
            'beam_size': _BEAM_SIZE,
            'beam_alpha': _BEAM_ALPHA,
            'greedy': {'predictions': preds.tolist(), 'logits': logits.tolist()},
            'beam': {
                'predictions': preds_beam.tolist(),
                'logits': logits_beam.tolist(),
                'scores': scores_beam.tolist(),
            },
            'weights': weights,
            'batch': batch,
        }
        save_expected_training_results(
            results_comparison_dir=CORRECT_RESULTS_DIR,
            correct_results=CORRECT_RESULTS,
            expected_results=outputs,
        )

    if not UPDATE_EXPECTED_RESULTS and COMPARE_EXPECTED_RESULTS:
        assert [
            k in ['greedy', 'beam', 'seed', 'smiles', 'num_tokens_to_generate', 'beam_size', 'beam_alpha']
            for k in outputs.keys()
        ]
        assert all(
            outputs[k] == val
            for k, val in zip(
                ['seed', 'smiles', 'num_tokens_to_generate', 'beam_size', 'beam_alpha'],
                [cfg.seed, _SMIS, _NUM_TOKENS_TO_GENERATE, _BEAM_SIZE, _BEAM_ALPHA],
            )
        ), 'Setup of the test does not match setup of the expected results'
        # Convert from list to tensor in order to compare.
        assert torch.equal(list_to_tensor(outputs['greedy']['predictions']), preds)
        assert torch.equal(list_to_tensor(outputs['greedy']['logits']), logits)
        assert torch.equal(list_to_tensor(outputs['beam']['predictions']), preds_beam)
        assert torch.equal(list_to_tensor(outputs['beam']['logits']), logits_beam)

    model.unfreeze()


if __name__ == '__main__':
    test_megamolbart_greedy_beam_search()
