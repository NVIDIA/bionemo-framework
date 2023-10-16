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

import logging
import os
import pathlib
from contextlib import contextmanager

import pytest
from hydra import compose, initialize
from rdkit import Chem

from bionemo.model.molecule.megamolbart import MegaMolBARTInference
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    check_model_exists,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

CONFIG_PATH = "../examples/molecule/megamolbart/conf"
PREPEND_CONFIG_DIR = os.path.abspath("../examples/conf")
MODEL_CLASS = MegaMolBARTInference
CHECKPOINT_PATH = "/model/molecule/megamolbart/megamolbart.nemo"

####

_INFERER = None
os.environ["PROJECT_MOUNT"] = os.environ.get("PROJECT_MOUNT", '/workspace/bionemo')
THIS_FILE_DIR = pathlib.Path(os.path.abspath(__file__)).parent

_SMIS = [
    'c1cc2ccccc2cc1',
    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
    'CC(=O)C(=O)N1CCC([C@H]2CCCCN2C(=O)c2ccc3c(n2)CCN(C(=O)OC(C)(C)C)C3)CC1',
]


def get_cfg(prepend_config_path, config_name, config_path='conf'):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    return cfg


@contextmanager
def load_model(inf_cfg):
    global _INFERER
    if _INFERER is None:
        _INFERER = MegaMolBARTInference(inf_cfg)
    yield _INFERER


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH)


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_smis_to_hiddens():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)
    with load_model(cfg) as inferer:
        hidden_state, pad_masks = inferer.seq_to_hiddens(_SMIS)

        assert hidden_state is not None
        assert hidden_state.shape[0] == len(_SMIS)
        assert hidden_state.shape[2] == inferer.model.cfg.max_position_embeddings
        assert pad_masks is not None


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_smis_to_embedding():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)

    with load_model(cfg) as inferer:
        embedding = inferer.seq_to_embeddings(_SMIS)

        assert embedding is not None
        assert embedding.shape[0] == len(_SMIS)
        assert embedding.shape[1] == inferer.model.cfg.max_position_embeddings


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_hidden_to_smis():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)

    with load_model(cfg) as inferer:
        hidden_state, pad_masks = inferer.seq_to_hiddens(_SMIS)
        infered_smis = inferer.hiddens_to_seq(hidden_state, pad_masks)
        log.info(f'Input SMILES and Infered: {_SMIS}, {infered_smis}')

        assert len(infered_smis) == len(_SMIS)

        for smi, infered_smi in zip(_SMIS, infered_smis):
            log.info(f'Input and Infered:{smi},  {infered_smi}')
            input_mol = Chem.MolFromSmiles(smi)
            infer_mol = Chem.MolFromSmiles(infered_smi)
            assert input_mol is not None and infer_mol is not None

            canonical_smi = Chem.MolToSmiles(input_mol, canonical=True)
            canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
            log.info(f'Canonical Input and Infered: {canonical_smi}, {canonical_infered_smi}')

            assert canonical_smi == canonical_infered_smi


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_greedy():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)

    with load_model(cfg) as inferer:
        samples = inferer.sample(
            num_samples=3,
            sampling_method="greedy-perturbate",
            scaled_radius=1,
            smis=_SMIS,
        )
        samples = set(samples)
        log.info('\n'.join(_SMIS))
        log.info('\n'.join(samples))
        valid_molecules = []
        for smi in set(samples):
            isvalid = False
            mol = Chem.MolFromSmiles(smi)
            if mol:
                isvalid = True
                valid_molecules.append(smi)
            log.info(f'Sample: {smi},  {isvalid}')

        log.info('Valid Molecules' + "\n".join(valid_molecules))
        log.info(f'Total samples = {len(samples)} unique samples {len(set(samples))}  valids {len(valid_molecules)}')

        if len(valid_molecules) < len(samples) * 0.3:
            log.warning("TOO FEW VALID SAMPLES")
        assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_topk():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)

    with load_model(cfg) as inferer:
        samples = inferer.sample(
            num_samples=3,
            sampling_method="topkp-perturbate",
            scaled_radius=0,
            topk=4,
            temperature=2,
            topp=0.0,
            smis=_SMIS,
        )
        # samples = set(samples)
        nl = "\n"
        log.info(f"INPUTS: \n{nl.join(_SMIS)}\n")
        log.info(f"SAMPLES: \n{nl.join(samples)}\n")
        valid_molecules = []
        for smi in set(samples):
            isvalid = False
            mol = Chem.MolFromSmiles(smi)
            if mol:
                isvalid = True
                valid_molecules.append(smi)
            log.info(f'Sample: {smi},  {isvalid}')

        log.info('Valid Molecules' + "\n".join(valid_molecules))
        log.info(f'Total samples = {len(samples)} unique samples {len(set(samples))}  valids {len(valid_molecules)}')

        if len(valid_molecules) < len(samples) * 0.3:
            log.warning("TOO FEW VALID SAMPLES")
        assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_topp():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)

    with load_model(cfg) as inferer:
        samples = inferer.sample(
            num_samples=3,
            sampling_method="topkp-perturbate",
            scaled_radius=0,
            topk=0,
            temperature=2,
            topp=0.9,
            smis=_SMIS,
        )
        # samples = set(samples)
        nl = "\n"
        log.info(f"INPUTS: \n{nl.join(_SMIS)}\n")
        log.info(f"SAMPLES: \n{nl.join(samples)}\n")
        valid_molecules = []
        for smi in set(samples):
            isvalid = False
            mol = Chem.MolFromSmiles(smi)
            if mol:
                isvalid = True
                valid_molecules.append(smi)
            log.info(f'Sample: {smi},  {isvalid}')

        log.info('Valid Molecules' + "\n".join(valid_molecules))
        log.info(f'Total samples = {len(samples)} unique samples {len(set(samples))}  valids {len(valid_molecules)}')

        if len(valid_molecules) < len(samples) * 0.3:
            log.warning("TOO FEW VALID SAMPLES")
        assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_beam_search():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)

    beam_size = 5

    with load_model(cfg) as inferer:
        samples = inferer.sample(
            num_samples=beam_size,
            sampling_method="beam-search-perturbate",
            beam_alpha=0,
            smis=_SMIS,
        )
        assert len(samples) == len(_SMIS)
        assert len(samples[0]) == beam_size

        nl = "\n"
        for smi_i, samples_i in zip(_SMIS, samples):
            log.info(f"INPUT: \n{smi_i}\n")
            log.info(f"SAMPLES: \n{nl.join(samples_i)}\n")

        samples_flat = [item for sublist in samples for item in sublist]
        valid_molecules = []
        for smi in set(samples_flat):
            isvalid = False
            mol = Chem.MolFromSmiles(smi)
            if mol:
                isvalid = True
                valid_molecules.append(smi)
            log.info(f'Sample: {smi},  {isvalid}')

        log.info('Valid Molecules' + "\n".join(valid_molecules))

        log.info(
            f'Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))} '
            f'valids {len(valid_molecules)}'
        )

        if len(valid_molecules) < len(samples_flat) * 0.3:
            log.warning("TOO FEW VALID SAMPLES")
        assert len(valid_molecules) != 0
