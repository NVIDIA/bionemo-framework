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


"""
In order to update test results or configs please use the following command:

UPDATE_EXPECTED_RESULTS=1 pytest examples/tests/test_model_pretrain_and_downstream.py
UPDATE_EXPECTED_CFG=1 pytest examples/tests/test_model_pretrain_and_downstream.py
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Type, TypedDict

import pytest
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import LightningModule

from bionemo.callbacks import setup_dwnstr_task_validation_callbacks
from bionemo.data.diffdock.data_manager import DataManager as DiffdockDataManager
from bionemo.data.equidock import DataManager
from bionemo.model.dna.dnabert.dnabert_model import DNABERTModel
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModel as DiffdockScoreModel,
)
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModelAllAtom as DiffdockConfidenceModel,
)
from bionemo.model.molecule.dsmbind.dsmbind_model import DSMBind
from bionemo.model.molecule.megamolbart import FineTuneMegaMolBART, MegaMolBARTModel, MegaMolBARTRetroModel
from bionemo.model.molecule.molmim import MolMIMModel
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.protein.equidock.equidock_model import EquiDock
from bionemo.model.protein.esm1nv import ESM1nvModel, ESM2nvModel
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.prott5nv import ProtT5nvModel
from bionemo.model.singlecell.downstream.finetuning import FineTuneGeneformerModel
from bionemo.model.singlecell.geneformer.model import GeneformerModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    check_expected_training_results,
    distributed_model_parallel_state,
    load_expected_training_results,
    resolve_cfg,
    teardown_apex_megatron_cuda,
)


# logger
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def bionemo_home() -> Path:
    try:
        x = os.environ["BIONEMO_HOME"]
    except KeyError:
        raise ValueError("Need to set BIONEMO_HOME in order to run unit tests! See docs for instructions.")
    else:
        yield Path(x).absolute()


@pytest.fixture(scope="session")
def config_path_for_tests(bionemo_home) -> str:
    yield str((bionemo_home / "examples" / "tests" / "conf").absolute())


# Pretraining, encoder finetuning and secondary structure validation-in-the-loop tests
class TrainingTestParams(TypedDict):
    config_name: str
    script_path: str
    model_cls: Type[LightningModule]
    model_size: int


TEST_PARAMS: List[TrainingTestParams] = [
    {
        "config_name": "megamolbart_test",
        "script_path": "examples/molecule/megamolbart/pretrain.py",
        "model_cls": MegaMolBARTModel,
        "model_size": 4047872,
    },
    {
        "config_name": "megamolbart_downstream_retro_test",
        "script_path": "examples/molecule/megamolbart/downstream_retro.py",
        "model_cls": MegaMolBARTRetroModel,
        "model_size": 45058048,
    },
    {
        "config_name": "megamolbart_physchem_test",
        "script_path": "examples/molecule/megamolbart/downstream_physchem.py",
        "model_cls": FineTuneMegaMolBART,
        "model_size": 66817,
    },
    {
        "config_name": "esm1nv_test",
        "script_path": "examples/protein/esm1nv/pretrain.py",
        "model_cls": ESM1nvModel,
        "model_size": 43612544,
    },
    {
        "config_name": "esm2nv_8M_test",
        "script_path": "examples/protein/esm2nv/pretrain.py",
        "model_cls": ESM2nvModel,
        "model_size": 7542848,
    },
    {
        "config_name": "prott5nv_sec_str_val_test",
        "script_path": "examples/protein/prott5nv/pretrain.py",
        "model_cls": ProtT5nvModel,
        "model_size": 198970496,
    },
    {
        "config_name": "prott5nv_test",
        "script_path": "examples/protein/prott5nv/pretrain.py",
        "model_cls": ProtT5nvModel,
        "model_size": 198970496,
    },
    {
        "config_name": "prott5nv_encoder_finetune_test",
        "script_path": "examples/protein/downstream/downstream_flip.py",
        "model_cls": FineTuneProteinModel,
        "model_size": 199145485,
    },
    {
        "config_name": "esm1nv_encoder_finetune_test",
        "script_path": "examples/protein/downstream/downstream_flip.py",
        "model_cls": FineTuneProteinModel,
        "model_size": 43787533,
    },
    {
        "config_name": "diffdock_score_test",
        "script_path": "examples/molecule/diffdock/train.py",
        "model_cls": DiffdockScoreModel,
        "model_size": 20248214,
    },
    {
        "config_name": "diffdock_confidence_test",
        "script_path": "examples/molecule/diffdock/train.py",
        "model_cls": DiffdockConfidenceModel,
        "model_size": 4769636,
    },
    {
        "config_name": "equidock_pretrain_test",
        "script_path": "examples/protein/equidock/pretrain.py",
        "model_cls": EquiDock,
        "model_size": 525671,
    },
    {
        "config_name": "equidock_finetune_test",
        "script_path": "examples/protein/equidock/pretrain.py",
        "model_cls": EquiDock,
        "model_size": 684074,
    },
    {
        "config_name": "openfold_initial_training_test",
        "script_path": "examples/protein/openfold/train.py",
        "model_cls": AlphaFold,
        "model_size": 93229082,
    },
    {
        "config_name": "dnabert_test",
        "script_path": "examples/dna/dnabert/pretrain.py",
        "model_cls": DNABERTModel,
        "model_size": 8121216,
    },
    {
        "config_name": "molmim_pretrain_xsmall_test",
        "script_path": "examples/molecule/molmim/pretrain.py",
        "model_cls": MolMIMModel,
        "model_size": 6288002,
    },
    {
        "config_name": "molmim_pretrain_continue_small_test",
        "script_path": "examples/molecule/molmim/pretrain.py",
        "model_cls": MolMIMModel,
        "model_size": 70349440,
    },
    {
        "config_name": "geneformer_pretrain_test",
        "script_path": "examples/singlecell/geneformer/pretrain.py",
        "model_cls": GeneformerModel,
        "model_size": 10300032,
    },
    # TODO get the following working with a small test case.
    # {
    #     "config_name": 'geneformer_finetune_perturb_test',
    #     "script_path": "examples/singlecell/geneformer/finetune_perturbseq.py",
    #     "model_cls": FineTuneGeneformerModel,
    #     "model_size": 15549510,
    # },
    # TODO test_model_training for DSMBind should be customized
    {
        "config_name": "dsmbind_training_test",
        "script_path": "examples/molecule/dsmbind/train.py",
        "model_cls": DSMBind,
        "model_size": 1019136,
    },
]

CONFIG_NAME = [params["config_name"] for params in TEST_PARAMS]
TRAINING_SCRIPT_PATH = [params["script_path"] for params in TEST_PARAMS]
MODEL_CLASS = [params["model_cls"] for params in TEST_PARAMS]
MODEL_PARAMETERS = [params["model_size"] for params in TEST_PARAMS]


@pytest.fixture(scope="module")
def expected_training_logs_path(bionemo_home) -> str:
    expected_logs_path = bionemo_home / "examples" / "tests" / "expected_results" / "trainer_logs"
    if not expected_logs_path.exists():
        expected_logs_path.mkdir(exist_ok=True)
    return str(expected_logs_path)


@pytest.fixture(scope="module")
def expected_configs_path(bionemo_home) -> str:
    expected_configs_path = bionemo_home / "examples" / "tests" / "expected_results" / "configs"
    if not expected_configs_path.exists():
        expected_configs_path.mkdir(exist_ok=True)
    return str(expected_configs_path)


@pytest.mark.parametrize("config_name", CONFIG_NAME)
def test_config_parameters(config_name: str, expected_configs_path: str, config_path_for_tests):
    """Load the config parameters and ensure they're identical to previous"""
    cfg = load_model_config(config_name=config_name, config_path=config_path_for_tests)
    expected_cfg_name = config_name + "-expected"
    if os.environ.get("UPDATE_EXPECTED_CFG", False):
        msg = f"Updating expected config in {expected_configs_path}/{expected_cfg_name}"
        logger.warning(msg)
        # will create a new comparison config
        OmegaConf.save(cfg, os.path.join(expected_configs_path, expected_cfg_name + ".yaml"))
        assert False, msg

    expected_cfg = load_model_config(config_name=expected_cfg_name, config_path=expected_configs_path)
    assert resolve_cfg(expected_cfg) == resolve_cfg(cfg), (
        f"Mismatch in config {expected_configs_path}/{expected_cfg_name}."
        f"\nIn order to update please use the folllowing command:\n UPDATE_EXPECTED_CFG=1 pytest examples/tests/test_model_pretrain_and_downstream.py"
    )


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.parametrize(
    "config_name, model_class, model_parameters",
    list(zip(CONFIG_NAME, MODEL_CLASS, MODEL_PARAMETERS)),
)
def test_model_size(config_name: str, model_class: LightningModule, model_parameters: int, config_path_for_tests):
    """Check that number of model weights are correct"""

    cfg = load_model_config(config_name=config_name, config_path=config_path_for_tests)
    with distributed_model_parallel_state():
        if model_class != DSMBind:  # DSMBind is not a LightningModule
            callbacks = setup_dwnstr_task_validation_callbacks(cfg)
            trainer = setup_trainer(cfg, callbacks=callbacks)
        if model_class == FineTuneProteinModel or model_class == FineTuneMegaMolBART:
            with open_dict(cfg):
                cfg.model.encoder_cfg = cfg  # TODO: why do we have to do this?
            model = model_class(cfg.model, trainer)
        elif model_class == DiffdockScoreModel or model_class == DiffdockConfidenceModel:
            DiffdockDataManager.reset_instances()
            data_manager = DiffdockDataManager(cfg)
            model = model_class(cfg=cfg, trainer=trainer, data_manager=data_manager)
        elif model_class == EquiDock:
            data_manager = DataManager(cfg)
            cfg.model.input_edge_feats_dim = data_manager.train_ds[0][0].edata["he"].shape[1]
            trainer = setup_trainer(cfg, callbacks=callbacks)
            model = EquiDock(cfg=cfg, trainer=trainer, data_manager=data_manager)
        elif model_class in {FineTuneGeneformerModel, GeneformerModel}:
            if model_class == FineTuneGeneformerModel:
                with open_dict(cfg):
                    cfg.model.encoder_cfg = cfg  # TODO: why do we have to do this?
            model = model_class(cfg.model, trainer)
        elif model_class == DSMBind:
            model = model_class(cfg.model)
            model.num_weights = sum(p.numel() for p in model.parameters())
        else:
            model = model_class(cfg.model, trainer)
        assert model.num_weights == model_parameters


config_to_test = [
    (config, script) for config, script in zip(CONFIG_NAME, TRAINING_SCRIPT_PATH) if config != "dsmbind_training_test"
]  # Skip the DSMBind training test for now since (1) we don't have validation setup and (2) DSMBind is not dependent on nemo.core.ModelPT.


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    "config_name, script_path",
    config_to_test,
)
def test_model_training(
    config_name: str, script_path: str, tmp_path: Path, expected_training_logs_path: str, config_path_for_tests
):
    """
    Run short model training and ensure key metrics are identical
    """
    # Lookup a free socket to fix errors with DDP on a single node, e.g. this pytest.
    # TODO(@cye): Why does this depend on how `pytest` is executed, i.e. with a single vs. multiple tests?
    # Some tests succeed when run in batch / fail otherwise, other tests succeed when run alone / fail otherwise.
    open_port = find_free_network_port()
    cmd = (
        f"export MASTER_PORT={open_port} && "
        f"python {script_path}  --config-path {config_path_for_tests} --config-name {config_name} "
        f"++exp_manager.exp_dir={tmp_path} "
        f"++create_trainer_metric_callback=True ++trainer_metric_callback_kwargs.log_path={tmp_path}"
    )
    print(cmd)
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    teardown_apex_megatron_cuda()
    # check if training was successful
    error_out = process_handle.stderr.decode("utf-8")
    assert process_handle.returncode == 0, f"Command failed:\n{cmd}\n Error log:\n{error_out}"

    # Comparing training results with expected training results.
    # SaveTrainerFinalMetricCallback saves training metrics to "{log_path}/trainer_logs.json" at the end of training
    trainer_logs_name = "trainer_logs.json"
    expected_logs_name = config_name + "-log.json"
    trainer_results = load_expected_training_results(str(tmp_path), trainer_logs_name)
    if os.environ.get("UPDATE_EXPECTED_RESULTS", False):
        # Updating expected trainer results
        expected_logs_path = f"{expected_training_logs_path}/{expected_logs_name}"
        msg = f"Updating expected training results in {expected_logs_path}"
        shutil.copy(f"{tmp_path}/{trainer_logs_name}", expected_logs_path)
        assert False, msg

    expected_results = load_expected_training_results(expected_training_logs_path, expected_logs_name)

    tolerance_overrides = {
        "grad_norm": 0.5,  # grad norm flucuates quite a bit in these small training runs
        "val_percent_invalid": 1.0,  # This varies a ton early in training, can be anywhere in 0-1.
        "val_character_accuracy": 2.0,  # This accuracy varries wildly early in training
        "reduced_train_hiddens_mim_log_prob_q_z_given_x": 2.0,  # This varies a ton early in training, can be anywhere in 0-1.
        # "3state_accuracy": 0.10,  # This accuracy is measured in percent
        # "resolved_accuracy": 0.05,  # This accuracy is measured in percent
        # "8state_accuracy": 0.05.0,  # This accuracy is measured in percent
        # "val_loss": 1.0,
        # "reduced_train_loss": 1.0,
    }

    check_expected_training_results(
        trainer_results,
        expected_results,
        rel_tol=0.2,
        err_msg="\nIn order to update please use the folllowing command:\n UPDATE_EXPECTED_RESULTS=1 pytest examples/tests/test_model_pretrain_and_downstream.py",
        test_rel_tol_overrides=tolerance_overrides,
    )
