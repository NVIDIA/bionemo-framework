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

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict

import pytest
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from pretrain import main, parser  # TODO: needs to be refactored to a package and imported!

from bionemo import geneformer
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import parse_kwargs_to_arglist
from bionemo.testing import megatron_parallel_state_utils


# TODO(@jstjohn) use fixtures for pulling down data and checkpoints
# python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss
bionemo2_root: Path = (
    # geneformer module's path is the most dependable --> don't expect this to change!
    Path(geneformer.__file__)
    # This gets us from 'sub-packages/bionemo-geneformer/src/bionemo/esm2/__init__.py' to 'sub-packages/bionemo-geneformer'
    .parent.parent.parent.parent
    # From here, we want to get to the root of the repository: _before_ sub-packages/
    .parent.parent
).absolute()
data_path: Path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/processed_data"


def test_bionemo2_rootdir():
    assert (bionemo2_root / "sub-packages").exists(), "Could not find bionemo2 root directory."
    assert (bionemo2_root / "sub-packages").is_dir(), "sub-packages is supposed to be a directory."
    data_error_str = (
        "Please download test data with:\n"
        "`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    )
    assert data_path.exists(), f"Could not find test data directory.\n{data_error_str}"
    assert data_path.is_dir(), f"Test data directory is supposed to be a directory.\n{data_error_str}"


@pytest.mark.skip("duplicate unittest")
def test_main_runs(tmpdir):
    result_dir = Path(tmpdir.mkdir("results"))

    with megatron_parallel_state_utils.distributed_model_parallel_state():
        main(
            data_dir=data_path,
            num_nodes=1,
            devices=1,
            seq_length=128,
            result_dir=result_dir,
            wandb_project=None,
            wandb_offline=True,
            num_steps=55,
            limit_val_batches=1,
            val_check_interval=1,
            num_dataset_workers=0,
            biobert_spec_option=BiobertSpecOption.bert_layer_local_spec,
            lr=1e-4,
            micro_batch_size=2,
            accumulate_grad_batches=2,
            cosine_rampup_frac=0.01,
            cosine_hold_frac=0.01,
            precision="bf16-mixed",
            experiment_name="test_experiment",
            resume_if_exists=False,
            create_tensorboard_logger=False,
        )

    assert (result_dir / "test_experiment").exists(), "Could not find test experiment directory."
    assert (result_dir / "test_experiment").is_dir(), "Test experiment directory is supposed to be a directory."
    children = list((result_dir / "test_experiment").iterdir())
    assert len(children) == 1, f"Expected 1 child in test experiment directory, found {children}."
    uq_rundir = children[0]  # it will be some date.
    assert (
        result_dir / "test_experiment" / uq_rundir / "checkpoints"
    ).exists(), "Could not find test experiment checkpoints directory."
    assert (
        result_dir / "test_experiment" / uq_rundir / "checkpoints"
    ).is_dir(), "Test experiment checkpoints directory is supposed to be a directory."
    assert (
        result_dir / "test_experiment" / uq_rundir / "nemo_log_globalrank-0_localrank-0.txt"
    ).is_file(), "Could not find experiment log."


@pytest.mark.skip("duplicate unittest")
def test_pretrain_cli(tmpdir):
    result_dir = Path(tmpdir.mkdir("results"))
    open_port = find_free_network_port()
    # NOTE: if you need to change the following command, please update the README.md example.
    cmd_str = f"""python  \
    {bionemo2_root}/scripts/singlecell/geneformer/pretrain.py     \
    --data-dir {data_path}     \
    --result-dir {result_dir}     \
    --experiment-name test_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 10 \
    --num-dataset-workers 0 \
    --num-steps 55 \
    --seq-length 128 \
    --limit-val-batches 2 \
    --micro-batch-size 2 \
    --accumulate-grad-batches 2
    """.strip()
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, f"Pretrain script failed: {cmd_str}"
    assert (result_dir / "test_experiment").exists(), "Could not find test experiment directory."


@pytest.fixture(scope="function")
def required_args_reference() -> Dict[str, str]:
    """
    This fixture provides a dictionary of required arguments for the pretraining script.

    It includes the following keys:
    - data_dir: The path to the data directory.

    Returns:
        A dictionary with the required arguments for the pretraining script.
    """
    return {
        "data_dir": "path/to/cellxgene_2023-12-15_small",
    }


def test_required_data_dir(required_args_reference):
    """
    Test data_dir is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("data_dir")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    with pytest.raises(SystemExit):
        parser.parse_args(arglist)


#### test expected behavior on parser ####
@pytest.mark.parametrize("limit_val_batches", [0.1, 0.5, 1.0])
def test_limit_val_batches_is_float(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as a float.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
        limit_val_batches (float): The value of limit_val_batches.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser.parse_args(arglist)


@pytest.mark.parametrize("limit_val_batches", ["0.1", "0.5", "1.0"])
def test_limit_val_batches_is_float_string(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as a string of float.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
        limit_val_batches (float): The value of limit_val_batches.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser.parse_args(arglist)


@pytest.mark.parametrize("limit_val_batches", None, "None")
def test_limit_val_batches_is_none(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as none.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    args = parser.parse_args(arglist)
    assert args.limit_val_batches is None


# TODO(@sichu) add datamodule unittest when limit_val_batches smaller than gbs
@pytest.mark.parametrize("limit_val_batches", [1, 2])
def test_limit_val_batches_is_int(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as integer.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
        limit_val_batches (int): The value of limit_val_batches.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser.parse_args(arglist)
