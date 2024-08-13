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

from lightning.fabric.plugins.environments.lightning import find_free_network_port
from pretrain import main  # TODO: needs to be refactored to a package and imported!

from bionemo import geneformer
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption


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
assert bionemo2_root != Path("/")
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


def test_main_runs(tmpdir):
    result_dir = Path(tmpdir.mkdir("results"))

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
        result_dir / "test_experiment" / uq_rundir / "hparams.yaml"
    ).is_file(), "Could not find experiment hparams."
    assert (
        result_dir / "test_experiment" / uq_rundir / "nemo_log_globalrank-0_localrank-0.txt"
    ).is_file(), "Could not find experiment log."


def test_pretrain_cli(tmpdir):
    result_dir = Path(tmpdir.mkdir("results"))
    open_port = find_free_network_port()
    # NOTE: if you need to change the following command, please update the README.md example.
    cmd_str = f"""python  \
    scripts/singlecell/geneformer/pretrain.py     \
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
    --micro-batch-size 2
    """.strip()
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=bionemo2_root,
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, f"Pretrain script failed: {cmd_str}"
    assert (result_dir / "test_experiment").exists(), "Could not find test experiment directory."
