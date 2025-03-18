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
import io
import os
import re
import shlex
import sqlite3
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.esm2.scripts.train_esm2 import get_parser, main
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import parse_kwargs_to_arglist
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state


def run_train_with_std_redirect(
    train_database_path,
    valid_database_path,
    parquet_train_val_inputs,
    result_dir,
    num_steps,
    val_check_interval,
    create_checkpoint_callback,
    create_tensorboard_logger,
    create_tflops_callback,
    resume_if_exists,
    wandb_project,
) -> str:
    """
    Run a function with output capture.
    """
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        train_small_esm2(
            train_database_path,
            valid_database_path,
            parquet_train_val_inputs,
            result_dir,
            num_steps,
            val_check_interval,
            create_checkpoint_callback,
            create_tensorboard_logger,
            create_tflops_callback,
            resume_if_exists,
            wandb_project,
        )

    train_stdout = stdout_buf.getvalue()
    train_stderr = stderr_buf.getvalue()
    print("Captured STDOUT:\n", train_stdout)
    print("Captured STDERR:\n", train_stderr)
    return train_stdout


def train_small_esm2(
    train_database_path,
    valid_database_path,
    parquet_train_val_inputs,
    result_dir,
    num_steps,
    val_check_interval,
    create_checkpoint_callback,
    create_tensorboard_logger,
    create_tflops_callback,
    resume_if_exists,
    wandb_project=None,
    limit_val_batches=1,
):
    train_cluster_path, valid_cluster_path = parquet_train_val_inputs
    with distributed_model_parallel_state():
        trainer = main(
            train_cluster_path=train_cluster_path,
            train_database_path=train_database_path,
            valid_cluster_path=valid_cluster_path,
            valid_database_path=valid_database_path,
            num_nodes=1,
            devices=1,
            min_seq_length=None,
            max_seq_length=128,
            result_dir=result_dir,
            experiment_name="esm2",
            wandb_project=wandb_project,
            wandb_offline=True,
            wandb_anonymous=True,
            num_steps=num_steps,
            scheduler_num_steps=None,
            warmup_steps=1,
            limit_val_batches=limit_val_batches,
            val_check_interval=val_check_interval,
            log_every_n_steps=1,
            num_dataset_workers=1,
            biobert_spec_option=BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
            lr=1e-4,
            micro_batch_size=2,
            accumulate_grad_batches=2,
            precision="bf16-mixed",
            resume_if_exists=resume_if_exists,
            create_tensorboard_logger=create_tensorboard_logger,
            create_tflops_callback=create_tflops_callback,
            num_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            ffn_hidden_size=4 * 4,
            create_checkpoint_callback=create_checkpoint_callback,
        )
    return trainer


@pytest.fixture
def dummy_protein_dataset(tmp_path):
    """Create a mock protein dataset."""
    db_file = tmp_path / "protein_dataset.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE protein (
            id TEXT PRIMARY KEY,
            sequence TEXT
        )
    """
    )

    proteins = [
        ("UniRef90_A", "ACDEFGHIKLMNPQRSTVWY"),
        ("UniRef90_B", "DEFGHIKLMNPQRSTVWYAC"),
        ("UniRef90_C", "MGHIKLMNPQRSTVWYACDE"),
        ("UniRef50_A", "MKTVRQERLKSIVRI"),
        ("UniRef50_B", "MRILERSKEPVSGAQLA"),
    ]
    cursor.executemany("INSERT INTO protein VALUES (?, ?)", proteins)

    conn.commit()
    conn.close()

    return db_file


@pytest.fixture
def dummy_parquet_train_val_inputs(tmp_path):
    """Create a mock protein train and val cluster parquet."""
    train_cluster_path = tmp_path / "train_clusters.parquet"
    train_clusters = pd.DataFrame(
        {
            "ur90_id": [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]],
        }
    )
    train_clusters.to_parquet(train_cluster_path)

    valid_cluster_path = tmp_path / "valid_clusters.parquet"
    valid_clusters = pd.DataFrame(
        {
            "ur50_id": ["UniRef50_A", "UniRef50_B", "UniRef50_A", "UniRef50_B"],  # 2 IDs more than confest
        }
    )
    valid_clusters.to_parquet(valid_cluster_path)
    return train_cluster_path, valid_cluster_path


@pytest.mark.parametrize("create_checkpoint_callback", [True, False])
def test_main_runs(tmp_path, dummy_protein_dataset, dummy_parquet_train_val_inputs, create_checkpoint_callback):
    val_check_interval = 2
    num_steps = 4
    trainer = train_small_esm2(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=num_steps,
        val_check_interval=val_check_interval,
        create_checkpoint_callback=create_checkpoint_callback,
        create_tensorboard_logger=True,
        create_tflops_callback=True,
        resume_if_exists=False,
        wandb_project=None,
    )

    experiment_dir = tmp_path / "esm2"
    assert experiment_dir.exists(), "Could not find experiment directory."
    assert experiment_dir.is_dir(), "Experiment directory is supposed to be a directory."
    log_dir = experiment_dir / "dev"
    assert log_dir.exists(), "Directory with logs does not exist"

    children = list(experiment_dir.iterdir())
    # ["checkpoints", "dev"] since wandb is disabled. Offline mode was causing troubles
    expected_children = 2 if create_checkpoint_callback else 1
    assert (
        len(children) == expected_children
    ), f"Expected {expected_children} child in the experiment directory, found {children}."

    if create_checkpoint_callback:
        checkpoints_dir = experiment_dir / "checkpoints"
        assert checkpoints_dir.exists(), "Checkpoints directory does not exist."
        # check if correct checkpoint was saved
        expected_checkpoint_suffix = f"step={num_steps -1}"
        matching_subfolders = [
            p
            for p in checkpoints_dir.iterdir()
            if p.is_dir() and (expected_checkpoint_suffix in p.name and "last" in p.name)
        ]
        assert (
            matching_subfolders
        ), f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."

    assert (log_dir / "nemo_log_globalrank-0_localrank-0.txt").is_file(), "Could not find experiment log."

    # Recursively search for files from tensorboard logger
    event_files = list(log_dir.rglob("events.out.tfevents*"))

    assert event_files, f"No TensorBoard event files found under {log_dir}"

    assert "val_ppl" in trainer.logged_metrics  # validation logging on by default
    assert "tflops_per_sec_per_gpu" in trainer.logged_metrics  # ensuring that tflops logger can be added
    assert "train_step_timing in s" in trainer.logged_metrics


@pytest.mark.slow
@pytest.mark.xfail(
    reason="ESM2 training fails to resume from checkpoints. "
    "Issue: https://github.com/NVIDIA/bionemo-framework/issues/757"
)
def test_main_stop_at_num_steps_and_continue(tmp_path, dummy_protein_dataset, dummy_parquet_train_val_inputs):
    val_check_interval = 2
    num_steps_first_run = 4
    num_steps_second_run = 6

    ######### FIRST TRAINING num_steps=num_steps_first_run #########
    train_stdout_first_run = run_train_with_std_redirect(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=num_steps_first_run,
        val_check_interval=val_check_interval,
        create_checkpoint_callback=True,
        create_tensorboard_logger=True,
        create_tflops_callback=True,
        resume_if_exists=True,
        wandb_project="test",
    )

    experiment_dir = tmp_path / "esm2"
    checkpoints_dir = experiment_dir / "checkpoints"
    assert experiment_dir.exists(), "Could not find experiment directory."
    assert experiment_dir.is_dir(), "Experiment directory is supposed to be a directory."
    wandb_dir = experiment_dir / "wandb"
    assert wandb_dir.exists(), "Directory with logs does not exist"

    assert f"Training epoch 0, iteration 0/{num_steps_first_run - 1}" in train_stdout_first_run
    # Extract and validate global steps
    global_steps_first_run = [int(m) for m in re.findall(r"\| global_step: (\d+) \|", train_stdout_first_run)]
    assert global_steps_first_run[0] == 0
    assert global_steps_first_run[-1] == (num_steps_first_run - 1)
    assert len(global_steps_first_run) == num_steps_first_run

    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    ######### SECOND TRAINING continues previous training since num_steps=num_steps_second_run #########
    # TODO: add that this should xfail at certain error message, refactor to check if when distributed is changed with redirect context menager,
    #  there maybe is not issue with process group
    train_stdout_second_run = run_train_with_std_redirect(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=num_steps_second_run,
        val_check_interval=val_check_interval,
        create_checkpoint_callback=True,
        create_tensorboard_logger=True,
        create_tflops_callback=True,
        resume_if_exists=True,
        wandb_project="test",
    )
    # Assertions
    assert f"Training epoch 0, iteration 0/{num_steps_second_run - 1}" not in train_stdout_second_run
    assert f"Training epoch 0, iteration {num_steps_first_run}/{num_steps_second_run - 1}" in train_stdout_second_run
    # Extract and validate global steps
    global_steps_second_run = [int(m) for m in re.findall(r"\| global_step: (\d+) \|", train_stdout_second_run)]
    assert global_steps_second_run[0] == num_steps_first_run
    assert global_steps_second_run[-1] == num_steps_second_run - 1
    assert len(global_steps_second_run) == (num_steps_second_run - num_steps_first_run)


@pytest.mark.parametrize("limit_val_batches", [0.0, 1.0, 4, None])
def test_val_dataloader_in_main_runs_with_limit_val_batches(
    tmp_path, dummy_protein_dataset, dummy_parquet_train_val_inputs, limit_val_batches
):
    # TODO: pydantic.
    """Ensures doesn't run out of validation samples whenever updating limit_val_batches logic.

    Args:
        monkeypatch: (MonkeyPatch): Monkey patch for environment variables.
        tmpdir (str): Temporary directory.
        dummy_protein_dataset (str): Path to dummy protein dataset.
        dummy_parquet_train_val_inputs (tuple[str, str]): Tuple of dummy protein train and val cluster parquet paths.
        limit_val_batches (Union[int, float, None]): Limit validation batches. None implies 1.0 as in PTL.
    """
    train_small_esm2(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=4,
        val_check_interval=2,
        create_checkpoint_callback=False,
        create_tensorboard_logger=False,
        create_tflops_callback=False,
        resume_if_exists=False,
        wandb_project=None,
        limit_val_batches=limit_val_batches,
    )


@pytest.mark.skip("duplicate with argparse, model and data unittests")
def test_pretrain_cli(tmpdir, dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    result_dir = Path(tmpdir.mkdir("results"))
    open_port = find_free_network_port()
    # NOTE: if you need to change the following command, please update the README.md example.
    cmd_str = f"""train_esm2     \
    --train-cluster-path {train_cluster_path} \
    --train-database-path {dummy_protein_dataset} \
    --valid-cluster-path {valid_cluster_path} \
    --valid-database-path {dummy_protein_dataset} \
    --result-dir {result_dir}     \
    --experiment-name test_experiment     \
    --num-gpus 1  \
    --num-nodes 1 \
    --val-check-interval 2 \
    --num-dataset-workers 1 \
    --num-steps 5 \
    --max-seq-length 128 \
    --limit-val-batches 1 \
    --val-check-interval 2 \
    --micro-batch-size 2 \
    --accumulate-grad-batches 2 \
    --num-layers 2 \
    --num-attention-heads 2 \
    --hidden-size 4 \
    --ffn-hidden-size 8
    """.strip()

    # a local copy of the environment
    env = dict(**os.environ)
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
    - train_cluster_path: The path to the training cluster parquet file.
    - train_database_path: The path to the training database file.
    - valid_cluster_path: The path to the validation cluster parquet file.
    - valid_database_path: The path to the validation database file.

    The values for these keys are placeholders and should be replaced with actual file paths.

    Returns:
        A dictionary with the required arguments for the pretraining script.
    """
    return {
        "train_cluster_path": "path/to/train_cluster.parquet",
        "train_database_path": "path/to/train.db",
        "valid_cluster_path": "path/to/valid_cluster.parquet",
        "valid_database_path": "path/to/valid.db",
    }


# TODO(@sichu) add test on dataset/datamodule on invalid path
def test_required_train_cluster_path(required_args_reference):
    """
    Test train_cluster_path is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("train_cluster_path")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arglist)


def test_required_train_database_path(required_args_reference):
    """
    Test train_database_path is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("train_database_path")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arglist)


def test_required_valid_cluster_path(required_args_reference):
    """
    Test valid_cluster_path is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("valid_cluster_path")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arglist)


def test_required_valid_database_path(required_args_reference):
    """
    Test valid_database_path is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("valid_database_path")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
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
    parser = get_parser()
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
    parser = get_parser()
    parser.parse_args(arglist)


@pytest.mark.parametrize("limit_val_batches", [None, "None"])
def test_limit_val_batches_is_none(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as none.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    args = parser.parse_args(arglist)
    assert args.limit_val_batches is None


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
    parser = get_parser()
    parser.parse_args(arglist)
