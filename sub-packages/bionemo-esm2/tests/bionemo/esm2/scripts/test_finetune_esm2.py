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


from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from nemo.lightning import io

from bionemo.core.data.load import load
from bionemo.esm2.model.finetune.dataset import InMemoryPerTokenValueDataset, InMemorySingleValueDataset
from bionemo.esm2.model.finetune.sequence_model import ESM2FineTuneSeqConfig
from bionemo.esm2.model.finetune.token_model import ESM2FineTuneTokenConfig
from bionemo.esm2.scripts.finetune_esm2 import finetune_esm2_entrypoint, get_parser, train_model
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.callbacks import MetricTracker


def data_to_csv(data, tmp_path):
    """Create a mock protein dataset."""
    csv_file = tmp_path / "protein_dataset.csv"
    # Create a DataFrame
    df = pd.DataFrame(data, columns=["sequences", "labels"])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.mark.parametrize("encoder_frozen", [True, False])
def test_esm2_finetune_token_classifier(
    tmp_path,
    dummy_data_per_token_classification_ft,
    encoder_frozen,
    n_steps_train: int = 50,
    seed: int = 42,
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        simple_ft_checkpoint, simple_ft_metrics, trainer = train_model(
            train_data_path=data_to_csv(dummy_data_per_token_classification_ft, tmp_path),
            valid_data_path=data_to_csv(dummy_data_per_token_classification_ft, tmp_path),
            experiment_name="finetune_new_head_token_classification",
            restore_from_checkpoint_path=str(load("esm2/8m:2.0")),
            num_steps=n_steps_train,
            num_nodes=1,
            devices=1,
            min_seq_length=None,
            max_seq_length=1024,
            result_dir=tmp_path / "finetune",
            limit_val_batches=2,
            val_check_interval=n_steps_train // 2,
            log_every_n_steps=n_steps_train // 2,
            num_dataset_workers=10,
            lr=1e-5,
            scale_lr_layer="classification_head",
            lr_multiplier=1e2,
            micro_batch_size=4,
            accumulate_grad_batches=1,
            resume_if_exists=False,
            precision="bf16-mixed",
            task_type="classification",
            encoder_frozen=encoder_frozen,
            dataset_class=InMemoryPerTokenValueDataset,
            config_class=ESM2FineTuneTokenConfig,
            metric_tracker=MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"]),
        )

        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert simple_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

        encoder_requires_grad = [
            p.requires_grad for name, p in trainer.model.named_parameters() if "classification_head" not in name
        ]
        assert (
            not all(encoder_requires_grad) == encoder_frozen
        ), f"Conflict in param requires_grad when encoder_frozen={encoder_frozen}"


@pytest.mark.parametrize("encoder_frozen", [True, False])
def test_esm2_finetune_regressor(
    tmp_path,
    dummy_data_single_value_regression_ft,
    encoder_frozen,
    n_steps_train: int = 50,
    seed: int = 42,
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        simple_ft_checkpoint, simple_ft_metrics, trainer = train_model(
            train_data_path=data_to_csv(dummy_data_single_value_regression_ft, tmp_path),
            valid_data_path=data_to_csv(dummy_data_single_value_regression_ft, tmp_path),
            experiment_name="finetune_new_head_regression",
            restore_from_checkpoint_path=str(load("esm2/8m:2.0")),
            num_steps=n_steps_train,
            num_nodes=1,
            devices=1,
            min_seq_length=None,
            max_seq_length=1024,
            result_dir=tmp_path / "finetune",
            limit_val_batches=2,
            val_check_interval=n_steps_train // 2,
            log_every_n_steps=n_steps_train // 2,
            num_dataset_workers=10,
            lr=1e-5,
            scale_lr_layer="regression_head",
            lr_multiplier=1e2,
            micro_batch_size=4,
            accumulate_grad_batches=1,
            resume_if_exists=False,
            precision="bf16-mixed",
            task_type="regression",
            encoder_frozen=encoder_frozen,
            dataset_class=InMemorySingleValueDataset,
            config_class=ESM2FineTuneSeqConfig,
            metric_tracker=MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"]),
        )

        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert simple_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

        encoder_requires_grad = [
            p.requires_grad for name, p in trainer.model.named_parameters() if "regression_head" not in name
        ]
        assert (
            not all(encoder_requires_grad) == encoder_frozen
        ), f"Conflict in param requires_grad when encoder_frozen={encoder_frozen}"


@pytest.mark.parametrize("encoder_frozen", [True, False])
def test_esm2_finetune_classifier(
    tmp_path,
    dummy_data_single_value_classification_ft,
    encoder_frozen,
    n_steps_train: int = 50,
    seed: int = 42,
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        simple_ft_checkpoint, simple_ft_metrics, trainer = train_model(
            train_data_path=data_to_csv(dummy_data_single_value_classification_ft, tmp_path),
            valid_data_path=data_to_csv(dummy_data_single_value_classification_ft, tmp_path),
            experiment_name="finetune_new_head_classification",
            restore_from_checkpoint_path=str(load("esm2/8m:2.0")),
            num_steps=n_steps_train,
            num_nodes=1,
            devices=1,
            min_seq_length=None,
            max_seq_length=1024,
            result_dir=tmp_path / "finetune",
            limit_val_batches=2,
            val_check_interval=n_steps_train // 2,
            log_every_n_steps=n_steps_train // 2,
            num_dataset_workers=10,
            lr=1e-5,
            scale_lr_layer="classification_head",
            lr_multiplier=1e2,
            micro_batch_size=4,
            accumulate_grad_batches=1,
            resume_if_exists=False,
            precision="bf16-mixed",
            task_type="classification",
            mlp_target_size=3,
            encoder_frozen=encoder_frozen,
            dataset_class=InMemorySingleValueDataset,
            config_class=ESM2FineTuneSeqConfig,
            metric_tracker=MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"]),
        )

        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert simple_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

        encoder_requires_grad = [
            p.requires_grad for name, p in trainer.model.named_parameters() if "classification_head" not in name
        ]
        assert (
            not all(encoder_requires_grad) == encoder_frozen
        ), f"Conflict in param requires_grad when encoder_frozen={encoder_frozen}"


@pytest.fixture
def mock_train_model():
    with patch("bionemo.esm2.scripts.finetune_esm2.train_model") as mock_train:
        yield mock_train


@pytest.fixture
def mock_parser_args():
    """Fixture to create mock arguments for the parser."""
    return [
        "--train-data-path",
        str(Path("train.csv")),
        "--valid-data-path",
        str(Path("valid.csv")),
        "--num-gpus",
        "1",
        "--num-nodes",
        "1",
        "--max-seq-length",
        "1024",
        "--result-dir",
        str(Path("./results")),
        "--lr",
        "0.001",
        "--task-type",
        "regression",
    ]


def test_finetune_esm2_entrypoint(mock_train_model, mock_parser_args):
    """Test the finetune_esm2_entrypoint function with mocked arguments."""
    with patch("sys.argv", ["finetune_esm2_entrypoint.py"] + mock_parser_args):
        finetune_esm2_entrypoint()

        # Check if train_model was called once
        mock_train_model.assert_called_once()

        # Check if the arguments were passed correctly
        called_kwargs = mock_train_model.call_args.kwargs
        assert called_kwargs["train_data_path"] == Path("train.csv")
        assert called_kwargs["valid_data_path"] == Path("valid.csv")
        assert called_kwargs["devices"] == 1
        assert called_kwargs["num_nodes"] == 1
        assert called_kwargs["max_seq_length"] == 1024
        assert called_kwargs["lr"] == 0.001
        assert called_kwargs["result_dir"] == Path("./results")


def test_get_parser():
    """Test the argument parser with all possible arguments."""
    parser = get_parser()
    args = parser.parse_args(
        [
            "--train-data-path",
            "train.csv",
            "--valid-data-path",
            "valid.csv",
            "--precision",
            "bf16-mixed",
            "--task-type",
            "classification",
            "--lr",
            "0.001",
            "--create-tensorboard-logger",
            "--resume-if-exists",
            "--result-dir",
            "./results",
            "--experiment-name",
            "esm2_experiment",
            "--wandb-entity",
            "my_team",
            "--wandb-project",
            "ft_project",
            "--wandb-tags",
            "tag1",
            "tag2",
            "--wandb-group",
            "group1",
            "--wandb-id",
            "1234",
            "--wandb-anonymous",
            "--wandb-log-model",
            "--wandb-offline",
            "--num-gpus",
            "2",
            "--num-nodes",
            "1",
            "--num-steps",
            "1000",
            "--num-dataset-workers",
            "4",
            "--val-check-interval",
            "500",
            "--log-every-n-steps",
            "100",
            "--min-seq-length",
            "512",
            "--max-seq-length",
            "1024",
            "--limit-val-batches",
            "2",
            "--micro-batch-size",
            "32",
            "--pipeline-model-parallel-size",
            "2",
            "--tensor-model-parallel-size",
            "2",
            "--accumulate-grad-batches",
            "2",
            "--save-last-checkpoint",
            "--metric-to-monitor-for-checkpoints",
            "val_loss",
            "--save-top-k",
            "5",
            "--restore-from-checkpoint-path",
            "./checkpoint",
            "--nsys-profiling",
            "--nsys-start-step",
            "10",
            "--nsys-end-step",
            "50",
            "--nsys-ranks",
            "0",
            "1",
            "--overlap-grad-reduce",
            "--no-overlap-param-gather",
            "--no-average-in-collective",
            "--grad-reduce-in-fp32",
            "--dataset-class",
            "InMemoryPerTokenValueDataset",
            "--config-class",
            "ESM2FineTuneTokenConfig",
            "--encoder-frozen",
            "--lr-multiplier",
            "1e2",
            "--scale-lr-layer",
            "dummy_layer",
        ]
    )

    # Assertions for all arguments
    assert args.train_data_path == Path("train.csv")
    assert args.valid_data_path == Path("valid.csv")
    assert args.precision == "bf16-mixed"
    assert args.task_type == "classification"
    assert args.lr == 0.001
    assert args.create_tensorboard_logger is True
    assert args.resume_if_exists is True
    assert args.result_dir == Path("./results")
    assert args.experiment_name == "esm2_experiment"
    assert args.wandb_entity == "my_team"
    assert args.wandb_project == "ft_project"
    assert args.wandb_tags == ["tag1", "tag2"]
    assert args.wandb_group == "group1"
    assert args.wandb_id == "1234"
    assert args.wandb_anonymous is True
    assert args.wandb_log_model is True
    assert args.wandb_offline is True
    assert args.num_gpus == 2
    assert args.num_nodes == 1
    assert args.num_steps == 1000
    assert args.num_dataset_workers == 4
    assert args.val_check_interval == 500
    assert args.log_every_n_steps == 100
    assert args.min_seq_length == 512
    assert args.max_seq_length == 1024
    assert args.limit_val_batches == 2
    assert args.micro_batch_size == 32
    assert args.pipeline_model_parallel_size == 2
    assert args.tensor_model_parallel_size == 2
    assert args.accumulate_grad_batches == 2
    assert args.save_last_checkpoint is True
    assert args.metric_to_monitor_for_checkpoints == "val_loss"
    assert args.save_top_k == 5
    assert args.restore_from_checkpoint_path == Path("./checkpoint")
    assert args.nsys_profiling is True
    assert args.nsys_start_step == 10
    assert args.nsys_end_step == 50
    assert args.nsys_ranks == [0, 1]
    assert args.overlap_grad_reduce is True
    assert args.no_overlap_param_gather is True
    assert args.no_average_in_collective is True
    assert args.grad_reduce_in_fp32 is True
    assert args.dataset_class == InMemoryPerTokenValueDataset
    assert args.config_class == ESM2FineTuneTokenConfig
    assert args.encoder_frozen is True
    assert args.lr_multiplier == 100
    assert args.scale_lr_layer == "dummy_layer"
