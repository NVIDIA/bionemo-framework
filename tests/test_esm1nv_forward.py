# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
This file tests the forward pass of ESM1.
"""

import json
import os
from pathlib import Path
from typing import Any, Iterator, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bionemo.model.loading import setup_inference
from bionemo.model.run_inference import predict_ddp
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import Deterministic, distributed_model_parallel_state


UPDATE_GOLDEN_VALUES = os.environ.get("UPDATE_GOLDEN_VALUES", "0") == "1"


@pytest.fixture(scope="module")
def golden_value_prepend_dir(bionemo_home: Path) -> Path:
    yield Path(bionemo_home / "tests" / "data" / "esm1_golden_values")


@pytest.fixture(scope="module")
def golden_value_heatmap_dir(golden_value_prepend_dir: Path) -> Path:
    yield Path(golden_value_prepend_dir / "heatmaps")


@pytest.fixture(scope="module")
def golden_values_fp16_json(golden_value_prepend_dir: Path) -> Path:
    yield Path(golden_value_prepend_dir / "revert_esm1nv_infer_golden_values_fp16.json")


@pytest.fixture(scope="module")
def golden_values_fp32_json(golden_value_prepend_dir: Path) -> Path:
    yield Path(golden_value_prepend_dir / "revert_esm1nv_infer_golden_values_fp32.json")


@pytest.fixture(scope="module")
def esm1nv_forward_config_path(bionemo_home: Path) -> Path:
    yield Path(bionemo_home / "examples" / "tests" / "conf")


def _load_config(precision: Literal["32", "16-mixed"], esm1nv_forward_config_path) -> DictConfig:
    cfg = load_model_config(config_name="esm1nv_infer", config_path=esm1nv_forward_config_path)
    cfg.trainer.precision = precision
    return cfg


def _load_model(cfg) -> Iterator[Tuple[Any, Any, DataLoader]]:
    with Deterministic(), distributed_model_parallel_state():
        model, trainer, dataloader = setup_inference(cfg)
        yield model, trainer, dataloader


@pytest.fixture(scope="module")
def cfg_32(esm1nv_forward_config_path):
    return _load_config("32", esm1nv_forward_config_path)


@pytest.fixture(scope="module")
def cfg_16mixed(esm1nv_forward_config_path):
    return _load_config("16-mixed", esm1nv_forward_config_path)


@pytest.fixture(scope="module")
def esm1_model_32(cfg_32):
    yield from _load_model(cfg_32)


@pytest.fixture(scope="module")
def esm1_model_16mixed(cfg_16mixed):
    yield from _load_model(cfg_16mixed)


def plot_heatmap(diff_array: np.ndarray, title: str, filepath: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(diff_array, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Predictions Index")
    plt.ylabel("Feature Index")
    plt.savefig(filepath)
    plt.close()


def create_diff_heatmaps(golden_predictions, new_predictions, tol=1e-1):
    # Initialize arrays for differences
    diff_embeddings = np.zeros((len(golden_predictions),) + golden_predictions[0]["embeddings"].shape)
    diff_ids = np.zeros(len(golden_predictions))
    diff_sequences = np.zeros(len(golden_predictions))

    # Iterate over predictions and compute differences
    for i, (golden, new) in enumerate(zip(golden_predictions, new_predictions)):
        # Compare ids
        if golden["id"] != new["id"]:
            diff_ids[i] = 1
        # Compare sequences
        if golden["sequence"] != new["sequence"]:
            diff_sequences[i] = 1
        # Compare embeddings
        diff_embeddings[i] = np.abs(golden["embeddings"] - new["embeddings"]) > tol

    return diff_ids, diff_sequences, diff_embeddings


def _test_esm1_inference_input_output_shapes_sizes_from_nemo_ckpt(cfg_infer, model, trainer, dataloader):
    """
    Run a forward pass on the ESM1 model using the online DAG check the shapes and sizes.
    Args:
        cfg_infer: The configuration for the inference.

    Important config arguments:
        cfg.model.seq_len: The maximum sequence length that can be used. This means the length of the amino acids in a protein sequence. This value is typically set to `1024`

    The inputs to the model are:
        {
        id: The id of the sequence. It is of shape: (batch_size, 1)
        sequence: The sequence of amino acids. It is of shape: (batch_size, sequence_length)
        }

    The outputs hold the following shape: A list of dictionaries. Each dictionary holds the following
    keys:
        embeddings: The embeddings for the model. We get one of these per each input tensor. They come out of the
            last transformer block. They are of shape: (hidden_size)
        hiddens: The hiddens for the model. We get one of these per each input tensor. The shape is (seq_length, hidden_size)
            Hiddens are the embeddings for each token in the sequence, and embeddings are embeddings for the whole sequence.
        sequence: The input sequence. It is of shape (sequence_length,)
        id: The id of the sequence. It is of shape (1,)
    """

    predictions = predict_ddp(model, dataloader, trainer, cfg_infer.model.downstream_task.outputs)

    # Confirm that we have an output for every input.
    assert len(predictions) == len(dataloader.dataset)

    # Confirm that the output is a list of dictionaries.
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], dict)

    # Confirm that embeddings are type numpy array
    assert isinstance(predictions[0]["embeddings"], np.ndarray)
    assert isinstance(predictions[0]["hiddens"], np.ndarray)

    # Confirm that the output dictionary has the expected keys.
    assert set(cfg_infer.model.downstream_task.outputs).issubset(set(predictions[0].keys()))

    # Confirm that the output dictionary has the expected values.
    sample = next(iter(dataloader))
    # Confirm that the input sample's sequence length matches the hidden layer output sequence length.
    assert len(sample["sequence"][0]) == len(predictions[0]["hiddens"])

    # Confirm that the shape of the embemddings are expected
    assert predictions[0]["embeddings"].shape == (cfg_infer.model.hidden_size,)
    assert predictions[0]["hiddens"].shape == (len(sample["sequence"][0]), cfg_infer.model.hidden_size)
    assert predictions[0]["embeddings"].shape == predictions[0]["hiddens"][-1].shape

    # Maybe remove this test. It is not necessary.
    assert sample["sequence"][0] == predictions[0]["sequence"]


def test_esm1_inference_input_output_shapes_sizes_from_nemo_ckpt_32(cfg_32, esm1_model_32):
    model, trainer, dataset = esm1_model_32
    _test_esm1_inference_input_output_shapes_sizes_from_nemo_ckpt(cfg_32, model, trainer, dataset)


def test_esm1_inference_input_output_shapes_sizes_from_nemo_ckpt_16mixed(cfg_16mixed, esm1_model_16mixed):
    model, trainer, dataset = esm1_model_16mixed
    _test_esm1_inference_input_output_shapes_sizes_from_nemo_ckpt(cfg_16mixed, model, trainer, dataset)


def _test_esm1nv_golden_value_json_and_overwrite(
    cfg_infer, heatmaps_dir, model, trainer, dataloader, tol, golden_json_filepath
):
    precision = cfg_infer.trainer.precision
    predictions = predict_ddp(model, dataloader, trainer, cfg_infer.model.downstream_task.outputs)

    sample1 = next(iter(dataloader))
    current_predictions = predictions[0]
    current_hiddens = current_predictions["hiddens"]
    current_embeddings = current_predictions["embeddings"]
    current_num_parameters = sum(p.numel() for p in model.parameters())
    if UPDATE_GOLDEN_VALUES:
        golden_json_filepath.parent.mkdir(exist_ok=True, parents=True)
        save_json = {
            "sample": sample1,
            "predictions": {
                "sequence": current_predictions["sequence"],
                "id": current_predictions["id"],
                "embeddings": current_embeddings.tolist(),
                "hiddens": current_hiddens.tolist(),
            },
            "num_parameters": current_num_parameters,
        }
        with open(golden_json_filepath, "w") as f:
            json.dump(save_json, f)

        assert False, f"Updated expected values at {golden_json_filepath}, rerun with UPDATE_GOLDEN_VALUES=0"

    else:
        assert (
            golden_json_filepath.exists()
        ), f"Expected values file not found at {golden_json_filepath}. Rerun with UPDATE_GOLDEN_VALUES=1 to create it."

        # Load golden values
        with open(golden_json_filepath, "r") as f:
            golden_values = json.load(f)

    # Convert predictions from json to array
    golden_predictions = golden_values["predictions"]
    golden_embeddings = np.array(golden_predictions["embeddings"])
    golden_hiddens = np.array(golden_predictions["hiddens"])

    # Compare sample1
    assert sample1 == golden_values["sample"]

    # Generate Heatmaps for Embeddings
    if not os.path.isdir(heatmaps_dir):
        os.makedirs(heatmaps_dir)

    diff_embeddings = np.abs(current_embeddings - golden_embeddings)

    # Compute the mean of these absolute differences
    mean_diff = np.mean(diff_embeddings)

    # Compute the mean of the absolute values of the original embeddings
    mean_current = np.mean(np.abs(current_embeddings))
    mean_golden = np.mean(np.abs(golden_embeddings))

    # Use the average of the two means as the baseline for percent difference
    baseline = (mean_current + mean_golden) / 2

    # Calculate the percent difference
    embedding_percent_diff = (mean_diff / baseline) * 100 if baseline != 0 else 0

    diff_embeddings = np.abs(current_embeddings - golden_embeddings).reshape(32, 24)

    plot_heatmap(
        diff_embeddings,
        f"Differences in Embeddings: {embedding_percent_diff:.2f}%",
        str(heatmaps_dir / f"diff_embeddings-{precision}.png"),
    )

    diff_hiddens = np.abs(current_hiddens - golden_hiddens)
    # Compute the mean of these absolute differences
    mean_diff = np.mean(diff_hiddens)

    # Compute the mean of the absolute values of the original hiddens
    mean_current = np.mean(np.abs(current_hiddens))
    mean_golden = np.mean(np.abs(golden_hiddens))

    # Use the average of the two means as the baseline for percent difference
    baseline = (mean_current + mean_golden) / 2

    # Calculate the percent difference
    hiddens_percent_diff = (mean_diff / baseline) * 100 if baseline != 0 else 0
    plot_heatmap(
        diff_hiddens,
        f"Differences in Hiddens: {hiddens_percent_diff:.2f}%",
        str(heatmaps_dir / f"diff_hiddens-{precision}.png"),
    )

    # Compare the first prediction vs the golden value prediction.
    assert np.allclose(current_embeddings, golden_embeddings, atol=tol)
    assert np.allclose(current_hiddens, golden_hiddens, atol=tol)
    assert current_predictions["sequence"] == golden_predictions["sequence"]
    assert current_predictions["id"] == golden_predictions["id"]

    # Check num parameters:
    golden_num_parameters = golden_values["num_parameters"]
    assert current_num_parameters == golden_num_parameters


@pytest.mark.needs_gpu
@pytest.mark.slow
def test_esm1nv_golden_value_json_and_overwrite_32(
    cfg_32, esm1_model_32, golden_values_fp32_json, golden_value_heatmap_dir
):
    model, trainer, dataset = esm1_model_32
    _test_esm1nv_golden_value_json_and_overwrite(
        cfg_32, golden_value_heatmap_dir, model, trainer, dataset, 1e-9, golden_values_fp32_json
    )


@pytest.mark.needs_gpu
@pytest.mark.slow
def test_esm1nv_golden_value_json_and_overwrite_16(
    cfg_16mixed, esm1_model_16mixed, golden_values_fp16_json, golden_value_heatmap_dir
):
    model, trainer, dataset = esm1_model_16mixed
    _test_esm1nv_golden_value_json_and_overwrite(
        cfg_16mixed, golden_value_heatmap_dir, model, trainer, dataset, 1e-1, golden_values_fp16_json
    )
