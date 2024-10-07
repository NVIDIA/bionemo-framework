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
This file tests the forward pass of ESM2.
"""

import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from bionemo.model.loading import setup_inference
from bionemo.model.run_inference import predict_ddp
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    Deterministic,
    distributed_model_parallel_state,
    reset_microbatch_calculator,
)


os.environ["BIONEMO_HOME"] = os.environ.get("BIONEMO_HOME", "/workspace/bionemo")
BIONEMO_HOME = pathlib.Path(os.environ["BIONEMO_HOME"])

CONFIG_PATH = BIONEMO_HOME / "examples" / "tests" / "conf"
PREPEND_CONFIG_DIR = BIONEMO_HOME / "examples" / "protein" / "esm2nv" / "conf"

GOLDEN_VALUE_PREPEND_DIR = pathlib.Path(
    BIONEMO_HOME / "examples" / "tests" / "test_data" / "expected_outputs" / "forward_golden_values" / "esm2nv"
)
# SET FILEPATHS
GOLDEN_VALUES_FP16PT = GOLDEN_VALUE_PREPEND_DIR / "revert_esm2nv_infer_golden_values_fp16.pt"
GOLDEN_VALUES_FP32PT = GOLDEN_VALUE_PREPEND_DIR / "revert_esm2nv_infer_golden_values_fp32.pt"
GOLDEN_VALUES_FP16JSON = GOLDEN_VALUE_PREPEND_DIR / "revert_esm2nv_infer_golden_values_fp16.json"
GOLDEN_VALUES_FP32JSON = GOLDEN_VALUE_PREPEND_DIR / "revert_esm2nv_infer_golden_values_fp32.json"

# HEATMAPS DIR
HEATMAPS_REPO_DIR = GOLDEN_VALUE_PREPEND_DIR / "heatmaps"

# Diffs
DIFF_FILEPATH_FP32 = f"{GOLDEN_VALUE_PREPEND_DIR}/differences-fp32.yaml"
DIFF_FILEPATH_FP16 = f"{GOLDEN_VALUE_PREPEND_DIR}/differences-fp16.yaml"

parameter_sets_json = [
    ("32", 1e-9, GOLDEN_VALUES_FP32JSON, HEATMAPS_REPO_DIR),
    ("16-mixed", 1e-1, GOLDEN_VALUES_FP16JSON, HEATMAPS_REPO_DIR),
]


@pytest.fixture()
def cfg(config_path_for_tests):
    cfg = load_model_config(config_name="esm2nv_data_test", config_path=config_path_for_tests)
    yield cfg
    reset_microbatch_calculator()
    torch.cuda.empty_cache()


# @pytest.fixture
# def model_and_configs(cfg):
#     reset_microbatch_calculator()
#     initialize_distributed_parallel_state()
#
#     trainer = setup_trainer(cfg)
#     model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
#     yield model, cfg
#     reset_microbatch_calculator()
#     torch.cuda.empty_cache()


@pytest.fixture()
def cfg_infer(config_path_for_tests):
    cfg = load_model_config(config_name="esm2nv_infer", config_path=config_path_for_tests)
    yield cfg
    reset_microbatch_calculator()
    torch.cuda.empty_cache()


def plot_heatmap(diff_array, title, filepath):
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


def compare_predictions(predictions, golden_predictions, tol=1e-9):
    differences = []

    for i, (pred, gold) in enumerate(zip(predictions, golden_predictions)):
        diff = {}
        if not all(np.allclose(pred[key], gold[key], atol=tol) for key in ["embeddings", "hiddens"]):
            # Calculate differences for embeddings and hiddens
            diff["embeddings"] = np.abs(pred["embeddings"] - gold["embeddings"]).mean()
            diff["hiddens"] = np.abs(pred["hiddens"] - gold["hiddens"]).mean()
        if pred["sequence"] != gold["sequence"]:
            diff["sequence"] = (pred["sequence"], gold["sequence"])
        if pred["id"] != gold["id"]:
            diff["id"] = (pred["id"], gold["id"])
        if diff:
            differences.append(diff)

    return differences


# TODO(@jomitchell) Add later for gradient-based unit tests.
# def test_esm2nv_model_forward_pass_returns_golden_values_output_tensor(cfg):
#     cfg.trainer.precision = 32

#     initialize_distributed_parallel_state()
#     trainer = setup_trainer(cfg, adjust_config=True)
#     model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
#     train_dataset = model.build_train_dataset(cfg.model, 10)
#     train_dataloader = model.build_pretraining_data_loader(train_dataset, 0, num_workers=0)
#     batch = next(iter(train_dataloader))

#     tokens = batch['text'].cuda()
#     batch['types'].cuda()
#     batch['is_random'].cuda()
#     padding_mask = batch['padding_mask'].cuda()
#     lm_labels = batch["labels"]
#     batch["loss_mask"]

#     # Note: Weights are unknown so we need to load them via a .nemo file

#     model.forward(
#         tokens,
#         padding_mask,  # Why is this int64.
#         None,
#         lm_labels,
#         checkpoint_activations_all_layers=None,
#         model=model,
#     )


def test_esm2_inference_input_output_shapes_sizes_from_nemo_ckpt(cfg_infer):
    """
    Run a forward pass on the ESM2 model using the online DAG check the shapes and sizes.
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
    cfg_infer.trainer.precision = "32"
    with Deterministic(), distributed_model_parallel_state():
        # with distributed_model_parallel_state():
        model, trainer, dataloader = setup_inference(cfg_infer)
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


@pytest.mark.needs_gpu
@pytest.mark.slow
@pytest.mark.parametrize("precision, tol, golden_json_filepath, heatmaps_dir", parameter_sets_json)
def test_esm2nv_golden_value_json_and_overwrite(cfg_infer, precision, tol, golden_json_filepath, heatmaps_dir):
    cfg_infer.trainer.precision = precision
    with Deterministic(), distributed_model_parallel_state():
        model, trainer, dataloader = setup_inference(cfg_infer)
        predictions = predict_ddp(model, dataloader, trainer, cfg_infer.model.downstream_task.outputs)
    sample1 = next(iter(dataloader))
    current_predictions = predictions[0]
    current_hiddens = current_predictions["hiddens"]
    current_embeddings = current_predictions["embeddings"]

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

    diff_embeddings = np.abs(current_embeddings - golden_embeddings).reshape(32, 40)

    plot_heatmap(
        diff_embeddings,
        f"Differences in Embeddings: {embedding_percent_diff:.2f}%",
        f"{heatmaps_dir}/diff_embeddings-{precision}.png",
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
        f"{heatmaps_dir}/diff_hiddens-{precision}.png",
    )

    # Compare the first prediction vs the golden value prediction.
    assert np.allclose(current_embeddings, golden_embeddings, atol=tol)
    assert np.allclose(current_hiddens, golden_hiddens, atol=tol)
    assert current_predictions["sequence"] == golden_predictions["sequence"]
    assert current_predictions["id"] == golden_predictions["id"]

    # Check num parameters:
    current_num_parameters = sum(p.numel() for p in model.parameters())
    golden_num_parameters = golden_values["num_parameters"]
    assert current_num_parameters == golden_num_parameters

    save_json = {}
    save_json["sample"] = sample1
    save_json["predictions"] = current_predictions
    save_json["predictions"]["embeddings"] = current_embeddings.tolist()
    save_json["predictions"]["hiddens"] = current_hiddens.tolist()
    save_json["num_parameters"] = current_num_parameters

    # Overwrite and save new golden values to json.
    # with open(golden_json_filepath, 'w') as f:
    #     json.dump(save_json, f)
