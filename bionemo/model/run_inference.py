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
Runs inference over all models. Supports extracting embeddings, and hiddens.

NOTE: If out of memory (OOM) error occurs, try spliting the data to multiple smaller files.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import click
import numpy as np
import pytorch_lightning as pl
import torch
from nemo.utils import logging
from nemo.utils.distributed import gather_objects
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bionemo.model.loading import setup_inference
from bionemo.utils.hydra import load_model_config


__all__: Sequence[str] = (
    "predict_ddp",
    "predict",
    "format_predictions",
    "extract_output_filename",
    "validate_output_filename",
    "main",
)


def predict_ddp(
    model: pl.LightningModule, dataloader: DataLoader, trainer: pl.Trainer, output_schema
) -> Optional[List[Dict[str, np.ndarray]]]:
    all_batch_predictions = predict(model, dataloader, trainer)
    partial_predictions = format_predictions(all_batch_predictions, output_schema)
    # collect all results when using DDP
    logging.info("Collecting results from all GPUs...")
    predictions = gather_objects(partial_predictions, main_rank=0)
    # all but rank 0 will return None
    if predictions is None:
        return None
    return predictions


def predict(model: pl.LightningModule, dataloader: DataLoader, trainer: pl.Trainer) -> List[Dict[str, torch.Tensor]]:
    # predict outputs for all sequences in batch mode
    all_batch_predictions = trainer.predict(
        model=model,
        dataloaders=dataloader,
        return_predictions=True,
    )

    if len(all_batch_predictions) == 0:
        raise ValueError("No predictions were made")

    return all_batch_predictions


def format_predictions(
    all_batch_predictions: List[Dict[str, torch.Tensor]], output_schema
) -> List[Dict[str, np.ndarray]]:
    # break batched predictions into individual predictions (list of dics)
    predictions: List[Dict[str, np.ndarray]] = []
    pred_keys = list(all_batch_predictions[0].keys())

    for batch_predictions in all_batch_predictions:
        batch_size = len(batch_predictions[pred_keys[0]])
        for i in range(batch_size):
            predictions.append({k: _convert_to_numpy(batch_predictions[k][i]) for k in pred_keys})

    # extract active hiddens if needed
    extract_hiddens = "hiddens" in output_schema
    # otherwise, remove hiddens from the output
    # but always remove 'mask'
    for p in predictions:
        if extract_hiddens:
            if "hiddens" in p and "mask" in p:
                p["hiddens"] = p["hiddens"][p["mask"]]
                del p["mask"]
        else:
            if "mask" in p:
                del p["mask"]
            if "hiddens" in p:
                del p["hiddens"]

    return predictions


def _convert_to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts a tensor or numpy array to a numpy array, gracefully handling bfloat16 and float16 types.

    Args:
        x (Union[torch.Tensor, np.ndarray]): The input tensor or numpy array.

    Returns:
        np.ndarray: The converted numpy array.

    Raises:
        None

    """
    if not torch.is_tensor(x):
        return x

    x = x.detach().cpu()

    # Check and convert dtype if it is bfloat16 or float16
    if x.dtype == torch.bfloat16 or x.dtype == torch.float16:
        x = x.float()  # Converting to float32 before numpy conversion

    # For other types, directly convert to numpy
    return x.numpy()


def _resolve_and_validate_output_filename(cfg: DictConfig, output_override: Optional[str], overwrite: bool) -> str:
    if output_override is not None:
        output_fname = output_override
        logging.info(f"Using user-supplied output filename: {output_fname}")
    else:
        output_fname = extract_output_filename(cfg)
        logging.info(f"Using config-supplied output filename: {output_fname}")

    validate_output_filename(output_fname, overwrite)

    return output_fname


def extract_output_filename(cfg: DictConfig) -> str:
    try:
        output_fname = cfg.model.data.output_fname
    except Exception as e:
        raise ValueError("Expecting model.data.output_fname to exist in configuration!") from e
    if output_fname is None or len(output_fname) == 0:
        raise ValueError("Output filename not specified in configuration! Specify one under: model.data.output_fname")
    return output_fname


def extract_output_format(cfg: DictConfig) -> str:
    try:
        output_format = cfg.model.data.output_format
    except Exception as e:
        raise ValueError("Expecting model.data.extract_output_format to exist in configuration!") from e
    if output_format is None or len(output_format) == 0:
        raise ValueError(
            "Output format not specified in configuration! Specify one under: model.data.extract_output_format"
        )
    return output_format


def validate_output_filename(output_fname: str, overwrite: bool) -> None:
    """Raises an exception if invalid or otherwise cannot write to location."""
    if len(output_fname) == 0:
        raise ValueError("Output filepath may not be empty!")

    if os.path.exists(output_fname):
        if not overwrite:
            raise ValueError(f"Output path {output_fname} already exists! Will not overwrite!")
        else:
            logging.warning(f"Overwriting output filename: {output_fname}")
    else:
        try:
            Path(output_fname).absolute().parent.mkdir(exist_ok=True)
        except Exception:
            logging.error("Failed to ensure that directory containing output exists!")
            raise


def main(config_path: str, config_name: str, output_override: Optional[str], overwrite: bool) -> None:
    print(f"Loading config from:                   {str(Path(config_path) / config_name)}")
    print(f"Override output location from config?: {output_override}")
    print("Overwrite output file if it exists?", {overwrite})
    print("-" * 80)

    cfg = load_model_config(config_name=config_name, config_path=config_path, logger=logging)

    output_fname = _resolve_and_validate_output_filename(cfg, output_override, overwrite)

    logging.info("Loading model from config")
    model, trainer, dataloader = setup_inference(cfg)

    predictions = predict_ddp(model, dataloader, trainer, cfg.model.downstream_task.outputs)
    if predictions is None:
        logging.info("From non-rank 0 process: exiting now. Rank 0 will gather results and write.")
        return
    # from here only rank 0 should continue

    logging.info(f"Saving {len(predictions)} samples to {output_fname}")
    with open(output_fname, "wb") as wb:
        pickle.dump(predictions, wb)


@click.command()
@click.option("--config-path", required=True, help="Path to Hydra config directory where configuration date lives.")
@click.option(
    "--config-name",
    default="infer.yaml",
    show_default=True,
    required=True,
    help="Name of YAML config file in --config-path to load from.",
)
@click.option(
    "--output",
    required=False,
    help="An override to where to write the pickled predictions. "
    "If unset, then the model.data.output_fname value from the configuration is used. "
    "In either case, the predictions are written to the output filename using Python's pickle object format. "
    "Note that this filepath must not exist: exsiting content will *not* be overwritten.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="If present, will overwrite the output file. Defaults to not overwrite and return a non-zero exit code.",
)
def entrypoint(config_path: str, config_name: str, output: Optional[str], overwrite: bool) -> None:  # pragma: no cover
    main(config_path, config_name, output, overwrite)


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
