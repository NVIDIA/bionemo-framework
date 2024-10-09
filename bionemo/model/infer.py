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
from typing import Sequence
from uuid import uuid4

import h5py
import numpy as np
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig
from tqdm import tqdm

from bionemo.model.loading import setup_inference
from bionemo.model.run_inference import (
    extract_output_filename,
    extract_output_format,
    predict_ddp,
    validate_output_filename,
)


__all__: Sequence[str] = ()  # pragma: no cover


@hydra_runner()
def entrypoint(cfg: DictConfig) -> None:
    """Provide --config-dir and --config-name when using from the CLI."""
    model, trainer, dataloader = setup_inference(cfg)

    try:
        output_fname: str = extract_output_filename(cfg)
    except ValueError:
        filename = "infer_output"
        # Convention: put into cfg's exp_manager.exp_dir if possible!
        try:
            output_fname = str(Path(cfg.exp_manager.exp_dir).absolute() / filename)
        except AttributeError:
            logging.warning("No exp_manager.exp_dir in configuration! Must write output to current directory.")
            output_fname = f"./{filename}"

    try:
        output_format: str = extract_output_format(cfg)
    except ValueError:  # backward compatibility: default to pickle
        output_format = "pkl"

    if output_format not in ["pkl", "h5"]:
        raise ValueError(f"Users must specify either pkl or h5 output format but {output_format} is given")

    if os.path.exists(output_fname):
        if output_format == "pkl":
            output_fname += f"--{uuid4()}.{output_format}"
            logging.warning(f"Configuration's output filename already exists: {output_fname}!")
        elif output_format == "h5":
            for emb in cfg.model.downstream_task.outputs:
                output_embeddings_path = f"{output_fname}_{emb}.h5"
                logging.warning(f"Configuration's output filename already exists: {output_embeddings_path}!")
    else:
        output_folderpath: str = os.path.dirname(output_fname)
        if not os.path.exists(output_folderpath):
            os.makedirs(output_folderpath, exist_ok=True)
            print(f"Directory of output filename '{output_folderpath}' created")

    validate_output_filename(output_fname, overwrite=False)

    # predict outputs for all sequences in batch mode
    predictions = predict_ddp(model, dataloader, trainer, cfg.model.downstream_task.outputs)
    if predictions is None:
        logging.info("From non-rank 0 process: exiting now. Rank 0 will gather results and write.")
        return
    # from here only rank 0 should continue

    logging.info(f"Saving {len(predictions)} samples to {output_fname}")
    if output_format == "pkl":
        with open(output_fname, "wb") as wb:
            pickle.dump(predictions, wb)
    elif output_format == "h5":
        # pack embeddings into h5 instead of pickle
        for emb in cfg.model.downstream_task.outputs:
            values_dict = {sample["id"]: sample[emb] for sample in predictions}
            output_embeddings_path = f"{output_fname}_{emb}.h5"
            with h5py.File(output_embeddings_path, "w") as output_embeddings_file:
                for idx, (seq_id, values) in enumerate(tqdm(values_dict.items(), f"converting {emb} to h5 format")):
                    if emb == "hiddens":
                        maxshape = values.shape
                    elif emb == "embeddings":
                        maxshape = len(values)
                    output_embeddings_file.create_dataset(
                        str(idx), data=np.array(values), compression="gzip", chunks=True, maxshape=(maxshape)
                    )
                    output_embeddings_file[str(idx)].attrs["original_id"] = seq_id
    else:
        raise ValueError(f"Users must specify either pkl or h5 output format but {output_format} is given")


if __name__ == "__main__":
    entrypoint()
