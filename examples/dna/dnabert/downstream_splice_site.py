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
from pathlib import PosixPath

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict

from bionemo.data.preprocess.dna.preprocess import (
    GRCh38Ensembl99FastaResourcePreprocessor,
    GRCh38Ensembl99GFF3ResourcePreprocessor,
)
from bionemo.model.dna.dnabert.splice_site_prediction import (
    SpliceSiteBERTPredictionModel,
)
from bionemo.model.utils import (
    InferenceTrainerBuilder,
    setup_trainer,
)
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector


@hydra_runner(config_path="conf", config_name="dnabert_config_splice_site")
def main(cfg):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    do_training = cfg.do_training
    # TODO: fixup the rest of these.
    do_prediction = cfg.get("do_prediction")

    with open_dict(cfg):
        cfg.model.encoder_cfg = cfg

    if not do_training:
        logging.info("************** Starting Preprocessing ***********")
        logging.warning(
            "For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose."
        )
        preprocessor = GRCh38Ensembl99GFF3ResourcePreprocessor(
            dest_directory=cfg.model.data.dataset_path,  # Set to $BIONEMO_HOME/data
            root_directory=cfg.model.data.root_directory,
            train_perc=cfg.model.data.train_perc,
            val_perc=cfg.model.data.val_perc,
            test_perc=cfg.model.data.test_perc,
            size=cfg.model.data.num_sites_to_include,
        )
        _data_paths = preprocessor.prepare()

        # Needed for our actual data loaders. Used inside SpliceSiteDataModule.get_fasta_files()
        fasta_preprocessor = GRCh38Ensembl99FastaResourcePreprocessor(
            root_directory=cfg.model.data.root_directory,  # Set to $BIONEMO_HOME/data
            dest_directory=cfg.model.data.dataset_path,
        )
        fasta_preprocessor.prepare()

        assert PosixPath(cfg.model.data.train_file) in _data_paths
        logging.info("*************** Finish Preprocessing ************")

    if do_prediction:
        if (predictions_file := cfg.get("predictions_output_file")) is None:
            raise ValueError("predictions_output_file must be specified if do_prediction=True")
        predictions_file = os.path.join(cfg.exp_manager.exp_dir, predictions_file)

    seed = cfg.model.seed
    np.random.seed(seed)
    pl.seed_everything(seed)

    trainer = setup_trainer(cfg, builder=InferenceTrainerBuilder() if not do_training else None)
    if cfg.restore_from_path:
        logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
        model = SpliceSiteBERTPredictionModel.restore_from(
            cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=BioNeMoSaveRestoreConnector()
        )
    else:
        model = SpliceSiteBERTPredictionModel(cfg.model, trainer)

    if do_training:
        trainer.fit(model)
        ckpt_path = "best"

    if do_prediction:
        if not do_training:
            model.data_setup()
            ckpt_path = None

        dataloader = model.predict_dataloader()
        predictions = trainer.predict(model, dataloaders=dataloader, ckpt_path=ckpt_path)
        dataset = model.predict_dataset
        predictions = reformat_predictions(predictions, dataset)
        pd.DataFrame(predictions).to_csv(predictions_file)
        logging.info(f"Predictions written to {predictions_file}!")


def reformat_predictions(predictions, dataset):
    predictions = torch.cat(predictions)
    pred_labels = torch.argmax(predictions, 1)

    # WARNING: this changes the behavior or `dataset` and is intended for use
    # only after the inference step has been completed. Set `do_transforms`
    # back to True if normal behavior is needed again.
    dataset.do_transforms = False
    predictions = [dict(**dataset[i], pred_label=pred_labels[i].item()) for i in range(len(dataset))]

    return predictions


if __name__ == "__main__":
    main()
