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
Entry point to DiffDock: training of score/confidence model.

modify parameters from conf/*.yaml
"""

import os
import warnings

from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from rdkit import RDLogger

from bionemo.data.diffdock.data_manager import DataManager
from bionemo.data.diffdock.embedding_preprocess import prep_embedding
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModel as CGScoreModel,
)
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModelAllAtom as AAScoreModel,
)
from bionemo.model.utils import setup_trainer


warnings.filterwarnings("ignore", module="torch_geometric.data.collate")


@hydra_runner(config_path="conf", config_name="train_score")
def main(cfg) -> None:
    """
    This is the main function conducting data preprocessing, model training
    for both score model and confidence model, and final model inference.
    """
    logging.info("\n\n************** Experiment Configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    if not cfg.model.confidence_mode:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    if cfg.get("seed", None) is not None:
        seed_everything(cfg.seed)

    if cfg.do_embedding_preprocessing:
        logging.info("************** Starting Embedding PreProcessing ***********")
        prep_embedding(cfg.protein_data)

    data_manager = DataManager(cfg)
    if cfg.do_preprocessing:
        logging.info("************** Starting Complex Graph PreProcessing ***********")
        data_manager.preprocess()

    if cfg.do_training and data_manager.datasets_ready:
        trainer = setup_trainer(cfg)

        if "all_atoms" in cfg.data and cfg.data.all_atoms:
            model = AAScoreModel(cfg=cfg, trainer=trainer, data_manager=data_manager)
        else:
            model = CGScoreModel(cfg=cfg, trainer=trainer, data_manager=data_manager)

        if cfg.model.confidence_mode:
            logging.info(">>> Configured Confidence Model ...")
        else:
            logging.info(">>> Configured Score Model ...")

        logging.info("************** Starting Training ***********")
        trainer.fit(model)


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    main()
