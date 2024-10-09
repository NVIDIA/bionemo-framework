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


import numpy as np
import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.data.preprocess.dna.preprocess import DNABERTPreprocess
from bionemo.model.dna.dnabert import DNABERTModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector


@hydra_runner(config_path="conf", config_name="dnabert_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    np.random.seed(cfg.model.seed)
    pl.seed_everything(cfg.model.seed)

    if cfg.do_training:
        trainer = setup_trainer(cfg)
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = DNABERTModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=BioNeMoSaveRestoreConnector()
            )
        else:
            model = DNABERTModel(cfg.model, trainer)
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("*************** Finish Training ************")
    else:
        logging.info("************** Starting Preprocessing ***********")
        logging.warning(
            "For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose."
        )

        preprocessor = DNABERTPreprocess(
            cfg.model.data.dataset_path,
            cfg.model.tokenizer.model,
            cfg.model.tokenizer.vocab_file,
            cfg.model.tokenizer.k,
            cfg.model.data.dataset,
        )
        preprocessor.preprocess()

        logging.info("*************** Finish Preprocessing ************")


if __name__ == "__main__":
    main()
