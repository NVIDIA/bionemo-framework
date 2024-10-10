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


from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.callbacks import setup_dwnstr_task_validation_callbacks
from bionemo.data import Zinc15Preprocess
from bionemo.model.molecule.megamolbart import MegaMolBARTModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector


@hydra_runner(config_path="conf", config_name="pretrain_xsmall_span_aug")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    callbacks = setup_dwnstr_task_validation_callbacks(cfg)

    trainer = setup_trainer(cfg, callbacks=callbacks)

    logging.info("************** Model parameters and their sizes ***********")

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = MegaMolBARTModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=BioNeMoSaveRestoreConnector()
            )
        else:
            model = MegaMolBARTModel(cfg.model, trainer)

        trainer.fit(model)

        for name, param in model.named_parameters():
            logging.info(f"{name}: {param.size()}")
            logging.info("***********************************************************")

    else:
        logging.info("************** Starting Data PreProcessing ***********")
        preproc = Zinc15Preprocess(root_directory=cfg.model.data.dataset_path)
        preproc.prepare_dataset(
            links_file=cfg.model.data.links_file,
            max_smiles_length=cfg.model.seq_length,
            output_dir=cfg.model.data.dataset_path,
        )

    if cfg.do_testing:
        logging.info("************** Starting Testing ***********")
        trainer.test(model)


if __name__ == "__main__":
    main()
