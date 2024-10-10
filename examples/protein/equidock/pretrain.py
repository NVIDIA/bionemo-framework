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


#!/bin/bash

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
Entry point to EquiDock.

modify parameters from conf/*.yaml

"""

import os

import torch
from nemo.core.config import hydra_runner
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.multiprocessing import set_start_method

from bionemo.data.equidock import DataManager
from bionemo.data.equidock.preprocess import preprocess
from bionemo.model.protein.equidock.equidock_model import EquiDock
from bionemo.model.utils import setup_trainer


os.environ["DGLBACKEND"] = "pytorch"

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


@hydra_runner(config_path="conf", config_name="pretrain")  # config_name options [finetune, pretrain]
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.seed)

    logging.info(f"\n Dataset name {cfg.data.data_name}")
    train_cfg = cfg.model.train_ds
    test_cfg = cfg.model.test_ds
    val_cfg = cfg.model.validation_ds

    if not cfg.do_training:
        # preprocessing the original raw data in EquiDock repo
        logging.info("\n Preprocessing validation dataset!")
        logging.info(f"\n{OmegaConf.to_yaml(val_cfg)}")
        preprocess(val_cfg)

        logging.info("\n Preprocessing testing dataset!")
        logging.info(f"\n{OmegaConf.to_yaml(test_cfg)}")
        preprocess(test_cfg)

        logging.info("\n Preprocessing training dataset!")
        logging.info(f"\n{OmegaConf.to_yaml(train_cfg)}")
        preprocess(train_cfg)

    else:
        if cfg.data.num_workers != 0:
            try:
                set_start_method("spawn", force=True)
            except RuntimeError:
                logging.info("Torch multiprocessing failed with spawn, set num_workers=0!")
                raise RuntimeError(
                    f"Torch multiprocessing failed with spawn, set num_workers=0, but it is {cfg.data.num_workers}"
                )

        data_manager = DataManager(cfg)
        cfg.model.input_edge_feats_dim = data_manager.train_ds[0][0].edata["he"].shape[1]
        logging.info(f"\n Dataset name {cfg.data.data_name} loaded!")

        trainer = setup_trainer(cfg)
        model = EquiDock(cfg=cfg, trainer=trainer, data_manager=data_manager)

        if cfg.model.get("restore_from_path", None) is not None:
            logging.info(f"\n\n************** Restoring model from {cfg.model.restore_from_path} ***********")

            model = model.restore_from(
                restore_path=cfg.model.restore_from_path,
                trainer=trainer,
                override_config_path=cfg,
                save_restore_connector=SaveRestoreConnector(),  # SaveRestoreConnector
            )
            # restore_from overriedes datasets and dataloaders
            model.reload_nemo_model(cfg, trainer, data_manager)

        logging.info(f"***** Training dataset {cfg.data.data_name} ***** ")
        logging.info(f"***** Number of model parameters: {model.num_weights} ***** ")
        trainer.fit(model=model)

        if cfg.do_testing:
            logging.info(f"***** Testing dataset {cfg.data.data_name} ***** ")
            trainer.test(model, verbose=True)


if __name__ == "__main__":
    main()
