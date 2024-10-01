# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.model.singlecell.geneformer.model import GeneformerModel
from bionemo.model.utils import create_geneformer_preprocessor, setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector


@hydra_runner(config_path="conf", config_name="geneformer_config")
def main(cfg) -> None:
    """
    Main function for pretraining the Geneformer model.

    Args:
        cfg (OmegaConf): Configuration object containing the experiment settings.

    Returns:
        None
    """
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    if cfg.get("seed_everything", True):
        pl.seed_everything(cfg.model.seed)
    if cfg.do_training:
        trainer = setup_trainer(cfg)
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = GeneformerModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=BioNeMoSaveRestoreConnector()
            )
        else:
            model = GeneformerModel(cfg.model, trainer)

        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("*************** Finish Training ************")
    else:
        logging.info("************** Starting Preprocessing ***********")
        # Path that the medians file gets saved to. Note that internally the tokenizer also controls saving its vocab based on a location in the config
        preprocessor = create_geneformer_preprocessor(
            cfg.model.data.compute_medians,
            cfg.model.data.train_dataset_path,
            cfg.model.data.medians_file,
            cfg.model.tokenizer.vocab_file,
        )
        match preprocessor.preprocess():
            case {"tokenizer": _, "median_dict": _}:
                logging.info("*************** Preprocessing Finished ************")
            case _:
                logging.error("Preprocessing failed.")


if __name__ == "__main__":
    main()
