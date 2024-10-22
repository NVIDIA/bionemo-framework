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
from bionemo.data import FLIPPreprocess
from bionemo.data.preprocess.protein.preprocess import ESM2Preprocess, FastaPreprocess
from bionemo.model.protein.esm1nv import esm1nv_model
from bionemo.model.utils import setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector


@hydra_runner(config_path="conf", config_name="pretrain_esm2_8M")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    callbacks = setup_dwnstr_task_validation_callbacks(cfg)

    if cfg.do_preprocessing:
        logging.info("************** Starting Preprocessing ***********")
        use_default_esm_pretraining_strategy = cfg.model.data.train.custom_pretraining_fasta_path is None

        if use_default_esm_pretraining_strategy:
            # Pretraining on uniref90 sequences sampled from uniref50 clusters
            # ref: https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf
            preprocessor = ESM2Preprocess()

            # Prepare training dataset
            preprocessor.prepare_dataset(
                uf50_datapath=cfg.model.data.train.uf50_datapath,
                uf90_datapath=cfg.model.data.train.uf90_datapath,
                cluster_mapping_tsv=cfg.model.data.train.cluster_mapping_tsv,
                uf50_output_dir=cfg.model.data.train.dataset_path,
                uf90_output_dir=cfg.model.data.train.uf90.uniref90_path,
                sort_fastas=cfg.model.data.train.sort_fastas,
                mode="train",
                num_preprocess_workers=cfg.model.data.preprocessing.num_preprocess_workers,
            )

            # Prepare val dataset
            preprocessor.prepare_dataset(
                uf50_datapath=cfg.model.data.val.uf50_datapath,
                uf50_output_dir=cfg.model.data.val.dataset_path,
                sort_fastas=False,
                mode="val",
            )

            # Prepare test dataset
            preprocessor.prepare_dataset(
                uf50_datapath=cfg.model.data.test.uf50_datapath,
                uf50_output_dir=cfg.model.data.test.dataset_path,
                sort_fastas=False,
                mode="test",
            )
        else:
            preprocessor = FastaPreprocess(
                root_directory=cfg.model.data.dataset_path,
            )

            # Prepare training dataset
            preprocessor.prepare_dataset(
                fasta_path=cfg.model.data.train.custom_pretraining_fasta_path,
                mode="train",
            )

            # Prepare val dataset
            preprocessor.prepare_dataset(
                fasta_path=cfg.model.data.val.custom_pretraining_fasta_path,
                mode="val",
            )

            # Prepare test dataset
            preprocessor.prepare_dataset(
                fasta_path=cfg.model.data.test.custom_pretraining_fasta_path,
                mode="test",
            )

        # Downloading and preprocessing data for downstream task validation
        if cfg.model.dwnstr_task_validation.enabled:
            flip_preprocessor = FLIPPreprocess()
            if "task_name" not in cfg.model.dwnstr_task_validation.dataset:
                task_name = cfg.model.dwnstr_task_validation.dataset.dataset_path.split("/")[-1]
            else:
                task_name = cfg.model.dwnstr_task_validation.dataset.task_name
            flip_preprocessor.prepare_dataset(
                output_dir=cfg.model.dwnstr_task_validation.dataset.dataset_path, task_name=task_name
            )

    if cfg.do_training:
        trainer = setup_trainer(cfg, callbacks=callbacks)
        logging.info("************** Starting Training ***********")
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = esm1nv_model.ESM2nvModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=BioNeMoSaveRestoreConnector()
            )
        else:
            model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
        trainer.fit(model)
        logging.info("************** Finished Training ***********")


if __name__ == "__main__":
    main()
