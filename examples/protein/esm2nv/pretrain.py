# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from bionemo.data import FLIPPreprocess
from bionemo.data.preprocess.protein.preprocess import ESM2Preprocess
from bionemo.model.protein.esm1nv import esm1nv_model
from bionemo.model.utils import setup_trainer
from bionemo.utils import BioNeMoSaveRestoreConnector
from bionemo.utils.callbacks.callback_utils import setup_callbacks


@hydra_runner(config_path="../../../examples/protein/esm2nv/conf", config_name="pretrain_esm2_8M")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    callbacks = setup_callbacks(cfg)

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
    else:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = ESM2Preprocess()
        # NOTE: Having issues? check that all the lock files are removed in the directories below.
        preprocessor.prepare_dataset(
            uf50_datapath=cfg.model.data.uf50_datapath,
            uf90_datapath=cfg.model.data.uf90_datapath,
            cluster_mapping_tsv=cfg.model.data.cluster_mapping_tsv,
            uf50_output_dir=cfg.model.data.dataset_path,
            uf90_output_dir=cfg.model.data.uf90.uniref90_path,
            val_size=cfg.model.data.val_size,
            test_size=cfg.model.data.test_size,
            sort_fastas=cfg.model.data.sort_fastas,
        )

        if cfg.model.dwnstr_task_validation.enabled:
            flip_preprocessor = FLIPPreprocess()
            if "task_name" not in cfg.model.dwnstr_task_validation.dataset:
                task_name = cfg.model.dwnstr_task_validation.dataset.dataset_path.split("/")[-1]
            else:
                task_name = cfg.model.dwnstr_task_validation.dataset.task_name
            flip_preprocessor.prepare_dataset(
                output_dir=cfg.model.dwnstr_task_validation.dataset.dataset_path, task_name=task_name
            )


if __name__ == '__main__':
    main()
