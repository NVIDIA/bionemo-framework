# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from bionemo.data import PhysChemPreprocess
from bionemo.model.molecule.megamolbart import FineTuneMegaMolBART
from bionemo.model.utils import (
    setup_trainer,
)


@hydra_runner(config_path="conf", config_name="finetune_config")
def main(cfg) -> None:
    logging.info("\n\n************* Fintune config ****************")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if cfg.do_training or cfg.do_testing:
        trainer = setup_trainer(cfg, builder=None)
        model = FineTuneMegaMolBART(cfg, trainer)

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("************** Finished Training ***********")
        if cfg.do_testing:
            if "test" in cfg.model.data.dataset:
                trainer.test(model)
            else:
                raise UserWarning(
                    "Skipping testing, test dataset file was not provided. Please specify 'test_ds.data_file' in yaml config"
                )
    else:
        logging.info("************** Starting Data PreProcessing ***********")
        PhysChemPreprocess().prepare_dataset(
            links_file=cfg.model.data.links_file, output_dir=cfg.model.data.preprocessed_data_path
        )

        if cfg.model.data.split_data:
            PhysChemPreprocess()._process_split(
                links_file=cfg.model.data.links_file,
                output_dir=cfg.model.data.preprocessed_data_path,
                test_frac=cfg.model.data.test_frac,
                val_frac=cfg.model.data.val_frac,
            )
        logging.info("************** Finished Data PreProcessing ***********")


if __name__ == '__main__':
    main()
