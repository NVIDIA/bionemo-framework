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


from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.core.config.schedulers import SchedulerParams
from nemo.core.optim.lr_scheduler import register_scheduler
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.optim.lr_scheduler import LinearLR

from bionemo.data.preprocess.dna.preprocess import BasenjiDatasetPreprocessor
from bionemo.model.dna.enformer import Enformer
from bionemo.model.utils import setup_trainer


@dataclass
class LinearLRParams(SchedulerParams):
    # warning: all params must be explicitly overriden in the config
    # as registered params are not passed down
    start_factor: float = 0.000001
    end_factor: float = 1.0
    total_iters: int = 5000
    last_epoch: int = -1
    verbose: bool = True


@hydra_runner(config_path="conf", config_name="enformer_pretrain")
def main(cfg) -> None:
    register_scheduler(name="LinearLR", scheduler=LinearLR, scheduler_params=LinearLRParams)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing required keys in config:\n{missing_keys}")

    if cfg.do_preprocess:
        preprocessor = BasenjiDatasetPreprocessor(cfg.model.data)
        preprocessor.process()

    np.random.seed(cfg.model.seed)
    pl.seed_everything(cfg.model.seed)
    trainer = setup_trainer(cfg, callbacks=[])

    if cfg.restore_from_path:
        # since fine-tuning has different head configuration, let's override model config
        enformer = Enformer.restore_from(restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer)
        enformer.setup_training_data(cfg.model.train_ds)
        enformer.setup_validation_data(cfg.model.validation_ds)
    else:
        enformer = Enformer(cfg.model, trainer)
    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(enformer)
        logging.info("*************** Finish Training ************")

    if cfg.do_testing:
        enformer.setup_test_data(cfg.model.test_ds)
        trainer.test(enformer)


if __name__ == "__main__":
    main()
