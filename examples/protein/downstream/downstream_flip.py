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


from nemo.collections.nlp.parts.nlp_overrides import (
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict

from bionemo.data import FLIPPreprocess
from bionemo.data.metrics import accuracy, mse, per_token_accuracy
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.utils import (
    setup_trainer,
)


@hydra_runner(config_path="../esm1nv/conf", config_name="downstream_flip_sec_str")  # ESM1
def main(cfg) -> None:
    logging.info("\n\n************* Finetune config ****************")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Do preprocessing if preprocess
    if cfg.do_preprocessing:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = FLIPPreprocess()
        preprocessor.prepare_all_datasets(output_dir=cfg.model.data.preprocessed_data_path)

    if cfg.do_training is False and cfg.do_testing is False:  # finish run without model instantiation
        return

    trainer = setup_trainer(cfg, builder=None, reset_accumulate_grad_batches=False)

    # Load model
    with open_dict(cfg):
        cfg.model.encoder_cfg = cfg

    if cfg.restore_from_path:
        logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
        model = FineTuneProteinModel.restore_from(
            cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )
    else:
        model = FineTuneProteinModel(cfg.model, trainer)

    metrics = {}
    metrics_args = {}
    for idx, name in enumerate(cfg.model.data.target_column):
        if cfg.model.data.task_type == "token-level-classification":
            metrics[name + "_accuracy"] = per_token_accuracy
            metrics_args[name + "_accuracy"] = {"label_id": idx}
        elif cfg.model.data.task_type == "classification":
            metrics[name + "_accuracy"] = accuracy
            metrics_args[name + "_accuracy"] = {}
        elif cfg.model.data.task_type == "regression":
            metrics[name + "_MSE"] = mse
            metrics_args[name + "_MSE"] = {}

    model.add_metrics(metrics=metrics, metrics_args=metrics_args)

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("************** Finished Training ***********")

    if cfg.do_testing:
        logging.info("************** Starting Testing ***********")
        if "test" in cfg.model.data.dataset:
            trainer.limit_train_batches = 0
            trainer.limit_val_batches = 0
            trainer.fit(model)
            trainer.test(model, ckpt_path=None)
        else:
            raise UserWarning(
                "Skipping testing, test dataset file was not provided. Please specify 'dataset.test' in yaml config"
            )
        logging.info("************** Finished Testing ***********")


if __name__ == "__main__":
    main()
