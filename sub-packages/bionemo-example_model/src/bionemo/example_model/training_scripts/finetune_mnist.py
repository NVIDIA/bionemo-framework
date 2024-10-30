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


import argparse
from pathlib import Path

from nemo.collections import llm
from nemo.lightning import NeMoLogger, resume
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from bionemo.example_model.lighting.lightning_basic import (
    BionemoLightningModule,
    ExampleFineTuneConfig,
    checkpoint_callback,
    data_module,
    trainer,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_ckpt_dirpath", type=str, help="The checkpoint directory after pre-training")
    args = parser.parse_args()

    name = "example"
    directory_name = "sample_models"
    save_dir = Path(directory_name) / "classifier"

    nemo_logger2 = NeMoLogger(
        log_dir=str(save_dir),
        name=name,
        tensorboard=TensorBoardLogger(save_dir=save_dir, name=name),
        ckpt=checkpoint_callback,
        extra_loggers=[CSVLogger(save_dir / "logs", name=name)],
    )

    lightning_module2 = BionemoLightningModule(
        config=ExampleFineTuneConfig(
            initial_ckpt_path=args.pretrain_ckpt_dirpath,
            initial_ckpt_skip_keys_with_these_prefixes={"digit_classifier"},
        )
    )

    llm.train(
        model=lightning_module2,
        data=data_module,
        trainer=trainer,
        log=nemo_logger2,
        resume=resume.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        ),
    )
    finetune_dir = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    print(finetune_dir)
