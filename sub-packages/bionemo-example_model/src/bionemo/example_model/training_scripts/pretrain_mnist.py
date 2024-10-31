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


from pathlib import Path

from nemo.collections import llm
from nemo.lightning import NeMoLogger, resume
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from bionemo.example_model.lightning.lightning_basic import (
    BionemoLightningModule,
    PretrainConfig,
    checkpoint_callback,
    data_module,
    trainer,
)


def run_pretrain(name: str, directory_name: str):
    """Run the pretraining step.

    Args:
        name: The experiment name.
        directory_name: The directory to write the output
    Returns:
        str: the path of the trained model.
    """
    # Setup the logger train the model
    nemo_logger = NeMoLogger(
        log_dir=str(save_dir),
        name=name,
        tensorboard=TensorBoardLogger(save_dir=directory_name, name=name),
        ckpt=checkpoint_callback,
        extra_loggers=[CSVLogger(save_dir / "logs", name=name)],
    )

    # Set up the training module
    lightning_module = BionemoLightningModule(config=PretrainConfig())

    # This trains the model
    llm.train(
        model=lightning_module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )


if __name__ == "__main__":
    directory_name = "sample_models"
    save_dir = Path(directory_name) / "pretrain"
    name = "example"
    pretrain_ckpt_dirpath = run_pretrain(name, directory_name)

    print(pretrain_ckpt_dirpath)
