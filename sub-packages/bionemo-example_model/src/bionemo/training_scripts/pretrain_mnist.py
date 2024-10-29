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


import tempfile

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger, resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from bionemo.core import BIONEMO_CACHE_DIR
from bionemo.example_model.lightning_basics import (
    BionemoLightningModule,
    MetricTracker,
    MNISTDataModule,
    PretrainConfig,
)


checkpoint_callback = nl_callbacks.ModelCheckpoint(
    save_last=True,
    save_on_train_epoch_end=True,
    monitor="reduced_train_loss",
    every_n_train_steps=25,
    always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
)

# Set up the data module
data_module = MNISTDataModule(data_dir=str(BIONEMO_CACHE_DIR), batch_size=128)

strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    ddp="megatron",
    find_unused_parameters=True,
    always_save_context=True,
)
metric_tracker = MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"])

trainer = nl.Trainer(
    accelerator="gpu",
    devices=1,
    strategy=strategy,
    limit_val_batches=5,
    val_check_interval=25,
    max_steps=500,
    max_epochs=10,
    num_nodes=1,
    log_every_n_steps=25,
    callbacks=[metric_tracker],
    plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
)

if __name__ == "__main__":
    temp_dir = tempfile.TemporaryDirectory()
    save_dir = temp_dir / "pretrain"
    name = "example"
    # Setup the logger train the model
    nemo_logger = NeMoLogger(
        log_dir=str(save_dir),
        name=name,
        tensorboard=TensorBoardLogger(save_dir=save_dir, name=name),
        ckpt=checkpoint_callback,
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

pretrain_ckpt_dirpath = checkpoint_callback.last_model_path.replace(".ckpt", "")
print(metric_tracker.collection_train["loss"])
print(metric_tracker.collection_val["logged_metrics"])
print(pretrain_ckpt_dirpath)
