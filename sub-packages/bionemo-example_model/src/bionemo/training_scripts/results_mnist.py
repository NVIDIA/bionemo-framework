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

from nemo import lightning as nl

from bionemo.core import BIONEMO_CACHE_DIR
from bionemo.example_model.lightning_basics import (
    BionemoLightningModule,
    ExampleFineTuneConfig,
    MNISTDataModule,
)
from bionemo.example_model.pretrain_mnist import data_module, strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_dir", type=str, help="The directory with the fine-tuned model. ")
    args = parser.parse_args()
    test_run_trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        num_nodes=1,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=None,
    )

    lightning_module3 = BionemoLightningModule(config=ExampleFineTuneConfig(initial_ckpt_path=args.finetune_dir))

    new_data_module = MNISTDataModule(
        data_dir=str(BIONEMO_CACHE_DIR), batch_size=len(data_module.mnist_test), output_log=False
    )

    results = test_run_trainer.predict(lightning_module3, datamodule=new_data_module)
    print(results)
