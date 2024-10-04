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
from typing import Tuple

import pytorch_lightning as pl
from nemo import lightning as nl

from bionemo.esm2.api import ESM2GenericConfig
from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.finetune_regressor import ESM2FineTuneSeqConfig, InMemorySingleValueDataset
from bionemo.llm.model.biobert.lightning import BioBertLightningModule


def infer_model(
    config: ESM2GenericConfig,
    data_module: pl.LightningDataModule,
    tokenizer: BioNeMoESMTokenizer = get_tokenizer(),
) -> Tuple[Path, pl.Callback, nl.Trainer]:
    """Infers a BioNeMo ESM2 model using PyTorch Lightning.

    Parameters:
        config (ESM2GenericConfig): The configuration for the ESM2 model.
        data_module (pl.LightningDataModule): The data module for training and validation.
        tokenizer (BioNeMoESMTokenizer, optional): The tokenizer to use. Defaults to `get_tokenizer()`.

    Returns:
        List[torch.Tensor]: A list of tensors containing the predictions of predict_dataset in datamodule
    """
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, ddp="megatron", find_unused_parameters=True
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        num_nodes=1,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )
    module = BioBertLightningModule(config=config, tokenizer=tokenizer)
    results = trainer.predict(module, datamodule=data_module)

    return results


if __name__ == "__main__":
    # create a List[Tuple] with (sequence, target) values
    artificial_sequence_data = [
        "TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
        "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN",
        "DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
        "LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
        "LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
        "SGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKT",
    ]
    data = [(seq, len(seq) / 100.0) for seq in artificial_sequence_data]

    dataset = InMemorySingleValueDataset(data)

    # NOTE: Due to the current limitation in inference of NeMo lightning module, partial batches with
    # size < global_batch_size are not being processed with predict_step(). Therefore we set the global to len(data)
    # and choose the micro_batch_size so that global batch size is divisible by micro batch size x data parallel size
    data_module = ESM2FineTuneDataModule(
        predict_dataset=dataset, global_batch_size=len(data), micro_batch_size=len(data)
    )

    config = ESM2FineTuneSeqConfig(
        # initial_ckpt_path = finetuned_checkpoint,  # supply the finetuned checkpoint path
        # initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=list)   # reset to avoid skipping the head params
    )

    results = infer_model(config, data_module)
    print(results)
