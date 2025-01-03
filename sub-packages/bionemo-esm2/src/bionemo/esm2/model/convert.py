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

import torch
from nemo.lightning import io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf

from bionemo.esm2.model.model import ESM2Config
from bionemo.llm.lightning import BionemoLightningModule
from bionemo.llm.model.biobert.lightning import biobert_lightning_module


@io.model_importer(BionemoLightningModule, "hf")
class HFESM2Importer(io.ModelConnector["AutoModelForMaskedLM", BionemoLightningModule]):
    def init(self) -> BionemoLightningModule:
        return biobert_lightning_module(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import AutoModelForMaskedLM

        source = AutoModelForMaskedLM.from_pretrained(str(self), trust_remote_code=True, torch_dtype="auto")
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted ESM-2 model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target): ...

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)), trust_remote_code=True)

    @property
    def config(self) -> ESM2Config:
        from transformers import AutoConfig as HFAutoConfig

        source = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        output = ESM2Config(
            num_layers=source.num_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            seq_length=source.seq_length,
            num_query_groups=source.multi_query_group_num,
            make_vocab_size_divisible_by=source.padded_vocab_size,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output
