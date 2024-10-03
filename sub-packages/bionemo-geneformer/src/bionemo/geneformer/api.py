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

import json
import tarfile
from dataclasses import dataclass
from typing import Sequence, Type

from bionemo.geneformer.model.finetune_token_regressor import FineTuneSeqLenBioBertConfig
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.model.biobert.connector import GenericBioBertNeMo1LightningModuleConnector
from bionemo.llm.model.biobert.model import BioBertGenericConfig, MegatronBioBertModel
from bionemo.llm.utils import iomixin_utils as iom


__all__: Sequence[str] = (
    "GeneformerModel",
    "GeneformerConfig",
    "FineTuneSeqLenBioBertConfig",
)

GeneformerModel = MegatronBioBertModel


@dataclass
class GeneformerConfig(BioBertGenericConfig[GeneformerModel], iom.IOMixinWithGettersSetters):
    """A geneformer config.

    The geneformer config overrides the parent config, and adds a leaf-level iomixin, please do not inherit from this
    directly, as your parameters will likely be reset to this method's parameters silently.
    """

    model_cls: Type[GeneformerModel] = GeneformerModel


class GeneformerNeMo1LightningModuleConnector(GenericBioBertNeMo1LightningModuleConnector[GeneformerModel]):
    @property
    def tokenizer(self):
        nemo1_settings = self.get_nemo1_config()
        fmt_vocab, vocab_tar_path = nemo1_settings["tokenizer"]["vocab_file"].split(":")
        assert fmt_vocab == "nemo"
        # TODO add another function to pull out the medians file from a nemo1 checkpoint, if the user wants it.
        #  It's not needed for checkpoint conversion though.
        # fmt_medians, medians_tar_path = nemo1_settings["data"]["medians_file"].split(":")
        # assert fmt_vocab == fmt_medians and fmt_vocab == "nemo"
        nemo1_path = str(self)
        with tarfile.open(nemo1_path, "r") as old_ckpt:
            vocab_gene_ens_dict = json.loads(old_ckpt.extractfile(f"./{vocab_tar_path}").readlines()[0])
        tokenizer = GeneTokenizer(**vocab_gene_ens_dict)
        return tokenizer

    def get_config_class(self) -> GeneformerConfig:
        return GeneformerConfig
