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


from typing import List, Optional

import torch
from torch.cuda.amp import autocast

from bionemo.model.core.infer import BaseEncoderInference
from bionemo.model.dna.dnabert.dnabert_model import DNABERTModel


class DNABERTInference(BaseEncoderInference):
    def __init__(
        self,
        cfg,
        model: DNABERTModel | None = None,
        freeze: bool = True,
        restore_path: Optional[str] = None,
        training: bool = False,
        adjust_config: bool = True,
        interactive: bool = False,
        inference_batch_size_for_warmup: Optional[int] = None,
    ):
        super().__init__(
            cfg=cfg,
            model=model,
            freeze=freeze,
            restore_path=restore_path,
            training=training,
            adjust_config=adjust_config,
            interactive=interactive,
            inference_batch_size_for_warmup=inference_batch_size_for_warmup,
        )

    def get_example_input_sequence(self) -> str:
        return "AAAAAATATATATAAAAAA"

    def _tokenize(self, sequences: List[str]):
        # parent pulls the tokenizer from the loaded model.
        token_ids = [self.model.tokenizer.text_to_ids(s) for s in sequences]

        return token_ids

    def tokenize(self, sequences: List[str]):
        """
        Note that the parent includes padding, likely not necessary for DNABERT.
        This actually fails if you let it call the parent super class, since it expects padding to be a thing.
        """
        return self._tokenize(sequences)

    def seq_to_hiddens(self, sequences: List[str]):
        token_ids = torch.tensor(self.tokenize(sequences), device=self.model.device)
        padding_mask = torch.ones(size=token_ids.size(), device=self.model.device)

        with autocast(enabled=True):
            output_tensor = self.model(token_ids, padding_mask, token_type_ids=None, lm_labels=None)

        # Padding mask gets used for automatically adjusting the length of the sequence with respect to padding tokens.
        #       DNABERT does not have a padding token, so this is redundant.
        return output_tensor, padding_mask

    def load_model(self, cfg, model=None, restore_path=None, strict: bool = True):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            ESM trained model
        """
        # control post-processing, load_model messes with our config so we need to bring it in here.
        model = super().load_model(cfg, model=model, restore_path=restore_path, strict=strict)
        # Hardcoded for DNABERT as there is no post-processing
        model.model.post_process = False
        return model
