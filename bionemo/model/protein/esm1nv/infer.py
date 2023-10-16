# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from typing import List

import torch
from torch.cuda.amp import autocast

from bionemo.model.core.infer import BaseEncoderDecoderInference


class ESM1nvInference(BaseEncoderDecoderInference):
    '''
    All inference functions
    '''

    def __init__(self, cfg, model=None, freeze=True, restore_path=None, training=False, adjust_config=True):
        super().__init__(
            cfg=cfg,
            model=model,
            freeze=freeze,
            restore_path=restore_path,
            training=training,
            adjust_config=adjust_config,
        )

    def _tokenize(self, sequences: List[str]):
        """
        ESM expects input format:

        encoder input ids - <BOS> + [tokens] + <EOS>
        """
        # Tokenize sequences and add <BOS> and <EOS> tokens
        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]
        token_ids = [torch.tensor([self.tokenizer.bos_id] + s + [self.tokenizer.eos_id]).cuda() for s in token_ids]

        return token_ids

    def seq_to_hiddens(self, sequences):
        '''
        Transforms Sequences into hidden state.
        Should be implemented in a child class, since it is model specific.
        This method returns hidden states and masks.
        Hidden states are returned for all tokens, including <BOS>, <EOS> and padding.
        <BOS>, <EOS> and padding are masked out.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for special tokens (<BOS> and <EOS>) and padded sections
        '''
        token_ids, enc_mask = self.tokenize(sequences)
        # FIXME this autocast shouldn't be needed
        with autocast(enabled=self.model.enable_autocast):
            hidden_states = self.model(token_ids, enc_mask, None)

        # ignore <BOS> and <EOS> tokens
        enc_mask[:, 0:2] = 0
        enc_mask = torch.roll(enc_mask, shifts=-1, dims=1)

        # Want to check actual value in the model and not in the config
        if self.model.model.post_process:
            hidden_states = hidden_states[0]

        return hidden_states, enc_mask

    def load_model(self, cfg, model=None, restore_path=None):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            ESM trained model
        """
        # control post-processing
        if model is None:
            post_process = cfg.model.post_process
        else:
            post_process = model.model.post_process
        model = super().load_model(cfg, model=model, restore_path=restore_path)

        model.model.post_process = post_process

        return model
