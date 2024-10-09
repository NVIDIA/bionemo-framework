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


from typing import List, Optional, Sequence, Tuple

import torch

from bionemo.model.core.infer import BaseEncoderInference


__all__: Sequence[str] = ("ESM1nvInference",)


class ESM1nvInference(BaseEncoderInference):
    """
    All inference functions
    """

    def __init__(
        self,
        cfg,
        model=None,
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
            needs_warmup=True,  # @jstjohn verified that this is currently required 5/7/2024. Try removing it on upgrade if you can.
        )

    def get_example_input_sequence(self) -> str:
        return "DAEFRHDSGYEVHHQKLVFF"

    def _tokenize(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        ESM expects input format:

        encoder input ids - <BOS> + [tokens] + <EOS>
        """
        # Tokenize sequences and add <BOS> and <EOS> tokens
        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]
        token_ids = [torch.tensor([self.tokenizer.bos_id] + s + [self.tokenizer.eos_id]).cuda() for s in token_ids]

        return token_ids

    def seq_to_hiddens(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
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
        """
        token_ids, enc_mask = self.tokenize(sequences)
        hidden_states = self.model.encode(token_ids, enc_mask, reconfigure_microbatch=not self.interactive)

        # ignore <BOS> and <EOS> tokens
        enc_mask[:, 0:2] = 0
        enc_mask = torch.roll(enc_mask, shifts=-1, dims=1)

        return hidden_states, enc_mask

    def load_model(self, cfg, model=None, restore_path=None, strict: bool = True):
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
        model = super().load_model(cfg, model=model, restore_path=restore_path, strict=strict)

        model.model.post_process = post_process

        return model
