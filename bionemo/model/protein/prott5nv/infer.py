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


from typing import List, Optional, Tuple

import torch

from bionemo.model.core.infer import BaseEncoderDecoderInference, BaseEncoderInference


class ProtT5nvInference(BaseEncoderDecoderInference):
    """
    All inference functions
    """

    def __init__(
        self,
        cfg,
        model=None,
        freeze=True,
        restore_path=None,
        training=False,
        adjust_config=True,
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

    def warmup(self, max_bs: int):
        # FIXME skip the decoder warmup for ProtT5nv because no one implemented the required hiddens_to_seq!
        #  copy/paste from the molecule implementation of hiddens_to_seq didn't work.
        return BaseEncoderInference.warmup(self, max_bs)

    def get_example_input_sequence(self) -> str:
        return "DAEFRHDSGYEVHHQKLVFF"

    def _tokenize(self, sequences: List[str]) -> List[str]:
        """
        ProtT5 expects input/output format:

        encoder input ids - [tokens] (without <BOS> and <EOS>)
        decoder input ids - <BOS> + [tokens]
        decoder output ids - [tokens] + <EOS>
        """
        # Tokenize sequences
        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]

        return token_ids

    def seq_to_hiddens(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms Sequences into hidden state.
        Should be implemented in a child class, since it is model specific.
        This method returns hidden states and masks.
        Hiddens states contain paddings but do not contain special tokens
        such as <BOS> and <EOS> tokens.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        """
        token_ids, enc_mask = self.tokenize(sequences)
        embedding = self.model.encode(
            tokens_enc=token_ids, enc_mask=enc_mask, reconfigure_microbatch=not self.interactive
        )

        return embedding, enc_mask
