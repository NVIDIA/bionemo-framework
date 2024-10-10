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


import logging
from typing import Any, List, Optional, Sequence, Tuple

import torch

from bionemo.model.core.infer import BaseEncoderDecoderInference, SeqsOrBatch


log = logging.getLogger(__name__)

__all__: Sequence[str] = ("MolInference",)


class MolInference(BaseEncoderDecoderInference):
    """
    All inference functions
    """

    def __init__(
        self,
        cfg,
        model: Optional[Any] = None,
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
        return "c1cc(C(O)=O)ccc1"  # Benzoic Acid

    def _tokenize(self, sequences: List[str]) -> List[int]:
        """
        Expected expects input/output format:

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
            hiddens (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        """
        token_ids, enc_mask = self.tokenize(sequences)
        hiddens = self.model.encode(
            tokens_enc=token_ids,
            enc_mask=enc_mask,
            reconfigure_microbatch=not self.interactive,
        )
        if self.model.cfg.encoder.arch == "perceiver":
            # Perceiver outputs a fixed number of embedding tokens (K) regardless of input position.
            #  thus the enc_mask should change to reflect this shape, and the fact that none of the K tokens
            #  should be masked.
            k: int = self.model.cfg.encoder.hidden_steps
            assert hiddens.shape[1] == k
            enc_mask = torch.ones(size=hiddens.shape[:2], dtype=enc_mask.dtype, device=enc_mask.device)

        if hiddens.shape[:2] != enc_mask.shape:
            raise ValueError(
                f"Not using perceiver but got mismatched shapes out of hiddens and enc_mask, please check! {hiddens.shape} vs {enc_mask.shape}"
            )
        return hiddens, enc_mask

    def hiddens_to_seq(
        self,
        hiddens: torch.Tensor,
        enc_mask: torch.Tensor,
        override_generate_num_tokens: Optional[int] = None,
        **kwargs,
    ) -> SeqsOrBatch:
        """
        Transforms hidden state into sequences (i.e., sampling in most cases).
        This class should be implemented in a child class, since it is model specific.
        This class should return the sequence with special tokens such as
         <BOS> and <EOS> tokens, if used.

        Args:
            hiddens (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
            override_generate_num_tokens (Optional[int]): If not provided, the model.max_position_embeddings is used for the number of tokens to generate.
                If provided, we override the saved model's values (may be useful for speed if you know you only need shorter sequences.)
        Returns:
            sequences (list[str]) or list[list[str]]): list of sequences
        """
        if enc_mask is not None:
            assert (
                enc_mask.shape == hiddens.shape[:2]
            ), f"Incompatible shapes between hiddens and mask: {hiddens.shape} vs {enc_mask.shape}"

        if override_generate_num_tokens is not None:
            num_tokens_to_generate = override_generate_num_tokens
        else:
            num_tokens_to_generate = self.model.cfg.max_position_embeddings

        predicted_tokens_ids, _ = self.model.decode(
            tokens_enc=None,
            enc_mask=enc_mask,
            num_tokens_to_generate=num_tokens_to_generate,
            enc_output=hiddens,
            reconfigure_microbatch=not self.interactive,
            **kwargs,
        )
        sequences = self.detokenize(tokens_ids=predicted_tokens_ids)
        return sequences
