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


import math
import os
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
import torch.distributed
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.bert.bert_lm_head import BertLMHead
from megatron.core.models.bert.pooler import Pooler
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer import spec_utils
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning import get_vocab_size
from torch import Tensor
from torch.optim import Optimizer

from bionemo.contrib.model.biobert.model import BioBertConfig, MegatronBioBertModel
from bionemo.contrib.model.biobert.transformer_specs import BiobertSpecOption, get_biobert_spec
from bionemo.contrib.model.esm2.embedding import ESM2Embedding


class ESM2Model(MegatronBioBertModel):
    """ESM2 Transformer language model.

    Args:
        config (TransformerConfig): transformer config
        num_tokentypes (int): Set to 2 when args.bert_binary_head is True, and 0 otherwise. Defaults to 0.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        tokenizer (AutoTokenizer): optional tokenizer object (currently only used in the constructor of ESM2Model)
        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include an output layer (used with pipeline parallelism)
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks
        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
            Defaults is 'learned_absolute'.
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_tokentypes: int,
        transformer_layer_spec: spec_utils.ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        tokenizer: AutoTokenizer = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        add_binary_head=True,
        return_embeddings=False,
        use_full_attention_mask=False,
    ):
        super(MegatronBioBertModel, self).__init__(config=config)
        self.post_process = post_process
        self.add_binary_head = add_binary_head
        if return_embeddings:
            assert self.post_process and self.add_binary_head
        # `b` = batch, `s` = sequence.
        # The old flash attention mechanism apparently wants you to use a b x 1 x s x s attention mask while
        #  the new one wants a b x 1 x 1 x s attention mask. This is a hack to allow us to switch between the two.
        self.use_full_attention_mask = use_full_attention_mask
        self.config: TransformerConfig = config
        self.transformer_layer_spec: spec_utils.ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.add_binary_head = add_binary_head
        self.return_embeddings = return_embeddings

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # Embeddings.
        if self.pre_process:
            # ESM2 Customization: ESM2Embedding instead of LanguageModelEmbedding
            self.embedding = ESM2Embedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                num_tokentypes=num_tokentypes,
                # ESM2 NEW ARGS
                token_dropout=self.config.token_dropout,
                use_attention_mask=self.config.use_attention_mask,
                mask_token_id=tokenizer.mask_id,
            )

        if self.position_embedding_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
            )

        # Transformer.
        self.encoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            # TODO: Make sure you are passing in the mpu_vocab_size properly
            self.lm_head = BertLMHead(
                config.hidden_size,
                config,
            )

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
            )

            self.binary_head = None
            if self.add_binary_head:
                # TODO: Shoudl switch this to TE ?
                self.binary_head = get_linear_layer(
                    config.hidden_size, 2, config.init_method, config.perform_initialization
                )

                self.pooler = Pooler(config.hidden_size, config.init_method, config, config.sequence_parallel)
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def embedding_forward(
        self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: Tensor = None, attention_mask: Tensor = None
    ):
        # ESM2 Customization: ESM2Embedding forward takes attention_mask
        # in addition to the args required by LanguageModelEmbedding
        return self.embedding(
            input_ids=input_ids, position_ids=position_ids, tokentype_ids=tokentype_ids, attention_mask=attention_mask
        )


def esm_gelu_func(x):
    """
    ESM2-specific gelu implementation from the original ESM repo.
    Using F.gelu yields subtly wrong results.

    Args:
        x (Tensor): input tensor of any given dimension
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@dataclass
class ESM2Config(BioBertConfig):
    num_layers: int = 33  # 650M
    hidden_size: int = 1280  # 650M
    num_attention_heads: int = 20
    ffn_hidden_size: int = 4 * 1280  # Transformer FFN hidden size. Usually 4 * hidden_size.
    hidden_dropout: int = 0  # ESM2 removes dropout from hidden layers and attention
    attention_dropout: float = 0.0  # ESM2 does not use attention dropout
    apply_residual_connection_post_layernorm: bool = False  # TODO: farhadr False is new default, True was BERT pub.
    layernorm_epsilon: float = 1.0e-5
    layernorm_zero_centered_gamma: bool = False  # TODO(@jstjohn) check this
    activation_func: Callable = esm_gelu_func  # ESM2 MLP

    init_method_std: float = 0.02
    apply_query_key_layer_scaling: bool = True  # TODO: farhadr check in esm2

    masked_softmax_fusion: bool = True  # Use a kernel that fuses the attention softmax with it's mask.
    fp16_lm_cross_entropy: bool = False  # Move the cross entropy unreduced loss calculation for lm head to fp16
    share_embeddings_and_output_weights: bool = True
    enable_autocast: bool = True  # This has to be set to True if we use the mixed precision plugin
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.esm2_bert_layer_local_spec.value
    position_embedding_type: Literal["learned_absolute", "rope"] = (
        "rope"  # ESM2 uses relative positional encoding 'ROPE' to extrapolate to longer sequences unseen during training
    )

    seq_length: int = 1024
    make_vocab_size_divisible_by: int = 128

    # embedding
    token_dropout: bool = True
    use_attention_mask: bool = True

    # core attention
    use_esm_attention: bool = True  # farhadr MR813
    attention_softmax_in_fp32: bool = True

    optimizer_fn: Optional[Callable[[MegatronBioBertModel], Optimizer]] = None

    def configure_model(self, tokenizer) -> ESM2Model:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        # The local specs all require the standard full attention mask. For transformer engine only the NVTE_FLASH_ATTN=0
        #  option requires this full attention mask.
        use_full_attention_mask: bool = os.getenv("NVTE_FLASH_ATTN") == "0" or self.biobert_spec_option in {
            BiobertSpecOption.bert_layer_local_spec,
            BiobertSpecOption.bert_layer_local_spec_with_qk_ln,
            BiobertSpecOption.esm2_bert_layer_local_spec,
        }

        do_next_sentence = False

        return ESM2Model(
            self,
            transformer_layer_spec=get_biobert_spec(self.biobert_spec_option, qk_layernorm=self.qk_layernorm),
            num_tokentypes=2 if do_next_sentence else 0,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            tokenizer=tokenizer,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            return_embeddings=False,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),  # set to False for inference
            add_binary_head=do_next_sentence,
            use_full_attention_mask=use_full_attention_mask,
        )


__all__ = ["ESM2Config", "ESM2Model"]
