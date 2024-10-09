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


# Imports needed for redefinition of TransformerLanguageModel
import torch
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import BertModel
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    PromptEncoderAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.language_model import (
    Embedding,
    ModelParallelConfig,
    Pooler,
    TransformerLanguageModel,
)
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.position_embedding import (
    ALiBiRelativePositionEmbedding,
    KERPLERelativePositionEmbedding,
    RotaryEmbedding,
    SandwichRelativePositionEmbedding,
)
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)

from bionemo.model.protein.esm1nv.transformer import ESMnvParallelTransformer


try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    LayerType = ApexGuardDefaults()  # noqa: F811

try:
    from megatron.core import parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

# Additional imports needed for BertModel
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import BertLMHead
from nemo.utils import logging


# TODO(dorotat, georgea) Refactor these part to use directly megatron.core
class ESMnvEmbedding(Embedding):
    def __init__(
        self,
        config: ModelParallelConfig,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        fp32_residual_connection=False,
        position_embedding_type="learned_absolute",
        transpose_batch_sequence=True,
        # BIONEMO NEW ARGS
        token_dropout=False,
        use_attention_mask=False,
        mask_token_id=None,
    ):
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            init_method=init_method,
            num_tokentypes=num_tokentypes,
            embedding_dropout_prob=embedding_dropout_prob,
            position_embedding_type=position_embedding_type,
            transpose_batch_sequence=transpose_batch_sequence,
            fp32_residual_connection=fp32_residual_connection,
        )
        self.token_dropout = token_dropout
        self.use_attention_mask = use_attention_mask
        if mask_token_id is None:
            mask_token_id = torch.nan
        self.mask_token_id = mask_token_id
        if self.token_dropout or self.use_attention_mask:
            logging.warning("Using custom ESM2 Embeddings instead of the default NeMo version")

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        words_embeddings = self.word_embeddings(input_ids)

        # BIONEMO: add custom logic for attention masking and token dropout
        # TODO(srabhi, georgea): refactor the custom ESMnvEmbedding module using Megatron Core when NeMo 1.21 is available
        embeddings_mask = None
        if attention_mask is not None and (self.token_dropout or self.use_attention_mask):
            embeddings_mask = ~attention_mask[:, 0, :, 0]

        if embeddings_mask is not None and self.token_dropout:
            words_embeddings = words_embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs
            src_lengths = embeddings_mask.sum(-1)
            is_mask_token = input_ids == self.mask_token_id
            mask_ratio_observed = (is_mask_token).sum(-1).float() / src_lengths
            words_embeddings = (
                words_embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
            ).to(words_embeddings.dtype)
        # END BIONEMO

        if self.position_embedding_type == "learned_absolute":
            assert position_ids is not None
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        elif self.position_embedding_type == "learned_parameters":
            embeddings = words_embeddings + self.position_embeddings
        else:
            embeddings = words_embeddings
        if token_type_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(token_type_ids)
        else:
            assert self.tokentype_embeddings is None

        # BIONEMO: include attention masking from ESM2
        # TODO(srabhi, georgea): refactor the custom ESMnvEmbedding module using Megatron Core when NeMo 1.21 is available
        if embeddings_mask is not None and self.use_attention_mask:
            embeddings = (embeddings * embeddings_mask.unsqueeze(-1)).to(embeddings.dtype)
        # END BIONEMO

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        if self.transpose_batch_sequence:
            embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.mappings.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.random.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings


class ESMnvTransformerLanguageModel(TransformerLanguageModel):
    # BIONEMO: This is copy/paste/edited from NeMo. Parts that are
    # modifications are annotated with `BIONEMO`.
    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        encoder_attn_mask_type,
        vocab_size,
        max_position_embeddings,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_tokentypes,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        add_decoder=False,
        decoder_attn_mask_type=AttnMaskType.causal,
        add_pooler=False,
        pre_process=True,
        post_process=True,
        megatron_amp_O2=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        normalization="layernorm",
        layernorm_epsilon=1e-5,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        bias=True,
        masked_softmax_fusion=True,
        activation="gelu",
        headscale=False,
        transformer_block_type="pre_ln",
        normalize_attention_scores=True,
        position_embedding_type="learned_absolute",
        rotary_percentage=1.0,
        multi_query_attention=False,
        share_embeddings_and_output_weights=True,  # to decide whether use a new linear layer to compute final scores (False) or reuse embedding weights (True)
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        megatron_legacy=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_layers_per_pipeline=None,
        transformer_engine=False,
        fp8=False,
        fp8_e4m3=False,
        fp8_hybrid=False,
        fp8_margin=0,
        fp8_interval=1,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo="max",
        reduce_amax=True,
        use_emha=False,
        ub_tp_comm_overlap=False,
        use_flash_attention=False,
        seq_len_interpolation_factor=None,
        rotary_base=10000,
        # BIONEMO: new arguments
        embedding_token_dropout=False,
        embedding_use_attention_mask=False,
        mask_token_id=None,
        # NEW BIONEMO ARGS
        use_esm_attention=False,
        esm_gelu=False,
        use_pt_layernorm=False,
        use_pt_mlp_out=False,
    ):
        super(TransformerLanguageModel, self).__init__(
            config=config, share_token_embeddings=share_embeddings_and_output_weights
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.sequence_parallel = config.sequence_parallel
        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), "hidden_size must be divisible by num_attention_heads if kv_channels is None"
            kv_channels = hidden_size // num_attention_heads

        # Embeddings.
        if self.pre_process:
            # BIONEMO: New embedding class
            # TODO(srabhi, georgea): refactor the custom ESMnvEmbedding module using Megatron Core when NeMo 1.21 is available
            self.embedding = ESMnvEmbedding(
                config=config,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_position_embeddings,
                init_method=self.init_method,
                num_tokentypes=self.num_tokentypes,
                embedding_dropout_prob=self.hidden_dropout,
                position_embedding_type=position_embedding_type,
                fp32_residual_connection=fp32_residual_connection,
                # BIONEMO: New arguments
                token_dropout=embedding_token_dropout,
                use_attention_mask=embedding_use_attention_mask,
                mask_token_id=mask_token_id,
            )
            self._embedding_key = "embedding"

        if position_embedding_type == "rope":
            rotary_dim = self.hidden_size // num_attention_heads if kv_channels is None else kv_channels
            assert 0 < rotary_percentage <= 1
            if rotary_percentage < 1:
                rotary_dim = int(rotary_dim * rotary_percentage)
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_dim,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                pretrained_max_position_embeddings=max_position_embeddings,
                rotary_base=rotary_base,
            )

        elif position_embedding_type == "alibi":
            # TODO: If this is used for encoder-decodemax_position_embeddingsr model, implement proper logic and following
            # addition for decoder. Currently it is only used for decoder model only.
            # Encoder-decoder model, such as T5 is implemented in token_level_encoder_decoder.py
            self.encoder_relative_position_embedding = ALiBiRelativePositionEmbedding(
                bidirectional=encoder_attn_mask_type != AttnMaskType.causal,
                num_attention_heads=num_attention_heads,
                layer_type=LayerType.encoder,
                num_attention_heads_alibi=None,
                max_seq_len=max_position_embeddings,
            )

        elif position_embedding_type == "kerple":
            # TODO: If this is used for encoder-decodemax_position_embeddingsr model, implement proper logic and following
            # addition for decoder. Currently it is only used for decoder model only.
            # Encoder-decoder model, such as T5 is implemented in token_level_encoder_decoder.py
            self.encoder_relative_position_embedding = KERPLERelativePositionEmbedding(
                bidirectional=encoder_attn_mask_type != AttnMaskType.causal,
                num_attention_heads=num_attention_heads,
                layer_type=LayerType.encoder,
                num_attention_heads_kerple=None,
                max_seq_len=max_position_embeddings,
            )
            assert use_flash_attention is False, "flash-attention not supported with kerple at this point"

        elif position_embedding_type == "sandwich":
            self.encoder_relative_position_embedding = SandwichRelativePositionEmbedding(
                bidirectional=encoder_attn_mask_type != AttnMaskType.causal,
                num_attention_heads=num_attention_heads,
                layer_type=LayerType.encoder,
                hidden_size=self.hidden_size // num_attention_heads if kv_channels is None else kv_channels,
                max_seq_len=max_position_embeddings,
            )

        # Transformer.
        self.encoder = ESMnvParallelTransformer(
            config=config,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=self.encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            normalization=normalization,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            megatron_amp_O2=megatron_amp_O2,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            bias=bias,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            activation=activation,
            headscale=headscale,
            transformer_block_type=transformer_block_type,
            normalize_attention_scores=normalize_attention_scores,
            multi_query_attention=multi_query_attention,
            megatron_legacy=megatron_legacy,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            transformer_engine=transformer_engine,
            fp8=fp8,
            fp8_e4m3=fp8_e4m3,
            fp8_hybrid=fp8_hybrid,
            fp8_margin=fp8_margin,
            fp8_interval=fp8_interval,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            reduce_amax=reduce_amax,
            use_emha=use_emha,
            ub_tp_comm_overlap=ub_tp_comm_overlap,
            position_embedding_type=position_embedding_type,
            use_flash_attention=use_flash_attention,
            # NEW BIONEMO ARGs
            use_esm_attention=use_esm_attention,
            esm_gelu=esm_gelu,
            use_pt_layernorm=use_pt_layernorm,
            use_pt_mlp_out=use_pt_mlp_out,
        )
        self._encoder_key = "encoder"

        # Decoder
        if self.add_decoder:
            self.decoder = ParallelTransformer(
                config=config,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                init_method=self.init_method,
                output_layer_init_method=self.output_layer_init_method,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                ffn_hidden_size=ffn_hidden_size,
                pre_process=self.pre_process,
                post_process=self.post_process,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                normalization=normalization,
                layernorm_epsilon=layernorm_epsilon,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                megatron_amp_O2=megatron_amp_O2,
                bias_activation_fusion=bias_activation_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                megatron_legacy=megatron_legacy,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
                transformer_engine=transformer_engine,
                position_embedding_type=position_embedding_type,
                use_flash_attention=use_flash_attention,
            )
            self._decoder_key = "decoder"

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method, sequence_parallel=self.sequence_parallel)
                self._pooler_key = "pooler"

            if not self.share_embeddings_and_output_weights:
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    self.hidden_size,
                    self.vocab_size,
                    config=config,
                    bias=False,  # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                    init_method=self.init_method,
                )
                self._output_layer_key = "output_layer"
        self.set_accepted_adapter_types([PromptEncoderAdapterConfig._target_])

    def forward(
        self,
        enc_input_ids,
        enc_position_ids,
        enc_attn_mask,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attn_mask=None,
        enc_dec_attn_mask=None,
        token_type_ids=None,
        layer_past=None,
        get_key_value=False,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden_only=False,
        encoder_input=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        # Embeddings.
        if self.pre_process and encoder_input is None:
            # BIONEMO: add attn_mask to embeddings call
            # TODO(srabhi, georgea): refactor the custom ESMnvEmbedding module using Megatron Core when NeMo 1.21 is available
            encoder_input = self.embedding(
                enc_input_ids, enc_position_ids, token_type_ids=token_type_ids, attention_mask=enc_attn_mask
            )
            if self.is_adapter_available():
                _sq, _bs, _hs = encoder_input.size()
                ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
                v = ptuning_adapter.virtual_tokens
                if ptuning_adapter and _sq >= v:  # The sequence should be longer the v to insert virtual embeddings.
                    virtual_embeddings = ptuning_adapter(_bs)
                    encoder_input = encoder_input[
                        v:, :, :
                    ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                    encoder_input = torch.concat([virtual_embeddings, encoder_input], dim=0)
        else:
            pass

        # enc_attn_mask: [1, 1, s, s]
        if inference_max_sequence_len is not None:
            enc_seq_length = inference_max_sequence_len
        elif self.encoder.input_tensor is not None:
            if self.sequence_parallel:
                enc_seq_length = (
                    self.encoder.input_tensor.size(0) * parallel_state.get_tensor_model_parallel_world_size()
                )
            else:
                enc_seq_length = self.encoder.input_tensor.size(0)
        else:
            if self.sequence_parallel:
                enc_seq_length = encoder_input.size(0) * parallel_state.get_tensor_model_parallel_world_size()
            else:
                enc_seq_length = encoder_input.size(0)

        rotary_pos_emb = None
        encoder_self_attention_relative_position_bias = None
        if self.position_embedding_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(enc_seq_length)
        elif (
            self.position_embedding_type == "alibi"
            or self.position_embedding_type == "sandwich"
            or self.position_embedding_type == "kerple"
        ):
            encoder_self_attention_relative_position_bias = self.encoder_relative_position_embedding(
                query_seq_length=enc_seq_length,
                key_seq_length=enc_seq_length,
            )
            # causal attention bias: [1, head, 1, k]
            # non-causal attention bias: [1, head, q, k]

        # encoder.
        if enc_hidden_states is None:
            encoder_output = self.encoder(
                encoder_input,
                enc_attn_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                rotary_pos_emb=(
                    (rotary_pos_emb, None, None) if rotary_pos_emb is not None else None
                ),  # This assumes that this being used as a GPT/BERT model only (no cross-attention)
                self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
            )
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output, pooling_sequence_index)

        # output_enc_hidden_only refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden_only:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder Embedding
        dec_embedding_output = self.embedding(dec_input_ids, dec_position_ids)
        # decoder
        decoder_output = self.decoder(
            dec_embedding_output,
            dec_attn_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output


def esm_get_language_model(
    config: ModelParallelConfig,
    hidden_size,
    ffn_hidden_size,
    num_layers,
    max_position_embeddings,
    num_tokentypes,
    add_pooler,
    vocab_size,
    num_attention_heads,
    encoder_attn_mask_type,
    apply_query_key_layer_scaling=False,
    kv_channels=None,
    init_method=None,
    scaled_init_method=None,
    add_decoder=False,
    decoder_attn_mask_type=AttnMaskType.causal,
    pre_process=True,
    post_process=True,
    init_method_std=0.02,
    megatron_amp_O2=False,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    ffn_dropout=0.0,
    precision=16,
    fp32_residual_connection=False,
    activations_checkpoint_method=None,
    activations_checkpoint_num_layers=1,
    normalization="layernorm",
    layernorm_epsilon=1e-5,
    bias_activation_fusion=True,
    masked_softmax_fusion=True,
    activation="gelu",
    headscale=False,
    transformer_block_type="pre_ln",
    normalize_attention_scores=True,
    position_embedding_type="learned_absolute",
    attention_type="multihead",
    share_embeddings_and_output_weights=True,
    rotary_percentage=1.0,
    multi_query_attention=False,
    bias_dropout_add_fusion=True,
    bias=True,
    persist_layer_norm=False,
    openai_gelu=False,
    onnx_safe=False,
    megatron_legacy=False,
    activations_checkpoint_granularity=None,
    activations_checkpoint_layers_per_pipeline=None,
    transformer_engine=False,
    fp8=False,
    fp8_e4m3=False,
    fp8_hybrid=False,
    fp8_margin=0,
    fp8_interval=1,
    fp8_amax_history_len=1024,
    fp8_amax_compute_algo="max",
    reduce_amax=True,
    use_emha=False,
    ub_tp_comm_overlap=False,
    use_flash_attention=False,
    seq_len_interpolation_factor=None,
    rotary_base=10000,
    # BIONEMO: new arguments
    embedding_token_dropout=False,
    embedding_use_attention_mask=False,
    mask_token_id=None,
    # NEW BIONEMO ARGS
    use_esm_attention=False,
    esm_gelu=False,
    use_pt_layernorm=False,
    use_pt_mlp_out=False,
):
    """Build language model and return along with the key to save."""
    if kv_channels is None:
        assert (
            hidden_size % num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads if kv_channels is None"
        kv_channels = hidden_size // num_attention_heads

    if init_method is None:
        init_method = init_method_normal(init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    # Language model.
    # BIONEMO: use custom class
    # TODO(srabhi, georgea): refactor the custom ESMnvTransformerLanguageModel module using Megatron Core when NeMo 1.21 is available
    language_model = ESMnvTransformerLanguageModel(
        config=config,
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        encoder_attn_mask_type=encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        kv_channels=kv_channels,
        ffn_hidden_size=ffn_hidden_size,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        megatron_amp_O2=megatron_amp_O2,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        precision=precision,
        fp32_residual_connection=fp32_residual_connection,
        activations_checkpoint_method=activations_checkpoint_method,
        activations_checkpoint_num_layers=activations_checkpoint_num_layers,
        normalization=normalization,
        layernorm_epsilon=layernorm_epsilon,
        bias_activation_fusion=bias_activation_fusion,
        bias_dropout_add_fusion=bias_dropout_add_fusion,
        bias=bias,
        rotary_percentage=rotary_percentage,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        masked_softmax_fusion=masked_softmax_fusion,
        activation=activation,
        headscale=headscale,
        transformer_block_type=transformer_block_type,
        normalize_attention_scores=normalize_attention_scores,
        position_embedding_type=position_embedding_type,
        multi_query_attention=multi_query_attention,
        persist_layer_norm=persist_layer_norm,
        openai_gelu=openai_gelu,
        onnx_safe=onnx_safe,
        megatron_legacy=megatron_legacy,
        activations_checkpoint_granularity=activations_checkpoint_granularity,
        activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
        transformer_engine=transformer_engine,
        fp8=fp8,
        fp8_e4m3=fp8_e4m3,
        fp8_hybrid=fp8_hybrid,
        fp8_margin=fp8_margin,
        fp8_interval=fp8_interval,
        fp8_amax_history_len=fp8_amax_history_len,
        fp8_amax_compute_algo=fp8_amax_compute_algo,
        reduce_amax=reduce_amax,
        use_emha=use_emha,
        ub_tp_comm_overlap=ub_tp_comm_overlap,
        use_flash_attention=use_flash_attention,
        seq_len_interpolation_factor=seq_len_interpolation_factor,
        rotary_base=rotary_base,
        # BIONEMO: add arguments
        embedding_token_dropout=embedding_token_dropout,
        embedding_use_attention_mask=embedding_use_attention_mask,
        mask_token_id=mask_token_id,
        # NEW BIONEMO ARGS
        use_esm_attention=use_esm_attention,
        esm_gelu=esm_gelu,
        use_pt_layernorm=use_pt_layernorm,
        use_pt_mlp_out=use_pt_mlp_out,
    )
    # key used for checkpoints.
    language_model_key = "language_model"

    return language_model, language_model_key


class ESMnvBertModel(BertModel):
    def __init__(
        self,
        config: ModelParallelConfig,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        fp16_lm_cross_entropy=False,
        megatron_amp_O2=False,
        hidden_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_layers_per_pipeline=None,
        layernorm_epsilon=1e-5,
        masked_softmax_fusion=False,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        add_binary_head=True,
        megatron_legacy=False,
        sequence_parallel=False,
        position_embedding_type="learned_absolute",
        # BIONEMO: use custom arguments
        share_embeddings_and_output_weights=True,
        embedding_token_dropout=False,
        embedding_use_attention_mask=False,
        mask_token_id=None,
        attention_dropout=0.1,
        normalize_attention_scores=True,
        use_esm_attention=False,
        esm_gelu=False,
        use_pt_layernorm=False,
        use_pt_mlp_out=False,
    ):
        super(BertModel, self).__init__(config=config)
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.sequence_parallel = sequence_parallel

        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

        # BIONEMO: use custom language model constructor
        # TODO(srabhi, georgea): refactor the custom esm_get_language_model method using Megatron Core when NeMo 1.21 is available
        self.language_model, self._language_model_key = esm_get_language_model(
            config=config,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            hidden_dropout=hidden_dropout,
            num_tokentypes=num_tokentypes,
            max_position_embeddings=max_position_embeddings,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            add_pooler=self.add_binary_head,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            init_method_std=init_method_std,
            megatron_amp_O2=megatron_amp_O2,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            layernorm_epsilon=layernorm_epsilon,
            masked_softmax_fusion=masked_softmax_fusion,
            bias_activation_fusion=bias_gelu_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            megatron_legacy=megatron_legacy,
            position_embedding_type=position_embedding_type,
            # BIONEMO: use new arguments
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            attention_dropout=attention_dropout,
            embedding_token_dropout=embedding_token_dropout,
            embedding_use_attention_mask=embedding_use_attention_mask,
            mask_token_id=mask_token_id,
            normalize_attention_scores=normalize_attention_scores,
            use_esm_attention=use_esm_attention,
            esm_gelu=esm_gelu,
            use_pt_layernorm=use_pt_layernorm,
            use_pt_mlp_out=use_pt_mlp_out,
        )

        self.initialize_word_embeddings(
            init_method=init_method_normal(init_method_std), vocab_size=vocab_size, hidden_size=hidden_size
        )

        if self.post_process:
            self.lm_head = BertLMHead(
                config,
                self.word_embeddings_weight().size(0),
                hidden_size,
                init_method,
                layernorm_epsilon,
                parallel_output,
                openai_gelu,
                onnx_safe,
            )
            self._lm_head_key = "lm_head"
            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = get_linear_layer(hidden_size, 2, init_method)
                self._binary_head_key = "binary_head"


class ESMnvMegatronBertModel(MegatronBertModel):
    def model_provider_func(self, pre_process, post_process):
        cfg = self.cfg
        num_tokentypes = 2 if cfg.bert_binary_head else 0
        # TODO(srabhi, georgea): refactor the custom ESMnvBertModel method using Megatron Core when NeMo 1.21 is available
        model = ESMnvBertModel(
            config=self.model_parallel_config,
            vocab_size=self.padded_vocab_size,
            hidden_size=cfg.hidden_size,
            max_position_embeddings=cfg.max_position_embeddings,
            num_layers=cfg.num_layers,
            num_attention_heads=cfg.num_attention_heads,
            apply_query_key_layer_scaling=cfg.get("apply_query_key_layer_scaling", True),
            kv_channels=cfg.get("kv_channels", None),
            ffn_hidden_size=cfg.ffn_hidden_size,
            num_tokentypes=num_tokentypes,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=cfg.get("init_method_std", 0.02),
            fp16_lm_cross_entropy=cfg.get("fp16_lm_cross_entropy", False),
            megatron_amp_O2=self.cfg.get("megatron_amp_O2", False),
            hidden_dropout=cfg.get("hidden_dropout", 0.1),
            precision=cfg.get("precision", 16),
            fp32_residual_connection=cfg.get("fp32_residual_connection", False),
            activations_checkpoint_granularity=self.cfg.get("activations_checkpoint_granularity", None),
            activations_checkpoint_method=self.cfg.get("activations_checkpoint_method", None),
            activations_checkpoint_num_layers=self.cfg.get("activations_checkpoint_num_layers", 1),
            activations_checkpoint_layers_per_pipeline=self.cfg.get(
                "activations_checkpoint_layers_per_pipeline", None
            ),
            layernorm_epsilon=cfg.get("layernorm_epsilon", 1e-5),
            masked_softmax_fusion=cfg.get("masked_softmax_fusion", True),
            bias_gelu_fusion=cfg.get("bias_gelu_fusion", True),
            bias_dropout_add_fusion=cfg.get("bias_dropout_add_fusion", True),
            onnx_safe=cfg.get("onnx_safe", False),
            add_binary_head=cfg.bert_binary_head,
            megatron_legacy=cfg.get("megatron_legacy", False),
            position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
            # BIONEMO: use custom flags
            share_embeddings_and_output_weights=self.cfg.get("share_embeddings_and_output_weights", True),
            embedding_token_dropout=self.cfg.get("embedding_token_dropout", False),
            embedding_use_attention_mask=self.cfg.get("embedding_use_attention_mask", False),
            mask_token_id=self.cfg.get("mask_token_id", None),
            attention_dropout=self.cfg.get("attention_dropout", 0.1),
            normalize_attention_scores=self.cfg.get("normalize_attention_scores", True),
            use_esm_attention=self.cfg.get("use_esm_attention", False),
            esm_gelu=self.cfg.get("esm_gelu", False),
            use_pt_layernorm=self.cfg.get("use_pt_layernorm", False),
            use_pt_mlp_out=self.cfg.get("use_pt_mlp_out", False),
        )

        return model
