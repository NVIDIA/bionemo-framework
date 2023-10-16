"""
Much of this file is copy/pasted directly from:
nemo/collections/nlp/modules/common/megatron/transformer.py
This files mainly serves to redirect the transformer's attention through
a custom bionemo attention layer.
"""

import torch
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    ParallelLinearAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.attention import ParallelAttention, ParallelChunkedCrossAttention
from nemo.collections.nlp.modules.common.megatron.layer_norm_1p import LayerNorm1P, LPLayerNorm
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.mlp import SwitchMLP
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.collections.nlp.parts import utils_funcs
from nemo.utils import logging


try:
    from apex.normalization import MixedFusedRMSNorm
    from apex.transformer.enums import AttnMaskType, AttnType, ModelType

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()  # noqa: F811

try:
    from megatron.core import parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

try:
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import TransformerLayer

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

    # fake missing class
    class TransformerLayer(ApexGuardDefaults):
        def __init__(self):
            super().__init__()

            logging.warning(
                "Transformer Engine was not found. transformer_engine.pytorch.transformer.TransformerLayer will not work. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )


## BIONEMO imports
from nemo.collections.nlp.modules.common.megatron.transformer import (
    AutocastTransformerLayer,
    ParallelTransformer,
    ParallelTransformerLayer_,
    remove_bias_from_layernorm,
)

from bionemo.model.protein.esm1nv.attention import ESMnvParallelAttention
from bionemo.model.protein.esm1nv.layernorm import esm_get_layer_norm
from bionemo.model.protein.esm1nv.mlp import ESMnvParallelMLP


## END BIONEMO


class ESMnvParallelTransformerLayer_(ParallelTransformerLayer_):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        fp32_residual_connection=False,
        precision=16,
        apply_query_key_layer_scaling=False,
        kv_channels=None,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.1,
        persist_layer_norm=False,
        use_cpu_initialization=False,
        megatron_amp_O2=False,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        gradient_accumulation_fusion=False,
        openai_gelu=False,
        onnx_safe=False,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        activation='gelu',
        megatron_legacy=False,
        bias=True,
        chunk_size=64,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        position_embedding_type='learned_absolute',
        multi_query_attention=False,
        headscale=False,
        activations_checkpoint_granularity=None,
        sequence_parallel=False,
        normalize_attention_scores=True,
        num_moe_experts=1,
        moe_frequency=1,
        moe_dropout=0.0,
        use_flash_attention=False,
        # NEW BIONEMO ARGS
        use_esm_attention=False,
        esm_gelu=False,
        use_pt_layernorm=False,
        use_pt_mlp_out=False,
    ):
        super(ParallelTransformerLayer_, self).__init__()

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        self.layer_number = layer_number
        self.layer_type = layer_type
        self.bias = bias
        self.transformer_block_type = transformer_block_type
        self.position_embedding_type = position_embedding_type
        self.param_dtype = utils_funcs.dtype_from_precision(precision, megatron_amp_O2)

        self.set_accepted_adapter_types([LinearAdapterConfig._target_, ParallelLinearAdapterConfig._target_])

        if not bias and bias_dropout_add_fusion:
            raise ValueError(
                'bias_dropout_add_fusion=True requires bias=True, found bias=False. Either set both to True or both to False.'
            )

        # the low_precision_layernorm does not require a bias term, whereas layernorm1p from apex
        # does require a bias, so it cannot be used for bias-less low precision LN such as in MPT-7B
        if normalization not in ['layernorm', 'layernorm1p', 'rmsnorm', 'low_precision_layernorm']:
            raise ValueError(f'normalization must be "layernorm", "layernorm1p" or "rmsnorm", found {normalization}')

        if transformer_block_type not in ['pre_ln', 'post_ln', 'normformer']:
            raise ValueError(
                f'transformer_block_type must be either "pre_ln" or "post_ln" or "normformer", found {transformer_block_type}'
            )

        self.fp32_residual_connection = fp32_residual_connection  # if true move residual connections to fp32
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bias_dropout_add_fusion = bias_dropout_add_fusion  # if true, enable bias dropout fusion

        # Self attention.
        # retrieval_decoder_after_self_attn skips the self attention
        if self.layer_type != LayerType.retrieval_decoder_after_self_attn:
            # Layernorm on the input data.
            if normalization == 'layernorm':
                self.input_layernorm = esm_get_layer_norm(
                    hidden_size,
                    layernorm_epsilon,
                    persist_layer_norm,
                    sequence_parallel,
                    use_pt_layernorm=use_pt_layernorm,
                )
            elif normalization == 'layernorm1p':
                self.input_layernorm = LayerNorm1P(
                    hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                )
            elif normalization == 'low_precision_layernorm':
                self.input_layernorm = LPLayerNorm(hidden_size, layernorm_epsilon)
            else:
                self.input_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)
            # for architectures such as MPT, there is no bias term even on the layernorms
            # this code allows us to remove the bias terms from the layernorm module
            # so that we can support MPT. However, certain apex-based LNs don't support
            # removing bias, so we also have to check for that
            if not bias and normalization not in ['layernorm', 'layernorm1p']:
                remove_bias_from_layernorm(self.input_layernorm)

            self.self_attention = ESMnvParallelAttention(
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                layer_number=0 if use_esm_attention else layer_number,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                attention_type=AttnType.self_attn,
                attn_mask_type=self_attn_mask_type,
                precision=precision,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                use_cpu_initialization=use_cpu_initialization,
                megatron_amp_O2=megatron_amp_O2,
                masked_softmax_fusion=masked_softmax_fusion,
                attention_dropout=attention_dropout,
                multi_query_attention=multi_query_attention,
                layer_type=layer_type,
                megatron_legacy=megatron_legacy,
                bias=bias,
                headscale=headscale,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                position_embedding_type=position_embedding_type,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                normalize_attention_scores=normalize_attention_scores,
                use_flash_attention=use_flash_attention,
                # NEW BIONEMO ARGS
                use_esm_attention=use_esm_attention,
            )

            if transformer_block_type == 'normformer':
                if normalization == 'layernorm':
                    self.post_attention_normformer_norm = esm_get_layer_norm(
                        hidden_size,
                        layernorm_epsilon,
                        persist_layer_norm,
                        use_pt_layernorm=use_pt_layernorm,
                    )
                else:
                    self.post_attention_normformer_norm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)

            if self.layer_type != LayerType.decoder_pre_mlp or self.transformer_block_type != 'post_ln':
                #  the post_attention_layernorm is used for layermorm after mlp
                # don't need it for decoder_pre_mlp and post_ln
                if normalization == 'layernorm':
                    self.post_attention_layernorm = esm_get_layer_norm(
                        hidden_size,
                        layernorm_epsilon,
                        persist_layer_norm,
                        sequence_parallel,
                        use_pt_layernorm=use_pt_layernorm,
                    )
                elif normalization == 'layernorm1p':
                    self.post_attention_layernorm = LayerNorm1P(
                        hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                    )
                elif normalization == 'low_precision_layernorm':
                    self.post_attention_layernorm = LPLayerNorm(hidden_size, layernorm_epsilon)
                else:
                    self.post_attention_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)
                if not bias and normalization not in ['layernorm', 'layernorm1p']:
                    remove_bias_from_layernorm(self.post_attention_layernorm)

        if self.layer_type == LayerType.decoder_pre_mlp:
            # skip MLP and cross attention
            return

        # the post_attention_layernorm is used for layermorm after mlp
        # need it for post_ln
        if self.layer_type == LayerType.retrieval_decoder_after_self_attn and self.transformer_block_type == 'post_ln':
            # Layernorm on the attention output
            if normalization == 'layernorm':
                self.post_attention_layernorm = esm_get_layer_norm(
                    hidden_size,
                    layernorm_epsilon,
                    persist_layer_norm,
                    sequence_parallel,
                    use_pt_layernorm=use_pt_layernorm,
                )
            elif normalization == 'layernorm1p':
                self.post_attention_layernorm = LayerNorm1P(
                    hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                )
            elif normalization == 'low_precision_layernorm':
                self.post_attention_layernorm = LPLayerNorm(hidden_size, layernorm_epsilon)
            else:
                self.post_attention_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)
            if not bias and normalization not in ['layernorm', 'layernorm1p']:
                remove_bias_from_layernorm(self.post_attention_layernorm)

        if self.layer_type == LayerType.decoder or self.layer_type == LayerType.retrieval_encoder:
            self.inter_attention = ParallelAttention(
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                layer_number=layer_number,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                attention_type=AttnType.cross_attn,
                attn_mask_type=AttnMaskType.padding,
                precision=precision,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                multi_query_attention=multi_query_attention,
                use_cpu_initialization=use_cpu_initialization,
                megatron_amp_O2=megatron_amp_O2,
                masked_softmax_fusion=masked_softmax_fusion,
                attention_dropout=attention_dropout,
                megatron_legacy=megatron_legacy,
                bias=bias,
                headscale=headscale,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                normalize_attention_scores=normalize_attention_scores,
            )
            # Normformer normalization
            if transformer_block_type == 'normformer':
                if normalization == 'layernorm':
                    self.post_inter_attention_normformer_norm = esm_get_layer_norm(
                        hidden_size,
                        layernorm_epsilon,
                        persist_layer_norm,
                        sequence_parallel,
                        use_pt_layernorm=use_pt_layernorm,
                    )
                elif normalization == 'layernorm1p':
                    self.post_inter_attention_normformer_norm = LayerNorm1P(
                        hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                    )
                else:
                    self.post_inter_attention_normformer_norm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)

            # Layernorm on the attention output.
            if normalization == 'layernorm':
                self.post_inter_attention_layernorm = esm_get_layer_norm(
                    hidden_size,
                    layernorm_epsilon,
                    persist_layer_norm,
                    sequence_parallel,
                    use_pt_layernorm=use_pt_layernorm,
                )
            elif normalization == 'layernorm1p':
                self.post_inter_attention_layernorm = LayerNorm1P(
                    hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                )
            else:
                self.post_inter_attention_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)
        elif (
            self.layer_type == LayerType.retrieval_decoder
            or self.layer_type == LayerType.retrieval_decoder_after_self_attn
        ):
            self.inter_attention = ParallelChunkedCrossAttention(
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                layer_number=layer_number,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                precision=precision,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                use_cpu_initialization=use_cpu_initialization,
                megatron_amp_O2=megatron_amp_O2,
                masked_softmax_fusion=masked_softmax_fusion,
                attention_dropout=attention_dropout,
                megatron_legacy=megatron_legacy,
                chunk_size=chunk_size,
                bias=bias,
                headscale=headscale,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )
            # Normformer normalization
            if transformer_block_type == 'normformer':
                if normalization == 'layernorm':
                    self.post_inter_attention_normformer_norm = esm_get_layer_norm(
                        hidden_size,
                        layernorm_epsilon,
                        persist_layer_norm,
                        sequence_parallel,
                        use_pt_layernorm=use_pt_layernorm,
                    )
                elif normalization == 'layernorm1p':
                    self.post_inter_attention_normformer_norm = LayerNorm1P(
                        hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                    )
                else:
                    self.post_inter_attention_normformer_norm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)

            # Layernorm on the attention output.
            if normalization == 'layernorm':
                self.post_inter_attention_layernorm = esm_get_layer_norm(
                    hidden_size,
                    layernorm_epsilon,
                    persist_layer_norm,
                    sequence_parallel,
                    use_pt_layernorm=use_pt_layernorm,
                )
            elif normalization == 'layernorm1p':
                self.post_inter_attention_layernorm = LayerNorm1P(
                    hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                )
            else:
                self.post_inter_attention_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)

        # MLP
        if num_moe_experts > 1 and self.layer_number % moe_frequency == 0:
            self.mlp = SwitchMLP(
                num_experts=num_moe_experts,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                use_cpu_initialization=use_cpu_initialization,
                dtype=self.param_dtype,
                bias_activation_fusion=bias_activation_fusion,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                activation=activation,
                bias=bias,
                transformer_block_type=transformer_block_type,
                normalization=normalization,
                layernorm_epsilon=layernorm_epsilon,
                persist_layer_norm=persist_layer_norm,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dropout=moe_dropout,
            )
        else:
            self.mlp = ESMnvParallelMLP(
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                use_cpu_initialization=use_cpu_initialization,
                dtype=self.param_dtype,
                bias_activation_fusion=bias_activation_fusion,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                activation=activation,
                bias=bias,
                transformer_block_type=transformer_block_type,
                normalization=normalization,
                layernorm_epsilon=layernorm_epsilon,
                persist_layer_norm=persist_layer_norm,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dropout=ffn_dropout,
                # BIONEOM ARGS
                esm_gelu=esm_gelu,
                use_pt_layernorm=use_pt_layernorm,
                use_pt_mlp_out=use_pt_mlp_out,
            )


class ESMnvParallelTransformerLayer(ESMnvParallelTransformerLayer_):
    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        fp32_residual_connection=False,
        precision=16,
        apply_query_key_layer_scaling=False,
        kv_channels=None,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.1,
        bias_dropout_add_fusion=True,
        persist_layer_norm=False,
        use_cpu_initialization=False,
        megatron_amp_O2=False,
        bias_activation_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        activation='gelu',
        megatron_legacy=False,
        bias=True,
        chunk_size=64,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        position_embedding_type='learned_absolute',
        multi_query_attention=False,
        headscale=False,
        activations_checkpoint_granularity=None,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        normalize_attention_scores=True,
        num_moe_experts=1,
        moe_frequency=1,
        moe_dropout=0.0,
        use_flash_attention=False,
        # NEW BIONEMO ARGS
        use_esm_attention=False,
        esm_gelu=False,
        use_pt_layernorm=False,
        use_pt_mlp_out=False,
    ):
        super(ESMnvParallelTransformerLayer, self).__init__(
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_attention_heads,
            layer_type=layer_type,
            self_attn_mask_type=self_attn_mask_type,
            fp32_residual_connection=fp32_residual_connection,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            persist_layer_norm=persist_layer_norm,
            use_cpu_initialization=use_cpu_initialization,
            megatron_amp_O2=megatron_amp_O2,
            bias_activation_fusion=bias_activation_fusion,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            activation=activation,
            megatron_legacy=megatron_legacy,
            bias=bias,
            chunk_size=chunk_size,
            normalization=normalization,
            transformer_block_type=transformer_block_type,
            position_embedding_type=position_embedding_type,
            headscale=headscale,
            multi_query_attention=multi_query_attention,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            normalize_attention_scores=normalize_attention_scores,
            num_moe_experts=num_moe_experts,
            moe_frequency=moe_frequency,
            moe_dropout=moe_dropout,
            use_flash_attention=use_flash_attention,
            # NEW BIONEMO ARGS
            use_esm_attention=use_esm_attention,
            esm_gelu=esm_gelu,
            use_pt_layernorm=use_pt_layernorm,
            use_pt_mlp_out=use_pt_mlp_out,
        )

        # Dtype for forward pass - ignore amp O2
        self.dtype = utils_funcs.dtype_from_precision(precision, megatron_amp_O2=None)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        enc_dec_attn_mask=None,
        rotary_pos_emb=None,
        layer_past=None,
        get_key_value=False,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        self_attention_relative_position_bias=None,
        cross_attention_relative_position_bias=None,
        checkpoint_core_attention=False,
    ):
        if self.dtype == torch.float32:
            return super().forward(
                hidden_states,
                attention_mask,
                encoder_output,
                enc_dec_attn_mask,
                layer_past,
                get_key_value,
                set_inference_key_value_memory,
                inference_max_sequence_len,
                rotary_pos_emb,
                self_attention_relative_position_bias,
                cross_attention_relative_position_bias,
                checkpoint_core_attention,
            )
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return super().forward(
                hidden_states,
                attention_mask,
                encoder_output,
                enc_dec_attn_mask,
                layer_past,
                get_key_value,
                set_inference_key_value_memory,
                inference_max_sequence_len,
                rotary_pos_emb,
                self_attention_relative_position_bias,
                cross_attention_relative_position_bias,
                checkpoint_core_attention,
            )


class ESMnvParallelTransformer(ParallelTransformer):
    """Transformer class."""

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        num_layers,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        apply_query_key_layer_scaling=False,
        kv_channels=None,
        layer_type=LayerType.encoder,  # it can be a list of types or single type
        self_attn_mask_type=AttnMaskType.padding,
        pre_process=True,
        post_process=True,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=None,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        use_cpu_initialization=False,
        megatron_amp_O2=False,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        gradient_accumulation_fusion=False,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        activation='gelu',
        model_type=ModelType.encoder_or_decoder,
        megatron_legacy=False,
        bias=True,
        chunk_size=64,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        position_embedding_type='learned_absolute',
        headscale=False,
        layer_number_offset=0,  # this is use only for attention norm_factor scaling
        activations_checkpoint_granularity=None,
        activations_checkpoint_layers_per_pipeline=None,
        sequence_parallel=False,
        transformer_engine=False,
        fp8=False,
        fp8_e4m3=False,
        fp8_hybrid=False,
        fp8_margin=0,
        fp8_interval=1,
        fp8_amax_history_len=1,
        fp8_amax_compute_algo='most_recent',
        reduce_amax=True,
        use_emha=False,
        ub_tp_comm_overlap=False,
        normalize_attention_scores=True,
        multi_query_attention=False,
        num_moe_experts=1,
        moe_frequency=1,
        moe_dropout=0.0,
        use_flash_attention=False,
        # NEW BIONEMO ARGS
        use_esm_attention=False,
        esm_gelu=False,
        use_pt_layernorm=False,
        use_pt_mlp_out=False,
    ):
        super(ParallelTransformer, self).__init__()

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        self.fp32_residual_connection = fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.self_attn_mask_type = self_attn_mask_type
        self.model_type = model_type
        self.normalization = normalization
        self.transformer_block_type = transformer_block_type
        self.layer_type = layer_type
        self.position_embedding_type = position_embedding_type
        self.multi_query_attention = multi_query_attention

        self.inference_current_sequence_len = 0
        self.inference_params = None

        self.activations_checkpoint_method = activations_checkpoint_method
        self.activations_checkpoint_num_layers = activations_checkpoint_num_layers
        self.activations_checkpoint_granularity = activations_checkpoint_granularity
        self.activations_checkpoint_layers_per_pipeline = activations_checkpoint_layers_per_pipeline

        if self.activations_checkpoint_granularity:
            if self.activations_checkpoint_granularity == 'selective':
                if self.activations_checkpoint_method == 'uniform':
                    logging.info(
                        (
                            'Using uniform activation checkpointing with granularity selective forces all layers to use checkpointing.'
                        )
                    )
                elif self.activations_checkpoint_method == 'block':
                    logging.info(
                        (
                            f'Using block activation checkpointing requires activations_checkpoint_num_layers to be set.'
                            f'Got: {self.activations_checkpoint_num_layers}. Setting to 1 by default.'
                        )
                    )
                else:
                    raise ValueError(
                        'activations_checkpoint_method should be "uniform" or "block" when using granularity selective.'
                    )
            elif self.activations_checkpoint_granularity == 'full':
                if self.activations_checkpoint_method in ['uniform', 'block']:
                    if not self.activations_checkpoint_num_layers:
                        logging.info(
                            (
                                f'Using uniform or block activation checkpointing requires activations_checkpoint_num_layers to be set.'
                                f'Got: {self.activations_checkpoint_num_layers}. Setting to 1 by default.'
                            )
                        )
                else:
                    raise ValueError(
                        'activations_checkpoint_method should be "uniform" or "block" when using granularity full.'
                    )
            else:
                raise ValueError('activations_checkpoint_granularity should be "selective" or "full".')

        self.sequence_parallel = sequence_parallel
        self.transformer_engine = transformer_engine
        self.fp8 = fp8
        self.fp8_e4m3 = fp8_e4m3
        self.fp8_hybrid = fp8_hybrid
        self.fp8_margin = fp8_margin
        self.fp8_interval = fp8_interval
        self.fp8_amax_history_len = fp8_amax_history_len
        self.fp8_amax_compute_algo = fp8_amax_compute_algo
        self.reduce_amax = reduce_amax

        self.fp8_recipe = None

        if self.fp8:
            if self.fp8_e4m3:
                fp8_format = recipe.Format.E4M3
            elif self.fp8_hybrid:
                fp8_format = recipe.Format.HYBRID
            self.fp8_recipe = recipe.DelayedScaling(
                margin=self.fp8_margin,
                interval=self.fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=self.fp8_amax_history_len,
                amax_compute_algo=self.fp8_amax_compute_algo,
                reduce_amax=reduce_amax,
            )

        self.is_first_microbatch = True
        self.microbatch_count = 0  # transformer engine forward needs to know if it is working on the first microbatch
        self.checkpoint_core_attention = (
            activations_checkpoint_granularity == 'selective'
        )  # transformer engine forward allows for more granular selective checkpointing

        if self.model_type == ModelType.encoder_or_decoder:
            assert (
                num_layers % parallel_state.get_pipeline_model_parallel_world_size() == 0
            ), 'num_layers must be divisible by pipeline_model_parallel_size'

        assert moe_frequency <= num_layers, 'MoE frequency must be <= number of transformer layers'
        # TODO: Add similar assert for encoder-decoder.

        self.num_layers = self.get_num_layers(num_layers)

        # Transformer layers.
        def build_layer(layer_number):
            if isinstance(layer_type, list):
                lt = layer_type[layer_number - 1]
            else:
                lt = layer_type

            if self.transformer_engine:
                return AutocastTransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    layernorm_epsilon=layernorm_epsilon,
                    num_attention_heads=num_attention_heads,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    hidden_dropout=hidden_dropout,
                    attention_dropout=attention_dropout,
                    layer_number=layer_number + layer_number_offset,
                    kv_channels=kv_channels,
                    self_attn_mask_type=self_attn_mask_type.name,
                    tp_size=parallel_state.get_tensor_model_parallel_world_size(),
                    params_dtype=torch.float32,  # dtype params are initialized in
                    get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
                    fuse_wgrad_accumulation=gradient_accumulation_fusion,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    seq_length=None,  # used for jit warmup
                    micro_batch_size=None,  # used for jit warmup
                    sequence_parallel=sequence_parallel,
                    apply_residual_connection_post_layernorm=False,
                    autocast_dtype=precision,
                    use_emha=use_emha,
                    ub_tp_comm_overlap=ub_tp_comm_overlap,
                    zero_centered_gamma=normalization == 'layernorm1p',
                )
            else:
                return ESMnvParallelTransformerLayer(
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    layer_number=layer_number + layer_number_offset,
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    num_attention_heads=num_attention_heads,
                    apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                    kv_channels=kv_channels,
                    layer_type=lt,
                    self_attn_mask_type=self_attn_mask_type,
                    precision=precision,
                    fp32_residual_connection=fp32_residual_connection,
                    layernorm_epsilon=layernorm_epsilon,
                    hidden_dropout=hidden_dropout,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    use_cpu_initialization=use_cpu_initialization,
                    megatron_amp_O2=megatron_amp_O2,
                    bias_activation_fusion=bias_activation_fusion,
                    bias_dropout_add_fusion=bias_dropout_add_fusion,
                    masked_softmax_fusion=masked_softmax_fusion,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    persist_layer_norm=persist_layer_norm,
                    position_embedding_type=position_embedding_type,
                    openai_gelu=openai_gelu,
                    onnx_safe=onnx_safe,
                    activation=activation,
                    megatron_legacy=megatron_legacy,
                    bias=bias,
                    chunk_size=chunk_size,
                    normalization=normalization,
                    transformer_block_type=transformer_block_type,
                    headscale=headscale,
                    activations_checkpoint_granularity=activations_checkpoint_granularity,
                    sequence_parallel=sequence_parallel,
                    normalize_attention_scores=normalize_attention_scores,
                    num_moe_experts=num_moe_experts,
                    moe_frequency=moe_frequency,
                    moe_dropout=moe_dropout,
                    use_flash_attention=use_flash_attention,
                    # NEW BIONEMO ARGS
                    use_esm_attention=use_esm_attention,
                    esm_gelu=esm_gelu,
                    use_pt_layernorm=use_pt_layernorm,
                    use_pt_mlp_out=use_pt_mlp_out,
                )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            assert num_layers % parallel_state.get_virtual_pipeline_model_parallel_world_size() == 0, (
                'num_layers_per_stage must be divisible by ' 'virtual_pipeline_model_parallel_size'
            )

            assert self.model_type.value != 2, 'virtual pipeline parallel currently only supported for GPT'

            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // parallel_state.get_virtual_pipeline_model_parallel_world_size()
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = parallel_state.get_virtual_pipeline_model_parallel_rank() * (
                num_layers // parallel_state.get_virtual_pipeline_model_parallel_world_size()
            ) + (parallel_state.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if (
                self.model_type == ModelType.encoder_and_decoder
                and parallel_state.get_pipeline_model_parallel_world_size() > 1
            ):
                pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = parallel_state.get_pipeline_model_parallel_split_rank()
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = parallel_state.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.transformer_block_type != 'post_ln':
            # Final layer norm before output.
            if normalization == 'layernorm':
                self.final_layernorm = esm_get_layer_norm(
                    hidden_size,
                    layernorm_epsilon,
                    persist_layer_norm,
                    sequence_parallel=sequence_parallel,
                    use_pt_layernorm=use_pt_layernorm,
                )
            elif normalization == 'layernorm1p':
                self.final_layernorm = LayerNorm1P(
                    hidden_size, layernorm_epsilon, sequence_parallel_enabled=sequence_parallel
                )
            elif normalization == 'low_precision_layernorm':
                self.final_layernorm = LPLayerNorm(hidden_size, layernorm_epsilon)
            else:
                self.final_layernorm = MixedFusedRMSNorm(hidden_size, layernorm_epsilon)
            # for architectures such as MPT, there is no bias term even on the layernorms
            # this code allows us to remove the bias terms from the layernorm module
            # so that we can support MPT. However, certain apex-based LNs don't support
            # removing bias, so we also have to check for that
            if not bias and normalization not in ['layernorm', 'layernorm1p']:
                remove_bias_from_layernorm(self.final_layernorm)
