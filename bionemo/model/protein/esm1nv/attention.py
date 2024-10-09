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


from typing import Callable, Optional, Union

import torch
from einops import rearrange
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    InfusedAdapterConfig,
    LoraKQVAdapterConfig,
    LoraKQVAdapterWeightTyingConfig,
    LoraKVAdapterConfig,
    LoraQAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.position_embedding.rotary_position_embedding import (
    apply_rotary_pos_emb,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
)
from nemo.collections.nlp.parts import utils_funcs


try:
    from apex.transformer.enums import AttnMaskType, AttnType
    from apex.transformer.utils import divide as safe_divide

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.model_parallel_config import ModelParallelConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    from flash_attn.flash_attn_triton import flash_attn_func

    HAVE_FLASH_ATTENTION = True

except (ImportError, ModuleNotFoundError):
    HAVE_FLASH_ATTENTION = False

    flash_attn_unpadded_func, flash_attn_func = None, None
    unpad_input, pad_input = None, None

# BIONEMO imports
from nemo.collections.nlp.modules.common.megatron.attention import CoreAttention, ParallelAttention
from nemo.utils import logging


# END BIONEMO


# TODO(dorotat, georgea) Refactor these part to use directly megatron.core
class ESMnvCoreAttention(CoreAttention):
    """Region where selective activation recomputation is applied.
    See Figure 3. in Reducing Activation Recomputation in Large Transformer Models
    https://arxiv.org/pdf/2205.05198.pdf for more details.

    """

    def __init__(self, *args, use_esm_attention=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = False
        self.use_esm_attention = use_esm_attention
        if self.use_esm_attention:
            logging.warning("Using custom ESM2 attention instead of the default NeMo version")

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        rotary_pos_emb=None,
        relative_position_bias=None,
        headscale_tensor=None,
        inference_mode=None,
    ):
        # b, np, sq, sk, hn
        _, _, sq, sk, _ = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
            query_layer.size(3),
        )

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================
        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[..., sq - 1, :sk].unsqueeze(2)
                else:
                    attention_mask = attention_mask[..., :sq, :sk]

        # ==================================================
        # Update attention bias. [b, np, sq, sk]
        # ==================================================
        if relative_position_bias is not None:
            relative_position_bias = relative_position_bias[
                :,
                self.num_attention_heads_partition_offset : self.num_attention_heads_partition_offset
                + self.num_attention_heads_per_partition,
                -sq:,
                -sk:,
            ]

        # ==================================================
        # Update query_layer, key_layer, value_layer
        # ==================================================
        # BIONEMO custom attention normalization
        # TODO(srabhi, georgea): refactor the custom ESMnvCoreAttention module using Megatron Core when NeMo 1.21 is available
        if self.use_esm_attention:
            query_layer = query_layer * self.hidden_size_per_attention_head**-0.5
        # END BIONEMO
        # TODO: figure out how to do this
        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if self.position_embedding_type.lower() == "xpos":
            query_layer = self.xpos(query_layer, offset=key_layer.shape[-2] - query_layer.shape[-2], downscale=False)
            key_layer = self.xpos(key_layer, offset=0, downscale=True)

        # ==================================================
        # query_layer [sq, b, np, hn]
        # key_layer   [sk, b, np, hn]
        # value_layer [sk, b, np, hn]
        # attention_mask [b, 1, sq, sk] or [b, s]
        # relative_position_bias [b, np, sq, sk]
        # context_layer [b, np, sq, hn]
        # ==================================================
        context_layer = self.attn_fn(
            query_layer, key_layer, value_layer, attention_mask, relative_position_bias, inference_mode
        )

        if self.return_intermediate:
            context_layer, attention_probs, attention_scores = context_layer

        if headscale_tensor is not None:
            context_layer = context_layer * headscale_tensor

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.return_intermediate:
            return (
                context_layer,
                query_layer,
                key_layer,
                rotary_pos_emb,
                attention_probs,
                attention_scores,
            )
        return context_layer

    def esm2_scale_mask_softmax(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[Union[float, int]] = None,
        mask_func: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Scale Mask Softmax function.

        Args:
            input: Tensor of shape (Batch, NP, SK, SQ). The input may or may not have already
                had a mask applied to it.
            mask: If a mask is to be applied, it will go here.
            scale: A scale factor that will be applied before the softmax.
            mask_func: An optional function to apply to the mask. If None, it is assumed that
                the input already had the mask applied to it.

        Returns:
            probs: Tensor of normalized probabilities after the softmax has been applied,
                of shape (Batch, NP, SK, SQ).
        """
        if self.attn_mask_type.name != "padding":
            raise ValueError(
                f"self.attn_mask_type: {self.attn_mask_type} is not 'padding'. Only 'padding' type is supported currently."
            )

        original_dtype = input.dtype  # Store original dtype
        if (original_dtype == torch.float16 or original_dtype == torch.bfloat16) and self.attention_softmax_in_fp32:
            input = input.float()  # Convert to float32 for softmax

        if scale is not None:
            input = input * scale  # Apply scaling

        if mask is not None and mask_func is not None:
            input = mask_func(input, mask)  # Apply mask function if provided

        probs = torch.nn.functional.softmax(input, dim=-1)  # Apply softmax

        if self.attention_softmax_in_fp32 and original_dtype in (torch.float16, torch.bfloat16):
            probs = probs.to(original_dtype)  # Convert back to original dtype if necessary

        return probs

    def torch_attention(self, query_layer, key_layer, value_layer, attention_mask, attention_bias, inference_mode):
        sq, b, np, hn = query_layer.shape
        sk = key_layer.shape[0]

        if self.multi_query_attention:
            query_layer = rearrange(query_layer, "sq b np hn -> b (np sq) hn")
            key_layer = rearrange(key_layer, "sk b 1 hn -> b hn sk")
            value_layer = rearrange(value_layer, "sv b np hn -> (b np) sv hn")
        else:
            query_layer = rearrange(query_layer, "sq b np hn -> (b np) sq hn")
            key_layer = rearrange(key_layer, "sk b np hn -> (b np) hn sk")
            value_layer = rearrange(value_layer, "sv b np hn -> (b np) sv hn")

        matmul_input_buffer = torch.zeros(
            query_layer.shape[0],
            query_layer.shape[1],
            key_layer.shape[2],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,
            key_layer,
            beta=0.0,
            alpha=(1.0 / self.norm_factor) if self.normalize_attention_scores else 1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(b, np, sq, sk)
        # TODO(srabhi, georgea): refactor the custom torch_attention module using Megatron Core when NeMo 1.21 is available
        if self.use_esm_attention:
            # NOTE: the slicing here is to make the attention_mask the same shape as the extended
            #  attention mask in ESM2. The multiplication by -3.4028e+38 is similarly motivated
            #  by ESM2's maskikng approach, which forces softmax of attention scores for
            #  masked entries to be close to 0.
            attention_probs = self.esm2_scale_mask_softmax(
                attention_scores + attention_mask[:, :, 0:1, :] * -3.4028e38
            )
        else:
            if attention_bias is not None:
                attention_scores += attention_bias

            attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with tensor_parallel.random.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # change view [b * np, sq, sk]
        attention_probs = rearrange(attention_probs, "b np sq sk -> (b np) sq sk")

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = rearrange(context_layer, "(b np) sq hn -> b np sq hn", np=np)

        if self.return_intermediate:
            return context_layer, attention_probs, attention_scores

        return context_layer


class ESMnvParallelAttention(ParallelAttention):
    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        layer_number,
        num_attention_heads,
        hidden_size,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        precision=16,
        apply_query_key_layer_scaling=False,
        kv_channels=None,
        megatron_amp_O2=False,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        layer_type=None,
        megatron_legacy=False,
        bias=True,
        headscale=False,
        position_embedding_type="learned_absolute",
        multi_query_attention=False,
        normalize_attention_scores=True,
        use_flash_attention=False,
        # NEW BIONEMO ARGS
        use_esm_attention=False,
    ):
        super(ParallelAttention, self).__init__(config=config)
        self.return_intermediate = False
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.normalize_attention_scores = normalize_attention_scores
        self.position_embedding_type = position_embedding_type
        self.multi_query_attention = multi_query_attention
        self.use_flash_attention = use_flash_attention

        self.megatron_legacy = megatron_legacy
        self.dtype = utils_funcs.torch_dtype_from_precision(precision, megatron_amp_O2)

        self.set_accepted_adapter_types(
            [
                InfusedAdapterConfig._target_,
                LoraKQVAdapterConfig._target_,
                LoraQAdapterConfig._target_,
                LoraKVAdapterConfig._target_,
                LoraKQVAdapterWeightTyingConfig._target_,
            ]
        )

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), "hidden_size must be divisible by num_attention_heads if kv_channels is None"
            kv_channels = hidden_size // num_attention_heads
        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
            self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                3 * projection_size,
                config=config,
                gather_output=False,
                init_method=init_method,
                bias=bias,
            )
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                projection_size,
                config=config,
                gather_output=False,
                init_method=init_method,
                bias=bias,
            )

            self.key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                2 * projection_size,
                config=config,
                gather_output=False,
                init_method=init_method,
                bias=bias,
            )

        self.core_attention = ESMnvCoreAttention(
            config=config,
            layer_number=self.layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=self.attention_type,
            attn_mask_type=self.attn_mask_type,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            multi_query_attention=multi_query_attention,
            normalize_attention_scores=normalize_attention_scores,
            position_embedding_type=position_embedding_type,
            use_flash_attention=use_flash_attention,
            # NEW BIONEMO ARGS
            use_esm_attention=use_esm_attention,
        )

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            hidden_size,
            config=config,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            bias=bias,
        )

        self.headscale = headscale
        if headscale:
            self.head_scale_tensor = torch.nn.Parameter(
                torch.ones(1, self.num_attention_heads_per_partition, 1, 1), requires_grad=True
            )

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

        # relative position embedding
        self.layer_type = layer_type
