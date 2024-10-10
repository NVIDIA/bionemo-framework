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


import torch
import torch.nn.functional as F
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    MLPInfusedAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.fused_bias_geglu import fused_bias_geglu
from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.layer_norm_1p import LayerNorm1P
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    ApproxGELUActivation,
    erf_gelu,
    squared_relu,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    openai_gelu as openai_gelu_func,
)


try:
    from apex.normalization import MixedFusedRMSNorm
    from apex.transformer import tensor_parallel

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


try:
    from megatron.core import tensor_parallel
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.parallel_state import get_tensor_model_parallel_world_size

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

# BIONEMO Imports
# END BIONEMO
# BIONEMO: copy gelu function from esm
import math

from nemo.collections.nlp.modules.common.megatron.mlp import ParallelMLP
from nemo.utils import logging

from bionemo.model.protein.esm1nv.layernorm import esm_get_layer_norm


def esm_gelu_func(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# END BIONEMO


# TODO(dorotat, georgea) Refactor these part to use directly megatron.core
class ESMnvParallelMLP(ParallelMLP):
    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        hidden_size,
        ffn_hidden_size,
        dtype=torch.float32,
        bias_activation_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        activation="gelu",
        bias=True,
        transformer_block_type="pre_ln",
        normalization="layernorm",
        layernorm_epsilon=1e-5,
        persist_layer_norm=False,
        dropout=0.0,
        # BIONEMO ARGS
        esm_gelu=False,
        use_pt_layernorm=False,
        use_pt_mlp_out=False,
        # END BIONEMO
    ):
        # TODO(srabhi, georgea): refactor the custom ESMnvParallelMLP module using Megatron Core when NeMo 1.21 is available
        super(ParallelMLP, self).__init__(config=config)
        self.activation = activation
        self.bias = bias
        self.transformer_block_type = transformer_block_type
        self.normalization = normalization
        self.layernorm_epsilon = layernorm_epsilon
        self.persist_layer_norm = persist_layer_norm
        self.activation = activation
        self.dropout = dropout
        self.dtype = dtype
        self.set_accepted_adapter_types([MLPInfusedAdapterConfig._target_])
        self.esm_layer = use_pt_mlp_out

        supported_activations = [
            "gelu",
            "geglu",
            "reglu",
            "swiglu",
            "squared-relu",
            "fast-geglu",
            "fast-swiglu",
            "fast-reglu",
            "approx-gelu",
        ]

        if activation not in supported_activations:
            raise ValueError(
                f"Activation {activation} not supported. Supported activations are {supported_activations}"
            )

        self.fast_glu_activation = activation in ["fast-geglu", "fast-swiglu", "fast-reglu"]

        # Project to 4h.
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            (
                ffn_hidden_size * 2 if self.fast_glu_activation else ffn_hidden_size
            ),  # NOTE: When using geglu, divide ffn dim by 2/3 to keep overall params the same.
            config=config,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=bias,
        )

        if activation in ["geglu", "reglu", "swiglu"]:
            # Separate linear layer for *GLU activations.
            # Source: https://github.com/huggingface/transformers/blob/bee361c6f1f7704f8c688895f2f86f6e5ff84727/src/transformers/models/t5/modeling_t5.py#L292
            self.dense_h_to_4h_2 = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                ffn_hidden_size,  # NOTE: When using *glu, divide ffn dim by 2/3 to keep overall params the same.
                config=config,
                gather_output=False,
                init_method=init_method,
                skip_bias_add=True,
                bias=bias,
            )

        self.glu_activation_family = activation in [
            "geglu",
            "reglu",
            "swiglu",
            "fast-geglu",
            "fast-reglu",
            "fast-swiglu",
        ]
        bias_activation_fusion_unavailable = activation in ["reglu", "swiglu"]

        if bias_activation_fusion_unavailable and bias_activation_fusion:
            raise ValueError(
                f"Cannot use bias_activation_fusion with {activation} activation. Please turn bias gelu fusion off."
            )

        if self.glu_activation_family and onnx_safe and self.bias_activation_fusion:
            raise ValueError(
                f"Cannot use onnx_safe with specificed activation function and bias_activation_fusion : {activation} Please turn onnx safe off."
            )

        if bias_activation_fusion and not bias:
            raise ValueError(
                "Cannot use bias_activation_fusion without bias terms. Please set bias=True or bias_activation_fusion=False."
            )

        self.bias_activation_fusion = bias_activation_fusion

        # Give openai_gelu precedence over other activations if set, for HF compatibility. Normally this is off and shouldn't affect regular model training.
        if openai_gelu:
            self.activation_func = openai_gelu_func
        # BIONEMO allow esm gelu
        elif esm_gelu:
            logging.warning("Using custom ESM2 GELU function instead of the default NeMo version")
            self.activation_func = esm_gelu_func
        elif activation in ["gelu", "geglu", "fast-geglu"]:
            self.activation_func = F.gelu
        elif activation == "approx-gelu":
            self.activation_func = ApproxGELUActivation
        elif onnx_safe:
            self.activation_func = erf_gelu
        elif activation in ["reglu", "fast-reglu"]:
            self.activation_func = F.relu
        elif activation in ["swiglu", "fast-swiglu"]:
            # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
            self.activation_func = F.silu
        elif activation == "squared-relu":
            self.activation_func = squared_relu

        # Project back to h.
        if use_pt_mlp_out:
            logging.warning(
                "Using PyTorch Linear instead of the default NeMo RowParallelLinear for `dense_4h_to_h` module"
            )
            self.dense_4h_to_h = torch.nn.Linear(ffn_hidden_size, hidden_size, bias=bias, dtype=dtype)
        else:
            self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
                ffn_hidden_size,
                hidden_size,
                config=config,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=True,
                bias=bias,
            )

        # Normformer normalization
        if transformer_block_type == "normformer":
            if normalization == "layernorm":
                self.normalization = esm_get_layer_norm(
                    ffn_hidden_size // get_tensor_model_parallel_world_size(),
                    layernorm_epsilon,
                    persist_layer_norm,
                    use_pt_layernorm=use_pt_layernorm,
                )
            elif normalization == "layernorm1p":
                self.normalization = LayerNorm1P(
                    ffn_hidden_size // get_tensor_model_parallel_world_size(),
                    layernorm_epsilon,
                    sequence_parallel_enabled=config.sequence_parallel,
                )
            else:
                self.normalization = MixedFusedRMSNorm(
                    ffn_hidden_size // get_tensor_model_parallel_world_size(), layernorm_epsilon
                )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.fast_glu_activation:
            intermediate_parallel, intermediate_parallel_2 = torch.chunk(intermediate_parallel, 2, dim=-1)
            if bias_parallel is not None:
                bias_parallel, bias_parallel_2 = torch.chunk(bias_parallel, 2, dim=-1)
        elif self.glu_activation_family and not self.fast_glu_activation:
            intermediate_parallel_2, bias_parallel_2 = self.dense_h_to_4h_2(hidden_states)

        if self.bias_activation_fusion:
            if self.activation == "gelu":
                intermediate_parallel = fused_bias_gelu(intermediate_parallel, bias_parallel)
            elif self.activation in ["geglu", "fast-geglu"]:
                intermediate_parallel = fused_bias_geglu(
                    intermediate_parallel, bias_parallel, intermediate_parallel_2, bias_parallel_2
                )

        elif self.glu_activation_family and not self.bias_activation_fusion:
            if bias_parallel is not None:
                intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel) * (
                    intermediate_parallel_2 + bias_parallel_2
                )
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel) * intermediate_parallel_2

        else:
            if bias_parallel is not None:
                intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        if self.dropout > 0:
            intermediate_parallel = F.dropout(intermediate_parallel, p=self.dropout, training=self.training)

        infused_adapter = self.get_adapter_module(AdapterName.MLP_INFUSED)
        if infused_adapter:
            intermediate_parallel = infused_adapter(intermediate_parallel)

        # Normformer normalization
        if self.transformer_block_type == "normformer":
            intermediate_parallel = self.normalization(intermediate_parallel)

        # [s, b, h] BIONEMO NEW, think about: output, output_bias = super().forward(hidden_states) and the inf statement
        if self.esm_layer:
            output = self.dense_4h_to_h(intermediate_parallel)
            output_bias = torch.zeros(size=(hidden_states.shape[-1],)).to(output.device)
        else:
            output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias
