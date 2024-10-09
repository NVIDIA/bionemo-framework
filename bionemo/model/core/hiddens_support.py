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


# NOTE: This module contains all the hiddens module related extentions.
# NOTE: This module should be imported by any model which uses the hiddens module.

import inspect
import math

import torch
from megatron.core import ModelParallelConfig, tensor_parallel
from nemo.collections.nlp.modules.common.megatron.hiddens import MegatronBaseHiddenTransform, register_hidden_transform
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal
from torch.distributions.uniform import Uniform


__all__ = ["SampledVarGaussianHiddenTransform", "InterpVarGaussianHiddenTransform"]

###########################################################################
## Hidden Transforms
###########################################################################


class ColumnParallelMLP(torch.nn.Module):
    """
    A simple multi-layer perceptron (MLP) with column parallelism.
    """

    def __init__(
        self,
        layer_sizes,
        gather_output=True,
        init_method_std=0.02,
        bias=True,
        config: ModelParallelConfig = None,
        activation=torch.nn.ReLU,
    ) -> None:
        super().__init__()
        # skip_bias_add=False,
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements")

        layers = []
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(
                tensor_parallel.ColumnParallelLinear(
                    input_size,
                    output_size,
                    gather_output=gather_output,
                    init_method=init_method_normal(init_method_std),
                    skip_bias_add=False,
                    bias=bias,
                    config=config,
                )
            )
            layers.append(activation())

        layers.pop()  # remove last activation

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # ColumnLinear returns output and bias, we ignore bias here (already added to output)
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]

        return x


class SampledVarGaussianHiddenTransform(MegatronBaseHiddenTransform):
    """
    Constructs a diagonal Gaussian distribution from the hidden states and samples from it using reparametrization.
    The variance is sampled during training, and is set to min_logvar during inference.
    Following MolMIM paper <https://arxiv.org/abs/2208.09016>
    """

    __NAME__ = "sampled_var_cond_gaussian"

    def __init__(
        self,
        hidden_size,
        ffn_hidden_size=None,
        min_logvar=-6,
        max_logvar=0,
        map_var_to_hiddens=True,
        init_method_std=0.02,
        name=__NAME__,
        model_parallel_cfg: ModelParallelConfig = None,
    ):
        super().__init__(name=name, model_parallel_cfg=model_parallel_cfg)
        # limit smaller allowed variance (for numerical stability)
        self.min_logvar = min_logvar
        # limit larger allowed variance (during training)
        self.max_logvar = max_logvar
        # map logvar to hiddens to condition z on the logvar value
        self.map_var_to_hiddens = map_var_to_hiddens
        self.hidden_size = hidden_size
        if ffn_hidden_size is None:
            ffn_hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size

        self.logvar_sampler = Uniform(self.min_logvar, self.max_logvar)

        # project logvar and hiddens to mean to condition z on the logvar value
        if map_var_to_hiddens:
            self.logvar_hiddens_to_mean = ColumnParallelMLP(
                [hidden_size + 1, hidden_size + 1, ffn_hidden_size],
                gather_output=True,
                init_method_std=init_method_std,
                bias=True,
                config=self.model_parallel_cfg,
            )
        else:
            # project hiddens to mean and log variance (support tensor parallelism)
            self.hiddens_to_mean = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                ffn_hidden_size,  # NOTE: When using *glu, divide ffn dim by 2/3 to keep overall params the same.
                gather_output=True,
                init_method=init_method_normal(init_method_std),
                skip_bias_add=False,
                bias=True,
                config=self.model_parallel_cfg,
            )

    @property
    def input_names(self):
        """
        Provide here all required inputs
        """
        return ["hiddens", "hiddens_mask"]

    @property
    def output_names(self):
        """
        Provide here all generated outputs
        """
        return ["z_mean", "z_logvar", "z", "z_log_prob"]

    def _transform(self, inputs, batch_data=None):
        """
        We expect here shapes to be [S x B x H] for Sequence, Batch, Hidden sizes (due to tensor parallel support).

        inputs:
            hiddens: accepts a tensor of shape [S x B x H]

        outputs:
            z: a sample from Gaussian a tensor of shape [S x B x H]
            z_mean: mean of Gaussian a tensor of shape [S x B x H]
            z_logvar: log variance of Gaussian a tensor of shape [S x B x H]
            z_log_prob: log probability of z over posterior log q(z|x) a tensor of shape [S x B x H]
        """
        hiddens = inputs["hiddens"]
        # S - sequence length, B - batch size
        S, B, _ = hiddens.shape
        # compute distribution's parameters (or use cached ones)
        if "z_mean" in inputs and "z_logvar" in inputs:
            z_mean = inputs["z_mean"]
            z_logvar = inputs["z_logvar"]
            # clamp logvar (for numerical stability)
            z_logvar = z_logvar.clamp(min=self.min_logvar)
        else:
            # sample logvar during training
            if self.training:
                # 1 x B x 1 (broadcastable to S x B x H), each element in the batch will have its own logvar per step.
                z_logvar = self.logvar_sampler.sample((1, B, 1)).to(hiddens).expand(S, B, 1)
            else:
                # use minimum logvar during inference
                z_logvar = torch.empty(S, B, 1).fill_(self.min_logvar).to(hiddens)

            # condition posterior on logvar
            if self.map_var_to_hiddens:
                # ColumnLinear returns output and bias, we ignore bias here (already added to hiddens)
                z_mean = self.logvar_hiddens_to_mean(torch.cat([hiddens, z_logvar], dim=-1))
            else:
                # ColumnLinear returns output and bias, we ignore bias here (already added to hiddens)
                z_mean, _ = self.hiddens_to_mean(hiddens)

            # expand z_logvar to match z_mean
            z_logvar = z_logvar.expand_as(z_mean)

        # sample z with reparametrization (or use cached one)
        if "z" in inputs:
            z = inputs["z"]
            z_log_prob = inputs.get("z_log_prob", None)
        else:
            e = torch.randn_like(z_mean)
            z = (z_logvar * 0.5).exp() * e + z_mean
            z_log_prob = None

        if z_log_prob is None:
            # compute log probability of z under a diagonal Gaussian distribution
            z_log_prob = -0.5 * (math.log(2 * math.pi) + z_logvar + (z - z_mean).pow(2) / z_logvar.exp())

        return {
            "z": z,  # [S x B x H]
            "z_mean": z_mean,  # [S x B x H]
            "z_logvar": z_logvar,  # [S x B x H]
            "z_log_prob": z_log_prob,  # [S x B x H]
        }


class InterpVarGaussianHiddenTransform(MegatronBaseHiddenTransform):
    """
    Constructs a diagonal Gaussian distribution from the hidden states and samples from it using reparametrization.
    The variance and mean are sampled during training by interpolating between a Normal and the posterior,
    and is set to min_logvar during inference.
    Following MolMIM paper <https://arxiv.org/abs/2208.09016>
    """

    __NAME__ = "interp_var_cond_gaussian"

    def __init__(
        self,
        hidden_size,
        ffn_hidden_size=None,
        min_logvar=-6,
        map_coef_to_hiddens=True,
        init_method_std=0.02,
        name=__NAME__,
        model_parallel_cfg: ModelParallelConfig = None,
    ):
        super().__init__(name=name, model_parallel_cfg=model_parallel_cfg)
        # limit smaller allowed variance (for numerical stability)
        self.min_logvar = min_logvar
        # map logvar to hiddens to condition z on the logvar value
        self.map_coef_to_hiddens = map_coef_to_hiddens
        self.hidden_size = hidden_size
        if ffn_hidden_size is None:
            ffn_hidden_size = hidden_size * 2
        self.ffn_hidden_size = ffn_hidden_size

        self.posterior_coef_sampler = Uniform(0, 1)

        # project logvar and hiddens to mean to condition z on the logvar value
        if map_coef_to_hiddens:
            self.posterior_coef_hiddens_to_mean_logvar = ColumnParallelMLP(
                [hidden_size + 1, hidden_size + 1, ffn_hidden_size],
                gather_output=True,
                init_method_std=init_method_std,
                bias=True,
                config=self.model_parallel_cfg,
            )
        else:
            # project hiddens to mean and log variance (support tensor parallelism)
            self.hiddens_to_mean_logvar = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                ffn_hidden_size,  # NOTE: When using *glu, divide ffn dim by 2/3 to keep overall params the same.
                gather_output=True,
                init_method=init_method_normal(init_method_std),
                skip_bias_add=False,
                bias=True,
                config=self.model_parallel_cfg,
            )

    @property
    def input_names(self):
        """
        Provide here all required inputs
        """
        return ["hiddens", "hiddens_mask"]

    @property
    def output_names(self):
        """
        Provide here all generated outputs
        """
        return ["z_mean", "z_logvar", "z", "z_log_prob"]

    def _transform(self, inputs, batch_data=None):
        """
        We expect here shapes to be [S x B x H] for Sequence, Batch, Hidden sizes (due to tensor parallel support).

        inputs:
            hiddens: accepts a tensor of shape [S x B x H]

        outputs:
            z: a sample from Gaussian a tensor of shape [S x B x H]
            z_mean: mean of Gaussian a tensor of shape [S x B x H]
            z_logvar: log variance of Gaussian a tensor of shape [S x B x H]
            z_log_prob: log probability of z over posterior log q(z|x) a tensor of shape [S x B x H]
        """
        hiddens = inputs["hiddens"]
        # S - sequence length, B - batch size
        S, B, _ = hiddens.shape
        # compute distribution's parameters (or use cached ones)
        if "z_mean" in inputs and "z_logvar" in inputs:
            z_mean = inputs["z_mean"]
            z_logvar = inputs["z_logvar"]
            # clamp logvar (for numerical stability)
            z_logvar = z_logvar.clamp(min=self.min_logvar)
        else:
            # sample logvar during training
            if self.training:
                # 1 x B x 1 (broadcastable to S x B x H), each element in the batch will have its own logvar per step.
                posterior_coef = self.posterior_coef_sampler.sample((1, B, 1)).to(hiddens).expand(S, B, 1)
            else:
                # use interpolation coef of 1 during training, this puts everything into the prediction
                posterior_coef = torch.ones(S, B, 1, device=hiddens.device, dtype=hiddens.dtype)
            # condition posterior on logvar
            if self.map_coef_to_hiddens:
                # ColumnLinear returns output and bias, we ignore bias here (already added to hiddens)
                z_mean, z_logvar = self.posterior_coef_hiddens_to_mean_logvar(
                    torch.cat([hiddens, posterior_coef], dim=-1)
                ).chunk(2, dim=-1)
            else:
                # ColumnLinear returns output and bias, we ignore bias here (already added to hiddens)
                z_mean, z_logvar = self.hiddens_to_mean_logvar(hiddens)[0].chunk(2, dim=-1)

            # expand posterior_coef to match z_mean and z_logvar
            posterior_coef = posterior_coef.expand_as(z_mean)

            # interpolate between Normal prior and posterior for final posterior
            #  e^0 = 1, we want variance to be 1 for the unit normal we're interpolating with
            z_logvar = z_logvar * posterior_coef + (1 - posterior_coef) * torch.zeros_like(z_logvar)
            z_logvar = z_logvar.clamp(min=self.min_logvar)

            z_mean = z_mean * posterior_coef + (1 - posterior_coef) * torch.zeros_like(z_mean)

        # sample z with reparametrization (or use cached one)
        if "z" in inputs:
            z = inputs["z"]
            z_log_prob = inputs.get("z_log_prob", None)
        else:
            e = torch.randn_like(z_mean)
            z = (z_logvar * 0.5).exp() * e + z_mean
            z_log_prob = None

        if z_log_prob is None:
            # compute log probability of z under a diagonal Gaussian distribution
            z_log_prob = -0.5 * (math.log(2 * math.pi) + z_logvar + (z - z_mean).pow(2) / z_logvar.exp())

        return {
            "z": z,  # [S x B x H]
            "z_mean": z_mean,  # [S x B x H]
            "z_logvar": z_logvar,  # [S x B x H]
            "z_log_prob": z_log_prob,  # [S x B x H]
        }


###########################################################################
## register all hidden transforms and losses
###########################################################################
for hidden_transform in [SampledVarGaussianHiddenTransform, InterpVarGaussianHiddenTransform]:
    register_hidden_transform(
        hidden_transform.__NAME__,
        inspect.getmodule(hidden_transform).__name__ + "." + hidden_transform.__qualname__,
    )

# NOTE: to register all hidden losses use: register_hidden_loss("loss_name", inspect.getmodule(LossClass).__name__ + '.' + LossClass.__qualname__)
