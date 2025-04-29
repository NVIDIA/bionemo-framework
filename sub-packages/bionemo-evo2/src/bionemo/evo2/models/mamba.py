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

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Type

import torch
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

# Import original MCoreMambaModel for subclassing
from megatron.core.models.mamba import MambaModel as MCoreMambaModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import init_method_normal
from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import make_upper_case, reweighted_cross_entropy
from nemo.collections.llm.gpt.model.ssm import (
    SSMConfig,
)
from nemo.lightning import get_vocab_size


def mamba_forward_step(model, batch) -> torch.Tensor:
    """Forward step function for Mamba models, similar to hyena_forward_step.

    Args:
        model: The Mamba model
        batch: Dictionary containing input batch data

    Returns:
        torch.Tensor: Output from the model forward pass
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
        "loss_mask": batch["loss_mask"],
    }
    forward_args["attention_mask"] = None
    return model(**forward_args)


class MambaModel(GPTModel):
    """Mamba model that extends GPTModel for integration with NeMo.

    Note that the loss calculation is handled by CustomMCoreMambaModel instead.
    """

    def get_inference_wrapper(self, params_dtype, inference_batch_times_seqlen_threshold) -> torch.Tensor:
        """Gets the inference wrapper for the Mamba model."""
        from megatron.core.models.mamba import MambaModel as MCoreMambaModel

        # Find MCoreMambaModel instance
        mcore_model = self.module
        while mcore_model:
            if isinstance(mcore_model, (MCoreMambaModel, Evo2StyleMCoreMambaModel)):
                break
            mcore_model = getattr(mcore_model, "module", None)
        if mcore_model is None or not isinstance(mcore_model, (MCoreMambaModel, Evo2StyleMCoreMambaModel)):
            raise ValueError("Mamba model instance not found in the model structure.")

        vocab_size = None
        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        else:
            raise ValueError("Unable to find vocab size.")

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
        )

        model_inference_wrapper = GPTInferenceWrapper(mcore_model, inference_wrapper_config)
        return model_inference_wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Forward pass that delegates to CustomMCoreMambaModel, which handles loss calculation."""
        extra_kwargs = {"packed_seq_params": packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,  # Pass labels to the Megatron module
            inference_params=inference_params,
            loss_mask=loss_mask,  # Pass loss_mask to the Megatron module
            **extra_kwargs,
        )

        # Return whatever CustomMCoreMambaModel.forward returns
        # (logits during inference, loss during training)
        return output_tensor


# Custom MCoreMambaModel with reweighted loss calculation
class Evo2StyleMCoreMambaModel(MCoreMambaModel):
    """Custom version of MCoreMambaModel that implements reweighted loss calculation.

    Note that this is similar to the HyenaModel for uppercase/lowercase handling.
    """

    def __init__(
        self,
        config: TransformerConfig,
        mamba_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        hybrid_attention_ratio: float = 0.0,
        hybrid_mlp_ratio: float = 0.0,
        hybrid_override_pattern: str | None = None,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        # Mamba with no attention has no need for position embeddings, so none is default
        position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        lowercase_loss_reweighting: float = 1.0,
        to_upper: str = "normalized_weighted",
        spike_no_more_embedding_init: bool = False,
        layernorm_embeddings: bool = False,
    ):
        """Ingest the config and create a CustomMCoreMambaModel instance.

        Args:
            config: TransformerConfig
            mamba_stack_spec: ModuleSpec
            vocab_size: int
            max_sequence_length: int
            pre_process: bool. Defaults to True.
            hybrid_attention_ratio: float. Defaults to 0.0.
            hybrid_mlp_ratio: float. Defaults to 0.0.
            hybrid_override_pattern: Override pattern for layer order in the mamba stack. Defaults to None.
            post_process: Apply post processing to the output. Defaults to True.
            fp16_lm_cross_entropy: Use the fp16 version of the loss calculation. Defaults to False.
            parallel_output: Keep output in sequence parallel. Defaults to True.
            share_embeddings_and_output_weights: Share weights between embedding and output layer. Defaults to False.
            position_embedding_type: If you want ROPE etc embeddings set to something other than "none".
                Defaults to "none".
            rotary_percent: Percent of the sequence length to use for the rotary embeddings. Defaults to 1.0.
            rotary_base: Base for the rotary embeddings. Defaults to 10000.
            scatter_embedding_sequence_parallel: Scatter the embedding to sequence parallel. Defaults to True.
            seq_len_interpolation_factor: Factor to interpolate the sequence length. Defaults to None.
            lowercase_loss_reweighting: Loss reweighting for lowercase characters. Defaults to 1.0.
            to_upper: How lowercase characters are handled in the loss calculation. Defaults to "normalized_weighted".
            spike_no_more_embedding_init: Initialize embeddings with sd=1 normal rather than the model default.
                Defaults to False.
            layernorm_embeddings: Apply layernorm to the embedding output as suggested in the spike no more paper.
                Defaults to False.
        """
        # Save any additional kwargs we might need
        self.lowercase_loss_reweighting = lowercase_loss_reweighting
        self.to_upper = to_upper
        # NOTE: the following code is copied from the MambaModel class in Megatron-LM's __init__
        #  This is so we can override the config specifically in the LanguageModelEmbedding.
        #  A better approach would be to make this a configuration in the parent class and call super().__init__
        super(MCoreMambaModel, self).__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.mamba_stack_spec: ModuleSpec = mamba_stack_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.hybrid_override_pattern = hybrid_override_pattern
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder
        self.layernorm_embeddings: bool = layernorm_embeddings
        if self.pre_process:
            if spike_no_more_embedding_init:
                embedding_config = deepcopy(self.config)
                embedding_config.init_method = init_method_normal(1.0)
            else:
                embedding_config = self.config
            self.embedding = LanguageModelEmbedding(
                config=embedding_config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
            )
            if self.layernorm_embeddings:
                self.post_embedding_norm = TENorm(
                    config=self.config,
                    hidden_size=config.hidden_size,
                    eps=config.layernorm_epsilon,
                )

        if self.position_embedding_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        self.decoder = build_module(
            mamba_stack_spec,
            self.config,
            pre_process=self.pre_process,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            post_process=self.post_process,
            dtype=config.params_dtype,
        )

        # Output
        if post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask=None,
        decoder_input=None,
        labels=None,
        loss_mask=None,
        inference_params=None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Forward pass with custom loss calculation for uppercase/lowercase reweighting.

        Note that this mimics the behavior in hyena_model.py lines 273-292.

        Forward function of the Mamba model. This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            if self.layernorm_embeddings:
                # TODO verify that this is correctly applied with Tensor +/- Sequence
                #   and Context parallel
                decoder_input = self.post_embedding_norm(decoder_input)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        rotary_pos_emb = None
        if self.position_embedding_type == "rope":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # The following assert will currently fail when running inference.
        # Commented out for now.
        # TODO (duncan/rwaleffe): (1) confirm that the externally-generated
        #   attention mask is not needed and is ignored by the model in
        #   inference mode, (2) reduce the size of the externally-generated
        #   attention mask to prevent CPU OOM (as we did for training), (3)
        #   force the attention mask passed to the model in inference mode to
        #   be None, so this assert will succeed.
        # assert attention_mask is None, "The attention mask is ignored and should be set to None"

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        # Apply reweighted loss calculation for uppercase/lowercase handling
        labels, lowercase_mask = make_upper_case(labels)
        loss = self.compute_language_model_loss(labels, logits)
        normalize_per_batch = True if self.to_upper == "normalized_weighted" else False
        loss = reweighted_cross_entropy(
            loss,
            (labels, loss_mask, lowercase_mask),
            lowercase_weight=self.lowercase_loss_reweighting,
            normalize_per_batch=normalize_per_batch,
        )
        return loss


def mamba_no_weight_decay_cond(name, param, exclude_embeddings: bool = False):
    """Condition for no weight decay for Mamba parameters.

    Note that this follows the same pattern as in the original Mamba implementation.
    """
    # Mamba-specific parameters that should not have weight decay
    if (
        name.endswith("dt_bias")
        or name.endswith("A_log")
        or name.endswith("D")
        or ("embedding" in name and exclude_embeddings)
        or getattr(param, "_no_weight_decay", False)
    ):
        no_wd = True
    # All other parameters - use default MCore behavior:
    # Do not regularize biases and norm parameters
    # (See megatron.core.optimizer._get_pram_groups)
    # TODO exclude embeddings
    else:
        no_wd = name.endswith(".bias") or len(param.shape) == 1
    return no_wd


def mamba_no_weight_decay_cond_with_embeddings(name, param):
    """Condition for no weight decay for Mamba parameters with embeddings.

    Note that this follows the same pattern as in the original Mamba implementation but also skips WD on embeddings.
    """
    return mamba_no_weight_decay_cond(name, param, exclude_embeddings=True)


@dataclass
class HybridMambaConfig8BEvo2Loss(SSMConfig):
    """Config for 8B hybrid Mamba model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    num_layers: int = 52
    seq_length: int = 8192
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    ffn_hidden_size: int = 21504
    num_attention_heads: int = 32
    init_method_std: float = 0.014
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    tokenizer_library: str = "byte-level"  # Use Evo2 tokenizer
    tokenizer_name: str = None
    mapping_type: str = "nvidia-hybrid-mamba"
    masked_softmax_fusion: bool = True
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    attention_softmax_in_fp32: bool = False
    vocab_size: int = 512
    first_last_layers_bf16: bool = True
    is_hybrid_model: bool = True
    forward_step_fn: Callable = mamba_forward_step
    data_step_fn: Callable = gpt_data_step
    # Set a reasonable default for to_upper to match HyenaModel behavior
    to_upper: str = "normalized_weighted"
    # Set lowercase loss reweighting factor
    lowercase_loss_reweighting: float = 1.0
    activation_func: Callable = lambda x: torch.square(F.relu(x))  # lambda x: torch.pow(F.relu(x), 2)
    # The trainer is responsible for using this when initializing the optimizer state:
    #  opt = MegatronOptimizerModule(opt_config, sched, no_weight_decay_cond=model_config.hyena_no_weight_decay_cond_fn)
    hyena_no_weight_decay_cond_fn: Callable = mamba_no_weight_decay_cond
    spike_no_more_embedding_init: bool = False
    layernorm_embeddings: bool = False

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "Evo2StyleMCoreMambaModel":
        """Override the configure_model method to properly configure a CustomMCoreMambaModel with Evo2 style loss.

        Args:
            tokenizer: Tokenizer to use with the model
            pre_process: Whether to include pre-processing in the model
            post_process: Whether to include post-processing in the model

        Returns:
            CustomMCoreMambaModel: Configured custom Mamba model instance
        """
        from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec

        # Set additional attributes that may be used during model initialization

        # Return our custom MCoreMambaModel with reweighted loss calculation
        return Evo2StyleMCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
            post_process=post_process or parallel_state.is_pipeline_last_stage(),
            # Pass our custom parameters for loss calculation
            to_upper=self.to_upper,
            lowercase_loss_reweighting=self.lowercase_loss_reweighting,
            spike_no_more_embedding_init=self.spike_no_more_embedding_init,
            layernorm_embeddings=self.layernorm_embeddings,
        )


# Dictionary mapping model size names to config classes
MAMBA_MODEL_OPTIONS: dict[str, Type[SSMConfig]] = {
    "hybrid_mamba_8b": HybridMambaConfig8BEvo2Loss,
}
