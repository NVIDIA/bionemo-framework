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

import contextlib
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable

import megatron.core.models.gpt.gpt_model
import torch
from megatron.core import parallel_state
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from nemo.collections import llm
from nemo.collections.llm.gpt.model.base import GPTModel, get_packed_seq_params, mtp_block_spec
from nemo.collections.llm.gpt.model.llama import Llama3Config, apply_rope_scaling
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import make_upper_case, reweighted_cross_entropy
from nemo.lightning import get_vocab_size
from nemo.utils.import_utils import safe_import
from typing_extensions import override

from bionemo.evo2.utils.loss.embedding_variance import SquaredErrorTargetedVarianceLoss


_, HAVE_TE = safe_import("transformer_engine")

# Gradient accumulation fusion may be enabled if available, for more information see:
# https://github.com/NVIDIA/Megatron-LM/blob/01945b98d1ea3a2acb5e8301e181a328104f4856/megatron/core/tensor_parallel/layers.py#L575
# TODO: Clean this up with a getter and install instructions
_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda  # noqa: F401  # pylint: disable=unused-import
except ImportError:
    _grad_accum_fusion_available = False

logger = logging.getLogger(__name__)


def evo2_gpt_forward_step(model, batch) -> torch.Tensor:
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
    if "attention_mask" not in batch:
        assert HAVE_TE, (
            "The dataloader did not provide an attention mask, however Transformer Engine was not detected. \
            This requires Transformer Engine's implementation of fused or flash attention."
        )
    else:
        forward_args["attention_mask"] = batch["attention_mask"]

    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)

    return model(**forward_args)


class Evo2GPTModel(GPTModel):
    """Mamba model that extends GPTModel for integration with NeMo.

    Note that the loss calculation is handled by CustomMCoreMambaModel instead.
    """

    @override
    def get_inference_wrapper(
        self, params_dtype, inference_batch_times_seqlen_threshold, inference_max_seq_length=8192
    ) -> GPTInferenceWrapper:
        """Gets the inference wrapper for the Mamba model."""
        model = self
        while model is not None:
            if getattr(model, "module", None) is not None:
                model = model.module
            else:
                break
        if not isinstance(model, megatron.core.models.gpt.gpt_model.GPTModel):
            raise ValueError("GPT model instance not found in the model structure.")

        vocab_size = None
        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        else:
            raise ValueError("Unable to find vocab size.")

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
            inference_max_seq_length=inference_max_seq_length,
        )

        model_inference_wrapper = GPTInferenceWrapper(model, inference_wrapper_config)
        return model_inference_wrapper

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        decoder_input: torch.Tensor | None = None,
        inference_context=None,
        packed_seq_params=None,
        inference_params=None,
        runtime_gather_output: bool | None = None,
        loss_mask: torch.Tensor | None = None,
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
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            loss_mask=loss_mask,  # Pass loss_mask to the Megatron module
            **extra_kwargs,
        )

        # Return whatever CustomMCoreMambaModel.forward returns
        # (logits during inference, loss during training)
        return output_tensor


# Custom MCoreMambaModel with reweighted loss calculation
class Evo2StyleMCoreGPTModel(megatron.core.models.gpt.gpt_model.GPTModel):
    """Custom version of MCoreMambaModel that implements reweighted loss calculation.

    Note that this is similar to the HyenaModel for uppercase/lowercase handling.
    """

    def __init__(self, *args, **kwargs):
        """Initializes `Evo2StyleMCoreMambaModel` with unique parameters for the Evo2 variant of `MCoreMambaModel`."""
        super().__init__(*args, **kwargs)
        if self.config.use_targeted_variance_loss:
            if not hasattr(self.config, "embedding_init_method_std"):
                logger.warning("embedding_init_method_std is not supported in this config, please upgrade Megatron-LM")
            # 1.0 is the suggested value for embedding_init_method_std from the
            # [Spike No More](https://arxiv.org/abs/2312.16903) paper.
            embedding_init_method_std: float = getattr(self.config, "embedding_init_method_std", 1.0)
            self.targeted_variance_loss = SquaredErrorTargetedVarianceLoss(
                loss_coeff=self.config.targeted_variance_loss_loss_coeff,
                var_target=embedding_init_method_std**2,
            )

    @override
    def forward(self, *args, labels: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None, **kwargs):
        """Forward pass that delegates to Evo2StyleMCoreGPTModel, which handles loss calculation."""
        _forward_out = super().forward(*args, labels=labels, loss_mask=loss_mask, **kwargs)
        if labels is None or not self.post_process:
            # These are the two short-circuit cases in megatron.core.models.gpt.gpt_model.GPTModel.forward
            # 1. labels is None
            #  -> return the logits transposed to batch_size x seq_len x vocab_size
            # 2. not self.post_process
            #  -> return the hidden states.
            return _forward_out
        # Now that the above is false, we know that _forward_out is the loss, as in:
        # loss = self.compute_language_model_loss(labels, logits)
        loss = _forward_out

        labels, lowercase_mask = make_upper_case(labels)
        normalize_per_batch = True if self.config.to_upper == "normalized_weighted" else False
        loss = reweighted_cross_entropy(
            loss,
            (labels, loss_mask, lowercase_mask),
            lowercase_weight=self.config.lowercase_loss_reweighting,
            normalize_per_batch=normalize_per_batch,
        )
        if self.training and self.config.use_targeted_variance_loss:
            # Only use this in training, not validation etc.
            var_loss = self.targeted_variance_loss(self.embedding.word_embeddings.weight)
            loss += var_loss
        return loss


def gpt_no_weight_decay_cond(name, param, exclude_embeddings: bool = False):
    """Condition for no weight decay for Mamba parameters.

    Note that this follows the same pattern as in the original Mamba implementation.
    """
    # Mamba-specific parameters that should not have weight decay
    if ("embedding" in name and exclude_embeddings) or getattr(param, "_no_weight_decay", False):
        no_wd = True
    # All other parameters - use default MCore behavior:
    # Do not regularize biases and norm parameters
    # (See megatron.core.optimizer._get_pram_groups)
    # TODO exclude embeddings
    else:
        no_wd = name.endswith(".bias") or len(param.shape) == 1
    return no_wd


def gpt_no_weight_decay_cond_with_embeddings(name, param):
    """Condition for no weight decay for Mamba parameters with embeddings.

    Note that this follows the same pattern as in the original Mamba implementation but also skips WD on embeddings.
    """
    return gpt_no_weight_decay_cond(name, param, exclude_embeddings=True)


@dataclass
class LLama31ConfigEvoLoss3B(llm.Llama3Config8B):
    """Config for 8B hybrid Mamba model."""

    # RoPE/context length related block:
    rotary_base: int = 500_000
    seq_length: int = 8192
    old_context_len: int = 8192  # should be set/updated based on the loaded checkpoint's seq_length if fine-tuning.
    scale_factor: float = 1.0  # should be the ratio between the old context length and the new seq_length
    low_freq_factor: float = 1.0  # this factor can be left as is when extending the context length
    high_freq_factor: float = 4.0  # this factor can be left as is when extending the context length

    # vocab_size: int = 512
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    embedding_init_method_std: float = 1.0

    init_method_std: float = 0.02
    hyena_no_weight_decay_cond_fn: Callable = gpt_no_weight_decay_cond  # TODO rename to something more general
    forward_step_fn: Callable = evo2_gpt_forward_step

    layernorm_embeddings: bool = False
    # If set to true, use targeted variance loss which encourages the word embedding weight variances
    # to be close to a target value (1.0).
    share_embeddings_and_output_weights: bool = False
    use_targeted_variance_loss: bool = False
    targeted_variance_loss_loss_coeff: float = 0.1
    spike_no_more_embedding_init: bool = True
    to_upper: str = "normalized_weighted"
    lowercase_loss_reweighting: float = 0.1

    def __post_init__(self):
        """Post-init logic for Evo2 to enable backwards compatibility with old configs."""
        # Specific post_init logic for Evo2 to enable backwards compatibility with old configs.
        if not hasattr(self, "embedding_init_method_std"):
            raise ValueError("embedding_init_method_std is not supported in this config, please upgrade Megatron-LM")
        if self.spike_no_more_embedding_init and self.embedding_init_method_std is None:
            logger.warning(
                "spike_no_more_embedding_init is deprecated, please set "
                "embedding_init_method_std=[desired_stdev] in the future. To get the old behavior set to 1.0. "
                "For now setting to 1.0."
            )
            self.embedding_init_method_std = 1.0
        # Continue with the remaining post-init logic defined in NemotronHConfigBase and/or TransformerConfig.
        super().__post_init__()

    @override
    def configure_model(
        self, tokenizer, pre_process=None, post_process=None, vp_stage: int | None = None
    ) -> Evo2StyleMCoreGPTModel:
        """Configure and instantiate a Megatron Core Llama 3.1 model.

        Extends the base configuration with Llama 3.1 specific RoPE scaling.

        Args:
            tokenizer: Tokenizer used with the model
            pre_process: Whether to include pre-processing in the model
            post_process: Whether to include post-processing in the model
            vp_stage: Virtual pipeline parallel stage (or None if not using virtual pipeline parallelism)

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        if self.enable_cuda_graph:
            assert HAVE_TE, "Transformer Engine is required for cudagraphs."
            assert getattr(self, "use_te_rng_tracker", False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(
            self, "account_for_loss_in_pipeline_split", False
        )
        is_pipeline_asymmetric |= (
            getattr(self, "num_layers_in_first_pipeline_stage", None)
            or getattr(self, "num_layers_in_last_pipeline_stage", None)
        ) is not None
        is_flexible_pp_layout = is_pipeline_asymmetric or (
            getattr(self, "pipeline_model_parallel_layout", None) is not None
        )
        if vp_size and not is_flexible_pp_layout:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        import inspect

        # During fake lightning initialization, pass 0 to bypass the assertion that vp_stage must be
        # non-None when using virtual pipeline model parallelism
        vp_stage = vp_stage or 0

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            # Check if the transformer_layer_spec function accepts vp_stage parameter
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        if self.vocab_size is not None:
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logging.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)
        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(torch.device, device="meta")

        if "mtp_block_spec" in inspect.signature(Evo2StyleMCoreGPTModel.__init__).parameters:
            kwargs = {"mtp_block_spec": mtp_block_spec(self, vp_stage=vp_stage)}
        else:
            kwargs = {}
        with model_init_device_context():
            model = Evo2StyleMCoreGPTModel(
                self,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process
                or parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
                post_process=post_process
                or parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                vp_stage=vp_stage,
                **kwargs,
            )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if self.use_transformer_engine_full_layer_spec:
            raise ValueError("use_transformer_engine_full_layer_spec is not supported in this config.")

        # Apply rope scaling for Llama3.1 model
        model.rotary_pos_emb.inv_freq = apply_rope_scaling(
            model.rotary_pos_emb.inv_freq,
            factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            old_context_len=self.old_context_len,
        )
        return model


# Dictionary mapping model size names to config classes
GPT_MODEL_OPTIONS: dict[str, type[Llama3Config]] = {
    "llama3_8b": LLama31ConfigEvoLoss3B,
}
