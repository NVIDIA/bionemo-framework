# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.utils import get_batch_on_this_cp_rank
from nemo.collections.llm.gpt.model.hyena import HyenaConfig
from nemo.collections.llm.gpt.model.hyena import HyenaModel as ForwardHyenaModel
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_model import HyenaModel as CoreHyenaModel
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import (
    make_upper_case,
    reweighted_cross_entropy,
)
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging

from ..loss.borzoi import BorzoiLoss
from .debug import debug_heads


class ParallelHeadTransform(IOMixin):
    """Parallel Head Transformer.

    Adds RNA expression head to transformer models (e.g., Evo2/Hyena) for multi-task training
    with parallel heads (e.g., DNA and RNA loss).

    This transform allows the model to learn from both DNA sequence modeling and RNA expression
    prediction simultaneously, supporting flexible weighting and toggling of these objectives.

    Args:
        dna_loss_weight (float): Weight assigned to the DNA (language modeling) loss component.
        rna_loss_weight (float): Weight assigned to the RNA expression prediction loss component.
        parallel_dna (bool): Whether to use the DNA head during training/evaluation.
        parallel_rna (bool): Whether to use the RNA head during training/evaluation.

    Inherits from:
        IOMixin - Required by the training framework to enable model saving and serialization.

    Notes:
        - Can be used to add additional predictor heads in future, such as protein expression, CHIPseq, etc.
        - The HyenaModel is modified in-place to add the RNA head and adjust forward logic while respecting the BioNemo
          IOMixin structure.

        ┌───────────────────────────────┐
        │ PyTorch Lightning		        │  ← Interact with this in predict/training scripts
        │ - training_step()		        │
        │ - configure_optimizers()      │
        │ - Easy API		            │
        ├───────────────────────────────┤
        │ NeMo Wrappers		            │  ← Framework integration
        │ - Config management           │
        │ - Checkpointing               │
        ├───────────────────────────────┤
        │ Megatron-Core                 │  ← Power under the hood
        │ - Tensor Parallelism          │
        │ - Pipeline Parallelism        │
        │ - Context Parallelism         │
        ├───────────────────────────────┤
        │ Core Model                    │  ← Our actual model, what we modify below
        │ - Embedding                   │
        │ - Hyena Decoder               │
        │ - Output Layer                │
        └───────────────────────────────┘
    """

    def __init__(
        self,
        dna_loss_weight: float = 1.0,
        rna_loss_weight: float = 1.0,
        pep_loss_weight: float = 1.0,
        parallel_dna: bool = True,
        parallel_rna: bool = True,
        parallel_pep: bool = False,
        **kwargs,
    ):
        """Initializes the ParallelHeadTransform with specified loss weights and toggles.

        Args:
            dna_loss_weight (float): Weight for DNA language modeling loss.
            rna_loss_weight (float): Weight for RNA expression prediction loss.
            pep_loss_weight (float): Weight for peptide mapping loss.
            parallel_dna (bool): Whether to enable DNA head.
            parallel_rna (bool): Whether to enable RNA head.
            parallel_pep (bool): Whether to enable peptide mapping head.
            kwargs: Additional keyword arguments for loss functions.
        """
        # Store configurable loss weights and toggles
        self.dna_loss_weight = dna_loss_weight
        self.rna_loss_weight = rna_loss_weight
        self.pep_loss_weight = pep_loss_weight
        self.parallel_dna = parallel_dna
        self.parallel_rna = parallel_rna
        self.parallel_pep = parallel_pep

        # Setup custom loss functions if rna or pep parallelism is enabled
        if self.parallel_rna:
            # Use Borzoi loss for RNA head
            self.rna_loss_fn = BorzoiLoss(**kwargs)
        if self.parallel_pep:
            # Use Borzoi loss for peptide head
            self.pep_loss_fn = BorzoiLoss(**kwargs)
        # Log model transform initialization
        logging.info("🚀 Parallel Head Transform Initialized")

    # ============================================================================
    # Update model to include parallel heads and modified forward logic
    # ============================================================================

    def __call__(self, model: nn.Module) -> nn.Module:
        """Applies the transform to a given model in-place.

        Adds an RNA-seq head if not already present, and modifies the forward pass logic to incorporate both RNA and
        DNA losses (if enabled).

        Args:
            model (nn.Module): The model to be transformed.

        Returns:
            nn.Module: The modified model, augmented with parallel RNA head and updated forward logic.
        """
        # Log current model state for debugging
        debug_heads("Model pre transform", model)

        logging.info("🔧 Applying Enhanced ParallelHeadTransform")

        # Extract the main model components
        core_model = self._get_core_hyena_model(model)
        forward_target = self._get_forward_target_model(model)
        config = self._get_config(model)
        model = self._transform_core_model(model, core_model, forward_target, config)

        # Return the updated model
        return model

    def _transform_core_model(
        self, model: nn.Module, core_model: nn.Module, forward_target: nn.Module | None, config: HyenaConfig
    ) -> nn.Module:
        """Apply any core model specific transforms.

        Args:
            model (nn.Module): The overall model.
            core_model (nn.Module): The core HyenaModel instance.
            forward_target (nn.Module): The model used during forward pass.
            config (HyenaConfig): The model configuration.

        Returns:
            nn.Module: The transformed model with parallel heads.
        """
        # Use parallel_state to determine if input_is_parallel should be True
        input_is_parallel = parallel_state.is_pipeline_first_stage()

        # Add RNA seq head to core model
        if not hasattr(core_model, "rna_seq_head") and self.parallel_rna:
            # Create RowParallelLinear for RNA-seq head because output is small
            core_model.rna_seq_head = tensor_parallel.RowParallelLinear(  # ColumnParallelLinear, alternative
                config.hidden_size,  # Input dim: model hidden state size
                1,  # Output dim: one value per token for RNA expression
                config=config,
                init_method=config.init_method,  # type: ignore # Weight initialization strategy
                bias=config.add_bias_output,  # Whether to add bias
                skip_bias_add=False,
                # gather_output=True,  # Ensures output is gathered across TP ranks <== Needed for ColumnParallelLinear
                input_is_parallel=input_is_parallel,
            )
            # Initialize bias to 0 if bias is used
            if config.add_bias_output:
                core_model.rna_seq_head.bias.data.zero_()

        # Add pep map head to core model
        if not hasattr(core_model, "pep_map_head") and self.parallel_pep:
            # Create RowParallelLinear for pep map head because output is small
            core_model.pep_map_head = tensor_parallel.RowParallelLinear(  # ColumnParallelLinear, alternative
                config.hidden_size,  # Input dim: model hidden state size
                1,  # Output dim: one value per token for RNA expression
                config=config,
                init_method=config.init_method,  # type: ignore # Weight initialization strategy
                bias=config.add_bias_output,  # Whether to add bias
                skip_bias_add=False,
                # gather_output=True,  # Ensures output is gathered across TP ranks <== Needed for ColumnParallelLinear
                input_is_parallel=input_is_parallel,
            )
            # Initialize bias to 0 if bias is used
            if config.add_bias_output:
                core_model.pep_map_head.bias.data.zero_()

        # Set attributes on core model
        core_model.dna_loss_weight = self.dna_loss_weight  # type: ignore
        core_model.rna_loss_weight = self.rna_loss_weight  # type: ignore
        core_model.pep_loss_weight = self.pep_loss_weight  # type: ignore
        core_model.parallel_dna = self.parallel_dna  # type: ignore
        core_model.parallel_rna = self.parallel_rna  # type: ignore
        core_model.parallel_pep = self.parallel_pep  # type: ignore

        # Set up forward target
        if forward_target and forward_target != core_model:
            # Copy attributes to forward target
            forward_target.dna_loss_weight = self.dna_loss_weight  # type: ignore
            forward_target.rna_loss_weight = self.rna_loss_weight  # type: ignore
            forward_target.pep_loss_weight = self.pep_loss_weight  # type: ignore
            forward_target.parallel_dna = self.parallel_dna  # type: ignore
            forward_target.parallel_rna = self.parallel_rna  # type: ignore
            forward_target.parallel_pep = self.parallel_pep  # type: ignore
            # Store reference to core model on forward target
            forward_target._core_model = core_model
            if self.parallel_rna:
                forward_target.rna_seq_head = core_model.rna_seq_head
            if self.parallel_pep:
                forward_target.pep_map_head = core_model.pep_map_head

            # Override the forward pass logic with multi-head awareness
            if not hasattr(forward_target, "_original_forward"):
                forward_target._original_forward = forward_target.forward  # type: ignore
                forward_target.forward = self._create_parallel_forward(forward_target, core_model)  # type: ignore

        # Log updated model structure
        debug_heads("Model post transform", model)

        # Return the updated model
        return model

    def _get_core_hyena_model(self, model):
        """Get the core HyenaModel from possibly wrapped model."""
        hyena_models = self._discover_all_hyena_models(model)

        # Find the core model
        core_models = [info for info in hyena_models if info["is_core"]]
        if core_models:
            return core_models[0]["model"]

        if hyena_models:
            return hyena_models[0]["model"]

        raise ValueError("❌ No HyenaModel found")

    def _get_forward_target_model(self, model: nn.Module) -> Optional[nn.Module]:
        """Get the HyenaModel that will be called during forward step.

        This is typically a wrapper model around the core HyenaModel.

        Args:
            model (nn.Module): The overall model.

        Returns:( nn.Module or None): The model used during forward pass, or None if not found.
        """
        hyena_models = self._discover_all_hyena_models(model)

        # Based on your debug output, the forward step calls the wrapper at level 1
        # Look for wrapper models first
        wrapper_models = [info for info in hyena_models if info["is_wrapper"]]
        if wrapper_models:
            # Usually the first wrapper in the hierarchy is the forward target
            target = wrapper_models[0]
            logging.info(f"✅ Forward target is wrapper at level {target['level']}")
            return target["model"]

        # Fallback to core model
        core_models = [info for info in hyena_models if info["is_core"]]
        if core_models:
            target = core_models[0]
            logging.info(f"⚠️ Using core model as forward target at level {target['level']}")
            return target["model"]

        return None

    def _discover_all_hyena_models(self, model):
        """Unwrap and discover hyena models.

        Find ALL HyenaModel instances in the model hierarchy.

        ### Note
            - Models have a unique ID per GPU in multi-GPU systems.
        """
        hyena_models = []
        current_model = model
        unwrap_count = 0

        logging.info("🔍 Discovering HyenaModel instances...")

        while unwrap_count < 6:  # Increased safety limit
            logging.info(f"   Level {unwrap_count}: {type(current_model)} (id: {id(current_model)})")

            # Check if this is ANY kind of HyenaModel
            type_name = type(current_model).__name__
            if "HyenaModel" in type_name:
                model_info = {
                    "model": current_model,
                    "level": unwrap_count,
                    "type": type(current_model),
                    "id": id(current_model),
                    "has_embedding": hasattr(current_model, "embedding"),
                    "has_decoder": hasattr(current_model, "decoder"),
                    "has_output_layer": hasattr(current_model, "output_layer"),
                    "has_forward": hasattr(current_model, "forward"),
                }

                # Determine if this is core or wrapper
                is_core = model_info["has_embedding"] and model_info["has_decoder"] and model_info["has_output_layer"]
                model_info["is_core"] = is_core
                model_info["is_wrapper"] = not is_core

                hyena_models.append(model_info)
                logging.info(f"   ✅ Found {'Core' if is_core else 'Wrapper'} HyenaModel at level {unwrap_count}")
                logging.info(f"      Has core attributes: {is_core}")

            # Try to unwrap further
            if hasattr(current_model, "module"):
                current_model = current_model.module
                unwrap_count += 1
            else:
                logging.info(f"   ⏹️ Cannot unwrap further at level {unwrap_count}")
                break

        logging.info(f"🔍 Discovery complete: Found {len(hyena_models)} HyenaModel instances")
        for i, info in enumerate(hyena_models):
            logging.info(f"   {i}: {'Core' if info['is_core'] else 'Wrapper'} at level {info['level']}")

        return hyena_models

    def _get_config(self, model) -> HyenaConfig:
        """Get the config from model, trying different attribute names and wrapper levels."""
        # Try transformer_config first (preferred)
        if hasattr(model, "transformer_config"):
            logging.info("✅ Using model.transformer_config")
            return model.transformer_config

        # Try config
        if hasattr(model, "config"):
            logging.info("✅ Using model.config")
            return model.config

        # Try to find config in the hierarchy
        current = model
        level = 0
        while hasattr(current, "module") and level < 5:
            current = current.module
            if hasattr(current, "transformer_config"):
                logging.info(f"✅ Using transformer_config from level {level + 1}")
                return current.transformer_config
            if hasattr(current, "config"):
                logging.info(f"✅ Using config from level {level + 1}")
                return current.config
            level += 1

        raise AttributeError(f"❌ Model {type(model)} has no accessible config in hierarchy")

    def _create_parallel_forward(self, target_model: ForwardHyenaModel, core_model: CoreHyenaModel):
        """Creates a custom forward function for a model with parallel DNA and RNA heads.

        This method overrides the default forward pass of the model to support both:
        - Language modeling (DNA) via cross-entropy loss
        - RNA-seq regression via mean squared error loss

        The returned function supports training and inference modes, and dynamically
        adjusts based on config and input availability.

        Args:
            target_model (ForwardHyenaModel): The wrapper model whose `forward` method will be replaced.
            core_model (CoreHyenaModel): The underlying transformer model (e.g., Hyena) used for computation.

        Returns:
            Callable: A function that can be used as a `forward()` method with support for
                      parallel multi-task objectives (DNA and RNA).
        """
        config = self._get_config(target_model)

        def parallel_forward(
            input_ids: torch.Tensor,
            position_ids: None | torch.Tensor = None,
            attention_mask: None | torch.Tensor = None,
            decoder_input: None | torch.Tensor = None,
            labels: None | torch.Tensor = None,
            loss_mask: None | torch.Tensor = None,
            rna_mask: None | torch.Tensor = None,
            pep_mask: None | torch.Tensor = None,
            inference_context: None = None,
            runtime_gather_output: Optional[bool] = None,
            rna_seq_targets: None | torch.Tensor = None,
            pep_map_targets: None | torch.Tensor = None,
            **kwargs,
        ):
            # -------------------------------------------
            # SHARED PREPROCESSING (Embeddings + Decoder)
            # -------------------------------------------

            # Step 1: Token embedding
            if decoder_input is not None:
                # Use pre-computed embedding if provided
                pass
            elif config.pre_process:
                decoder_input = core_model.embedding(input_ids=input_ids, position_ids=position_ids)
            else:
                decoder_input = None  # Skip embedding if pre-process is disabled

            # Step 2: Apply Rotary Embedding (if configured)
            rotary_pos_emb = None
            if config.position_embedding_type == "rope":
                rotary_seq_len = core_model.rotary_pos_emb.get_rotary_seq_len(
                    inference_context,  # type: ignore
                    core_model.decoder,  # type: ignore
                    decoder_input,  # type: ignore
                    config,
                    None,  # type: ignore
                )
                rotary_pos_emb = core_model.rotary_pos_emb(rotary_seq_len)

            # Step 3: Pass through the decoder
            hidden_states = core_model.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
            )

            # If model is not post-processing (e.g., encoder-only), return decoder output
            if not config.post_process:
                return hidden_states

            # -------------------------------------------
            # PARALLEL HEAD COMPUTATION (DNA, RNA, etc)
            # -------------------------------------------

            # DNA laguage model head
            logits = None
            if target_model.parallel_dna:
                output_weight = None
                # Shared weights as embedding layer
                if (
                    hasattr(core_model, "share_embeddings_and_output_weights")
                    and core_model.share_embeddings_and_output_weights
                ):
                    output_weight = core_model.shared_embedding_or_output_weight()

                logits, _ = core_model.output_layer(hidden_states, weight=output_weight)

            # RNA seq head
            rna_seq_logits = None
            if target_model.parallel_rna:
                # Processed through core model
                rna_seq_logits, _ = core_model.rna_seq_head(hidden_states)  # type: ignore # Output dim: (batch, seq_len, 1)
                rna_seq_logits = rna_seq_logits.squeeze(-1)

            # PEP map head
            pep_map_logits = None
            if target_model.parallel_pep:
                # Processed through core model
                pep_map_logits, _ = core_model.pep_map_head(hidden_states)  # type: ignore
                pep_map_logits = pep_map_logits.squeeze(-1)

            # -------------------------------------------
            # INFERENCE MODE (no labels)
            # -------------------------------------------
            if labels is None:
                inference = {
                    "logits": logits.transpose(0, 1).contiguous() if logits is not None else None,
                    "rna_seq_logits": rna_seq_logits,
                    "pep_map_logits": pep_map_logits,
                }
                print(f"Inference :\n {inference}")
                return inference

            # -------------------------------------------
            # TRAINING MODE: LOSS COMPUTATION
            # -------------------------------------------
            total_loss = None

            # DNA Language Modeling Loss (cross-entropy)
            if target_model.parallel_dna and labels is not None and logits is not None:
                # Convert labels to handle uppercase/lowercase weighting
                labels, lowercase_mask = make_upper_case(labels)

                # Compute standard LM loss
                loss = core_model.compute_language_model_loss(labels, logits)

                # Normalize or reweight lowercase loss tokens
                normalize_per_batch = config.to_upper == "normalized_weighted"
                loss = reweighted_cross_entropy(
                    loss,
                    (labels, loss_mask, lowercase_mask),
                    lowercase_weight=0.1,
                    normalize_per_batch=normalize_per_batch,
                )

                # Weight and accumulate DNA loss
                loss *= target_model.dna_loss_weight  # type: ignore
                total_loss = loss if total_loss is None else total_loss + loss

            # RNA-Seq Loss (Borzoi Loss)
            if target_model.parallel_rna and rna_seq_logits is not None and rna_seq_targets is not None:
                # Ensure matching dtypes
                rna_seq_targets = rna_seq_targets.to(dtype=rna_seq_logits.dtype)

                # Align shape if transposed
                if rna_seq_logits.shape != rna_seq_targets.shape:
                    if (
                        rna_seq_logits.shape[0] == rna_seq_targets.shape[1]
                        and rna_seq_logits.shape[1] == rna_seq_targets.shape[0]
                    ):
                        rna_seq_logits = rna_seq_logits.transpose(0, 1)

                # Compute Borzoi loss
                rna_seq_loss = self.rna_loss_fn.compute(
                    predictions=rna_seq_logits, targets=rna_seq_targets, mask=rna_mask
                )

                # Weight and accumulate rna seq loss
                rna_seq_loss *= target_model.rna_loss_weight  # type: ignore
                total_loss = rna_seq_loss if total_loss is None else total_loss + rna_seq_loss

            # PEP-MAP Loss (Borzoi Loss)
            if target_model.parallel_pep and pep_map_logits is not None and pep_map_targets is not None:
                # Ensure matching dtypes
                pep_map_targets = pep_map_targets.to(dtype=pep_map_logits.dtype)

                # Align shape if transposed
                if pep_map_logits.shape != pep_map_targets.shape:
                    if (
                        pep_map_logits.shape[0] == pep_map_targets.shape[1]
                        and pep_map_logits.shape[1] == pep_map_targets.shape[0]
                    ):
                        pep_map_logits = pep_map_logits.transpose(0, 1)

                # Compute MSE loss (element-wise)
                pep_map_loss = self.pep_loss_fn.compute(
                    predictions=pep_map_logits, targets=pep_map_targets, mask=pep_mask
                )

                # Weight and accumulate pep map loss
                pep_map_loss *= target_model.pep_loss_weight  # type: ignore
                total_loss = pep_map_loss if total_loss is None else total_loss + pep_map_loss

            # Convert loss to correct dtype for model stability (bfloat16)
            if total_loss is not None:
                total_loss = total_loss.to(dtype=torch.bfloat16)

            return total_loss

        return parallel_forward


def parallel_head_forward_step_fn(model, batch: Dict[str, Any]) -> torch.Tensor:
    """Executes a forward pass through the model.

    This function is designed to be modular and compatible with models like Hyena
    or GPT-style transformer models that use standard input names (`input_ids`, `position_ids`, etc.),
    while also optionally handling RNA-seq targets if present. Modified to support parallel heads. Including:
        - DNA language modeling
        - RNA expression prediction
        - Peptide mapping prediction

    Args:
        model (torch.nn.Module): The model to be evaluated. Expected to support keyword arguments
                                like `input_ids`, `position_ids`, `attention_mask`, `labels`,
                                `loss_mask`, and optionally `rna_seq_targets`.
        batch (dict): A dictionary of input tensors for the forward pass. Must include:
            - tokens (torch.Tensor): Tokenized input sequences.
            - position_ids (torch.Tensor): Positional encodings.
            - attention_mask (torch.Tensor, optional): Attention mask for padding/masking.
            - labels (torch.Tensor): Ground truth labels for loss calculation.
            - loss_mask (torch.Tensor): Mask to indicate which token losses to consider.
            - rna_seq_targets (torch.Tensor, optional): Additional output targets for RNA prediction tasks.

    Returns:
        torch.Tensor: The model's output from the forward pass (e.g., loss or logits).
    """
    # Prepare forward args
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "labels": batch.get("labels", None),
        "loss_mask": batch.get("loss_mask", None),
        "rna_mask": batch.get("rna_mask", None),
        "pep_mask": batch.get("pep_mask", None),
    }

    # Safely add rna_seq_targets to forward args
    if "rna_seq_targets" in batch:
        forward_args["rna_seq_targets"] = batch.get("rna_seq_targets", None)

    # Safely add pep_map_targets to forward args
    if "pep_map_targets" in batch:
        forward_args["pep_map_targets"] = batch.get("pep_map_targets", None)

    # Safely add rna_mask if present
    if "rna_mask" in batch:
        forward_args["rna_mask"] = batch.get("rna_mask", None)

    # Safely add pep_mask if present
    if "pep_mask" in batch:
        forward_args["pep_mask"] = batch.get("pep_mask", None)

    # Perform the forward pass using keyword argument unpacking
    result = model(**forward_args)

    # Return model output (typically loss or prediction logits)
    return result


def parallel_head_data_step_fn(dataloader_iter, use_mtp=False) -> dict[str, torch.Tensor]:
    """Retrieve and process the next batch of data from a dataloader.

    Inner function used during training steps to retrieve and process the next batch of data from a dataloader. Supports
    modular tensor routing across pipeline stages. This function handles the data loading step for GPT models, managing
    pipeline parallelism by distributing data appropriately across pipeline stages.

    Args:
        dataloader_iter: Iterator over the dataloader
        use_mtp: Whether the Multi-Token Prediction Module is used. Input needs to be passed
                into the last pipeline stage if mtp is used.

    Returns:
        dict[str, torch.Tensor]: Processed batch with required tensors moved to appropriate devices

    Note:
        To add additional arguments to forward, make sure to add to `required_device_keys` by making the key value
        exists in `_batch`.
            - For example:
                ```
                if "rna_seq_targets" in _batch:
                    required_device_keys.add("rna_seq_targets")
                ```
    """
    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    # Fetch the next batch from the dataloader
    batch = next(dataloader_iter)

    # Handle both single-dict and (dict, _, _) tuple formats
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch  # type: ignore

    # Sets for determining which keys need to be transferred to which device
    required_device_keys = set()  # Tensors needed on GPU
    required_host_keys = set()  # Tensors needed on CPU

    # Add target keys if they exist
    if "rna_seq_targets" in _batch:
        required_device_keys.add("rna_seq_targets")
    if "pep_map_targets" in batch:
        required_device_keys.add("pep_map_targets")
    # Add masks if they exist
    if "rna_mask" in _batch:
        required_device_keys.add("rna_mask")
    if "pep_mask" in batch:
        required_device_keys.add("pep_mask")

    # cu_seqlens-related values needed for FlashAttention or similar ops
    required_device_keys.add("attention_mask")
    if "cu_seqlens" in _batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    # If we're in the first pipeline stage or using MTP, we need input tokens
    if parallel_state.is_pipeline_first_stage() or use_mtp:
        required_device_keys.update(("tokens", "position_ids"))
        # RNA head loss may be computed in the final stage
        if "rna_seq_targets" in _batch:
            required_device_keys.add("rna_seq_targets")  # TODO: Not sure if needed here...
        if "pep_map_targets" in batch:
            required_device_keys.add("pep_map_targets")  # TODO: Not sure if needed here...
        # Also add masks if they exist
        if "rna_mask" in _batch:
            required_device_keys.add("rna_mask")  # TODO: Not sure if needed here...
        if "pep_mask" in batch:
            required_device_keys.add("pep_mask")  # TODO: Not sure if needed here...

    # If we're in the last pipeline stage, we need output labels for loss computation
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))
        # RNA head loss may be computed in the final stage
        if "rna_seq_targets" in _batch:
            required_device_keys.add("rna_seq_targets")
        if "pep_map_targets" in batch:
            required_device_keys.add("pep_map_targets")
        # Also add masks if they exist
        if "rna_mask" in _batch:
            required_device_keys.add("rna_mask")
        if "pep_mask" in batch:
            required_device_keys.add("pep_mask")

    # Dictionary that will hold the appropriately placed tensors (CPU/GPU/None)
    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_cp_rank(_batch_required_keys)

    return output
