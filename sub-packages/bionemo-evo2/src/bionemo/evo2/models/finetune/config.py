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

"""Configuration classes for Evo2 fine-tuning."""

from dataclasses import dataclass, field
from typing import List, Optional, Type

import torch
from megatron.core.models.bert.bert_lm_head import BERTMLMLossWithReduction
from nemo import io as iom
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS

from bionemo.evo2.models.finetune.loss import (
    ClassifierLossReduction,
    RegressorLossReduction,
    TokenClassifierLossReduction,
)
from bionemo.evo2.models.finetune.sequence_model import (
    Evo2FineTuneSeqModel,
    MambaFineTuneSeqModel,
)
from bionemo.evo2.models.finetune.token_model import (
    Evo2FineTuneTokenModel,
    MambaFineTuneTokenModel,
)
from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS
from bionemo.llm.model.config import TorchmetricsConfig


@dataclass
class Evo2FineTuneSeqConfig(iom.IOMixinWithGettersSetters):
    """Configuration for sequence-level fine-tuning.

    This configuration class sets up the model, loss function, and training
    parameters for sequence-level regression or classification tasks.
    """

    # Model configuration
    model_type: str = "hyena"  # "hyena" or "mamba"
    model_size: str = "7b"
    model_cls: Optional[Type] = None  # Will be set based on model_type
    initial_ckpt_path: Optional[str] = None
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(
        default_factory=lambda: ["regression_head", "classification_head"]
    )

    # Task configuration
    task_type: str = "regression"  # "regression" or "classification"
    encoder_frozen: bool = True

    # MLP head parameters
    mlp_ft_dropout: float = 0.25
    mlp_hidden_size: int = 256
    mlp_target_size: int = 1  # For regression or number of classes for classification

    # Training parameters
    params_dtype: torch.dtype = torch.bfloat16
    pipeline_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1

    # Metrics
    train_metric: Optional[TorchmetricsConfig] = None
    valid_metric: Optional[TorchmetricsConfig] = None

    # Additional transformer config attributes needed
    hidden_size: int = field(init=False)
    ft_dropout: float = field(init=False)

    def __post_init__(self):
        """Post-initialization to set model class and parameters."""
        # Set model class based on model type
        if self.model_type == "hyena":
            self.model_cls = Evo2FineTuneSeqModel
            # Get hidden size from model config
            if self.model_size in HYENA_MODEL_OPTIONS:
                model_config = HYENA_MODEL_OPTIONS[self.model_size]
                self.hidden_size = model_config.hidden_size
        elif self.model_type == "mamba":
            self.model_cls = MambaFineTuneSeqModel
            if self.model_size in MAMBA_MODEL_OPTIONS:
                model_config = MAMBA_MODEL_OPTIONS[self.model_size]
                self.hidden_size = model_config.hidden_size
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.ft_dropout = self.mlp_ft_dropout

    def get_loss_reduction_class(self) -> Type[BERTMLMLossWithReduction]:
        """Get the appropriate loss reduction class based on task type.

        Returns:
            Loss reduction class for the specified task type
        """
        if self.task_type == "regression":
            return RegressorLossReduction
        elif self.task_type == "classification":
            return ClassifierLossReduction
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def configure_model(self, tokenizer=None, pre_process=True, post_process=True):
        """Configure and return the model instance.

        Args:
            tokenizer: Tokenizer to use
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism

        Returns:
            Configured model instance
        """
        # Get base model configuration
        if self.model_type == "hyena":
            base_config = HYENA_MODEL_OPTIONS[self.model_size]
        else:
            base_config = MAMBA_MODEL_OPTIONS[self.model_size]

        # Merge with fine-tuning config
        merged_config = type(base_config)(**base_config.__dict__)

        # Add fine-tuning specific attributes
        merged_config.task_type = self.task_type
        merged_config.encoder_frozen = self.encoder_frozen
        merged_config.mlp_ft_dropout = self.mlp_ft_dropout
        merged_config.mlp_hidden_size = self.mlp_hidden_size
        merged_config.mlp_target_size = self.mlp_target_size
        merged_config.ft_dropout = self.ft_dropout

        # Set parallelism
        merged_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        merged_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size

        # Create model
        return self.model_cls(
            config=merged_config,
            pre_process=pre_process,
            post_process=post_process,
        )


@dataclass
class Evo2FineTuneTokenConfig(Evo2FineTuneSeqConfig):
    """Configuration for token-level fine-tuning.

    This configuration extends the sequence-level config with additional
    parameters specific to token-level classification tasks.
    """

    # CNN head parameters
    cnn_dropout: float = 0.25
    cnn_hidden_size: int = 32
    cnn_num_classes: int = 3

    # Override skip keys for token-level tasks
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(
        default_factory=lambda: ["token_classification_head"]
    )

    def __post_init__(self):
        """Post-initialization to set token model class."""
        super().__post_init__()
        # Set token model class
        if self.model_type == "hyena":
            self.model_cls = Evo2FineTuneTokenModel
        elif self.model_type == "mamba":
            self.model_cls = MambaFineTuneTokenModel

    def get_loss_reduction_class(self) -> Type[BERTMLMLossWithReduction]:
        """Get the token classification loss reduction class.

        Returns:
            TokenClassifierLossReduction class
        """
        return TokenClassifierLossReduction

    def configure_model(self, tokenizer=None, pre_process=True, post_process=True):
        """Configure and return the token-level model instance.

        Args:
            tokenizer: Tokenizer to use
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism

        Returns:
            Configured model instance
        """
        # Get base configuration
        model = super().configure_model(tokenizer, pre_process, post_process)

        # Add CNN-specific attributes to config
        if hasattr(model, "config"):
            model.config.cnn_dropout = self.cnn_dropout
            model.config.cnn_hidden_size = self.cnn_hidden_size
            model.config.cnn_num_classes = self.cnn_num_classes

        return model
