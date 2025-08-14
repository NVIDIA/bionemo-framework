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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Temporal Geneformer model for next-cell prediction."""

import logging
from dataclasses import dataclass, field
from typing import List, Type

from bionemo.geneformer.model.finetune_token_regressor import (
    FineTuneSeqLenBioBertConfig,
    MegatronBioBertFineTuneSeqLengthModel,
)
from bionemo.llm.api import MegatronLossType
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.model.loss import BERTMLMLossWithReduction


__all__ = ["TemporalGeneformerConfig", "TemporalGeneformerLoss", "TemporalGeneformerModel"]

logger = logging.getLogger(__name__)


class TemporalGeneformerLoss(BERTMLMLossWithReduction):
    """Custom loss for temporal Geneformer that handles temporal masking."""

    def __init__(self, validation_step: bool = False, val_drop_last: bool = True) -> None:
        """Initialize the temporal Geneformer loss function.

        Args:
            validation_step: Whether this is being used for validation
            val_drop_last: Whether to drop the last incomplete batch during validation
        """
        super().__init__(validation_step, val_drop_last)

    def forward(self, batch, forward_out):
        """Forward pass for temporal loss computation."""
        # Get the standard BERT MLM loss
        loss_dict = super().forward(batch, forward_out)

        # Add temporal-specific loss components if needed
        # For now, we use the standard MLM loss but could extend with temporal consistency losses

        return loss_dict


class TemporalGeneformerModel(MegatronBioBertFineTuneSeqLengthModel):
    """Temporal Geneformer model for next-cell prediction fine-tuning."""

    def __init__(self, config, *args, include_hiddens: bool = False, post_process: bool = True, **kwargs):
        """Initialize the temporal Geneformer model.

        Args:
            config: Model configuration
            *args: Additional positional arguments passed to parent class
            include_hiddens: Whether to include hidden states in output
            post_process: Whether this is the final pipeline stage
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(config, *args, include_hiddens=include_hiddens, post_process=post_process, **kwargs)
        logger.info("Initialized TemporalGeneformerModel for next-cell prediction")

    def forward(self, *args, **kwargs):
        """Forward pass for temporal Geneformer."""
        # The temporal logic is handled in the dataset - here we just use the standard forward
        # The model architecture remains the same, but the training data includes temporal sequences
        output = super().forward(*args, **kwargs)
        return output


@dataclass
class TemporalGeneformerConfig(FineTuneSeqLenBioBertConfig):
    """Configuration for temporal Geneformer fine-tuning.

    This extends the standard fine-tuning configuration to support loading
    from pretrained Geneformer checkpoints for temporal next-cell prediction.
    """

    # Model class for temporal Geneformer
    model_cls: Type[TemporalGeneformerModel] = TemporalGeneformerModel

    # Loss function for temporal training
    loss_reduction_class: Type[MegatronLossType] = TemporalGeneformerLoss

    # Skip regression head during checkpoint loading by default since we're fine-tuning
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["regression_head"])

    # Default configuration for temporal fine-tuning
    # These can be overridden when creating the config
    seq_length: int = 2048
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1

    def get_loss_reduction_class(self) -> Type[TemporalGeneformerLoss]:
        """Return the loss function class."""
        return TemporalGeneformerLoss

    def model_class(self) -> Type[TemporalGeneformerModel]:
        """Return the model class."""
        return TemporalGeneformerModel


def create_temporal_geneformer_config(
    initial_ckpt_path: str,
    seq_length: int = 2048,
    hidden_dropout: float = 0.1,
    attention_dropout: float = 0.1,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec,
    **kwargs,
) -> TemporalGeneformerConfig:
    """Create a temporal Geneformer configuration for fine-tuning.

    Args:
        initial_ckpt_path: Path to pretrained Geneformer checkpoint
        seq_length: Maximum sequence length
        hidden_dropout: Hidden layer dropout rate
        attention_dropout: Attention dropout rate
        biobert_spec_option: BioBERT architecture specification
        **kwargs: Additional configuration parameters

    Returns:
        Configured TemporalGeneformerConfig instance
    """
    return TemporalGeneformerConfig(
        initial_ckpt_path=initial_ckpt_path,
        seq_length=seq_length,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        biobert_spec_option=biobert_spec_option,
        **kwargs,
    )
