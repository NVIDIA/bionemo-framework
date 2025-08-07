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

"""Token-level fine-tuned Evo2 models."""

from nemo.collections.llm.gpt.model.hyena import HyenaModel

from bionemo.evo2.models.finetune.heads import MegatronConvHead
from bionemo.evo2.models.mamba import MambaModel


class Evo2FineTuneTokenModel(HyenaModel):
    """Fine-tuned Evo2 (Hyena) model for token-level tasks."""

    def __init__(self, config, *args, post_process: bool = True, **kwargs):
        """Initialize the fine-tuned model.

        Args:
            config: Model configuration
            post_process: Whether this is the last stage in pipeline parallelism
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, *args, post_process=post_process, **kwargs)

        # Freeze encoder parameters if requested
        if config.encoder_frozen:
            for name, param in self.named_parameters():
                if "output_layer" not in name:  # Keep output layer trainable
                    param.requires_grad = False

        if post_process:
            self.token_classification_head = MegatronConvHead(config)

    def forward(self, *args, **kwargs):
        """Forward pass through the model.

        Args:
            *args: Positional arguments for parent forward
            **kwargs: Keyword arguments for parent forward

        Returns:
            Dictionary containing model outputs and token-level predictions
        """
        output = super().forward(*args, **kwargs)

        if self.post_process and "embeddings" in output:
            embeddings = output["embeddings"]  # [b, s, h]
            output["token_classifier_output"] = self.token_classification_head(embeddings)

        return output


class MambaFineTuneTokenModel(MambaModel):
    """Fine-tuned Mamba model for token-level tasks."""

    def __init__(self, config, *args, post_process: bool = True, **kwargs):
        """Initialize the fine-tuned model.

        Args:
            config: Model configuration
            post_process: Whether this is the last stage in pipeline parallelism
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, *args, post_process=post_process, **kwargs)

        if config.encoder_frozen:
            for name, param in self.named_parameters():
                if "output_layer" not in name:
                    param.requires_grad = False

        if post_process:
            self.token_classification_head = MegatronConvHead(config)

    def forward(self, *args, **kwargs):
        """Forward pass through the model.

        Args:
            *args: Positional arguments for parent forward
            **kwargs: Keyword arguments for parent forward

        Returns:
            Dictionary containing model outputs and token-level predictions
        """
        output = super().forward(*args, **kwargs)

        if self.post_process and "embeddings" in output:
            embeddings = output["embeddings"]
            output["token_classifier_output"] = self.token_classification_head(embeddings)

        return output
