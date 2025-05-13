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

# Copyright The Lightning AI team.
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

from typing import Any
import torch
from lightning.pytorch import Callback 


class TEVCallback(Callback):
    """Callback for logging TEV statistics before each optimizer step."""

    @torch.no_grad()
    def on_before_optimizer_step(
        self, trainer, pl_module, optimizer) -> None:
        """Called before each optimizer step during training.

        This method calculates and logs Token Embedding Variance (TEV) statistics:
        1. Gets all named parameters from the model
        2. Finds the embedding parameter (expects exactly one embedding layer)
        3. Calculates the token embedding variance (TEV) - the variance of each token embedding
        4. Logs the mean and standard deviation of TEV values

        Args:
            trainer: The Lightning trainer instance
            pl_module: The current Lightning module being trained
            optimizer: The optimizer being used

        Raises:
            ValueError: If no embedding layer is found or if multiple embedding layers are found
        """
        # Get all named parameters from the model
        named_params = dict(pl_module.named_parameters())

        # Find all parameter keys containing 'embed'
        embed_keys = [key for key in named_params.keys() if 'embed' in key]

        # Validate we have exactly one embedding layer
        if len(embed_keys) == 0:
            raise ValueError("No embed keys found.")
        if len(embed_keys) > 1:
            raise ValueError("Multiple embed keys found.")

        # Get the embedding parameter
        embed = named_params[embed_keys[0]]

        # Calculate token embedding variance (TEV)
        # First center the embeddings by subtracting the mean
        # Then calculate the mean squared deviation (variance)
        # Finally take the square root to get standard deviation
        tev = torch.sqrt(torch.mean(torch.pow(embed - embed.mean(dim=0), 2), dim=0))

        # Calculate statistics of the TEV values
        tev_mean = torch.mean(tev).item()
        tev_sd = torch.std(tev).item()

        # Log the TEV statistics
        pl_module.log("tev_mean", tev_mean, on_step=True, on_epoch=False)
        pl_module.log("tev_sd", tev_sd, on_step=True, on_epoch=False)
