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

        named_params = dict(pl_module.named_parameters())
        embed_keys = [key for key in named_params.keys() if 'embed' in key]
        if len(embed_keys) == 0:
            raise ValueError("No embed keys found.")
        if len(embed_keys) > 1:
            raise ValueError("Multiple embed keys found.")
        embed = named_params[embed_keys[0]]

        tev = torch.sqrt(torch.mean(torch.pow(embed - embed.mean(dim=0), 2), dim=0))
        tev_mean = torch.mean(tev)
        tev_sd = torch.std(tev)

        trainer.log("tev_mean", tev_mean, on_step=True, on_epoch=False)
        trainer.log("tev_sd", tev_sd, on_step=True, on_epoch=False)
