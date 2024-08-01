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
from typing import Sequence

import transformer_engine as te
from megatron.core.transformer.transformer_config import TransformerConfig


__all__: Sequence[str] = ("TELayerNorm",)


class TELayerNorm(te.pytorch.LayerNorm):
    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        """A wrapper around transformer engine layernorm that allows it to be initialized with a TransformerConfig.
            This allows this method to be used in a megatron layerspec.

        Args:
            config (TransformerConfig): The megatron config. This is used for extracing sequence_parallel and zero_centered_gamma.
                The rest of the config is not used.
        """
        # Eps tends to get passed through properly, as does hidden_size, but not other params from the config.
        super().__init__(
            *args,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
            sequence_parallel=config.sequence_parallel,
            **kwargs,
        )
