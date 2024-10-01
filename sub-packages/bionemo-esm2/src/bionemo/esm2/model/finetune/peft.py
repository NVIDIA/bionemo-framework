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


# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm import fn
from nemo.collections.llm.fn.mixin import FNMixin
from nemo.collections.llm.peft.lora import LoRA
from torch import nn


class ESM2LoRA(LoRA):
    """LoRA for the BioNeMo2 ESM Model."""

    def __call__(self, model: nn.Module) -> nn.Module:
        """This method is called when the object is called as a function.

        Args:
            model: The input model.

        Returns:
            The modified model.
        """
        fn.walk(model, self.selective_freeze)
        fn.walk(model, self.transform)
        return model

    def selective_freeze(self, m: nn.Module, name=None, prefix=None):
        """Freezes specific modules in the given model.

        Args:
            m (nn.Module): The model to selectively freeze.
            name (str): The name of the module to freeze. Valid values are "encoder" and "embedding".
            prefix (str): The prefix of the module to freeze.

        Returns:
            nn.Module: The modified model with the specified modules frozen.

        See Also:
            nemo.collections.llm.fn.mixin.FNMixin
        """
        if name in ["encoder", "embedding"]:
            FNMixin.freeze(m)
        return m
