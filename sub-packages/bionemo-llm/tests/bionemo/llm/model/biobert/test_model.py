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


from typing import Type
from unittest import mock

from bionemo.llm.api import MegatronLossType
from bionemo.llm.model.biobert.model import BioBertConfig, MegatronBioBertModel
from bionemo.llm.utils import iomixin_utils as iom
from bionemo.testing import megatron_parallel_state_utils


class MockConfig(BioBertConfig[MegatronBioBertModel, MegatronLossType], iom.IOMixinWithGettersSetters):
    model_cls: Type[MegatronBioBertModel] = MegatronBioBertModel
    pass


def test_biobert_model_initialized():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        tokenizer = mock.MagicMock()
        tokenizer.vocab_size = 32
        config = MockConfig()
        model = config.configure_model(tokenizer)

        assert isinstance(model, MegatronBioBertModel)
