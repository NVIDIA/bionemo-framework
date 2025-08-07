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

"""Fine-tuning components for Evo2 models."""

from bionemo.evo2.models.finetune.config import (
    Evo2FineTuneSeqConfig,
    Evo2FineTuneTokenConfig,
)
from bionemo.evo2.models.finetune.datamodule import Evo2FineTuneDataModule
from bionemo.evo2.models.finetune.dataset import (
    InMemoryNucleotideDataset,
    InMemoryPerTokenValueDataset,
    InMemorySingleValueDataset,
)
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


__all__ = [
    "ClassifierLossReduction",
    "Evo2FineTuneDataModule",
    "Evo2FineTuneSeqConfig",
    "Evo2FineTuneSeqModel",
    "Evo2FineTuneTokenConfig",
    "Evo2FineTuneTokenModel",
    "InMemoryNucleotideDataset",
    "InMemoryPerTokenValueDataset",
    "InMemorySingleValueDataset",
    "MambaFineTuneSeqModel",
    "MambaFineTuneTokenModel",
    "RegressorLossReduction",
    "TokenClassifierLossReduction",
]
