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

import torch
from transformers.trainer_pt_utils import get_parameter_names


def compute_accuracy(preds, labels, ignore_index=-100) -> tuple[int, int]:
    """Calculate the accuracy."""
    preds_labels = torch.argmax(preds, dim=-1)
    mask = labels != ignore_index
    correct = (preds_labels == labels) & mask

    return correct.sum().item(), mask.sum().item()


def get_parameter_names_with_lora(model):
    """Get layers with non-zero weight decay.

    This function reuses the Transformers' library function
    to list all the layers that should have weight decay.
    """
    forbidden_name_patterns = [
        r"bias",
        r"layernorm",
        r"rmsnorm",
        r"(?:^|\.)norm(?:$|\.)",
        r"_norm(?:$|\.)",
        r"\.lora_[AB]\.",
    ]

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)

    return decay_parameters


SS3_ID2LABEL = {0: "H", 1: "E", 2: "C"}

SS3_LABEL2ID = {
    "H": 0,
    "I": 0,
    "G": 0,
    "E": 1,
    "B": 1,
    "S": 2,
    "T": 2,
    "~": 2,
    "C": 2,
    "L": 2,
}  # '~' denotes coil / unstructured

SS8_ID2LABEL = {0: "H", 1: "I", 2: "G", 3: "E", 4: "B", 5: "S", 6: "T", 7: "C"}

SS8_LABEL2ID = {
    "H": 0,
    "I": 1,
    "G": 2,
    "E": 3,
    "B": 4,
    "S": 5,
    "T": 6,
    "~": 7,
    "C": 7,
    "L": 7,
}  # '~' denotes coil / unstructured
