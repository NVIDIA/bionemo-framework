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


from typing import Dict, Literal, Sequence

import torch


__all__: Sequence[str] = (
    "PrecisionTypes",
    "get_autocast_dtype",
)


# NOTE(SKH) our precision types are a mess, but we inherit this problem from NeMo and Megatron.
PrecisionTypes = Literal["fp16", "bf16", "fp32", "bf16-mixed", "fp32-mixed", "16-mixed", "fp16-mixed", 16, 32]
precision_to_dtype: Dict[PrecisionTypes, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "16-mixed": torch.float16,
    "fp16-mixed": torch.float16,
    "bf16-mixed": torch.bfloat16,
    "fp32-mixed": torch.float32,
    16: torch.float16,
    32: torch.float32,
}

# NOTE(SKH) these do not have a perfect 1-1 relationship, but we can use this to serialize/deserialize dtypes in ModelConfigs since its ultimately converted with precision_to_dtype.
dtype_to_precision: Dict[torch.dtype, PrecisionTypes] = {v: k for k, v in precision_to_dtype.items()}


def get_autocast_dtype(precision: PrecisionTypes) -> torch.dtype:
    """Returns the torch dtype corresponding to the given precision.

    Args:
        precision: The precision type.

    Returns:
        torch.dtype: The torch dtype corresponding to the given precision.

    Raises:
        ValueError: If the precision is not supported.
    """
    # TODO move this to a utilities folder, or find/import the function that does this in NeMo
    if precision in precision_to_dtype:
        return precision_to_dtype[precision]
    else:
        raise ValueError(f"Unsupported precision: {precision}")
