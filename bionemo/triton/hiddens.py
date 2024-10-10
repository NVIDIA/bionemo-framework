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


from typing import Callable, Sequence, TypedDict

import numpy as np
from pytriton.decorators import batch

from bionemo.model.core.infer import M
from bionemo.triton.types_constants import HIDDENS, MASK
from bionemo.triton.utils import decode_str_batch


__all_: Sequence[str] = (
    "HiddenResponse",
    "HiddenInferFn",
    "triton_hidden_infer_fn",
)


class HiddenResponse(TypedDict):
    hiddens: np.ndarray
    mask: np.ndarray


HiddenInferFn = Callable[[np.ndarray], HiddenResponse]


def triton_hidden_infer_fn(model: M) -> HiddenInferFn:
    @batch
    def infer_fn(sequences: np.ndarray) -> HiddenResponse:
        seqs = decode_str_batch(sequences)

        hidden, mask = model.seq_to_hiddens(seqs)

        response: HiddenResponse = {
            HIDDENS: hidden.detach().cpu().numpy(),
            MASK: mask.detach().cpu().numpy(),
        }
        return response

    return infer_fn
