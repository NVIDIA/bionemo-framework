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

from typing import Callable, List, Sequence, TypedDict

import numpy as np
from pytriton.decorators import batch

from bionemo.model.core.infer import M
from bionemo.triton.types_constants import GENERATED
from bionemo.triton.utils import decode_str_batch


__all__: Sequence[str] = (
    "SamplingInferFn",
    "SamplingResponse",
    "triton_sampling_infer_fn",
)


class SamplingResponse(TypedDict):
    generated: np.ndarray


SamplingInferFn = Callable[[np.ndarray], SamplingResponse]


def triton_sampling_infer_fn(model: M) -> SamplingInferFn:
    """Produces a PyTriton-compatible inference function that uses a bionemo encoder-decoder model to generate new
    protein sequences given an input sequence as a starting point."""

    @batch
    def infer_fn(sequences: np.ndarray) -> SamplingResponse:
        seqs = decode_str_batch(sequences)

        generated: List[str] = model.sample(seqs=seqs, return_embedding=False, sampling_method="greedy-perturbate")

        response: SamplingResponse = {GENERATED: np.char.encode(generated, "utf-8").reshape((len(sequences), -1))}
        return response

    return infer_fn
