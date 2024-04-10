# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
