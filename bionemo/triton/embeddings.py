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
import torch
from model_navigator.package.package import Package
from pytriton.decorators import batch

from bionemo.model.core.infer import M
from bionemo.model.protein.esm1nv.esm1nv_model import ESM1nvModel
from bionemo.triton.types_constants import EMBEDDINGS, StrArray
from bionemo.triton.utils import decode_str_batch


__all_: Sequence[str] = (
    "NavEmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingInferFn",
    "triton_embedding_infer_fn",
    "nav_triton_embedding_infer_fn",
    "mask_postprocessing_fn",
)


class NavEmbeddingRequest(TypedDict):
    tokens: np.ndarray
    mask: np.ndarray


class EmbeddingResponse(TypedDict):
    embeddings: np.ndarray


EmbeddingInferFn = Callable[[StrArray], EmbeddingResponse]


def triton_embedding_infer_fn(model: M) -> EmbeddingInferFn:
    @batch
    def infer_fn(sequences: np.ndarray) -> EmbeddingResponse:
        seqs = decode_str_batch(sequences)

        embedding = model.seq_to_embeddings(seqs)

        response: EmbeddingResponse = {
            EMBEDDINGS: embedding.detach().cpu().numpy(),
        }
        return response

    return infer_fn


def nav_triton_embedding_infer_fn(model: M, runner: Package) -> EmbeddingInferFn:
    postprocess = mask_postprocessing_fn(model)

    @batch
    def infer_fn(sequences: np.ndarray) -> EmbeddingResponse:
        seqs = decode_str_batch(sequences)

        tokens_enc, enc_mask = model.tokenize(seqs)
        inp: NavEmbeddingRequest = {
            "tokens": tokens_enc.cpu().detach().numpy(),
            "mask": enc_mask.cpu().detach().numpy(),
        }

        hidden_states = runner.infer(inp)
        hidden_states = torch.tensor(hidden_states["embeddings"], device="cuda")
        enc_mask = postprocess(enc_mask)

        embedding = model.hiddens_to_embedding(hidden_states, enc_mask)

        response: EmbeddingResponse = {
            EMBEDDINGS: embedding.cpu().numpy(),
        }
        return response

    return infer_fn


def mask_postprocessing_fn(model: M) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(model, ESM1nvModel):

        def postprocess(enc_mask: torch.Tensor) -> torch.Tensor:
            enc_mask[:, 0:2] = 0
            enc_mask = torch.roll(enc_mask, shifts=-1, dims=1)
            return enc_mask

    else:

        def postprocess(enc_mask: torch.Tensor) -> torch.Tensor:
            return enc_mask

    return postprocess
