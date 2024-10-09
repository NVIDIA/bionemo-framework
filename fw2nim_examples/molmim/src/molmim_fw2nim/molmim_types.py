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

from fw2nim.config.types import BaseModel
from pydantic import Field


__all__: Sequence[str] = (
    "MolMimSequences",
    "MolMimEmbeddings",
    "MolMimGenerated",
    "MolMimHiddens",
    "MolMimControlOptIn",
    "MolMimControlOptOut",
)


class MolMimSequences(BaseModel):
    sequences: list[str] = Field(
        ...,
        numpy_dtype="bytes",
        shape_for_triton=(-1, 1),
        reshape_from_triton=(-1,),
    )


class MolMimEmbeddings(BaseModel):
    embeddings: list[list[float]] = Field(
        ...,
        numpy_dtype="float32",
        triton_shape=(-1, 512),
    )


class MolMimGenerated(BaseModel):
    generated: list[str] = Field(
        ...,
        numpy_dtype="bytes",
        triton_shape=(-1,),
    )


class MolMimHiddens(BaseModel):
    hiddens: list[list[float]] = Field(
        ...,
        numpy_dtype="float32",
        shape_for_triton=(-1, 1, 512),
        reshape_from_triton=(-1, 512),
    )
    mask: list[list[bool]] = Field(
        ...,
        numpy_dtype="bool",
        triton_shape=(-1, 1),
    )


class MolMimControlOptIn(BaseModel):
    smi: str = Field(
        ...,
        numpy_dtype="bytes",
    )
    algorithm: str = Field(..., numpy_dtype="bytes", triton_shape=(1, 1))
    num_molecules: int = Field(..., numpy_dtype="int32", triton_shape=(1, 1))
    property_name: str = Field(..., numpy_dtype="bytes", triton_shape=(1, 1))
    minimize: bool = Field(..., numpy_dtype="bool", triton_shape=(1, 1))
    min_similarity: float = Field(..., numpy_dtype="float32", triton_shape=(1, 1))
    particles: int = Field(..., numpy_dtype="int32", triton_shape=(1, 1))
    iterations: int = Field(..., numpy_dtype="int32", triton_shape=(1, 1))
    radius: float = Field(..., numpy_dtype="float32", triton_shape=(1, 1))


class MolMimControlOptOut(BaseModel):
    samples: list[str] = Field(
        ...,
        numpy_dtype="bytes",
        reshape_from_triton=(-1,),
    )
    scores: list[float] = Field(
        ...,
        numpy_dtype="float32",
        triton_shape=(-1,),
        reshape_from_triton=(-1,),
    )
    score_type: str = Field(..., numpy_dtype="bytes", triton_shape=(1, 1))
