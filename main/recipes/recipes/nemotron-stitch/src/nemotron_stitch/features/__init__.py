# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Encoder registry, content-addressed cache, chunking, and pooling."""

from nemotron_stitch.features.cache import (
    CacheBuilder,
    CacheContract,
    CacheMissError,
    ImmutableEmbeddingCache,
    open_feature_cache,
)
from nemotron_stitch.features.chunking import (
    Chunk,
    coverage_weights,
    plan_chunks,
    stitch_hidden_states,
    strip_special_tokens,
)
from nemotron_stitch.features.pooling import pool_positions
from nemotron_stitch.features.preparation import InvalidRow, Partition, prepare_features
from nemotron_stitch.features.protocol import FrozenEncoder
from nemotron_stitch.features.registry import EncoderRegistry, EncoderSpec
from nemotron_stitch.features.types import EncoderOutput, FeatureRef, Unit

__all__ = [
    "CacheBuilder",
    "CacheContract",
    "CacheMissError",
    "Chunk",
    "EncoderOutput",
    "EncoderRegistry",
    "EncoderSpec",
    "FeatureRef",
    "FrozenEncoder",
    "ImmutableEmbeddingCache",
    "InvalidRow",
    "Partition",
    "Unit",
    "coverage_weights",
    "plan_chunks",
    "pool_positions",
    "open_feature_cache",
    "prepare_features",
    "stitch_hidden_states",
    "strip_special_tokens",
]
