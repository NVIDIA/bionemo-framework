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

"""Projector zoo, named registry, scatter, and trainability."""

from nemotron_stitch.projector.multimodal import MultimodalProjector
from nemotron_stitch.projector.projectors import (
    Mlp2xGeluNormProjector,
    Mlp2xGeluProjector,
    Perceiver3DProjector,
    Projector,
    build_projector,
    register_projector,
)
from nemotron_stitch.projector.scatter import (
    dense_to_flat,
    derive_flat_indices,
    derive_indices,
    explicit_to_flat,
    scatter_flat,
)

__all__ = [
    "Mlp2xGeluNormProjector",
    "Mlp2xGeluProjector",
    "MultimodalProjector",
    "Perceiver3DProjector",
    "Projector",
    "build_projector",
    "dense_to_flat",
    "derive_flat_indices",
    "derive_indices",
    "explicit_to_flat",
    "register_projector",
    "scatter_flat",
]
