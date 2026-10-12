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

"""Shared contract constants for the package and its consumers.

The schema version and format discriminator are recorded in every manifest
and checked on load, so a checkpoint written by an older package fails loudly
rather than silently mis-scattering. Bumping ``SCHEMA_VERSION``
requires shipping a converter for the previous version in the same change
(AGENTS.md).
"""

from __future__ import annotations

# Artifact schema. A new counter for a new format, starting at
# 1. Readers dispatch on the discriminator below, never on this integer: a
# legacy consumer schema also wrote "schema_version": 1.
SCHEMA_VERSION = 1
# The on-disk format discriminator keeps the pre-rename package name: it is an
# artifact contract, not a distribution name. Changing it would be a schema
# change (bump SCHEMA_VERSION, ship a converter) that breaks every artifact
# written before the rename for zero functional gain.
MANIFEST_FORMAT = "nemotron-add-modality/mm-projector-manifest"

# Flat forward kwargs: per-projector encoder features, plus an
# optional explicit index override for collators that need exact control.
MM_FEATURES_PREFIX = "mm_features__"
MM_TOKEN_INDICES_PREFIX = "mm_token_indices__"

# Model-config key carrying the per-projector placeholder token ids used to
# derive and validate scatter targets from input_ids. Each
# projector owns a distinct id, so interleaved multi-projector prompts route
# by token identity and a misrouted index fails closed.
MM_PLACEHOLDER_TOKEN_IDS_KEY = "mm_placeholder_token_ids"

# Projector ownership modes.
OWNERSHIP_MODULE = "module"
OWNERSHIP_SIDECAR = "sidecar"
OWNERSHIP_MODES = frozenset({OWNERSHIP_MODULE, OWNERSHIP_SIDECAR})

# Trainability families. "lora" is the only family that may be
# called an adapter.
TRAINABILITY_FAMILIES = frozenset({"projector", "extra", "lora"})

# vLLM plugin modes.
VLLM_MODES = frozenset({"encode", "projected"})

# Sidecar on-disk layout.
MANIFEST_FILENAME = "mm-projector-manifest.json"
PROJECTOR_FILENAME = "mm-projector.safetensors"
EXTRA_STATE_DIR = "extra"
ADAPTER_FILENAME = "adapter_model.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
