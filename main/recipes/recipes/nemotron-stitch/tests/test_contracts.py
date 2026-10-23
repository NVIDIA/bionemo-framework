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

"""contracts.py is a real module with stable contract values; pin its values."""

from __future__ import annotations

from nemotron_stitch import contracts


def test_schema_version_starts_at_one() -> None:
    # A new counter for a new format; bumping it requires
    # shipping a converter for the previous version in the same change.
    assert contracts.SCHEMA_VERSION == 1


def test_manifest_format_discriminator() -> None:
    # A bare integer cannot discriminate: gr-v1 also writes schema_version 1.
    assert contracts.MANIFEST_FORMAT == "nemotron-add-modality/mm-projector-manifest"


def test_forward_kwarg_prefixes() -> None:
    assert contracts.MM_FEATURES_PREFIX == "mm_features__"
    assert contracts.MM_TOKEN_INDICES_PREFIX == "mm_token_indices__"


def test_ownership_modes() -> None:
    assert set(contracts.OWNERSHIP_MODES) == {"module", "sidecar"}


def test_trainability_families() -> None:
    assert set(contracts.TRAINABILITY_FAMILIES) == {"projector", "extra", "lora"}


def test_vllm_modes() -> None:
    assert set(contracts.VLLM_MODES) == {"encode", "projected"}
