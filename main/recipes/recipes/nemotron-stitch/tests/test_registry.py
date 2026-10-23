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

"""Idempotent out-of-tree model registration against a registry double."""

from __future__ import annotations

import pytest
from torch import nn

from nemotron_stitch.automodel.registry import register_models

ARCHITECTURE = "ExampleProjectedForConditionalGeneration"


class _Model(nn.Module):
    pass


def _registry(calls: list | None = None):
    class Registry:
        model_arch_name_to_cls = {}

        @classmethod
        def register(cls, name, model_class, *, exist_ok):
            if calls is not None:
                calls.append((name, model_class, exist_ok))
            cls.model_arch_name_to_cls[name] = model_class

    return Registry


def test_out_of_tree_registration_is_idempotent():
    calls = []
    registry = _registry(calls)

    assert register_models(ARCHITECTURE, _Model, registry=registry)
    assert register_models(ARCHITECTURE, _Model, registry=registry)
    assert registry.model_arch_name_to_cls == {ARCHITECTURE: _Model}
    assert calls == [(ARCHITECTURE, _Model, False)]


def test_out_of_tree_registration_rejects_conflicting_class():
    registry = _registry()
    registry.model_arch_name_to_cls[ARCHITECTURE] = object

    with pytest.raises(RuntimeError, match="already registered"):
        register_models(ARCHITECTURE, _Model, registry=registry)
