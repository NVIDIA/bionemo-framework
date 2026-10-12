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

"""resolve_lm_hidden_size: one width resolver for the training and serving paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemotron_stitch.lm_config import resolve_lm_hidden_size


def test_plain_hidden_size():
    assert resolve_lm_hidden_size(SimpleNamespace(hidden_size=3136)) == 3136


def test_text_config_and_llm_config_spellings():
    assert resolve_lm_hidden_size(SimpleNamespace(text_config=SimpleNamespace(hidden_size=64))) == 64
    assert resolve_lm_hidden_size(SimpleNamespace(llm_config=SimpleNamespace(hidden_size=64))) == 64


def test_agreeing_composite_spellings_resolve_to_the_shared_value():
    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=64),
        llm_config=SimpleNamespace(hidden_size=64),
    )
    assert resolve_lm_hidden_size(config) == 64


def test_missing_width_fails_closed():
    with pytest.raises(ValueError, match="does not expose an LM hidden size"):
        resolve_lm_hidden_size(SimpleNamespace(vocab_size=32000))


def test_conflicting_widths_fail_closed():
    config = SimpleNamespace(hidden_size=64, text_config=SimpleNamespace(hidden_size=128))
    with pytest.raises(ValueError, match="conflicting LM hidden sizes"):
        resolve_lm_hidden_size(config)
