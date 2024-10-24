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


import io

from pytest import raises

from infra_bionemo.new_project.utils import ask_yes_or_no


def test_ask_yes_or_no(monkeypatch):
    with raises(ValueError):
        ask_yes_or_no("")

    with monkeypatch.context() as ctx:
        ctx.setattr("sys.stdin", io.StringIO("y"))
        assert ask_yes_or_no("hello world?")

    with monkeypatch.context() as ctx:
        ctx.setattr("sys.stdin", io.StringIO("n"))
        assert not ask_yes_or_no("hello world?")

    with monkeypatch.context() as ctx:
        ctx.setattr("sys.stdin", io.StringIO("loop once\ny"))
        assert ask_yes_or_no("hello world?")
