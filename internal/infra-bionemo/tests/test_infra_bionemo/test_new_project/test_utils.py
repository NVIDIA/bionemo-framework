# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
