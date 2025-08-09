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


import tempfile

from infra_bionemo.lint_xfail_fixme import check_fixme_issues, check_xfail_issues


def create_temp_file(content: str, prefix: str | None = None) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", prefix=prefix) as temp_file:
        temp_file.write(content.encode("utf-8"))
        temp_file_path = temp_file.name
    return temp_file_path


def test_xfail_with_issue():
    content = """
    @pytest.mark.xfail(reason="Known issue https://github.com/NVIDIA/bionemo-framework/issues/123")
    def test_example():
        assert False
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", prefix="test_") as temp_file:
        temp_file.write(content.encode("utf-8"))
        temp_file_path = temp_file.name
    temp_file_path = create_temp_file(content=content, prefix="test_")
    errors = check_xfail_issues(temp_file_path)
    assert len(errors) == 0


def test_xfail_without_issue():
    content = """
    @pytest.mark.xfail(reason="Known issue with batch processing")
    def test_example():
        assert False
    """
    temp_file_path = create_temp_file(content=content, prefix="test_")
    errors = check_xfail_issues(temp_file_path)
    assert len(errors) == 1
    assert "missing required issue link" in errors[0][1]


def test_fixme_with_issue():
    content = """
    # FIXME: Handle edge case with empty input https://github.com/NVIDIA/bionemo-framework/issues/456
    def example_function():
        pass
    """
    temp_file_path = create_temp_file(content=content)
    errors = check_fixme_issues(temp_file_path)
    assert len(errors) == 0


def test_fixme_without_issue():
    content = """
    # FIXME: Handle edge case with empty input
    def example_function():
        pass
    """
    temp_file_path = create_temp_file(content=content)
    errors = check_fixme_issues(temp_file_path)
    assert len(errors) == 1
    assert "missing required issue link" in errors[0][1]


def test_todo_comment():
    content = """
    # TODO: Handle edge case with empty input
    def example_function():
        pass
    """
    temp_file_path = create_temp_file(content=content)
    errors = check_fixme_issues(temp_file_path)
    assert len(errors) == 0


def test_todo_comment_with_fix():
    content = """
    # TODO: Handle edge case with empty input. To add fix
    def example_function():
        pass
    """
    temp_file_path = create_temp_file(content=content)
    errors = check_fixme_issues(temp_file_path)
    assert len(errors) == 1
