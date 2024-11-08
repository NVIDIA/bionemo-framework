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


import re
from pathlib import Path

import pydantic
import pytest

from bionemo.core.data.resource import Resource, get_all_resources


def test_get_all_resources_returns_valid_entries():
    resources = get_all_resources()
    assert len(resources) > 0
    assert all(isinstance(resource, Resource) for resource in resources.values())


def test_get_all_resources_returns_combines_multiple_yamls(tmp_path: Path):
    (tmp_path / "resources1.yaml").write_text(
        """
        - tag: "foo"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        """
    )

    (tmp_path / "resources2.yml").write_text(
        """
        - tag: "foo2"
          ngc: "bar"
          ngc_registry: "model"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        """
    )

    resources = get_all_resources(tmp_path)
    assert len(resources) == 2


def test_get_all_resources_returns_assigns_correct_tag(tmp_path: Path):
    (tmp_path / "file_name.yaml").write_text(
        """
        - tag: "tag_name"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        """
    )

    resources = get_all_resources(tmp_path)
    assert "file_name/tag_name" in resources


def test_get_all_resources_fails_with_slash_in_tag(tmp_path: Path):
    (tmp_path / "file_name.yaml").write_text(
        """
        - tag: "tag/name"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        """
    )

    with pytest.raises(pydantic.ValidationError):
        get_all_resources(tmp_path)


def test_get_all_resources_errors_on_duplicate_tag(tmp_path: Path):
    (tmp_path / "file_name.yaml").write_text(
        """
        - tag: "tag_name"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        - tag: "tag_name"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        """
    )

    with pytest.raises(ValueError, match=re.escape("Duplicate resource tags found!: ['file_name/tag_name']")):
        get_all_resources(tmp_path)


def test_get_all_resources_errors_on_missing_ngc_registry(tmp_path: Path):
    (tmp_path / "file_name.yaml").write_text(
        """
        - tag: "tag_name"
          ngc: "bar"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
        """
    )

    with pytest.raises(
        pydantic.ValidationError, match="ngc_registry must be provided if ngc is not None: file_name/tag_name"
    ):
        get_all_resources(tmp_path)


def test_get_all_resources_errors_on_invalid_ngc_registry(tmp_path: Path):
    (tmp_path / "file_name.yaml").write_text(
        """
        - tag: "tag_name"
          ngc: "bar"
          ngc_registry: "foo"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
        """
    )

    with pytest.raises(pydantic.ValidationError, match="Input should be 'model' or 'resource'"):
        get_all_resources(tmp_path)


def test_get_all_resources_with_valid_registry(tmp_path: Path):
    (tmp_path / "file_name.yaml").write_text(
        """
        - tag: "tag_name"
          ngc: "bar"
          ngc_registry: "resource"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
        """
    )

    resource = get_all_resources(tmp_path)
    assert resource["file_name/tag_name"].ngc_registry == "resource"
