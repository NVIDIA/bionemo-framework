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
from pathlib import Path

import tomli_w
from pytest import raises

from infra_bionemo.new_project.exe.bionemo_subpackage import main as main_bionemo_sub
from infra_bionemo.new_project.exe.namespace import main as main_namespace
from infra_bionemo.new_project.exe.simple import main as main_simple


def test_create_namespace_cli(tmpdir):
    (Path(tmpdir) / "file").touch()
    # not a dir!
    with raises(ValueError):
        main_namespace(namespace="acme", module="rocket", location=f"{str(tmpdir)}/file", no_test_append=False)
    (Path(tmpdir) / "file").unlink()

    main_namespace(namespace="acme", module="rocket", location=str(tmpdir), no_test_append=False)

    location = Path(str(tmpdir)) / "acme-rocket"
    assert location.is_dir()
    assert (location / "src").is_dir()
    assert (location / "src" / "acme").is_dir()
    assert not (location / "src" / "acme" / "__init__.py").exists()
    assert (location / "src" / "acme" / "rocket").is_dir()
    assert (location / "src" / "acme" / "rocket" / "__init__.py").is_file()
    assert (location / "tests").is_dir()
    assert (location / "tests" / "test_acme").is_dir()
    assert (location / "tests" / "test_acme" / "test_rocket").is_dir()
    assert (location / "tests" / "test_acme" / "test_rocket" / "test_TODO_acme_rocket.py").is_file()
    assert (location / "README.md").is_file()
    assert not (location / "setup.py").exists()
    assert (location / "pyproject.toml").is_file()
    assert (location / "requirements.txt").is_file()
    assert (location / "requirements-test.txt").is_file()
    assert (location / "requirements-dev.txt").is_file()


def test_create_simple_cli(tmpdir):
    (Path(tmpdir) / "file").touch()
    # not a dir!
    with raises(ValueError):
        main_simple(project_name="simple", location=f"{str(tmpdir)}/file")
    (Path(tmpdir) / "file").unlink()

    main_simple(project_name="simple", location=str(tmpdir))

    location = Path(str(tmpdir)) / "simple"
    assert location.is_dir()
    assert (location / "src").is_dir()
    assert (location / "src" / "simple").is_dir()
    assert (location / "src" / "simple" / "__init__.py").is_file()
    assert (location / "tests").is_dir()
    assert (location / "tests" / "test_simple").is_dir()
    assert (location / "tests" / "test_simple" / "test_TODO_simple.py").is_file()
    assert (location / "README.md").is_file()
    assert not (location / "setup.py").exists()
    assert (location / "pyproject.toml").is_file()
    assert (location / "requirements.txt").is_file()
    assert (location / "requirements-test.txt").is_file()
    assert (location / "requirements-dev.txt").is_file()


def test_create_bionemo_cli(tmpdir, monkeypatch):
    # not a dir!
    with raises(ValueError):
        main_bionemo_sub(
            project_name="bionemo-supermodel",
            loc_sub_pack=f"{str(tmpdir)}/file",
            relax_name_check=False,
        )

    # no sub-packages dir!
    with raises(ValueError):
        main_bionemo_sub(
            project_name="bionemo-supermodel",
            loc_sub_pack=str(tmpdir),
            relax_name_check=False,
        )

    sub_packages = Path(tmpdir) / "sub-packages"
    sub_packages.mkdir(parents=True, exist_ok=True)

    # no bionemo-fw dir!
    with raises(ValueError):
        main_bionemo_sub(
            project_name="bionemo-supermodel",
            loc_sub_pack=str(sub_packages),
            relax_name_check=False,
        )

    (sub_packages / "bionemo-fw").mkdir(parents=True, exist_ok=True)

    # no pyproject.toml in bionemo-fw dir!
    with raises(ValueError):
        main_bionemo_sub(
            project_name="bionemo-supermodel",
            loc_sub_pack=str(sub_packages),
            relax_name_check=False,
        )

    # create & add basic pyproject.toml structure that's checked
    bionemo_fw_pyproject_toml = sub_packages / "bionemo-fw" / "pyproject.toml"
    bionemo_fw_pyproject_toml.touch(exist_ok=True)
    with open(str(bionemo_fw_pyproject_toml), "wt") as wt:
        wt.write(tomli_w.dumps({"project": {"dependencies": []}}))

    with monkeypatch.context() as ctx:
        ctx.setattr("sys.stdin", io.StringIO("y"))
        main_bionemo_sub(
            project_name="bionemo-supermodel",
            loc_sub_pack=str(sub_packages),
            relax_name_check=False,
        )

    location = sub_packages / "bionemo-supermodel"
    assert location.is_dir()
    assert (location / "src").is_dir()
    assert (location / "src" / "bionemo").is_dir()
    assert not (location / "src" / "bionemo" / "__init__.py").exists()
    assert (location / "src" / "bionemo" / "supermodel").is_dir()
    assert (location / "src" / "bionemo" / "supermodel" / "__init__.py").is_file()
    assert (location / "tests").is_dir()
    assert (location / "tests" / "bionemo").is_dir()
    assert (location / "tests" / "bionemo" / "supermodel").is_dir()
    assert (location / "tests" / "bionemo" / "supermodel" / "test_TODO_bionemo_supermodel.py").is_file()
    assert (location / "README.md").is_file()
    assert not (location / "setup.py").exists()
    assert (location / "pyproject.toml").is_file()
    assert not (location / "requirements.txt").exists()
    assert not (location / "requirements-test.txt").exists()
    assert not (location / "requirements-dev.txt").exists()
