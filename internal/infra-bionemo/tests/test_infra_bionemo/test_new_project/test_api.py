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


from pathlib import Path

from pytest import mark, raises

from infra_bionemo.new_project.api import (
    Dir,
    File,
    bionemo_subproject_structure,
    check,
    convert,
    create_on_filesystem,
    namespace_py_project_structure,
    py_project_structure,
)


def test_simple_file_and_dir():
    f = File("start.txt", contents="Hello world!")
    d = Dir("location", contents=[f])
    assert f in d.contents


def test_create_on_filesystem_errors(tmpdir):
    with raises(ValueError):
        create_on_filesystem(Path("./.12341251820y67451-does-not-exist"), File("_", contents=""))
    with raises(TypeError):
        create_on_filesystem(Path(tmpdir), None)  # type:ignore


def test_check_errors():
    with raises(ValueError):
        check("")

    with raises(ValueError):
        check("1not-ok")

    with raises(ValueError):
        check("NOT-OK")

    with raises(ValueError):
        check("not_ok")

    with raises(ValueError):
        check("not$ok")


@mark.parametrize(
    "project_name",
    [
        "bionemo",
        "bionemo-geneformer",
        "infra-bionemo",
        "super-awesome-tools",
    ],
)
def test_project_name_check(project_name):
    check(project_name)  # raises error
    with raises(ValueError):
        check("not a valid project name")


@mark.parametrize(
    "input_,expected",
    [
        ("bionemo", "bionemo"),
        (" bionemo-geneformer  ", "bionemo_geneformer"),
        ("infra-bionemo", "infra_bionemo"),
        ("super-awesome-tools", "super_awesome_tools"),
    ],
)
def test_module_name_convert(input_, expected):
    actual = convert(input_)
    assert actual == expected, f"{input_=} did not convert into {expected=}, instead was {actual=}"


def test_bionemo_subproject():
    d = bionemo_subproject_structure("geneformer-extras", ["bionemo-llm", "bionemo-geometric"])
    _assert_has_core_toplevel(d)
    assert not _has_file_by_name("setup.py", d, descend=False)
    assert not _has_file_by_name("requirements.txt", d, descend=False)
    assert not _has_file_by_name("requirements-test.txt", d, descend=False)
    assert not _has_file_by_name("requirements-dev.txt", d, descend=False)
    assert _has_file_by_name("__init__.py", d, descend=True)
    assert _has_file_by_name("test_TODO_bionemo_geneformer_extras.py", d, descend=True)
    assert _has_dir_by_name("bionemo", d, descend=True)
    assert _has_dir_by_name("geneformer_extras", d, descend=True)


def test_namespace_project():
    d = namespace_py_project_structure("bionemo", "geneformer-extras", ["nemo", "megatron"])
    _assert_has_core_toplevel(d)
    assert _has_file_by_name("requirements.txt", d, descend=False)
    assert _has_file_by_name("setup.py", d, descend=False)
    assert not _has_file_by_name("requirements-test.txt", d, descend=False)
    assert not _has_file_by_name("requirements-dev.txt", d, descend=False)
    assert _has_file_by_name("__init__.py", d, descend=True)
    assert _has_file_by_name("test_TODO_bionemo_geneformer_extras.py", d, descend=True)
    assert _has_dir_by_name("bionemo", d, descend=True)
    assert _has_dir_by_name("geneformer_extras", d, descend=True)

    d = namespace_py_project_structure(
        "bionemo",
        "geneformer-extras",
        ["nemo", "megatron"],
        add_test_reqs=True,
        add_dev_reqs=True,
        prefix_test_dirs=True,
    )
    assert _has_file_by_name("requirements-test.txt", d, descend=False)
    assert _has_file_by_name("requirements-dev.txt", d, descend=False)
    assert _has_dir_by_name("test_bionemo", d, descend=True)
    assert _has_dir_by_name("test_geneformer_extras", d, descend=True)


def test_simple_project():
    d = py_project_structure("infra-bionemo", ["nltk"])
    _assert_has_core_toplevel(d)
    assert _has_file_by_name("requirements.txt", d, descend=False)
    assert not _has_file_by_name("setup.py", d, descend=False)
    assert _has_file_by_name("requirements-test.txt", d, descend=False)
    assert _has_file_by_name("requirements-dev.txt", d, descend=False)
    assert _has_file_by_name("__init__.py", d, descend=True)
    assert _has_file_by_name("test_TODO_infra_bionemo.py", d, descend=True)
    assert _has_dir_by_name("infra_bionemo", d, descend=True)

    d = py_project_structure(
        "infra-bionemo",
        ["nltk"],
        add_setup_py=True,
        prefix_test_dirs=True,
    )
    assert _has_file_by_name("setup.py", d, descend=False)
    assert _has_dir_by_name("test_infra_bionemo", d, descend=True)


def _assert_has_core_toplevel(x: Dir):
    assert _has_file_by_name("README.md", x, descend=False)
    assert _has_file_by_name("pyproject.toml", x, descend=False)
    assert _has_dir_by_name("src", x, descend=False)
    assert _has_dir_by_name("tests", x, descend=False)


def _has_file_by_name(f_or_name: File | str, x: Dir, descend: bool) -> bool:
    match f_or_name:
        case File(name, _):
            filename: str = name
        case str():
            filename = f_or_name
        case _:
            raise TypeError(f"Expecting f_or_name to be File or str, not {type(f_or_name)}, {f_or_name=}")

    for c in x.contents:
        if isinstance(c, File):
            if c.name == filename:
                return True
        if descend and isinstance(c, Dir):
            found = _has_file_by_name(filename, c, descend=True)
            if found:
                return True
    return False


def _has_dir_by_name(d_or_name: Dir | str, x: Dir, descend: bool) -> bool:
    match d_or_name:
        case Dir(name, _):
            dirname: str = name
        case str():
            dirname = d_or_name
        case _:
            raise TypeError(f"Expecting d_or_name to be Dir or str, not {type(d_or_name)}, {d_or_name=}")

    for c in x.contents:
        if isinstance(c, Dir):
            if c.name == dirname:
                return True
            if descend:
                found = _has_dir_by_name(dirname, c, descend=True)
                if found:
                    return True
    return False
