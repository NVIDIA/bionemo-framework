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


import gzip
import io
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import Mock, patch

import ngcbpc.environ
import pytest

from bionemo.core.data.load import default_ngc_client, default_pbss_client, load
from bionemo.core.data.resource import get_all_resources


def test_load_raises_error_on_invalid_tag(tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "bar"
          pbss: "s3://test/bar"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    with pytest.raises(ValueError, match="Resource 'invalid/tag' not found."):
        load("invalid/tag", resources=get_all_resources(tmp_path), cache_dir=tmp_path)


def test_load_cli():
    # It looks like there's some issues with our NGC resources, but this is blocking CI. TODO: Revert to ngc when these
    # resources are available.
    # FIXME (dorotat): set source=ngc once the access issue with NGC is resolved (https://github.com/NVIDIA/bionemo-framework/issues/682)
    result = subprocess.run(
        ["download_bionemo_data", "--source", "pbss", "single_cell/testdata-20240506"],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr (optional)
        text=True,  # Return output as string rather than bytes
    )
    assert result.returncode == 0
    path = Path(result.stdout.strip())
    assert path.exists()
    assert path.is_dir()
    assert str(path).startswith("/")
    assert str(path).endswith(".untar")


def test_get_resources_cli():
    result = subprocess.run(
        ["download_bionemo_data", "--list-resources"],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr (optional)
        text=True,  # Return output as string rather than bytes
    )
    resources = result.stdout.strip()
    result.check_returncode()
    assert resources


def test_load_raises_with_invalid_source(tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "bar"
          pbss: "s3://test/bar"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    with pytest.raises(ValueError, match="Source 'invalid' not supported."):
        load("foo/bar", source="invalid", resources=get_all_resources(tmp_path), cache_dir=tmp_path)  # type: ignore


def test_load_raises_with_no_ngc_url(tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "bar"
          pbss: "s3://test/bar"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    with pytest.raises(ValueError, match="Resource 'foo/bar' does not have an NGC URL."):
        load("foo/bar", source="ngc", resources=get_all_resources(tmp_path), cache_dir=tmp_path)  # type: ignore


@patch("bionemo.core.data.load._s3_download")
def test_load_with_file(mocked_s3_download, tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "bar"
          pbss: "s3://test/bar"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    mocked_s3_download.side_effect = lambda _1, output_file, _2: Path(output_file).write_text("test")
    file_path = load("foo/bar", resources=get_all_resources(tmp_path), cache_dir=tmp_path, source="pbss")
    assert file_path.is_file()
    assert file_path.read_text() == "test"


@patch("bionemo.core.data.load._s3_download")
def test_load_with_gzipped_file(mocked_s3_download, tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "baz"
          pbss: "s3://test/baz.gz"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    def write_compressed_text(_1, output_file: str, _2):
        with gzip.open(output_file, "wt") as f:
            f.write("test")

    mocked_s3_download.side_effect = write_compressed_text

    file_path = load("foo/baz", resources=get_all_resources(tmp_path), cache_dir=tmp_path, source="pbss")
    assert file_path.is_file()
    assert file_path.read_text() == "test"


@patch("bionemo.core.data.load._s3_download")
def test_load_with_gzipped_file_no_decomp(mocked_s3_download, tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "baz"
          pbss: "s3://test/baz.gz"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
          decompress: false
        """
    )

    def write_compressed_text(_1, output_file: str, _2):
        with gzip.open(output_file, "wt") as f:
            f.write("test")

    mocked_s3_download.side_effect = write_compressed_text

    file_path = load("foo/baz", resources=get_all_resources(tmp_path), cache_dir=tmp_path, source="pbss")

    # Assert the file remained compressed.
    assert file_path.is_file()
    with gzip.open(file_path, "rt") as f:
        assert f.read() == "test"


@patch("bionemo.core.data.load._s3_download")
def test_load_with_tar_directory(mocked_s3_download, tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "dir"
          pbss: "s3://test/dir.tar"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    def write_compressed_dir(_1, output_file: str, _2):
        # Create a text file in memory
        text_content = "test"
        text_file = io.BytesIO(text_content.encode("utf-8"))

        # Create a tarfile
        with tarfile.open(output_file, "w") as tar:
            # Create a TarInfo object for the file
            info = tarfile.TarInfo(name="test_file")
            info.size = len(text_content)

            # Add the file to the tarfile
            tar.addfile(info, text_file)

    mocked_s3_download.side_effect = write_compressed_dir

    file_path = load("foo/dir", resources=get_all_resources(tmp_path), cache_dir=tmp_path, source="pbss")
    assert file_path.is_dir()
    assert (file_path / "test_file").read_text() == "test"


@patch("bionemo.core.data.load._s3_download")
def test_load_with_tar_directory_no_unpack(mocked_s3_download, tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "dir"
          pbss: "s3://test/dir.tar"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
          unpack: false
        """
    )

    def write_tarfile_dir(_1, output_file: str, _2):
        # Create a text file in memory
        text_content = "test"
        text_file = io.BytesIO(text_content.encode("utf-8"))

        # Create a tarfile
        with tarfile.open(output_file, "w") as tar:
            # Create a TarInfo object for the file
            info = tarfile.TarInfo(name="test_file")
            info.size = len(text_content)

            # Add the file to the tarfile
            tar.addfile(info, text_file)

    mocked_s3_download.side_effect = write_tarfile_dir

    file_path = load("foo/dir", resources=get_all_resources(tmp_path), cache_dir=tmp_path, source="pbss")

    # Assert the file stays as a tarfile.
    assert file_path.is_file()
    with tarfile.open(file_path, "r") as tar:
        tar.extractall(tmp_path / "extracted")
        assert (tmp_path / "extracted/test_file").read_text() == "test"


@patch("bionemo.core.data.load._s3_download")
def test_load_with_targz_directory(mocked_s3_download, tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "dir.gz"
          pbss: "s3://test/dir.tar.gz"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    def write_compressed_dir(_1, output_file: str, _2):
        # Create a text file in memory
        text_content = "test"
        text_file = io.BytesIO(text_content.encode("utf-8"))

        # Create a tarfile
        with tarfile.open(output_file, "w") as tar:
            # Create a TarInfo object for the file
            info = tarfile.TarInfo(name="test_file")
            info.size = len(text_content)

            # Add the file to the tarfile
            tar.addfile(info, text_file)

    mocked_s3_download.side_effect = write_compressed_dir

    file_path = load("foo/dir.gz", resources=get_all_resources(tmp_path), cache_dir=tmp_path, source="pbss")
    assert file_path.is_dir()
    assert (file_path / "test_file").read_text() == "test"


def test_default_pbss_client():
    client = default_pbss_client()
    assert client.meta.endpoint_url == "https://pbss.s8k.io"


@patch("ngcbpc.api.authentication.Authentication.validate_api_key")
def test_default_ngc_client_raises_when_api_key_invalid(mocked_validate_api_key, monkeypatch):
    monkeypatch.setattr(ngcbpc.environ, "NGC_CLI_API_KEY", "invalidapikey")  # pragma: allowlist secret
    mocked_validate_api_key.return_value = False
    with pytest.raises(ValueError, match="Invalid apikey for NGC service location"):
        default_ngc_client(use_guest_if_api_key_invalid=False)


@patch("ngcbpc.api.authentication.Authentication.validate_api_key")
def test_default_ngc_client_returns_guest_key_when_api_key_invalid(mocked_validate_api_key, monkeypatch):
    monkeypatch.setattr(ngcbpc.environ, "NGC_CLI_API_KEY", "invalidapikey")  # pragma: allowlist secret
    mocked_validate_api_key.return_value = False
    client = default_ngc_client()
    assert client.api_key == "no-apikey"  # pragma: allowlist secret


@patch("bionemo.core.data.load.default_ngc_client")
def test_load_with_file_from_ngc_model(mocked_get_ngc_client, tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "ngc_model"
          pbss: "s3://test/bar"
          ngc: model/foo/bar:1.0
          ngc_registry: model
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    def mocked_ngc_download(url, destination, file_patterns):
        ngc_dirname = Path(url).name.replace(":", "_v")
        file_name = file_patterns[0]

        (Path(destination) / ngc_dirname).mkdir(parents=True, exist_ok=True)
        (Path(destination) / ngc_dirname / file_name).write_text("test")

    mocked_ngc_client = Mock()
    mocked_ngc_client.registry.model.download_version.side_effect = mocked_ngc_download
    mocked_get_ngc_client.return_value = mocked_ngc_client

    file_path = load("foo/ngc_model", resources=get_all_resources(tmp_path), source="ngc", cache_dir=tmp_path)
    assert file_path.is_file()
    assert file_path.read_text() == "test"

    mocked_ngc_client.registry.model.download_version.assert_called_once()


@patch("bionemo.core.data.load.default_ngc_client")
def test_load_with_file_from_ngc_resource(mocked_get_ngc_client, tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "ngc_resource"
          pbss: "s3://test/bar"
          ngc: model/foo/bar:1.0
          ngc_registry: resource
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
          """
    )

    def mocked_ngc_download(url, destination, file_patterns):
        ngc_dirname = Path(url).name.replace(":", "_v")
        file_name = file_patterns[0]

        (Path(destination) / ngc_dirname).mkdir(parents=True, exist_ok=True)
        (Path(destination) / ngc_dirname / file_name).write_text("test")

    mocked_ngc_client = Mock()
    mocked_ngc_client.registry.resource.download_version.side_effect = mocked_ngc_download
    mocked_get_ngc_client.return_value = mocked_ngc_client

    file_path = load("foo/ngc_resource", resources=get_all_resources(tmp_path), source="ngc", cache_dir=tmp_path)
    assert file_path.is_file()
    assert file_path.read_text() == "test"

    mocked_ngc_client.registry.resource.download_version.assert_called_once()
