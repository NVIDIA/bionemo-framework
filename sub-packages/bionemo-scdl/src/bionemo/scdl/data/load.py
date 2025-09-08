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

import contextlib
import functools
import itertools
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Sequence

import nest_asyncio
import platformdirs
import pooch
import pydantic
import yaml
from botocore.config import Config
from registry.api.utils import RegistryTarget
from tqdm import tqdm


if TYPE_CHECKING:
    import ngcsdk

logger = pooch.get_logger()


def _get_cache_dir() -> Path:
    """Get the cache directory for downloaded resources."""
    if cache_dir := os.getenv("BIONEMO_CACHE_DIR"):
        return Path(cache_dir)

    cache_dir = Path(platformdirs.user_cache_dir(appname="bionemo", appauthor="nvidia"))

    try:
        cache_dir.mkdir(exist_ok=True, parents=True)
    except PermissionError as ex:
        raise PermissionError(
            f"Permission denied creating a cache directory at {cache_dir}. Please set BIONEMO_CACHE_DIR to a directory "
            "you have write access to."
        ) from ex
    return cache_dir


BIONEMO_CACHE_DIR = _get_cache_dir()


def _validate_ngc_resource(value: str) -> str:
    return str(RegistryTarget(value, "Pattern should be in format [org/[team/]]name[:version]"))


class Resource(pydantic.BaseModel):
    """Class that represents a remote resource for downloading and caching test data."""

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    tag: Annotated[str, pydantic.StringConstraints(pattern=r"^[^/]*/[^/]*$")]  # Only slash between filename and tag.
    """A unique identifier for the resource. The file(s) will be accessible via load("filename/tag")."""

    ngc: Annotated[str, pydantic.AfterValidator(_validate_ngc_resource)] | None = None
    """The NGC URL for the resource.

    Should be in format [org/[team/]]name[:version]. If None, the resource is not available on NGC.
    """

    ngc_registry: Literal["model", "resource"] | None = None
    """The NGC resource type (model or resource) for the data. Must be provided if ngc is not None."""

    pbss: Annotated[pydantic.AnyUrl, pydantic.UrlConstraints(allowed_schemes=["s3"])]
    """The PBSS (NVIDIA-internal) URL of the resource."""

    sha256: str | None
    """The SHA256 checksum of the resource. If None, the SHA will not be checked on download (not recommended)."""

    owner: pydantic.NameEmail
    """The owner or primary point of contact for the resource, in the format "Name <email>"."""

    description: str | None = None
    """A description of the file(s)."""

    unpack: Literal[False, None] = None
    """Whether the resource should be unpacked after download. If None, will defer to the file extension."""

    decompress: Literal[False, None] = None
    """Whether the resource should be decompressed after download. If None, will defer to the file extension."""

    @pydantic.model_validator(mode="after")
    def _validate_ngc_registry(self):
        if self.ngc and not self.ngc_registry:
            raise ValueError(f"ngc_registry must be provided if ngc is not None: {self.tag}")
        return self


@functools.cache
def get_all_resources(resource_path: Path | None = None) -> dict[str, Resource]:
    """Return a dictionary of all resources."""
    if not resource_path:
        # Default to the local scdl resources directory
        resource_path = Path(__file__).parent / "resources"

    resources_files = itertools.chain(resource_path.glob("*.yaml"), resource_path.glob("*.yml"))

    all_resources = [resource for file in resources_files for resource in _parse_resource_file(file)]

    resource_list = pydantic.TypeAdapter(list[Resource]).validate_python(all_resources)
    resource_dict = {resource.tag: resource for resource in resource_list}

    if len(resource_dict) != len(resource_list):
        # Show the # of and which ones are duplicated so that a user can begin debugging and resolve the issue.
        tag_counts = Counter([resource.tag for resource in resource_list])
        raise ValueError(f"Duplicate resource tags found!: {[tag for tag, count in tag_counts.items() if count > 1]}")

    return resource_dict


def _parse_resource_file(file) -> list[dict[str, Any]]:
    with file.open("r") as f:
        resources = yaml.safe_load(f)
        for resource in resources:
            resource["tag"] = f"{file.stem}/{resource['tag']}"
        return resources


__all__: Sequence[str] = (
    "NGCDownloader",
    "Resource",
    "default_ngc_client",
    "default_pbss_client",
    "get_all_resources",
    "load",
)
SourceOptions = Literal["ngc", "pbss"]
DEFAULT_SOURCE: SourceOptions = os.environ.get("BIONEMO_DATA_SOURCE", "ngc")  # type: ignore


def default_pbss_client():
    """Create a default S3 client for PBSS."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required to download from PBSS.")

    retry_config = Config(retries={"max_attempts": 10, "mode": "standard"})
    return boto3.client("s3", endpoint_url="https://pbss.s8k.io", config=retry_config)


def _s3_download(url: str, output_file: str | Path, _: pooch.Pooch) -> None:
    """Download a file from PBSS."""
    # Parse S3 URL to get bucket and key
    parts = url.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])

    with contextlib.closing(default_pbss_client()) as s3:
        object_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        progress_bar = tqdm(total=object_size, unit="B", unit_scale=True, desc=url)

        # Define callback
        def progress_callback(bytes_transferred):
            progress_bar.update(bytes_transferred)

        # Download file from S3
        s3.download_file(bucket, key, output_file, Callback=progress_callback)


def default_ngc_client(use_guest_if_api_key_invalid: bool = True) -> "ngcsdk.Client":
    """Create a default NGC client.

    This should load the NGC API key from ~/.ngc/config, or from environment variables passed to the docker container.
    """
    import ngcsdk

    client = ngcsdk.Client()

    try:
        client.configure()

    except ValueError as e:
        if use_guest_if_api_key_invalid:
            logger.error(f"Error configuring NGC client: {e}, signing in as guest.")
            client = ngcsdk.Client("no-apikey")
            client.configure(
                api_key="no-apikey",  # pragma: allowlist secret
                org_name="no-org",
                team_name="no-team",
                ace_name="no-ace",
            )

        else:
            raise

    return client


@dataclass
class NGCDownloader:
    """A class to download files from NGC in a Pooch-compatible way.

    NGC downloads are typically structured as directories, while pooch expects a single file. This class
    downloads a single file from an NGC directory and moves it to the desired location.
    """

    filename: str
    ngc_registry: Literal["model", "resource"]

    def __call__(self, url: str, output_file: str | Path, _: pooch.Pooch) -> None:
        """Download a file from NGC."""
        client = default_ngc_client()
        nest_asyncio.apply()

        download_fns = {
            "model": client.registry.model.download_version,
            "resource": client.registry.resource.download_version,
        }

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # NGC seems to always download to a specific directory that we can't specify ourselves.
        ngc_dirname = Path(url).name.replace(":", "_v")

        with tempfile.TemporaryDirectory(dir=output_file.parent) as temp_dir:
            download_fns[self.ngc_registry](url, temp_dir, file_patterns=[self.filename])
            shutil.move(Path(temp_dir) / ngc_dirname / self.filename, output_file)


def load(
    model_or_data_tag: str,
    source: SourceOptions = DEFAULT_SOURCE,
    resources: dict[str, Resource] | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download a resource from PBSS or NGC.

    Args:
        model_or_data_tag: A pointer to the desired resource. Must be a key in the resources dictionary.
        source: Either "pbss" (NVIDIA-internal download) or "ngc" (NVIDIA GPU Cloud). Defaults to "pbss".
        resources: A custom dictionary of resources. If None, the default resources will be used. (Mostly for testing.)
        cache_dir: The directory to store downloaded files. Defaults to BIONEMO_CACHE_DIR. (Mostly for testing.)

    Raises:
        ValueError: If the desired tag was not found, or if an NGC url was requested but not provided.

    Returns:
        A Path object pointing either at the downloaded file, or at a decompressed folder containing the
        file(s).

    Examples:
        For a resource specified in 'filename.yaml' with tag 'tag', the following will download the file:
        >>> load("filename/tag")
        PosixPath(/tmp/bionemo/downloaded-file-name)
    """
    if resources is None:
        # Get resources from the local scdl data directory
        resources = get_all_resources()

    if cache_dir is None:
        cache_dir = BIONEMO_CACHE_DIR

    if model_or_data_tag not in resources:
        raise ValueError(f"Resource '{model_or_data_tag}' not found.")

    if source == "ngc" and resources[model_or_data_tag].ngc is None:
        raise ValueError(f"Resource '{model_or_data_tag}' does not have an NGC URL.")

    resource = resources[model_or_data_tag]
    filename = str(resource.pbss).split("/")[-1]

    extension = "".join(Path(filename).suffixes)
    processor = _get_processor(extension, resource.unpack, resource.decompress)

    if source == "pbss":
        download_fn = _s3_download
        url = resource.pbss

    elif source == "ngc":
        assert resource.ngc_registry is not None
        download_fn = NGCDownloader(filename=filename, ngc_registry=resource.ngc_registry)
        url = resource.ngc

    else:
        raise ValueError(f"Source '{source}' not supported.")

    download = pooch.retrieve(
        url=str(url),
        fname=f"{resource.sha256}-{filename}",
        known_hash=resource.sha256,
        path=cache_dir,
        downloader=download_fn,
        processor=processor,
    )

    # Pooch by default returns a list of unpacked files if they unpack a zipped or tarred directory. Instead of that, we
    # just want the unpacked, parent folder.
    if isinstance(download, list):
        return Path(processor.extract_dir)  # type: ignore

    else:
        return Path(download)


def _get_processor(extension: str, unpack: bool | None, decompress: bool | None):
    """Get the processor for a given file extension.

    If unpack and decompress are both None, the processor will be inferred from the file extension.

    Args:
        extension: The file extension.
        unpack: Whether to unpack the file.
        decompress: Whether to decompress the file.

    Returns:
        A Pooch processor object.
    """
    if extension in {".gz", ".bz2", ".xz"} and decompress is None:
        return pooch.Decompress()

    elif extension in {".tar", ".tar.gz"} and unpack is None:
        return pooch.Untar()

    elif extension == ".zip" and unpack is None:
        return pooch.Unzip()

    else:
        return None
