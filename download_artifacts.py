# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Script to download pretrained models from NGC or PBSS."""

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


ALL_KEYWORD = "all"
DATA_SOURCE_CONFIG = Path("artifact_paths.yaml")
ArtifactSource = Literal["ngc", "pbss"]


#####################################################
# Define the structure of the DATA_SOURCE_CONFIG file
#####################################################
class SymlinkConfig(BaseModel):
    source: Path
    target: Path


class ArtifactConfig(BaseModel):
    ngc: Optional[str] = None
    pbss: Optional[str] = None
    symlink: Optional[SymlinkConfig] = None
    relative_download_dir: Optional[Path] = None
    extra_args: Optional[str] = None
    untar_dir: Optional[str] = None
    md5sum: Optional[str] = None


class Config(BaseModel):
    models: Dict[str, ArtifactConfig]
    data: Dict[str, ArtifactConfig]
    """
    @model_validator(mode="after")
    def check_download_source_exists(cls, values):
        for model_name, model_config in values.models.items():
            if model_config.ngc is None and model_config.pbss is None:
                raise ValueError(f"Model {model_name} doesn't have a NGC or PBSS download path.")

        for data_name, data_config in values.data.items():
            if data_config.ngc is None and data_config.pbss is None:
                raise ValueError(f"Data {data_name} doesn't have a NGC or PBSS download path.")

        return values
    """


#####################################################
# End config definition
#####################################################


def streamed_subprocess_call(cmd: str, stream_stdout: bool = False) -> Tuple[str, str, int]:
    """
    Run a command in a subprocess, streaming its output and handling errors.

    Args:
        cmd (str): The bash command to be executed.
        stream_stdout (bool, optional): If True, print the command's stdout during execution.

    Returns:
        (str, str, int): The stdout string, stderr string, and return code integer.
    Raises:
        CalledProcessError: If the subprocess exits with a non-zero return code.

    Note:
        This function uses subprocess.Popen to run the specified command.
        If `stream_stdout` is True, the stdout of the subprocess will be streamed to the console.
        ANSI escape sequences used by certain commands may interfere with the output.

    Example:
        >>> streamed_subprocess_call("ls -l", stream_stdout=True)
        Running command: ls -l
        total 0
        -rw-r--r-- 1 user user 0 Dec 10 12:00 example.txt
        Done.
    """
    stdout: List[str] = []
    stderr: str = ""
    print(f"Running command: {cmd}\n")
    with Popen(cmd, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            # TODO: ngc uses ANSI escape sequences to inline-refresh their console logs,
            # and it doesn't play well with using \r in python. Need to figure out how to
            # prevent logspam here.
            stdout.append(line)
            if stream_stdout:
                print(line, end="")
        p.wait()

        if p.returncode != 0:
            stderr = p.stderr.read()
            print(stderr, file=sys.stderr)
        else:
            print("\nDone.")
    return "".join(stdout), stderr, p.returncode


def get_available_models(config: Config, source: ArtifactSource) -> List[str]:
    """
    Get a list of models that are available from a given source.

    Args:
        config (Config): The artifacts configuration.
        source (str): The source of the models to download, "ngc" or "pbss".

    Returns:
        List: The list of models available in the given source.
    """
    available_models = []
    for model in list(config.models.keys()):
        if getattr(config.models[model], source):
            available_models.append(model)
    return available_models


def check_ngc_cli():
    """Checks if NGC CLI is present on the system."""
    _, _, exit_code = streamed_subprocess_call("ngc --version", stream_stdout=True)
    if exit_code == 0:
        return True
    else:
        return False


def check_and_install_ngc_cli(ngc_install_dir="/tmp"):
    """
    Checks if NGC CLI is present on the system, and installs if it isn't present.
    """
    print(f"Installing NGC CLI to {ngc_install_dir}")
    # Very important to run wget in quiet mode, or else an I/O deadlock happens.
    # It's this issue, and it's not trivial to fix without breaking the streaming
    # functionality of streamed_subprocess_call.
    # https://stackoverflow.com/questions/39477003/python-subprocess-popen-hanging
    INSTALL_COMMAND = (
        f"wget -q -O /tmp/ngccli_linux.zip --content-disposition "
        f"https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.38.0/files/ngccli_linux.zip && "
        f"unzip -o /tmp/ngccli_linux.zip -d {ngc_install_dir} && "
        f"chmod u+x {ngc_install_dir}/ngc-cli/ngc && "
        f"rm /tmp/ngccli_linux.zip"
    )
    stdout, stderr, exit_code = streamed_subprocess_call(INSTALL_COMMAND, stream_stdout=True)
    if exit_code == 0:
        print("NGC CLI successfully installed.")
    else:
        raise ValueError(f"Unable to install NGC CLI: {stderr}")


def download_artifacts(
    config: Config,
    artifact_list: List,
    source: ArtifactSource,
    download_dir_base: Path,
    stream_stdout: bool = False,
    artifact_type: str = "model",
) -> None:
    """
    Download models or data from a given source.

    Args:
        config (Config): The artifacts configuration.
        artifact_list (List): A list of model or data names to download that should be present in
                           the config.
        source (str): The source of the models to download, "ngc" or "pbss".
        download_dir_base (str): The target local directory for download.
        stream_stdout (bool): If true, stream the subprocess calls to stdout.
        artifact_type (str): Whether it's a model or data that we are downloading. Should be model or data
    """
    if len(artifact_list) == 0:
        raise ValueError("Must supply non-empty model or data list for download!")
    if source == "ngc":
        if check_ngc_cli():
            ngc_call_command = "ngc"
        else:
            NGC_INSTALL_DIR = "/tmp"
            check_and_install_ngc_cli(NGC_INSTALL_DIR)
            ngc_call_command = str(os.path.join(NGC_INSTALL_DIR, "ngc-cli/ngc"))
    if artifact_type == "model":
        conf = config.models
    else:
        conf = config.data
    for download_artifact in artifact_list:
        artifact_source_path = getattr(conf[download_artifact], source)
        if not artifact_source_path:
            print(f"Warning: {download_artifact} does not have a {source} URL; skipping download.")
            continue

        if conf[download_artifact].relative_download_dir:
            complete_download_dir = download_dir_base / conf[download_artifact].relative_download_dir
        else:
            complete_download_dir = download_dir_base

        if source == "ngc":
            # NGC seems to always download to a specific directory that we can't
            # specify ourselves
            ngc_dirname = Path(os.path.split(artifact_source_path)[1].replace(":", "_v"))
            ngc_dirname = complete_download_dir / ngc_dirname

            # TODO: this assumes that it's a model for now.
            command = f"mkdir -p {str(complete_download_dir)} && {ngc_call_command} registry model download-version {artifact_source_path} --dest {str(complete_download_dir)} && mv {str(ngc_dirname)}/* {str(complete_download_dir)}/ && rm -d {str(ngc_dirname)}"
            file_name = artifact_source_path.split("/")[-1]
        elif source == "pbss":
            command = f"aws s3 cp {str(artifact_source_path)} {str(complete_download_dir)}/ --endpoint-url https://pbss.s8k.io"
            file_name = artifact_source_path.split("/")[-1]
        if conf[download_artifact].extra_args:
            extra_args = conf[download_artifact].extra_args
            command = f"{command} {extra_args}"

        execute_download(stream_stdout, conf, download_artifact, complete_download_dir, command, file_name, source)

        if artifact_type == "data" and conf[download_artifact].untar_dir is not None:
            tar_file = f"{str(complete_download_dir)}/{file_name}"
            if Path(tar_file).is_file():
                with tarfile.open(tar_file) as tar:
                    extract_path = f"{str(complete_download_dir)}"
                    if conf[download_artifact].untar_dir:
                        extract_path = f"{extract_path}/{conf[download_artifact].untar_dir}"
                    tar.extractall(path=extract_path)
                Path(tar_file).unlink()

        # Create symlinks, if necessary
        if conf[download_artifact].symlink:
            source_file = conf[download_artifact].symlink.source
            target_file = complete_download_dir / conf[download_artifact].symlink.target
            target_dir = target_file.parent
            command = f"mkdir -p {target_dir} && ln -sf {str(source_file)} {str(target_file)}"
            _, stderr, retcode = streamed_subprocess_call(command, stream_stdout=True)
            if retcode != 0:
                raise ValueError(f"Failed to symlink {source_file=} to {target_file=}; {stderr=}")


@retry(
    wait=wait_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(ValueError),
    stop=stop_after_attempt(3),
    reraise=True,
)
def execute_download(
    stream_stdout: bool,
    conf: Dict[str, ArtifactConfig],
    download_artifact: str,
    complete_download_dir: Path,
    command: List[str],
    file_name: str,
    source: str,
) -> None:
    """Execute the download command and check the MD5 checksum of the downloaded file."""

    _, stderr, retcode = streamed_subprocess_call(command, stream_stdout)
    if retcode != 0:
        raise ValueError(f"Failed to download {download_artifact=}! {stderr=}")

    if source == "pbss" and conf[download_artifact].md5sum:
        downloaded_md5sum = _md5_checksum(Path(complete_download_dir) / file_name)
        if downloaded_md5sum != conf[download_artifact].md5sum:
            raise ValueError(
                f"MD5 checksum mismatch for {download_artifact=}! Expected "
                f"{conf[download_artifact].md5sum}, got {downloaded_md5sum}"
            )


def load_config(config_file: Path = DATA_SOURCE_CONFIG) -> Config:
    """
    Loads the artifacts file into a dictionary.

    Return:
        (Config): The configuration dictionary that specifies where and how to download models.
    """
    with open(DATA_SOURCE_CONFIG, "rt") as rt:
        config_data = yaml.safe_load(rt)
    return Config(**config_data)


def main():
    """
    Script to download pretrained checkpoints from PBSS (SwiftStack) or NGC.

    After the models are downloaded, symlinked paths are created. The models and symlinks
    are all defined in DATA_SOURCE_CONFIG.
    """
    config = load_config()
    all_models_list = list(config.models.keys())
    all_data_list = list(config.data.keys())
    parser = argparse.ArgumentParser(description="Pull pretrained model checkpoints and corresponding data.")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=all_models_list + [ALL_KEYWORD],
        help="Name of the model (optional if downloading all models)",
    )

    parser.add_argument(
        "--data",
        nargs="*",
        choices=all_data_list + [ALL_KEYWORD],
        help="Name of the data (optional if downloading all data)",
    )

    parser.add_argument(
        "--model_dir",
        default=".",
        type=str,
        help="Directory into which download and symlink the model.",
    )

    parser.add_argument(
        "--data_dir",
        default=".",
        type=str,
        help="Directory into which download and symlink the data.",
    )

    parser.add_argument(
        "--source",
        choices=list(ArtifactSource.__args__),
        default="ngc",
        help="Pull model from NVIDIA GPU Cloud (NGC) or SwiftStack (internal). Default is NGC.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print model download progress.")
    args = parser.parse_args()
    if args.models:
        if ALL_KEYWORD in args.models:
            download_list = all_models_list
        else:
            download_list = args.models

        download_artifacts(
            config, download_list, args.source, Path(args.model_dir), args.verbose, artifact_type="model"
        )
    if args.data:
        if ALL_KEYWORD in args.data:
            download_list = all_data_list
        else:
            download_list = args.data
        download_artifacts(config, download_list, args.source, Path(args.data_dir), args.verbose, "data")

    if not (args.models or args.data):
        print("No models or data were selected to download.")


def _md5_checksum(file_path: Path) -> str:
    """Calculate the MD5 checksum of a file.

    Args:
        file_path (Path): The path to the file to checksum.

    Returns:
        str: The MD5 checksum of the file.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


if __name__ == "__main__":
    main()
