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

import os
import shlex
import subprocess
from pathlib import Path

from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.testing.data.load import load


data_path: Path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data"


def test_bionemo2_rootdir():
    data_error_str = (
        "Please download test data with:\n"
        "`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    )
    assert data_path.exists(), f"Could not find test data directory.\n{data_error_str}"
    assert data_path.is_dir(), f"Test data directory is supposed to be a directory.\n{data_error_str}"


def test_pretrain_cli_from_ckpt(tmpdir):
    # Same as test_pretrain, but includes a checkpoint to initialize from.
    data_path: Path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data"
    result_dir = Path(tmpdir.mkdir("results"))

    open_port = find_free_network_port()
    config = "/workspaces/bionemo-fw-ea/test_config.json"
    # Invoke with blocking
    checkpoint_path: Path = load("geneformer/10M_240530:2.0")
    cmd_str = f"""bionemo-geneformer-recipe --dest {config} --recipe test --data-path {data_path} --result-dir {result_dir} --initial-ckpt-path {checkpoint_path}""".strip()
    # continue when finished
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    # Now do pretrain
    if result.returncode != 0:
        print(f"{cmd_str=}")
        print(f"{result.stdout=}")
        print(f"{result.stderr=}")
        raise Exception(f"Pretrain recipe failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")

    cmd_str = f"""bionemo-geneformer-train --conf {config}""".strip()
    env = dict(**os.environ)  # a local copy of the environment
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Exception(f"Pretrain script failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")
    # NOTE this looks a lot like a magic value. But we also could do json.loads(config)['experiment_config']['experiment_name']
    assert (result_dir / "test-experiment").exists(), "Could not find test experiment directory."


def test_pretrain_cli(tmpdir):
    """trains from scratch"""
    data_path: Path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data"
    result_dir = Path(tmpdir.mkdir("results"))

    open_port = find_free_network_port()
    config = "test_config.json"
    # Invoke with blocking
    cmd_str = f"""bionemo-geneformer-recipe --dest {config} --recipe test --data-path {data_path} --result-dir {result_dir}""".strip()
    # continue when finished
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    # Now do pretrain
    if result.returncode != 0:
        print(f"{cmd_str=}")
        print(f"{result.stdout=}")
        print(f"{result.stderr=}")
        raise Exception(f"Pretrain recipe failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")

    cmd_str = f"""bionemo-geneformer-train --conf {config}""".strip()
    env = dict(**os.environ)  # a local copy of the environment
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Exception(f"Pretrain script failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")
    # NOTE this looks a lot like a magic value. But we also could do json.loads(config)['experiment_config']['experiment_name']
    assert (result_dir / "test-experiment").exists(), "Could not find test experiment directory."


def test_finetune_cli(tmpdir):
    """Uses CLI to invoke the entrypoint"""
    data_path: Path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data"
    result_dir = Path(tmpdir.mkdir("results"))
    checkpoint_path: Path = load("geneformer/10M_240530:2.0")

    open_port = find_free_network_port()

    # TODO use relative path when the test is working.
    config = "test_config.json"

    # TODO add initial path
    cmd_str = f"""bionemo-geneformer-recipe --dest {config} --recipe test-finetune --data-path {data_path} --result-dir {result_dir} --initial-ckpt-path {checkpoint_path}""".strip()
    # continue when finished
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    import sys

    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    # Now do pretrain
    if result.returncode != 0:
        print(f"{cmd_str=}")
        print(f"{result.stdout=}")
        print(f"{result.stderr=}")
        raise Exception(f"Pretrain recipe failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")

    # TODO gotta set the right config options here.
    # TODO set the parsing flag
    cmd_str = f"""bionemo-geneformer-train --conf {config} """.strip()
    env = dict(**os.environ)  # a local copy of the environment
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    print("starting the training invocation of evil")
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        # capture_output=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        raise Exception(f"Pretrain script failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")
    # NOTE this looks a lot like a magic value. But we also could do json.loads(config)['experiment_config']['experiment_name']
    assert (result_dir / "test-experiment").exists(), "Could not find test experiment directory."