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
import shutil
from typing import Optional

import pytest
import yaml
from git import Repo

from internal.jet.run_jet_downstream_pipeline import run_jet_downstream_pipeline
from internal.jet.utils.modify_jet_workloads_config import CONFIG_BUILD_NAME_DOCKER_IMAGE, CONFIG_NAME_RECIPE_PERF


# It downloads git repo and connects with JET CI by JET API as well as pulls and pushes to JET Workloads gitlab repo.
# Was used it for testing the components of JET pipeline and should not be executed in BioNeMo CI.
# It can be made it more pytest friendly if needed
pytest.skip(
    "Testing internal tools that access various NVIDIA APIs and rely on internet connection", allow_module_level=True
)

JET_REF = "bionemo/new-model-dev"
JET_REF_EPH = "test_branch"
JET_REF_TEST = "test_branch"
JET_WORKLOADS_PATH = os.path.join(os.path.dirname(os.getcwd()), "test_repo", "workloads-registry")

GIT_REPO = "repo_name_temp"
GIT_BRANCH = "branch_name_tmp"
DOCKERFILE = "/TMP/Dockerfile"
DOCKER_IMAGE = "docker_image_tmp"

CONFIG_PATH = "config_path_tmp"
CONFIG_NAME = "config_name_tmp"
SCRIPT_PATH = "script_path_tmp"

EXTRA_OVERWRITES = "extra_overwrites_tmp"
D_WARMUP = 20
MODEL = "model_tmp"
VARIANT = "variant_tmp"
BATCH_SIZE = [2, 4]
NODES = [1, 4]
GPUS = [1, 8]
PRECISIONS = ["16-mixed", 32]
SETUP_JET_CI = False
DRY_RUN = True

JET_REPO_URL = "https://gitlab-master.nvidia.com/dl/jet/workloads-registry"


def check_if_jet_workloads_path_exists():
    print(JET_WORKLOADS_PATH)
    assert os.path.isdir(JET_WORKLOADS_PATH)


def check_if_jet_config_exists():
    assert os.path.exists(os.path.join(JET_WORKLOADS_PATH, "builds", CONFIG_BUILD_NAME_DOCKER_IMAGE))
    assert os.path.exists(os.path.join(JET_WORKLOADS_PATH, "recipes", CONFIG_NAME_RECIPE_PERF))


def check_if_docker_image_matches():
    with open(os.path.join(JET_WORKLOADS_PATH, "builds", CONFIG_BUILD_NAME_DOCKER_IMAGE), "r+") as stream:
        config = next(yaml.load_all(stream, Loader=yaml.BaseLoader))
        assert config["spec"]["source"]["image"] == DOCKER_IMAGE


def check_if_dockerfile_spec_matches():
    with open(os.path.join(JET_WORKLOADS_PATH, "builds", CONFIG_BUILD_NAME_DOCKER_IMAGE), "r+") as stream:
        config = next(yaml.load_all(stream, Loader=yaml.BaseLoader))

        assert config["spec"]["source"]["repo"] == GIT_REPO
        assert config["spec"]["source"]["ref"] == GIT_BRANCH
        assert config["spec"]["source"]["dockerfile"] == DOCKERFILE
        assert ~("image" in config["spec"]["source"].keys())


def check_if_test_spec_matches():
    with open(os.path.join(JET_WORKLOADS_PATH, "recipes", CONFIG_NAME_RECIPE_PERF), "r+") as stream:
        config = next(yaml.load_all(stream, Loader=yaml.BaseLoader))

        assert config["spec"]["config_path"] == CONFIG_PATH
        assert config["spec"]["config_name"] == CONFIG_NAME
        assert config["spec"]["script_path"] == SCRIPT_PATH
        assert config["spec"]["model"] == MODEL
        assert config["spec"]["variant"] == VARIANT
        assert config["spec"]["dllogger_warmup"] == str(D_WARMUP)
        assert config["spec"]["extra_overwrites"] == EXTRA_OVERWRITES


def check_if_cases_spec_matches(batch_size, nodes, gpus, precision):
    with open(os.path.join(JET_WORKLOADS_PATH, "recipes", CONFIG_NAME_RECIPE_PERF), "r+") as stream:
        config = next(yaml.load_all(stream, Loader=yaml.BaseLoader))
        assert (
            [int(a) for a in config["products"][0]["batch_size"]] == batch_size
            if isinstance(batch_size, list)
            else [batch_size]
        )
        assert [int(a) for a in config["products"][0]["nodes"]] == nodes if isinstance(nodes, list) else [nodes]
        assert [int(a) for a in config["products"][0]["gpus"]] == gpus if isinstance(gpus, list) else [gpus]
        assert (
            [int(a) for a in config["products"][0]["precision"]] == precision
            if isinstance(precision, list)
            else [precision]
        )


def check_if_ephemeral_branch_exists(ref: Optional[str] = None):
    repo = Repo(path=JET_WORKLOADS_PATH)
    prefix = "ephemeral/bionemo"
    if ref is not None:
        assert f"{prefix}/{ref}" == repo.active_branch.name
    else:
        assert repo.active_branch.name.startswith(prefix)


if __name__ == "__main__":
    if os.path.exists(JET_WORKLOADS_PATH):
        shutil.rmtree(JET_WORKLOADS_PATH)
    config_kwargs = {"config_path": CONFIG_PATH, "config_name": CONFIG_NAME, "script_path": SCRIPT_PATH}

    test_kwargs = {
        "model": MODEL,
        "variant": VARIANT,
        "extra_overwrites": EXTRA_OVERWRITES,
        "dllogger_warmup": D_WARMUP,
    }

    cases_kwargs = {"batch_size": BATCH_SIZE, "nodes": NODES, "gpus": GPUS, "precision": PRECISIONS}

    run_jet_downstream_pipeline(
        jet_workloads_ref_default=JET_REF,
        jet_workloads_repo_path=JET_WORKLOADS_PATH,
        jet_workloads_ref_ephemeral=JET_REF_EPH,
        git_repo=GIT_REPO,
        git_branch=GIT_BRANCH,
        dockerfile=DOCKERFILE,
        setup_jet_api=SETUP_JET_CI,
        dry_run=DRY_RUN,
        **config_kwargs,
        **test_kwargs,
        **cases_kwargs,
    )

    check_if_jet_workloads_path_exists()
    check_if_ephemeral_branch_exists(ref=JET_REF_EPH)
    check_if_jet_config_exists()
    check_if_dockerfile_spec_matches()
    check_if_test_spec_matches()
    check_if_cases_spec_matches(**cases_kwargs)

    print(f"Deleting {JET_WORKLOADS_PATH}")
    shutil.rmtree(os.path.dirname(JET_WORKLOADS_PATH))

    run_jet_downstream_pipeline(
        jet_workloads_ref_default=JET_REF,
        jet_workloads_repo_path=JET_WORKLOADS_PATH,
        docker_image=DOCKER_IMAGE,
        model=MODEL,
        setup_jet_api=SETUP_JET_CI,
        dry_run=DRY_RUN,
        config_path=CONFIG_PATH,
        **{k: v[0] for k, v in cases_kwargs.items()},
    )
    check_if_jet_workloads_path_exists()
    check_if_ephemeral_branch_exists()
    check_if_jet_config_exists()
    check_if_docker_image_matches()
    check_if_cases_spec_matches(**{k: v[0] for k, v in cases_kwargs.items()})

    print(f"Deleting {JET_WORKLOADS_PATH}")
    shutil.rmtree(os.path.dirname(JET_WORKLOADS_PATH))

    try:
        run_jet_downstream_pipeline(
            jet_workloads_ref_default=JET_REF,
            jet_workloads_repo_path=JET_WORKLOADS_PATH,
            setup_jet_api=SETUP_JET_CI,
            dry_run=DRY_RUN,
        )
    except AssertionError as e:
        assert "The name of the model" in str(e)

    try:
        run_jet_downstream_pipeline(
            jet_workloads_ref_default=JET_REF,
            jet_workloads_repo_path=JET_WORKLOADS_PATH,
            docker_image=DOCKER_IMAGE,
            git_branch=GIT_BRANCH,
            model=MODEL,
            setup_jet_api=SETUP_JET_CI,
            dry_run=DRY_RUN,
        )
    except AssertionError as e:
        assert "Docker container is specified either by" in str(e)
