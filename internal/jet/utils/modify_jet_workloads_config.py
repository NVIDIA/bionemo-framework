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


import logging
import os
from typing import List, Optional, Union

import ruamel.yaml as yaml


ALLOWED_BUILD_CONFIG_KWARGS_KEYS = ["docker_image", "git_repo", "git_branch", "dockerfile"]
ALLOWED_RECIPE_CONFIG_KWARGS_KEYS = [
    "config_path",
    "config_name",
    "script_path",
    "model",
    "variant",
    "extra_overwrites",
    "dllogger_warmup",
    "batch_size",
    "nodes",
    "gpus",
    "precision",
]
CONFIG_TYPES = ["build", "recipe"]

CONFIG_BUILD_NAME_DOCKER_IMAGE = "bionemo.yaml"
CONFIG_NAME_RECIPE_PERF = "config.yaml"
CONFIG_NAME_RECIPE_CONV = "config_conv.yaml"


def modify_jet_workloads_config(
    jet_workloads_repo_path: str,
    docker_image: Optional[str] = None,
    git_repo: Optional[str] = None,
    git_branch: Optional[str] = None,
    dockerfile: Optional[str] = None,
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    script_path: Optional[str] = None,
    variant: Optional[str] = None,
    model: Optional[str] = None,
    extra_overwrites: Optional[str] = None,
    dllogger_warmup: Optional[int] = None,
    nodes: Optional[Union[List[int], int]] = None,
    gpus: Optional[Union[List[int], int]] = None,
    precision: Optional[Union[List[int], int]] = None,
    batch_size: Optional[Union[List[int], int]] = None,
):
    """
    Modifies recipe or build related configs in local copy of JET Workloads Registry given the branch name jet_workload_ref
    Args:
        jet_workloads_repo_path: path to the local copy of JET Workloads Registry
        docker_image: name of the docker image
        git_repo: git repository url to build docker image from
        git_branch: git branch to use to build docker image
        dockerfile: path in the repository to the Dockerfile to build the container
        config_path: path to the folder with primary training/testing configs, relative to $BIONEMO_HOME
        config_name: hydra config name to use that can be found under config_path
        script_path: path to a folder with training/testing scripts to use, relative to $BIONEMO_HOME
        variant: name of a training/testing script to use (without .py extension), relative to the script_path
        model: name of the model to be tested
        extra_overwrites: additional training configs to be passed
        dllogger_warmup: int, warmup steps of DLLOGGER before the metrics are logged
        nodes: number(s) of nodes to be tested
        gpus: number(s) of devices (gpus) to be tested
        precision: precision(s) to be tested
        batch_size: batch size(s) to be tested

    Returns:

    """

    assert os.path.isdir(jet_workloads_repo_path), (
        f"JET Workloads Repository does not exists: " f"{jet_workloads_repo_path}"
    )

    if docker_image:
        _modify_jet_workloads_config(
            config_type="build",
            config_name=CONFIG_BUILD_NAME_DOCKER_IMAGE,
            jet_workloads_repo_path=jet_workloads_repo_path,
            config_kwargs={"docker_image": docker_image},
        )
    if git_repo and git_branch and dockerfile:
        _modify_jet_workloads_config(
            config_type="build",
            config_name=CONFIG_BUILD_NAME_DOCKER_IMAGE,
            jet_workloads_repo_path=jet_workloads_repo_path,
            config_kwargs={"git_repo": git_repo, "git_branch": git_branch, "dockerfile": dockerfile},
        )

    config_kwargs = {
        "config_path": config_path,
        "config_name": config_name,
        "script_path": script_path,
        "model": model,
        "variant": variant,
        "extra_overwrites": extra_overwrites,
        "dllogger_warmup": dllogger_warmup,
        "batch_size": batch_size,
        "nodes": nodes,
        "gpus": gpus,
        "precision": precision,
    }

    if any(v is not None for _, v in config_kwargs.items()):
        _modify_jet_workloads_config(
            config_type="recipe",
            config_name=CONFIG_NAME_RECIPE_PERF,
            jet_workloads_repo_path=jet_workloads_repo_path,
            config_kwargs=config_kwargs,
        )


def _modify_jet_workloads_config(
    config_type: str, config_name: str, jet_workloads_repo_path: str = None, config_kwargs: dict = {}
) -> None:
    """
    Modifies either recipe or build related configs
    Args:
        config_type: either recipe (test and model spec) or build (docker image build spec)
        config_name: name of the config to
        jet_workloads_repo_path: path to the local copy of JET Workloads Registry, default to parent dir of cwd
        config_kwargs: arguments to be updated as key=value
    """
    assert config_type in CONFIG_TYPES, f"config_type must be either of {CONFIG_TYPES}"
    if config_type == "build":
        workloads_config_path = os.path.join(jet_workloads_repo_path, "builds", config_name)
        assert all(key in ALLOWED_BUILD_CONFIG_KWARGS_KEYS for key in config_kwargs.keys())
        _modify_jet_workloads_build_config(workloads_config_path=workloads_config_path, **config_kwargs)
    else:
        assert all(key in ALLOWED_RECIPE_CONFIG_KWARGS_KEYS for key in config_kwargs.keys())
        workloads_config_path = os.path.join(jet_workloads_repo_path, "recipes", config_name)
        _modify_jet_workloads_recipe_config(workloads_config_path=workloads_config_path, **config_kwargs)


def _modify_jet_workloads_recipe_config(
    workloads_config_path: str,
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    script_path: Optional[str] = None,
    variant: Optional[str] = None,
    model: Optional[str] = None,
    extra_overwrites: Optional[str] = None,
    dllogger_warmup: Optional[int] = None,
    nodes: Optional[Union[List[int], int]] = None,
    gpus: Optional[Union[List[int], int]] = None,
    precision: Optional[Union[List[int], int]] = None,
    batch_size: Optional[Union[List[int], int]] = None,
) -> None:
    """
    Replaces default model configuration in the jet test with user-specific specification of the model.

    Recipe configs in JET Workload specify tests to be run in JET.
    For more info about the config structure, please refer to configs in
    https://gitlab-master.nvidia.com/dl/jet/workloads-registry/-/tree/bionemo/training-unit-tests/recipes
    Args:
        workloads_config_path: path to the workloads config with the recipe/test definition
        config_path: path to the folder with primary training/testing configs, relative to $BIONEMO_HOME
        config_name: hydra config name to use that can be found under config_path
        script_path: path to a folder with training/testing scripts to use, relative to $BIONEMO_HOME
        variant: name of a training/testing script to use (without .py extension), relative to the script_path
        model: name of the model to be tested
        extra_overwrites: additional training configs to be passed
        dllogger_warmup: int, warmup steps of DLLOGGER before the metrics are logged
        nodes: number(s) of nodes to be tested
        gpus: number(s) of devices (gpus) to be tested
        precision: precision(s) to be tested
        batch_size: batch size(s) to be tested

    """

    assert os.path.exists(workloads_config_path), f"JET workload config:{workloads_config_path} does not exist"
    logging.info(f"Overwriting model specification in the jet config:{workloads_config_path}")

    with open(workloads_config_path, "r") as stream:
        config = yaml.load(stream)

    if config_path is not None:
        config["spec"]["config_path"] = config_path

    if config_name is not None:
        config["spec"]["config_name"] = config_name

    if script_path is not None:
        config["spec"]["script_path"] = script_path

    if model is not None:
        config["spec"]["model"] = model

    if variant is not None:
        config["spec"]["variant"] = variant

    if dllogger_warmup is not None:
        config["spec"]["dllogger_warmup"] = dllogger_warmup

    if extra_overwrites is not None:
        config["spec"]["extra_overwrites"] = extra_overwrites

    if batch_size is not None:
        config["products"][0]["batch_size"] = batch_size if isinstance(batch_size, list) else [batch_size]

    if nodes is not None:
        config["products"][0]["nodes"] = nodes if isinstance(nodes, list) else [nodes]

    if gpus is not None:
        config["products"][0]["gpus"] = gpus if isinstance(gpus, list) else [gpus]

    if precision is not None:
        config["products"][0]["precision"] = precision if isinstance(precision, list) else [precision]

    with open(workloads_config_path, "w") as stream:
        yaml.dump(config, stream)


def _modify_jet_workloads_build_config(
    workloads_config_path: str,
    docker_image: Optional[str] = None,
    git_repo: Optional[str] = None,
    git_branch: Optional[str] = None,
    dockerfile: Optional[str] = None,
) -> None:
    """
    Modifies a build config in the jet workload repository given a config path and fields to update.
    Build configs in JET Workload are used to build docker images that are at the later stage run with
    the model tests.

    For more info about the config structure, please refer to configs in
    https://gitlab-master.nvidia.com/dl/jet/workloads-registry/-/tree/bionemo/training-unit-tests/builds
    Args:
        workloads_config_path: path to the config
        docker_image: name of the docker image
        git_repo: git repository url to build docker image from
        git_branch: git branch to use to build docker image
        dockerfile: path to the Dockerfile to use ot buil the docker image
    """
    assert os.path.exists(workloads_config_path), f"JET workload config:{workloads_config_path} does not exist"
    logging.info(f"Modifying build specification in the jet config:{workloads_config_path}")

    with open(workloads_config_path, "r") as stream:
        config = yaml.load(stream)

    if docker_image:
        config["spec"]["source"]["image"] = docker_image

    if git_repo and git_branch and dockerfile:
        config["spec"]["source"].pop("image")
        config["spec"]["source"]["repo"] = git_repo
        config["spec"]["source"]["ref"] = git_branch
        config["spec"]["source"]["dockerfile"] = dockerfile

    with open(workloads_config_path, "w") as stream:
        yaml.dump(config, stream)
