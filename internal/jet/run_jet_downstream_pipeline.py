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
from argparse import ArgumentParser
from typing import List, Optional, Union

from internal.jet.utils.handlers import JetGitRepoHandler, JetPipelineHandler
from internal.jet.utils.modify_jet_workloads_config import modify_jet_workloads_config


# using logging not from NeMo to run this script outside a container
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def run_jet_downstream_pipeline(
    jet_workloads_ref_default: str,
    jet_workloads_repo_path: str,
    jet_workloads_ref_ephemeral: Optional[str] = None,
    git_repo: Optional[str] = None,
    git_branch: Optional[str] = None,
    dockerfile: Optional[str] = None,
    docker_image: Optional[str] = None,
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
    jet_filter: Optional[str] = None,
    setup_jet_api: bool = False,
    dry_run: bool = False,
):
    """
    Runs JET tests for a model or docker specified by the arguments.

    Requires installed JET API https://gitlab-master.nvidia.com/dl/jet/api. Can be executed outside docker container.

    The script:
    1. Clones the Jet Workloads Registry repository to jet_workloads_repo_path and setup jet_workloads_ref_ephemeral
    2. Modifies default tests configs with optional new parameters passed by git_repo, git_branch, docker_image, model_type,
       variant or batch_size
    3. Pushes the jet_workloads_ref_ephemeral with modified configs to JET Workloads Registry
    4. Run tests in JET CI under JET reference equal to jet_workloads_ref_ephemeral and prints out pipeline's details

    ## USAGE
    # Running tests of a model that is already part of BioNeMo FW and is present in dev:latest-devel docker

    python internal/jet/run_jet_downstream_pipeline.py --model <MODEL_TYPE> --config_path <CONFIG_PATH>
    --config_name <CONFIG_NAME> --script_path <SCRIPT_PATH> --variant <VARIANT> --nodes <NODE1> <NODE2>
    --gpus <GPU1> <GPUS2> --precision <PREC1> <PREC2> --batch_size <BATCH_SIZE1> <BATCH_SIZE2>
    --extra_overwrites <ADDITIONAL_TRAIN_CONFIGS>

    ie
    python internal/jet/run_jet_downstream_pipeline.py --model megamolbart --config_path examples/molecule/megamolbart/conf
    --config_name pretrain_xsmall_span_aug --script_path examples/molecule/megamolbart --variant pretrain
    --nodes 1 --gpus 8 --batch_size 8 --precision "16-mixed" 32   --extra_overwrites "trainer.max_steps=200 ++trainer.check_val_every_n_epoch=None ++model.data.dataset_path=$BIONEMO_HOME/examples/tests/test_data/molecule
    ++model.data.dataset.train=x000 ++model.data.dataset.val=x000 ++model.data.dataset.test=x000 ++model.dwnstr_task_validation.enabled=False"


    # Running tests for the new model run on a customised docker from DOCKER_IMAGE
    python internal/jet/run_jet_downstream_pipeline.py --image <DOCKER_IMAGE> --model <MODEL_TYPE> --config_path <CONFIG_PATH>
    --config_name <CONFIG_NAME> --script_path <SCRIPT_PATH> --variant <VARIANT> --nodes <NODE1> <NODE2>
    --gpus <GPU1> <GPUS2> --precision <PREC1> <PREC2> --batch_size <BATCH_SIZE1> <BATCH_SIZE2>
    --extra_overwrites <ADDITIONAL_TRAIN_CONFIGS>

    ie
    python internal/jet/run_jet_downstream_pipeline.py --image "gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel"
    --model megamolbart --config_path examples/molecule/megamolbart/conf --config_name pretrain_xsmall_span_aug
    --script_path examples/molecule/megamolbart --variant pretrain --nodes 1 --gpus 8 --batch_size 8 --precision "16-mixed" 32
    --extra_overwrites "trainer.max_steps=200 ++trainer.check_val_every_n_epoch=None ++model.data.dataset_path=$BIONEMO_HOME/examples/tests/test_data/molecule
    ++model.data.dataset.train=x000 ++model.data.dataset.val=x000 ++model.data.dataset.test=x000 ++model.dwnstr_task_validation.enabled=False"

    ie
    # Running tests for the new test run on a customised docker build from Dockerfile stored on a git repo and git branch
    python internal/jet/run_jet_downstream_pipeline.py --git_repo <GIT_REPO> --git_branch <GIT_BRANCH>
    --dockerfile <PATH_DOCKERFILE> --model <MODEL_TYPE> --config_path <CONFIG_PATH>
    --config_name <CONFIG_NAME> --script_path <SCRIPT_PATH> --variant <VARIANT> --nodes <NODE1> <NODE2>
    --gpus <GPU1> <GPUS2> --precision <PREC1> <PREC2> --batch_size <BATCH_SIZE1> <BATCH_SIZE2>
    --extra_overwrites <ADDITIONAL_TRAIN_CONFIGS>

    python internal/jet/run_jet_downstream_pipeline.py --git_repo "https://gitlab-master.nvidia.com/clara-discovery/bionemo.git"
    --git_branch dev  --dockerfile setup/Dockerfile  --model megamolbart --config_path examples/molecule/megamolbart/conf
    --config_name pretrain_xsmall_span_aug --script_path examples/molecule/megamolbart --variant pretrain
    --nodes 1 --gpus 8 --batch_size 8 --precision "16-mixed" 32  --extra_overwrites "trainer.max_steps=200 ++trainer.check_val_every_n_epoch=None ++model.data.dataset_path=$BIONEMO_HOME/examples/tests/test_data/molecule
    ++model.data.dataset.train=x000 ++model.data.dataset.val=x000 ++model.data.dataset.test=x000 ++model.dwnstr_task_validation.enabled=False"


    Args:
        jet_workloads_ref_default: reference to a branch in JET Workloads Registry with default tests
        jet_workloads_repo_path: optional, path to the JET Workloads Registry to use or downloads the repo to
        jet_workloads_ref_ephemeral: optional, name of ephemeral reference (branch) to run the JET test on
                                     (if not provided, the reference name will be created)
        docker_image: name of the docker image
        git_repo: git repository url to build docker image
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
        jet_filter: query used in JET Api to filter workloads in the JET config and run only subset of them.
                   It has the pandas format and should not contain double quotation mark (replaced to single ones).
                   More info about the syntax in https://jet.nvidia.com/docs/workloads/filtering/
        setup_jet_api: a flag that determines whether to install JET API and set it up for the first time users
        dry_run: a flag that determines whether to execute a test run
                 (without uploading to JET Workloads Registry and running pipelines in JET CI)

    JET Workloads Registry: https://gitlab-master.nvidia.com/dl/jet/workloads-registry
    """
    if all(
        arg is None
        for arg in [
            docker_image,
            git_repo,
            git_branch,
            config_path,
            config_name,
            script_path,
            model,
            extra_overwrites,
            dllogger_warmup,
            nodes,
            gpus,
            precision,
            batch_size,
        ]
    ):
        # if no arguments are passed to modify existing config in JET Workloads Registry, the test related to
        # jet_workloads_ref_default is triggered
        logging.info(f"Running JET tests for jet workloads reference: {jet_workloads_ref_default}")
        modify_config = False
    else:
        modify_config = True

    if modify_config:
        if docker_image or git_repo or git_branch:
            assert (docker_image is not None) ^ (
                git_repo is not None and git_branch is not None and dockerfile is not None
            ), (
                "Docker container is specified either by docker_image or the build args: git_repo, "
                "git_branch and dockerfile"
            )
        if variant is not None and variant.endswith(".py"):
            variant = variant[:-3]

        repo_handler = JetGitRepoHandler(
            jet_workloads_ref_default=jet_workloads_ref_default,
            jet_workloads_repo_path=jet_workloads_repo_path,
            jet_workloads_ref_ephemeral=jet_workloads_ref_ephemeral,
        )
        repo_handler.setup_local_jet_workloads_repo()
        repo_handler.setup_local_ephemeral_branch()

        jet_workload_ref = repo_handler.jet_workloads_ref_ephemeral
        jet_workloads_repo_path = repo_handler.jet_workloads_repo_path

        modify_jet_workloads_config(
            jet_workloads_repo_path=jet_workloads_repo_path,
            git_repo=git_repo,
            git_branch=git_branch,
            dockerfile=dockerfile,
            docker_image=docker_image,
            config_path=config_path,
            config_name=config_name,
            script_path=script_path,
            variant=variant,
            model=model,
            extra_overwrites=extra_overwrites,
            dllogger_warmup=dllogger_warmup,
            nodes=nodes,
            gpus=gpus,
            precision=precision,
            batch_size=batch_size,
        )

        repo_handler.push_changes_to_remote_ephemeral_branch(dry_run=dry_run)
    else:
        jet_workload_ref = jet_workloads_ref_default
    jet_pipeline_runner = JetPipelineHandler(jet_workloads_ref=jet_workload_ref)
    if setup_jet_api:
        jet_pipeline_runner.setup_jet_api()

    if jet_filter is not None:
        jet_filter = jet_filter.replace('"', "'")
        jet_filter = f" and {jet_filter}"
    else:
        jet_filter = ""
    jet_filter = f"\"type == 'recipe'{jet_filter}\""
    jet_pipeline_runner.get_workload_info(jet_filter=jet_filter, dry_run=dry_run)
    jet_pipeline_runner.run_jet_pipeline(jet_filter=jet_filter, dry_run=dry_run)

    print(f"The pipeline was run for Jet Workloads Registry reference: {jet_workload_ref}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Arguments defining workload in JET Workloads Registry
    parser.add_argument(
        "--jet_ref",
        type=str,
        default="bionemo/new-model-dev",
        help="Reference to branch in JET Workloads Registry with default tests",
    )
    parser.add_argument(
        "--jet_ref_eph", type=str, default=None, help="Name of ephemeral reference (branch) to run the JET test on"
    )
    parser.add_argument(
        "--jet_repo_path",
        type=str,
        default=None,
        help="Path to the JET Workloads Registry to use or downloads the repo to",
    )
    # Arguments specifying how to obtain or build docker for a test
    parser.add_argument(
        "--git_repo",
        type=str,
        default=None,
        help='Repository name to copy the "bionemo/ci" folder from to the docker image',
    )
    parser.add_argument(
        "--git_branch",
        type=str,
        default=None,
        help='Branch name of the repository to copy the "bionemo/ci" folder from to the docker image',
    )
    parser.add_argument(
        "--dockerfile",
        type=str,
        default="internal/Dockerfile-devel",
        help='Branch name of the repository to copy the "bionemo/ci" folder from to the docker image',
    )
    parser.add_argument("--image", type=str, default=None, help="Docker image to use to run JET tests")

    # Arguments specifying command script to to use in the tests
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the folder with primary configs that is relative to bionemo "
        'ie "examples/molecule/megamolbart/conf"',
    )
    parser.add_argument(
        "--config_name", type=str, default=None, help="Filename on the config to use under config_path"
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default=None,
        help="Path to the folder with the model-specific training scripts that " "should be relative to bionemo.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="pretrain",
        help="Type of the training case defined as "
        "the name of a one training script that can be found "
        'in script_path, ie "pretrain".'
        'The training command is defined as "{script_path}/{variant}.py"',
    )

    parser.add_argument("--model", type=str, default=None, help='Name of the model to run tests for ie "megamolbart"')
    parser.add_argument(
        "--extra_overwrites", type=str, default=None, help="Configs that overwrite the training/test configuration"
    )
    # Arguments specifying tested cases for models
    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=None,
        help="List of ints that specify different numbers of nodes to test",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="List of ints that specify different numbers of devices per node  to test",
    )
    parser.add_argument(
        "--precision", nargs="+", type=int, default=None, help="List of ints that specify different precisions to test"
    )
    parser.add_argument(
        "--batch_size",
        nargs="+",
        type=int,
        default=None,
        help="List of ints that specify different batch sizes to test",
    )

    # Other arguments
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Query in the pandas format used to filter workloads in the JET config and run only subset of them",
    )
    parser.add_argument(
        "--setup_jet_api",
        action="store_true",
        default=False,
        help="A flag that determines whether to install JET API and set it up for the first time users",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False, help="A flag that determines whether to execute a test run"
    )

    args = parser.parse_args()

    run_jet_downstream_pipeline(
        jet_workloads_ref_default=args.jet_ref,
        jet_workloads_repo_path=args.jet_repo_path,
        jet_workloads_ref_ephemeral=args.jet_ref_eph,
        git_repo=args.git_repo,
        git_branch=args.git_branch,
        dockerfile=args.dockerfile,
        docker_image=args.image,
        config_path=args.config_path,
        config_name=args.config_name,
        script_path=args.script_path,
        variant=args.variant,
        model=args.model,
        extra_overwrites=args.extra_overwrites,
        nodes=args.nodes,
        gpus=args.gpus,
        precision=args.precision,
        batch_size=args.batch_size,
        jet_filter=args.filter,
        setup_jet_api=args.setup_jet_api,
        dry_run=args.dry_run,
    )
