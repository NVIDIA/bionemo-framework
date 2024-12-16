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


from dataclasses import dataclass
import os
import nemo_run as run
from typing import Optional, Type

import argparse

import yaml

from bionemo.geneformer.run.config_models import (
    ExposedFineTuneSeqLenBioBertConfig,
    ExposedGeneformerPretrainConfig,
    GeneformerPretrainingDataConfig,
)
from bionemo.geneformer.run.main import args_to_args_dict, defer_load
from bionemo.geneformer.run.nemo_run import build_nrargs, NRArgs

from bionemo.llm.run.config_models import MainConfig
from bionemo.llm.train import NsysConfig, train

def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    identity: str,
    time: str = "01:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."
        )

    mounts = []
    # Custom mounts are defined here.
    if custom_mounts:
        mounts.extend(custom_mounts)

    # Env vars for jobs are configured here
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir, # This is where the results of the run will be stored by default.
            identity=identity
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor

def main():
    from nemo_run import Partial
    from bionemo.geneformer.run.argument_parser import parse_args

    args = parse_args()
    args_dict = args_to_args_dict(args)
    recipe = Partial(defer_load, build_nrargs(args_dict))

    # or use a simple executor
    executor = run.LocalExecutor()

    # NOTE: slurm stuff below.
    identity="/home/bionemo/.ssh/id_ed25519"
    # OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
    DRACO="cs-oci-ord-login-03"
    # NOTE, how we mount determines the data and results path we would like to push in.
    # SRC: 
    #   /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/data/cellxgene_2023-12-15/processed_data
    #   /lustre:/lustre is the easiest mount

    CUSTOM_MOUNTS = [
        "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/results/bionemo2_geneformer_pretraining/bionemo2_geneformer_pretraining:/results",
        "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/data:/workspace/data",
        "/lustre:/lustre"
    ]

    # TODO how do we get nodes and devices out of our config?
    _executor = slurm_executor(
        user='skothenhill',
        identity=identity,
        host=DRACO,
        remote_job_dir='/home/skothenhill/20240924-bionemo2/nemorun',
        account='healthcareeng_bionemo',
        partition='polar',
        nodes=1,
        devices=8,
        custom_mounts = CUSTOM_MOUNTS,
        container_image="nvcr.io/nvidia/clara/bionemo-framework:nightly",
        custom_env_vars={"WANDB_API_KEY": os.environ.get('WANDB_API_KEY', '')}
    )

    # Submit a partial object
    # There is a way to do this with explicit experiment management but idk how.
    run.run(recipe, executor=executor, detach=True, dryrun=False)


if __name__ == "__main__":
    main()
