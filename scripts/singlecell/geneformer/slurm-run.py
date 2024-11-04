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


import nemo_run as run
from typing import Optional

from bionemo.geneformer.run.factories import default_adam_optimizer_with_cosine_annealing_recipe, default_trainer_config, experiment_config_recipe, geneformer10M_pretraining_recipe, pretrain_partial, simple_parallel_recipe, small_data_config

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
    identity="/home/bionemo/.ssh/id_ed25519"
    # OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
    DRACO="cs-oci-ord-login-03"
    CUSTOM_MOUNTS = [
        "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/results/bionemo2_geneformer_pretraining/bionemo2_geneformer_pretraining:/results",
        "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/data:/workspaces/bionemo-fw-ea/data",
    ]
    executor = slurm_executor(
        user='skothenhill',
        identity=identity,
        host=DRACO,
        remote_job_dir='/home/skothenhill/20240924-bionemo2/nemorun',
        account='healthcareeng_bionemo',
        partition='polar',
        nodes=1,
        devices=8,
        custom_mounts = CUSTOM_MOUNTS,
        container_image="nvcr.io/nvidian/cvai_bnmo_trng/bionemo:bionemo2-758aaecc65031530751c095c727eac58ffd5188b",
    )

    model_config = geneformer10M_pretraining_recipe()
    data_config = small_data_config(data_dir="/workspaces/bionemo-fw-ea/data/cellxgene_2023-12-15_small/processed_data") 
    parallel_config=simple_parallel_recipe()
    training_config = default_trainer_config()
    optim_config=default_adam_optimizer_with_cosine_annealing_recipe()
    experiment_config=experiment_config_recipe()
    data_config.seq_length=128
    data_config.micro_batch_size=8
    parallel_config.num_devices=8
    training_config.precision='bf16-mixed'
    training_config.max_steps=1000
    recipe = pretrain_partial(
        model_config=model_config,
        data_config=data_config,
        parallel_config=parallel_config,
        training_config=training_config,
        optim_config=optim_config,
        experiment_config=experiment_config,
        resume_if_exists=False,
    )
    # Submit a partial object
    # There is a way to do this with explicit experiment management but idk how.
    run.run(recipe, executor=executor, detach=True, dryrun=False)

main()