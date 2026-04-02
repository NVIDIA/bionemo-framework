#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Lepton Job submission script for Lingua 7B BF16 + FP32 master weights (DTensorFusedAdam)."""

import hydra
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import EnvValue, EnvVar, LeptonContainer
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from leptonai.api.v2.client import APIClient
from omegaconf import DictConfig, OmegaConf


def _resolve_scheduling_target(client, cfg: DictConfig):
    """Resolve node group and resource shape."""
    desired_node_group = str(cfg.node_group).strip()
    resource_shape = str(cfg.resource_shape).strip()

    node_groups = client.nodegroup.list_all()
    node_group_map = {ng.metadata.name: ng for ng in node_groups}

    if desired_node_group not in node_group_map:
        available = ", ".join(sorted(node_group_map.keys()))
        raise SystemExit(f"Node group '{desired_node_group}' not found.\nAvailable: {available}")

    chosen_group = node_group_map[desired_node_group]
    valid_node_ids = {n.metadata.id_ for n in client.nodegroup.list_nodes(chosen_group.metadata.id_)}

    # Filter out excluded nodes if specified
    if getattr(cfg, "exclude_nodes", None):
        exclude_set = set(cfg.exclude_nodes)
        original_count = len(valid_node_ids)
        valid_node_ids = valid_node_ids - exclude_set
        if original_count != len(valid_node_ids):
            print(f"  Excluding {original_count - len(valid_node_ids)} nodes: {sorted(exclude_set)}")
            print(f"  Remaining valid nodes: {len(valid_node_ids)}")

    return chosen_group, valid_node_ids, resource_shape


def launch_single_job(client, cfg: DictConfig):
    """Launch a single multinode job."""
    chosen_group, valid_node_ids, resource_shape = _resolve_scheduling_target(client, cfg)

    total_gpus = cfg.num_nodes * cfg.gpus_per_node
    gbs = cfg.micro_batch_size * cfg.grad_acc_steps * total_gpus

    training_script = f"""#!/bin/bash
set -e

echo "=========================================="
echo "Lingua 7B BF16 + FP32 Master Weights (DTensorFusedAdam)"
echo "Nodes: {cfg.num_nodes}"
echo "GPUs per node: {cfg.gpus_per_node}"
echo "Total GPUs: {total_gpus}"
echo "Micro batch size: {cfg.micro_batch_size}"
echo "Grad acc steps: {cfg.grad_acc_steps}"
echo "GBS: {gbs}"
echo "Hydra config: {cfg.hydra_config}"
echo "Steps: {cfg.num_train_steps:,}"
echo "=========================================="

# Initialize Lepton environment for distributed training
wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
chmod +x init.sh
source init.sh

export MASTER_PORT=29400
export NCCL_TIMEOUT_MS=1800000
export HF_HOME=/data/savithas/cache

cd {cfg.code_path}

echo "Installing dependencies..."
pip install -r requirements.txt

huggingface-cli login --token ${{HF_TOKEN}}
wandb login ${{WANDB_API_KEY}}

echo "=========================================="
echo "Starting multinode training..."
echo "=========================================="

torchrun \\
  --nnodes=$NNODES \\
  --nproc_per_node={cfg.gpus_per_node} \\
  --node_rank=$NODE_RANK \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  {cfg.train_script} \\
  --config-name={cfg.hydra_config} \\
  num_train_steps={cfg.num_train_steps} \\
  dataset.micro_batch_size={cfg.micro_batch_size} \\
  grad_acc_steps={cfg.grad_acc_steps} \\
  dataset.load_dataset_kwargs.path="{cfg.dataset_path}" \\
  dataset.load_dataset_kwargs.data_files="{cfg.data_files}" \\
  checkpoint.ckpt_dir={cfg.checkpoint_dir} \\
  checkpoint.save_every_n_steps={cfg.save_every_n_steps} \\
  checkpoint.resume_from_checkpoint=true \\
  wandb.project="{cfg.wandb_project}" \\
  wandb.name="{cfg.wandb_name}"

echo "=========================================="
echo "Training completed!"
echo "=========================================="
"""

    command = ["bash", "-c", training_script]

    env_vars = [
        EnvVar(name="WANDB_API_KEY", value_from=EnvValue(secret_name_ref=cfg.wandb_secret)),
        EnvVar(name="HF_TOKEN", value_from=EnvValue(secret_name_ref=cfg.hf_secret)),
    ]

    mounts = [
        {
            "path": "/BioNeMo",
            "mount_path": "/data",
            "from": "node-nfs:fs1",
        },
    ]

    job_spec = LeptonJobUserSpec(
        resource_shape=resource_shape,
        affinity=LeptonResourceAffinity(
            allowed_dedicated_node_groups=[chosen_group.metadata.id_],
            allowed_nodes_in_node_group=valid_node_ids,
        ),
        container=LeptonContainer(
            image=cfg.container.image,
            command=command,
        ),
        completions=cfg.num_nodes,
        parallelism=cfg.num_nodes,
        envs=env_vars,
        image_pull_secrets=[cfg.container.registry_auth],
        mounts=mounts,
    )

    job = LeptonJob(spec=job_spec, metadata=Metadata(id=cfg.job_name))
    try:
        launched_job = client.job.create(job)
        if launched_job.status:
            print(f"Job launched: {cfg.job_name}")
            print(
                f"  View at: https://dashboard.dgxc-lepton.nvidia.com/workspace/vfco61g2/compute/jobs/detail/"
                f"{launched_job.metadata.id_}/replicas/list"
            )
            return True
    except Exception as e:
        print(f"ERROR submitting job {cfg.job_name}: {e}")
        return False


@hydra.main(version_base=None, config_path="lepton_configs", config_name="")
def main(cfg: DictConfig):
    """Main function."""
    client = APIClient()
    OmegaConf.resolve(cfg)
    print(f"Launching job: {cfg.job_name}")
    if launch_single_job(client, cfg):
        print("Job submitted successfully!")
    else:
        print("Job submission failed!")
        exit(1)


if __name__ == "__main__":
    main()
