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

"""Simple Lepton Job submission script for John's OG2 7B FP8 experiment.

Version with validation logging enabled every 500 steps.
"""

from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import EnvValue, EnvVar, LeptonContainer
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from leptonai.api.v2.client import APIClient


def main():
    """Submit John's OG2 7B FP8 job with validation enabled."""
    # Configuration
    job_name = "savithas-johns-og2-7b-v5-val"
    resource_shape = "gpu.8xh100-sxm"
    node_group_name = "yo-bom-lepton-001"
    num_nodes = 6

    # Nodes to exclude (ECC errors)
    exclude_nodes = {
        "node-ip-10-50-80-195",
        "node-ip-10-50-81-231",
    }

    # Initialize client
    print("Initializing Lepton client...")
    client = APIClient()

    # Get node group
    print(f"Finding node group: {node_group_name}")
    node_groups = client.nodegroup.list_all()
    node_group_map = {ng.metadata.name: ng for ng in node_groups}

    if node_group_name not in node_group_map:
        print(f"ERROR: Node group '{node_group_name}' not found!")
        print(f"Available node groups: {list(node_group_map.keys())}")
        return

    node_group = node_group_map[node_group_name]
    print(f"Found node group with ID: {node_group.metadata.id_}")

    # Get valid node IDs
    valid_node_ids = set()
    node_ids = client.nodegroup.list_nodes(node_group.metadata.id_)
    for node in node_ids:
        if node.metadata.id_ not in exclude_nodes:
            valid_node_ids.add(node.metadata.id_)
    print(f"Found {len(valid_node_ids)} valid nodes (excluded {len(exclude_nodes)} bad nodes)")

    # John's training script with validation enabled
    training_script = """#!/bin/bash
set -e

echo "=========================================="
echo "John's OG2 7B FP8 - With Validation"
echo "Nodes: 6"
echo "GPUs per node: 8"
echo "Total GPUs: 48"
echo "Val check interval: 500 steps"
echo "Checkpoint: save-top-k=3 (best 3 by val loss)"
echo "=========================================="

# Download the environment setup script from Lepton's GitHub repository
wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
chmod +x init.sh
source init.sh

# Explicit port setting (matching working Lingua script)
export MASTER_PORT=29400
export NCCL_TIMEOUT_MS=1800000  # 30 minutes

nvidia-smi

# Debug: Print environment variables
echo "=== Environment Variables ==="
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
echo "==========================="

wandb login $WANDB_API_KEY

# Create output directories
mkdir -p /data/savithas/johns_og2_repro_v5_val/window_logs
mkdir -p /data/savithas/johns_og2_repro_v5_val/results
mkdir -p /data/savithas/johns_og2_repro_v5_val/checkpoints

echo "Starting multinode training..."

# Using STATIC torchrun (not rendezvous) - matching working Lingua script pattern
torchrun \\
  --nnodes=$NNODES \\
  --nproc_per_node=8 \\
  --node_rank=$NODE_RANK \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  --no-python \\
train_evo2 \\
  --sharded-eden-data \\
  --seq-length 8192 \\
  --stride 7992 \\
  --sequence-db-dir /data/bcr_eden/OG2_database_splits \\
  --train-window-db /data/bcr_eden/OG2_database_splits/og2__train__short.sqlite \\
  --val-window-db /data/bcr_eden/OG2_database_splits/og2__validation__short.sqlite \\
  --test-window-db /data/bcr_eden/OG2_database_splits/og2__test__short.sqlite \\
  --log-windows \\
  --window-log-dir /data/savithas/johns_og2_repro_v5_val/window_logs \\
  --num-nodes 6 \\
  --model-size 7B \\
  --devices 8 \\
  --grad-acc-batches 4 \\
  --max-steps 182314 \\
  --seed 42 \\
  --sequence-parallel \\
  --enable-preemption \\
  --no-fp32-residual-connection \\
  --fp8 \\
  --ckpt-async-save \\
  --clip-grad 1 \\
  --create-tflops-callback \\
  --overlap-grad-reduce \\
  --log-num-zeros-in-grad \\
  --spike-no-more-embedding-init \\
  --no-calculate-per-token-loss \\
  --save-top-k 3 \\
  --ckpt-dir /data/savithas/johns_og2_repro_v5_val/checkpoints \\
  --wandb-project llama3-metagenome-7b \\
  --wandb-group savithas_johns_og2_repro \\
  --wandb-id savithas-johns-repro-v5-val \\
  --wandb-run-name savithas_johns_og2_7b_fp8_v5_val \\
  --experiment-name savithas-johns-og2-repro-v5-val \\
  --lr 3e-05 \\
  --wd 0.1 \\
  --attention-dropout 0 \\
  --hidden-dropout 0 \\
  --min-lr 6e-07 \\
  --warmup-steps 2500 \\
  --limit-val-batches 40 \\
  --val-check-interval 500 \\
  --result-dir /data/savithas/johns_og2_repro_v5_val/results \\
  --tensor-parallel-size 4 \\
  --context-parallel-size 1 \\
  --pipeline-model-parallel-size 1 \\
  --workers 1 \\
  --micro-batch-size 8
"""

    # Create command
    command = ["bash", "-c", training_script]

    # Create job specification
    print("Creating job specification...")
    job_spec = LeptonJobUserSpec(
        resource_shape=resource_shape,
        affinity=LeptonResourceAffinity(
            allowed_dedicated_node_groups=[node_group.metadata.id_],
            allowed_nodes_in_node_group=valid_node_ids,
        ),
        container=LeptonContainer(
            image="nvcr.io/nvidian/cvai_bnmo_trng/bionemo-framework:nightly-250924",
            command=command,
        ),
        completions=num_nodes,
        parallelism=num_nodes,
        envs=[
            EnvVar(
                name="WANDB_API_KEY",
                value_from=EnvValue(secret_name_ref="wandb.savithas"),
            ),
        ],
        image_pull_secrets=["lepton-nvidia-cvai-bnmo-trng"],
        # NFS mounts - matching working Lingua script
        mounts=[
            {
                "path": "/BioNeMo",
                "mount_path": "/data",
                "from": "node-nfs:fs1",
            },
        ],
    )

    # Create job object
    job = LeptonJob(spec=job_spec, metadata=Metadata(id=job_name))

    # Submit the job
    print(f"Submitting job '{job_name}' with {num_nodes} nodes...")
    try:
        launched_job = client.job.create(job)
        print("✓ Job submitted successfully!")
        print(f"  Job ID: {launched_job.metadata.id_}")
        if launched_job.status:
            print(f"  Initial Status: {launched_job.status.state}")
        print(
            f"\nMonitor at: https://dashboard.dgxc-lepton.nvidia.com/workspace/vfco61g2/compute/jobs/detail/{launched_job.metadata.id_}/replicas/list"
        )
    except Exception as e:
        print(f"ERROR submitting job: {e}")
        return


if __name__ == "__main__":
    main()
