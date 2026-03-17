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

"""Lepton job submission script that runs Claude Code as an autonomous training agent.

Claude Code is installed in the container, given a prompt describing the training job,
and runs torchrun autonomously. It saves a JSON state summary to NFS on completion.
"""

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

    return chosen_group, valid_node_ids, resource_shape


def _build_agent_prompt(cfg: DictConfig) -> str:
    """Read claude_agent_prompt.txt and fill in config values."""
    import pathlib

    prompt_path = pathlib.Path(__file__).parent / "claude_agent_prompt.txt"
    template = prompt_path.read_text()

    return template.format(
        code_path=cfg.code_path,
        train_script=cfg.train_script,
        hydra_config=cfg.hydra_config,
        num_train_steps=cfg.num_train_steps,
        checkpoint_dir=cfg.checkpoint_dir,
        save_every_n_steps=cfg.save_every_n_steps,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        agent_state_dir=cfg.agent_state_dir,
    )


def launch_claude_agent_job(client, cfg: DictConfig):
    """Launch a single-node job that runs Claude Code as the training agent."""
    chosen_group, valid_node_ids, resource_shape = _resolve_scheduling_target(client, cfg)

    agent_prompt = _build_agent_prompt(cfg)

    # Git branch checkout logic (rank 0 only, but single-node so always rank 0)
    git_branch = cfg.get("git_branch", "")
    repo_root = cfg.get("repo_root", "/data/savithas/bionemo-framework")

    git_sync_script = ""
    if git_branch:
        git_sync_script = f"""
echo "=========================================="
echo "Syncing to branch: {git_branch}"
echo "=========================================="
cd {repo_root}
find .git -name "*.lock" -delete 2>/dev/null || true
git fetch origin
git checkout {git_branch}
git pull origin {git_branch}
echo "Git sync complete! Commit: $(git rev-parse HEAD)"
echo "=========================================="
"""

    # Escape the prompt for embedding in bash (use a heredoc)
    container_script = f"""#!/bin/bash
set -e

echo "=========================================="
echo "Claude Code Agent - Lepton GPU Node"
echo "GPUs: {cfg.gpus_per_node}x H100"
echo "=========================================="

# 1. Initialize Lepton environment
wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
chmod +x init.sh
source init.sh

export HF_HOME=/data/savithas/cache
{git_sync_script}
# 2. Install Node.js 22 LTS
echo "Installing Node.js 22 LTS..."
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
echo "Node.js version: $(node --version)"
echo "npm version: $(npm --version)"

# 3. Install Claude Code CLI
echo "Installing Claude Code..."
npm install -g @anthropic-ai/claude-code
echo "Claude Code installed: $(claude --version)"

# 4. Install Python training requirements
cd {cfg.code_path}
pip install -r requirements.txt

# 5. Login to wandb
wandb login ${{WANDB_API_KEY}}

# 6. Create agent state directory
mkdir -p {cfg.agent_state_dir}

# 7. Ensure checkpoint directory exists
mkdir -p {cfg.checkpoint_dir}

# 8. Create non-root user (Claude Code refuses --dangerously-skip-permissions as root)
echo "Creating non-root user for Claude Code..."
useradd -m -s /bin/bash claude-agent
# Give the user access to code, checkpoints, and agent state dirs
chown -R claude-agent:claude-agent {cfg.agent_state_dir}
chown -R claude-agent:claude-agent {cfg.checkpoint_dir}

# 9. Write the agent prompt to a file
cat > /tmp/agent_prompt.txt << 'AGENT_PROMPT_EOF'
{agent_prompt}
AGENT_PROMPT_EOF
chmod 644 /tmp/agent_prompt.txt

# 10. Write a wrapper script for the non-root user
cat > /tmp/run_claude.sh << 'WRAPPER_EOF'
#!/bin/bash
set -e
export ANTHROPIC_AUTH_TOKEN="${{ANTHROPIC_AUTH_TOKEN}}"
export ANTHROPIC_BASE_URL="${{ANTHROPIC_BASE_URL}}"
export WANDB_API_KEY="${{WANDB_API_KEY}}"
export HF_HOME=/data/savithas/cache
export PATH=/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${{LD_LIBRARY_PATH:-}}

cd {cfg.code_path}

echo "Testing Claude Code authentication..."
claude --dangerously-skip-permissions \
  --model {cfg.claude_model} \
  -p "Say OK if you can read this." 2>&1 | head -5
echo "Auth check complete."

echo "=========================================="
echo "Starting Claude Code agent..."
echo "=========================================="

PROMPT=$(cat /tmp/agent_prompt.txt)
claude --dangerously-skip-permissions \
  --model {cfg.claude_model} \
  -p "$PROMPT"

echo "=========================================="
echo "Claude Code agent finished."
echo "=========================================="
WRAPPER_EOF
chmod 755 /tmp/run_claude.sh

# 11. Run as non-root user, passing env vars through
su - claude-agent -w ANTHROPIC_AUTH_TOKEN,ANTHROPIC_BASE_URL,WANDB_API_KEY \
  -c "bash /tmp/run_claude.sh"
"""

    command = ["bash", "-c", container_script]

    env_vars = [
        EnvVar(name="ANTHROPIC_AUTH_TOKEN", value_from=EnvValue(secret_name_ref=cfg.anthropic_secret)),
        EnvVar(name="ANTHROPIC_BASE_URL", value=cfg.anthropic_base_url),
        EnvVar(name="WANDB_API_KEY", value_from=EnvValue(secret_name_ref=cfg.wandb_secret)),
    ]

    nfs_source_path = cfg.get("nfs", {}).get("source_path", "/BioNeMo")
    nfs_mount_path = cfg.get("nfs", {}).get("mount_path", "/data")
    nfs_source = cfg.get("nfs", {}).get("nfs_source", "node-nfs:fs1")

    mounts = [
        {
            "path": nfs_source_path,
            "mount_path": nfs_mount_path,
            "from": nfs_source,
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
        completions=1,
        parallelism=1,
        envs=env_vars,
        image_pull_secrets=[cfg.container.registry_auth],
        mounts=mounts,
    )

    job = LeptonJob(spec=job_spec, metadata=Metadata(id=cfg.job_name))

    try:
        launched_job = client.job.create(job)
        if launched_job.status:
            print(f"  Job launched: {cfg.job_name}")
            workspace_id = cfg.get("workspace_id", "vfco61g2")
            print(
                f"  View at: https://dashboard.dgxc-lepton.nvidia.com/workspace/"
                f"{workspace_id}/compute/jobs/detail/{launched_job.metadata.id_}/replicas/list"
            )
            return True
    except Exception as e:
        print(f"  ERROR submitting job {cfg.job_name}: {e}")
        return False


@hydra.main(version_base=None, config_path="lepton_configs", config_name="claude_agent_demo")
def main(cfg: DictConfig):
    """Submit a Lepton job that runs Claude Code as an autonomous training agent."""
    print("=" * 60)
    print(f"Claude Code Agent Demo - Job: {cfg.job_name}")
    print("=" * 60)
    print(f"  Model: {cfg.claude_model}")
    print(f"  Training config: {cfg.hydra_config}")
    print(f"  Steps: {cfg.num_train_steps}")
    print(f"  Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"  Agent state dir: {cfg.agent_state_dir}")

    if cfg.get("git_branch"):
        print(f"  Git branch: {cfg.git_branch}")

    print()

    client = APIClient()
    OmegaConf.resolve(cfg)

    success = launch_claude_agent_job(client, cfg)
    if not success:
        print("\nJob submission failed!")
        exit(1)

    print("\nClaude agent job submitted successfully!")
    print(f"Check {cfg.agent_state_dir}/run_summary.json for results after completion.")


if __name__ == "__main__":
    main()
