#!/usr/bin/env python3
"""
Lepton Job submission script with Hydra configuration

Demo: python launch_job.py --config-name "evo2_finetune_lora" job_name="evo2-finetune-lora-job"
"""

import hydra
from omegaconf import DictConfig
from leptonai.api.v2.client import APIClient
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import EnvVar, EnvValue, LeptonContainer, Mount, MountOptions
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec


def construct_mount(path: str, mount_path: str, from_: str = "node-nfs:lepton-shared-fs") -> Mount:
    """Construct a Mount object for a given path, mount_path, and source."""
    # note, the from_="node-nfs:lepton-shared-fs" is not yet documented in the API docs
    mount = {
        "path": path,
        "mount_path": mount_path,
        "from": from_,
    }
    return Mount(**mount)


def construct_env_var(env_var) -> EnvVar:
    """Construct an EnvVar object from a config entry, supporting both secrets and literals."""
    if 'value_from' in env_var:
        return EnvVar(
            name=env_var.name,
            value_from=EnvValue(secret_name_ref=env_var.value_from),
        )
    else:
        return EnvVar(
            name=env_var.name,
            value=env_var.value,
        )


def wrap_with_wandb_copy(script: str) -> str:
    return f"""set -euo pipefail

# run user script; capture exit code
set +e
(
{script}
)
RC=$?
set -e

# target dir: /data/model_convergence_tests/job_logs/<job_name>
JOB_DIR="/data/model_convergence_tests/job_logs/${{LEPTON_JOB_NAME:-unknown-job}}"
mkdir -p "$JOB_DIR"

# resolve wandb latest run dir
WB_DIR="${{WANDB_DIR:-wandb}}"
RUN_DIR=""
if [ -L "$WB_DIR/latest-run" ]; then
  RUN_DIR=$(readlink -f "$WB_DIR/latest-run" || true)
elif [ -f "$WB_DIR/latest-run" ]; then
  RID=$(cat "$WB_DIR/latest-run" || true)
  [ -n "${{RID:-}}" ] && RUN_DIR="$WB_DIR/run-$RID"
fi

# copy the two files if present (directly under $JOB_DIR; no wandb subdir)
LOGS_FOUND=0
if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR/files" ]; then
  cp -f "$RUN_DIR/files/wandb-summary.json"  "$JOB_DIR/" 2>/dev/null && LOGS_FOUND=1 || true
  cp -f "$RUN_DIR/files/wandb-metadata.json" "$JOB_DIR/" 2>/dev/null && LOGS_FOUND=1 || true
fi

if [ "$LOGS_FOUND" -eq 1 ]; then
  echo "wandb logs found and saved to $JOB_DIR"
else
  echo "wandb logs not found; nothing saved to $JOB_DIR"
fi

exit "$RC"
"""


@hydra.main(version_base=None, config_path="../configs", config_name="")
def main(cfg: DictConfig):

    # Initialize client
    client = APIClient()

    # Get node group
    node_groups = client.nodegroup.list_all()
    node_group_map = {ng.metadata.name: ng for ng in node_groups}

    if cfg.node_group_name not in node_group_map:
        print(f"ERROR: Node group '{cfg.node_group_name}' not found!")
        print(f"Available node groups: {list(node_group_map.keys())}")
        return

    node_group = node_group_map[cfg.node_group_name]
    print(f"Found node group with ID: {node_group.metadata.id_}")

    # Get valid node IDs
    valid_node_ids = set()
    node_ids = client.nodegroup.list_nodes(node_group.metadata.id_)
    for node in node_ids:
        valid_node_ids.add(node.metadata.id_)

    # Create command
    command = ["bash", "-c", wrap_with_wandb_copy(cfg.script)]

    # Build environment variables, if in config
    env_vars = []
    if hasattr(cfg, "environment_variables") and cfg.environment_variables:
        for env_var in cfg.environment_variables:
            env_vars.append(construct_env_var(env_var))

    # Build mounts, if in config
    mounts = []
    if hasattr(cfg, "mounts") and cfg.mounts:
        mounts = [construct_mount(path=m.path, mount_path=m.mount_path, from_=m.from_) for m in cfg.mounts]

    # Create job specification
    job_spec = LeptonJobUserSpec(
        resource_shape=cfg.resource_shape,
        affinity=LeptonResourceAffinity(
            allowed_dedicated_node_groups=[node_group.metadata.id_],
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

    # Create job object
    job = LeptonJob(spec=job_spec, metadata=Metadata(id=cfg.job_name))

    try:
        launched_job = client.job.create(job)
        if launched_job.status:
            print(f"Initial Status: {launched_job.status.state}")
    except Exception as e:
        print(f"ERROR submitting job: {e}")
        return


if __name__ == "__main__":
    main()
