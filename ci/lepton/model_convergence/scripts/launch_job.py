#!/usr/bin/env python3
"""
Lepton Job submission script with Hydra configuration

Demo: python launch_job.py --config-name "evo2_finetune_lora" job_name="evo2-finetune-lora-job"
"""

import hydra
import json

from omegaconf import DictConfig
from typing import Dict
from types import SimpleNamespace

from leptonai.api.v2.client import APIClient
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import EnvVar, EnvValue, LeptonContainer, Mount, MountOptions
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from omegaconf import OmegaConf
from omegaconf import DictConfig as HydraDictConfig, ListConfig

# todo: move to some defaults file
default_mounts = [
    {"path": "/BioNeMo", "mount_path": "/BioNeMo", "from_": "node-nfs:lepton-shared-fs"},
]

default_env_vars = [
    {"name": "KRATOS_SSA_URL", "value_from": "KRATOS_SSA_URL"},
    {"name": "KRATOS_SSA_CLIENT_ID", "value_from": "KRATOS_SSA_CLIENT_ID"},
    {"name": "KRATOS_SSA_SECRET", "value_from": "KRATOS_SSA_SECRET"},
    {"name": "LEP_LOGIN_CREDENTIALS", "value_from": "LEP_LOGIN_CREDENTIALS"},
]


# todo: make utils file? also, add cfg checks to make sure these are used in the config before calling
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
    if hasattr(env_var, 'value_from') and env_var.value_from is not None:
        return EnvVar(
            name=env_var.name,
            value_from=EnvValue(secret_name_ref=env_var.value_from),
        )
    else:
        return EnvVar(
            name=env_var.name,
            value=env_var.value,
        )


bionemo_mount = construct_mount(
    path=default_mounts[0]['path'], mount_path=default_mounts[0]['mount_path'], from_=default_mounts[0]['from_']
)

bionemo_env_vars = [construct_env_var(SimpleNamespace(**env_var)) for env_var in default_env_vars]
print('bionemo_env_vars', bionemo_env_vars)


def wrap_with_wandb_copy(
    script: str,
    result_dir: str = "pretraining_demo",
    experiment_name: str = "evo2",
    dashboard_info: Dict[str, str] = None,
) -> str:
    if isinstance(dashboard_info, (HydraDictConfig, ListConfig)):
        dashboard_info = OmegaConf.to_container(dashboard_info, resolve=True)
    if dashboard_info is None:
        dashboard_info = {}

    # serialize after conversion
    dashboard_json = json.dumps(dashboard_info, separators=(",", ":"))

    return f"""set -euo pipefail

# Get job name
JOB_NAME="${{LEPTON_JOB_NAME:-unknown-job}}"

# Run the training script
set +e
(
{script}
)
RC=$?
set -e

# Authenticate to Lepton
echo "Authenticating to Lepton..."
pip install -q leptonai >/dev/null 2>&1 || pip install leptonai
lep login -c "$LEP_LOGIN_CREDENTIALS" || true

echo "Job name: $JOB_NAME"

echo "Getting lepton job details (JSON)"
JOB_INFO="$(
  lep job get --id "$JOB_NAME" 2>/dev/null \
  | awk '
    BEGIN {{ json=""; depth=0; started=0 }}
    {{
      for (i=1; i<=length($0); i++) {{
        ch = substr($0, i, 1)
        if (ch == "{{") {{ depth++; started=1 }}
        if (started)      json = json ch
        if (ch == "}}") {{
          depth--
          if (started && depth == 0) {{ print json; exit }}
        }}
      }}
    }}' \
  | jq -c '
    {{
      metadata: {{
        id: .metadata.id,
        name: .metadata.name,
        created_at: .metadata.created_at,
        created_by: .metadata.created_by,
        owner: .metadata.owner,
        visibility: .metadata.visibility
      }},
      spec: {{
        resource_shape: .spec.resource_shape,
        affinity: {{
          allowed_dedicated_node_groups: .spec.affinity.allowed_dedicated_node_groups,
          allowed_nodes_in_node_group: .spec.affinity.allowed_nodes_in_node_group
        }},
        container_image: .spec.container.image,
        completions: .spec.completions,
        parallelism: .spec.parallelism,
        envs: .spec.envs,
        mounts: .spec.mounts,
        image_registry_auth: .spec.image_pull_secrets,
        ttl_seconds_after_finished: .spec.ttl_seconds_after_finished,
        log_enable_collection: .spec.log.enable_collection
      }},
      status: {{
        job_name: .status.job_name,
        state: .status.state,
        ready: .status.ready,
        active: .status.active,
        failed: .status.failed,
        succeeded: .status.succeeded,
        creation_time: .status.creation_time,
        completion_time: .status.completion_time
      }}
    }}
  ' 2>/dev/null
)"

# Normalize to valid JSON or default to {{}}
JOB_INFO_JSON="$(printf '%s' "$JOB_INFO" | jq -c . 2>/dev/null || echo '{{}}')"

# Stash dashboard info JSON (provided from Python)
DASHBOARD_INFO_JSON='{dashboard_json}'

# Copy W&B files from known location based on result_dir and experiment_name
JOB_DIR="/BioNeMo/model_convergence_tests/job_logs/$JOB_NAME"
mkdir -p "$JOB_DIR"

WANDB_BASE_DIR="{result_dir}/{experiment_name}/wandb"
WANDB_FOUND=0
if [ -d "$WANDB_BASE_DIR" ]; then
    LATEST_RUN=$(ls -td "$WANDB_BASE_DIR"/run-* 2>/dev/null | head -n1)
    if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/files" ]; then
        if [ -f "$LATEST_RUN/files/wandb-summary.json" ]; then
            cp -v "$LATEST_RUN/files/wandb-summary.json" "$JOB_DIR/wandb-summary.json"
            WANDB_FOUND=1
        fi
        if [ -f "$LATEST_RUN/files/wandb-metadata.json" ]; then
            cp -v "$LATEST_RUN/files/wandb-metadata.json" "$JOB_DIR/wandb-metadata.json"
        fi
    fi
fi

if [ $WANDB_FOUND -eq 1 ] && [ -f "$JOB_DIR/wandb-summary.json" ]; then
    echo "Combining W&B JSON files and uploading to Kratos..."

    METADATA_JSON=$(cat "$JOB_DIR/wandb-metadata.json" 2>/dev/null || echo '{{}}')
    SUMMARY_JSON=$(cat "$JOB_DIR/wandb-summary.json" 2>/dev/null || echo '{{}}')

    COMBINED_JSON=$(jq -n \
        --arg m "$METADATA_JSON" \
        --arg s "$SUMMARY_JSON" \
        --argjson job_info "$JOB_INFO_JSON" \
        --argjson dashboard_info "$DASHBOARD_INFO_JSON" \
        '
        . + {{
          job_name: env.LEPTON_JOB_NAME,
          metadata: ($m | fromjson? // {{}}),
          summary:  ($s | fromjson? // {{}}),
          job_info: $job_info,
          dashboard_info: $dashboard_info
        }}
        ')

    echo "$COMBINED_JSON" > "$JOB_DIR/wandb-combined.json"
    echo "Combined W&B JSON saved to: $JOB_DIR/wandb-combined.json"

    TELEMETRY_URL="https://prod.analytics.nvidiagrid.net"
    COLLECTOR_ID="bionemo-convergence-lepton-logs-kratos.telemetry.lepton-poc-v001.prod"
    SOURCE="bionemo-wandb-logs"
    TYPE="wandb-training-metrics"
    SUBJECT="$JOB_NAME"

    UUID=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "$(date +%s)-$-$RANDOM")
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

    echo "Sending telemetry event to Kratos..."

    if [ -z "${{KRATOS_SSA_CLIENT_ID:-}}" ] || [ -z "${{KRATOS_SSA_SECRET:-}}" ] || [ -z "${{KRATOS_SSA_URL:-}}" ]; then
        echo "Warning: Kratos credentials not found. Skipping telemetry upload."
    else
        ENCODED_CREDS=$(echo -n "${{KRATOS_SSA_CLIENT_ID}}:${{KRATOS_SSA_SECRET}}" | base64 | tr -d '\\n')
        TOKEN_RESPONSE=$(curl -sS --request POST \
            -H "Content-Type: application/x-www-form-urlencoded" \
            -H "Authorization: Basic $ENCODED_CREDS" \
            "https://${{KRATOS_SSA_URL}}/token?grant_type=client_credentials&scope=telemetry-write" 2>&1)
        ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.access_token' 2>/dev/null)

        if [ -n "$ACCESS_TOKEN" ] && [ "$ACCESS_TOKEN" != "null" ]; then
            JSON_PAYLOAD=$(jq -n \
                --arg id "$UUID" \
                --arg time "$TIMESTAMP" \
                --arg source "$SOURCE" \
                --arg type "$TYPE" \
                --arg subject "$SUBJECT" \
                --argjson data "$COMBINED_JSON" \
                '{{ 
                  "specversion": "1.0",
                  "id": $id,
                  "time": $time,
                  "source": $source,
                  "type": $type,
                  "subject": $subject,
                  "data": $data
                }}')

            RESPONSE=$(curl -sS --request POST \
                -H "Content-Type: application/cloudevents+json" \
                -H "Authorization: Bearer ${{ACCESS_TOKEN}}" \
                "${{TELEMETRY_URL}}/api/v2/topic/${{COLLECTOR_ID}}" \
                --data "$JSON_PAYLOAD" 2>&1)

            if [ $? -eq 0 ]; then
                echo "✓ Event sent successfully to Kratos (ID: $UUID)"
                [ -n "$RESPONSE" ] && echo "Response: $RESPONSE"
            else
                echo "Failed to send event to Kratos"
                echo "Response: $RESPONSE"
            fi
        else
            echo "Error: Failed to get Kratos access token"
            echo "Token response: $TOKEN_RESPONSE"
        fi
    fi
else
    echo "Skipping Kratos upload - W&B files not found"
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

    # Extract result_dir and experiment_name from config
    # Use the same defaults as in the training script
    result_dir = cfg.get('result_dir', 'pretraining_demo')
    experiment_name = cfg.get('experiment_name', 'evo2')

    print(f"Using result_dir: {result_dir}")
    print(f"Using experiment_name: {experiment_name}")
    print(f"W&B files will be searched in: {result_dir}/{experiment_name}/wandb/")

    # Create command with the extracted parameters
    command = [
        "bash",
        "-c",
        wrap_with_wandb_copy(
            cfg.script,
            result_dir=result_dir,
            experiment_name=experiment_name,
            dashboard_info=cfg.dashboard_info,
        ),
    ]

    # Build environment variables, if in config
    env_vars = []
    if hasattr(cfg, "environment_variables") and cfg.environment_variables:
        for env_var in cfg.environment_variables:
            env_vars.append(construct_env_var(env_var))

    # Build mounts, if in config
    mounts = []
    if hasattr(cfg, "mounts") and cfg.mounts:
        mounts = [construct_mount(path=m.path, mount_path=m.mount_path, from_=m.from_) for m in cfg.mounts]

    print('env_vars', env_vars)
    print('mounts', mounts)

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
            print("Batch Job Launched\n")
            print(
                f"View at https://dashboard.dgxc-lepton.nvidia.com/workspace/vfco61g2/compute/jobs/detail/{launched_job.metadata.id_}/replicas/list"
            )
    except Exception as e:
        print(f"ERROR submitting job: {e}")
        return


if __name__ == "__main__":
    main()
