#!/usr/bin/env python3
"""
Lepton Job submission script with Hydra configuration

Demo: python launch_job.py --config-name "evo2_finetune_lora" job_name="evo2-finetune-lora-job"
"""

import hydra
from omegaconf import DictConfig
from typing import Dict
from leptonai.api.v2.client import APIClient
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import EnvVar, EnvValue, LeptonContainer, Mount, MountOptions
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from omegaconf import OmegaConf
from omegaconf import DictConfig as HydraDictConfig, ListConfig


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


import json
from typing import Dict


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

ls


# Run the training script
set +e
(
{script}
)
RC=$?
set -e

echo "pwd"
pwd

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
echo "pwd"
pwd

echo "running ls -a bionemo-framework/recipes/esm2_native_te_mfsdp"
ls -a bionemo-framework/recipes/esm2_native_te_mfsdp

echo "Current working directory: $(pwd)"
echo "Looking for W&B files..."
WANDB_FOUND=0
WANDB_SUMMARY=""
WANDB_METADATA=""

# Look for wandb in the known location - try both absolute and relative paths
if [ -d "/workspace/bionemo-framework/recipes/esm2_native_te_mfsdp/wandb" ]; then
    WANDB_DIR="/workspace/bionemo-framework/recipes/esm2_native_te_mfsdp/wandb"
elif [ -d "bionemo-framework/recipes/esm2_native_te_mfsdp/wandb" ]; then
    WANDB_DIR="bionemo-framework/recipes/esm2_native_te_mfsdp/wandb"
else
    WANDB_DIR=""
fi

if [ -n "$WANDB_DIR" ] && [ -d "$WANDB_DIR" ]; then
    echo "Found wandb directory at: $WANDB_DIR"
    echo "Contents of $WANDB_DIR:"
    ls -la "$WANDB_DIR"
    
    # Look for both offline-run-* and run-* patterns
    LATEST_RUN=$(ls -td "$WANDB_DIR"/offline-run-* "$WANDB_DIR"/run-* 2>/dev/null | head -n1)
    
    if [ -n "$LATEST_RUN" ]; then
        echo "Found latest run: $LATEST_RUN"
        echo "Contents of $LATEST_RUN:"
        ls -la "$LATEST_RUN"
        
        # Check for files in the main run directory first
        if [ -f "$LATEST_RUN/wandb-summary.json" ]; then
            WANDB_SUMMARY="$LATEST_RUN/wandb-summary.json"
            WANDB_FOUND=1
            echo "✓ Found wandb-summary.json at: $WANDB_SUMMARY"
        fi
        
        if [ -f "$LATEST_RUN/wandb-metadata.json" ]; then
            WANDB_METADATA="$LATEST_RUN/wandb-metadata.json"
            echo "✓ Found wandb-metadata.json at: $WANDB_METADATA"
        fi
        
        # If not found in main directory, check the files subdirectory
        if [ "$WANDB_FOUND" -eq 0 ] && [ -d "$LATEST_RUN/files" ]; then
            echo "Checking $LATEST_RUN/files:"
            ls -la "$LATEST_RUN/files"
            
            if [ -f "$LATEST_RUN/files/wandb-summary.json" ]; then
                WANDB_SUMMARY="$LATEST_RUN/files/wandb-summary.json"
                WANDB_FOUND=1
                echo "✓ Found wandb-summary.json at: $WANDB_SUMMARY"
            fi
            
            if [ -f "$LATEST_RUN/files/wandb-metadata.json" ]; then
                WANDB_METADATA="$LATEST_RUN/files/wandb-metadata.json"
                echo "✓ Found wandb-metadata.json at: $WANDB_METADATA"
            fi
        fi
    else
        echo "No W&B runs found"
    fi
else
    echo "W&B directory does not exist at expected locations"
    echo "Checked: /workspace/bionemo-framework/recipes/esm2_native_te_mfsdp/wandb"
    echo "Checked: bionemo-framework/recipes/esm2_native_te_mfsdp/wandb"
fi
echo "Now what?"
if [ $WANDB_FOUND -eq 1 ] && [ -n "$WANDB_SUMMARY" ]; then
    echo "Combining W&B JSON files and uploading to Kratos..."

    METADATA_JSON=$(cat "$WANDB_METADATA" 2>/dev/null || echo '{{}}')
    SUMMARY_JSON=$(cat "$WANDB_SUMMARY" 2>/dev/null || echo '{{}}')

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

    echo "$COMBINED_JSON" > wandb-combined.json
    echo "Combined W&B JSON saved to: wandb-combined.json"

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
                --arg source "bionemo-wandb-logs" \
                --arg type "wandb-training-metrics" \
                --arg subject "$JOB_NAME" \
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
                "https://prod.analytics.nvidiagrid.net/api/v2/topic/bionemo-convergence-lepton-logs-kratos.telemetry.lepton-poc-v001.prod" \
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
