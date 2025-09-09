#!/usr/bin/env python3
"""
Lepton Job submission script with Hydra configuration

Demo: python launch_job.py --config-name "evo2_finetune_lora" job_name="evo2-finetune-lora-job"
"""

import hydra
import json
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from leptonai.api.v2.client import APIClient
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import EnvVar, EnvValue, LeptonContainer, MountOptions
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from omegaconf import DictConfig as HydraDictConfig, ListConfig

from utils import construct_mount, construct_env_var


def wrap_script_with_logging(
    script: str,
    dashboard_info: Dict[str, str] = None,
    recipe_subdir: str = "esm2_native_te_mfsdp",
    all_config_json: str = "{}",
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
pip install -q leptonai >/dev/null 2>&1 || pip install leptonai
lep login -c "$LEP_LOGIN_CREDENTIALS" || true

# Get lepton job details
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

JOB_INFO_JSON="$(printf '%s' "$JOB_INFO" | jq -c . 2>/dev/null || echo '{{}}')"
ALL_CONFIG_JSON='{all_config_json}'
DASHBOARD_INFO_JSON='{dashboard_json}'

# Look for W&B files
WANDB_DIR="/workspace/bionemo-framework/recipes/{recipe_subdir}/wandb"
WANDB_FOUND=0
WANDB_SUMMARY=""
WANDB_METADATA=""

if [ -d "$WANDB_DIR" ]; then
    # Use latest-run symlink or find most recent run
    if [ -L "$WANDB_DIR/latest-run" ]; then
        LATEST_RUN="$WANDB_DIR/latest-run"
    else
        LATEST_RUN=$(ls -td "$WANDB_DIR"/run-* "$WANDB_DIR"/offline-run-* 2>/dev/null | head -n1)
    fi
    
    if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/files" ]; then
        if [ -f "$LATEST_RUN/files/wandb-summary.json" ]; then
            WANDB_SUMMARY="$LATEST_RUN/files/wandb-summary.json"
            WANDB_METADATA="$LATEST_RUN/files/wandb-metadata.json"
            WANDB_FOUND=1
        fi
    fi
fi

if [ "$WANDB_FOUND" = "1" ] && [ -n "$WANDB_SUMMARY" ]; then
    echo "Uploading W&B metrics to Kratos..."
    
    METADATA_JSON=$(cat "$WANDB_METADATA" 2>/dev/null || echo '{{}}')
    SUMMARY_JSON=$(cat "$WANDB_SUMMARY" 2>/dev/null || echo '{{}}')

    COMBINED_JSON=$(jq -n \
        --arg m "$METADATA_JSON" \
        --arg s "$SUMMARY_JSON" \
        --argjson job_info "$JOB_INFO_JSON" \
        --argjson dashboard_info "$DASHBOARD_INFO_JSON" \
        --argjson all_config "$ALL_CONFIG_JSON" \
        '
        . + {{
          job_name: env.LEPTON_JOB_NAME,
          metadata: ($m | fromjson? // {{}}),
          summary:  ($s | fromjson? // {{}}),
          job_info: $job_info,
          dashboard_info: $dashboard_info,
          config: $all_config
        }}
        ')

    echo "$COMBINED_JSON" > wandb-combined.json

    UUID=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "$(date +%s)-$-$RANDOM")
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

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
            else
                echo "Failed to send event to Kratos: $RESPONSE"
            fi
        else
            echo "Error: Failed to get Kratos access token"
        fi
    fi
else
    echo "W&B metrics not found - skipping Kratos upload"
fi

exit "$RC"
"""


def launch_single_job(client, cfg: DictConfig):
    """Launch a single job with the given configuration."""

    # Get node group
    node_groups = client.nodegroup.list_all()
    node_group_map = {ng.metadata.name: ng for ng in node_groups}

    if cfg.node_group_name not in node_group_map:
        print(f"ERROR: Node group '{cfg.node_group_name}' not found!")
        print(f"  Job: {cfg.job_name}")
        return False

    node_group = node_group_map[cfg.node_group_name]

    # Get valid node IDs
    valid_node_ids = set()
    node_ids = client.nodegroup.list_nodes(node_group.metadata.id_)
    for node in node_ids:
        valid_node_ids.add(node.metadata.id_)

    full_cfg_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))

    # Create command with the extracted parameters
    command = [
        "bash",
        "-c",
        wrap_script_with_logging(
            cfg.script,
            dashboard_info=cfg.dashboard_info if hasattr(cfg, 'dashboard_info') else None,
            recipe_subdir=cfg.recipe_subdir if hasattr(cfg, 'recipe_subdir') else "esm2_native_te_mfsdp",
            all_config_json=full_cfg_json,
        ),
    ]

    # Build environment variables
    env_vars = []

    # Add any additional environment variables from config
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
            print(f"  ✓ Job launched: {cfg.job_name}")
            print(
                f"    View at: https://dashboard.dgxc-lepton.nvidia.com/workspace/vfco61g2/compute/jobs/detail/{launched_job.metadata.id_}/replicas/list"
            )
            return True
    except Exception as e:
        print(f"  ERROR submitting job {cfg.job_name}: {e}")
        return False


@hydra.main(version_base=None, config_path="../configs", config_name="")
def main(cfg: DictConfig):
    """Main function that handles both single and multi-product launches."""

    # Initialize client
    client = APIClient()

    # Disable struct mode at the beginning to allow flexible merging
    OmegaConf.set_struct(cfg, False)

    # Check if products key exists for multi-job launch
    if hasattr(cfg, 'products') and cfg.products:
        print(f"Launching {len(cfg.products)} jobs from products configuration...")
        successful_jobs = 0
        failed_jobs = 0

        for i, product in enumerate(cfg.products, 1):
            # Create a copy of the base config without resolving interpolations
            base_cfg_dict = OmegaConf.to_container(cfg, resolve=False)

            # Remove products from the base config to avoid recursion
            if 'products' in base_cfg_dict:
                del base_cfg_dict['products']

            # Convert product to dict
            product_dict = OmegaConf.to_container(product, resolve=False)

            # Merge the dictionaries
            merged_dict = {**base_cfg_dict, **product_dict}

            # Create new OmegaConf object from merged dict
            product_cfg = OmegaConf.create(merged_dict)

            # Generate job name as recipe_subdir-model_name, replacing underscores and slashes with hyphens
            recipe_subdir = product_cfg.recipe_subdir.replace('_', '-').replace('/', '-')
            model_name = product_dict['model_name'].replace('_', '-').replace('/', '-')
            product_cfg.job_name = f"{model_name}".lower()

            print(f"\n[{i}/{len(cfg.products)}] Launching: {product_cfg.job_name}")

            # Now resolve all interpolations after everything is merged
            resolved_cfg = OmegaConf.to_container(product_cfg, resolve=True)
            product_cfg = OmegaConf.create(resolved_cfg)

            # Launch the job
            if launch_single_job(client, product_cfg):
                successful_jobs += 1
            else:
                failed_jobs += 1

        # Summary
        print(f"\n{'='*50}")
        print(f"Job Launch Summary:")
        print(f"  Successful: {successful_jobs}")
        print(f"  Failed: {failed_jobs}")
        print(f"  Total: {len(cfg.products)}")

    else:
        # Single job launch (original behavior)
        print(f"Launching single job: {cfg.job_name}")
        launch_single_job(client, cfg)


if __name__ == "__main__":
    main()
