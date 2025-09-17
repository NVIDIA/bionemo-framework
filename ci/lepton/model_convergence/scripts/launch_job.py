#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Lepton Job submission script with Hydra configuration.

Demo: python launch_job.py --config-name "evo2_finetune_lora" job_name="evo2-finetune-lora-job"
"""

import json
import re

import hydra
from lepton_utils import construct_env_var, construct_mount
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import LeptonContainer
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from leptonai.api.v2.client import APIClient
from omegaconf import DictConfig, OmegaConf


def wrap_script_with_logging(
    script: str,
    all_config_json: str = "{}",
) -> str:
    """Wraps a shell script with logging and error handling for Lepton job execution.

    This function generates a shell script that:
      - Sets strict error handling (`set -euo pipefail`).
      - Retrieves the Lepton job name from the environment variable `LEPTON_JOB_NAME`.
      - Executes the provided training script with error trapping, capturing the return code.
      - (Additional logging and post-processing steps are appended after this wrapper.)

    Args:
        script (str): The shell script to be executed as the main training or job script.
        all_config_json (str, optional): A JSON string of the full configuration for logging or debugging.
            Defaults to "{}".

    Returns:
        str: A shell script string with logging and error handling wrapped around the provided script.
    """
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

echo "pwd"
pwd

echo "ls"
ls

echo "commit in bionemo-framework"
(cd bionemo-framework && git log -1 || true)
# Always grab the exact commit currently checked out in the framework repo
COMMIT_SHA="$(cd bionemo-framework && git rev-parse HEAD 2>/dev/null || true)"
echo "Resolved framework commit: ${{COMMIT_SHA:-<none>}}"

# Authenticate to Lepton
pip install -q leptonai >/dev/null 2>&1 || pip install -q leptonai || true
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

# Ingest provided config JSON
ALL_CONFIG_JSON='{all_config_json}'
if echo "$ALL_CONFIG_JSON" | jq -e . >/dev/null 2>&1; then
  ALL_CONFIG_JSON_UPDATED="$(printf '%s' "$ALL_CONFIG_JSON" | jq -c '.')"
else
  echo "Warning: ALL_CONFIG_JSON is not valid JSON. Using empty object."
  ALL_CONFIG_JSON_UPDATED='{{}}'
fi

# Inject/overwrite the resolved framework commit (only if we actually got one)
if [ -n "${{COMMIT_SHA:-}}" ]; then
  ALL_CONFIG_JSON_UPDATED="$(printf '%s' "$ALL_CONFIG_JSON_UPDATED" | jq -c --arg commit "$COMMIT_SHA" '.commit_sha = $commit')"
fi

# Extract values from config (with sensible defaults)
RECIPE_SUBDIR="$(printf '%s' "$ALL_CONFIG_JSON_UPDATED" | jq -r '.recipe_subdir // "esm2_native_te_mfsdp"')"

# ---------------------------
# Collect NVIDIA SMI as JSON (no cuda_version in --query-gpu)
# ---------------------------
set +e
NVIDIA_SMI_BIN="$(command -v nvidia-smi || echo /usr/bin/nvidia-smi)"
NVIDIA_SMI_JSON="[]"
for GPU_FIELDS in \
'index,uuid,name,driver_version,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,clocks.gr,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,compute_mode' \
'index,uuid,name,driver_version,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,clocks.current.graphics,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,compute_mode' \
'index,uuid,name,driver_version,pci.bus_id,memory.total,memory.free,memory.used,utilization.gpu'; do
  RAW_SMI="$("$NVIDIA_SMI_BIN" --query-gpu="$GPU_FIELDS" --format=csv,noheader,nounits 2>/dev/null || true)"
  if [ -n "$RAW_SMI" ]; then
    NVIDIA_SMI_JSON="$(
      GPU_FIELDS="$GPU_FIELDS" python3 - <<'PY' 2>/dev/null || true
import os, sys, csv, json
keys = [s.strip() for s in os.environ.get("GPU_FIELDS","").split(",") if s.strip()]
rows = []
for r in csv.reader(sys.stdin):
    if not r:
        continue
    vals = [x.strip() for x in r]
    if len(vals) < len(keys):
        vals += [None]*(len(keys)-len(vals))
    rows.append(dict(zip(keys, vals[:len(keys)])))
print(json.dumps(rows))
PY
    <<< "$RAW_SMI"
    )"
    if [ -n "$NVIDIA_SMI_JSON" ] && [ "$NVIDIA_SMI_JSON" != "[]" ]; then
      break
    fi
  fi
done

RAW_APPS="$("$NVIDIA_SMI_BIN" --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [ -n "$RAW_APPS" ]; then
  NVIDIA_COMPUTE_APPS_JSON="$(
    python3 - <<'PY' 2>/dev/null || true
import sys, csv, json
rows=[]
for r in csv.reader(sys.stdin):
    if not r:
        continue
    gpu_uuid = r[0].strip() if len(r)>0 else None
    # pid as int where possible
    pid = None
    if len(r)>1:
        try: pid = int(r[1].strip())
        except: pid = None
    process  = r[2].strip() if len(r)>2 else None
    used_mem = r[3].strip() if len(r)>3 else None
    rows.append({{"gpu_uuid": gpu_uuid, "pid": pid, "process_name": process, "used_memory": used_mem}})
print(json.dumps(rows))
PY
    <<< "$RAW_APPS"
  )"
else
  NVIDIA_COMPUTE_APPS_JSON="[]"
fi

# Driver/CUDA at top level from -q (stable across versions)
DRIVER_VERSION="$("$NVIDIA_SMI_BIN" -q 2>/dev/null | awk -F': ' '/Driver Version/ {{print $2; exit}}')"
CUDA_VERSION="$("$NVIDIA_SMI_BIN" -q 2>/dev/null | awk -F': ' '/CUDA Version/ {{print $2; exit}}')"
NVIDIA_DRIVER_INFO="$(jq -n --arg dv "$DRIVER_VERSION" --arg cv "$CUDA_VERSION" 'def nn($x): if ($x|length)>0 then $x else null end; {{driver_version: nn($dv), cuda_version: nn($cv)}}' 2>/dev/null || echo '{{}}')"
set -e

# Look for W&B files
WANDB_DIR="/workspace/bionemo-framework/recipes/$RECIPE_SUBDIR/wandb"
WANDB_FOUND=0
WANDB_SUMMARY=""
WANDB_METADATA=""

if [ -d "$WANDB_DIR" ]; then
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
        --argjson all_config "$ALL_CONFIG_JSON_UPDATED" \
        --argjson nvidia_smi "$NVIDIA_SMI_JSON" \
        --argjson nvidia_compute_apps "$NVIDIA_COMPUTE_APPS_JSON" \
        --argjson nvidia_driver "$NVIDIA_DRIVER_INFO" \
        '
        . + {{
          job_name: env.LEPTON_JOB_NAME,
          metadata: ($m | fromjson? // {{}}),
          summary:  ($s | fromjson? // {{}}),
          job_info: $job_info,
          config: $all_config,
          nvidia_smi: $nvidia_smi,
          nvidia_compute_apps: $nvidia_compute_apps,
          nvidia_driver: $nvidia_driver
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

    requested = []
    run_only = getattr(cfg, "run_only", "")
    if isinstance(run_only, str) and run_only.strip():
        requested = [s.strip() for s in re.split(r"[,\s]+", run_only) if s.strip()]

    if requested and getattr(cfg, "products", None):
        want = set(requested)
        filtered = [p for p in cfg.products if str(getattr(p, "config", "")) in want]
        if filtered:
            cfg.products = filtered
            print(f"Selected product subset: {', '.join(str(getattr(p, 'config', '')) for p in filtered)}")
        else:
            raise SystemExit(
                f"No products matched {sorted(want)}. "
                f"Available: {sorted(str(getattr(p, 'config', '')) for p in cfg.products)}"
            )

    # Check if products key exists for multi-job launch
    if hasattr(cfg, "products") and cfg.products:
        print(f"Launching {len(cfg.products)} jobs from products configuration...")
        successful_jobs = 0
        failed_jobs = 0

        for i, product in enumerate(cfg.products, 1):
            # Create a copy of the base config without resolving interpolations
            base_cfg_dict = OmegaConf.to_container(cfg, resolve=False)

            # Remove products from the base config to avoid recursion
            if "products" in base_cfg_dict:
                del base_cfg_dict["products"]

            # Convert product to dict
            product_dict = OmegaConf.to_container(product, resolve=False)

            # Merge the dictionaries
            merged_dict = {**base_cfg_dict, **product_dict}

            # Create new OmegaConf object from merged dict
            product_cfg = OmegaConf.create(merged_dict)

            # Generate job name using recipe_subdir and config value
            # Extract the base recipe name from recipe_subdir (e.g., "geneformer" from "geneformer_native_te_mfsdp_fp8")
            recipe_parts = product_cfg.recipe_subdir.split("_")
            base_recipe_name = recipe_parts[0] if recipe_parts else product_cfg.recipe_subdir

            # Create job name as base_recipe_name-config (e.g., "geneformer-10m")
            config_name = product_dict["config"].replace("_", "-").replace("/", "-")
            product_cfg.job_name = f"{base_recipe_name}-{config_name}".lower()

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
        print(f"\n{'=' * 50}")
        print("Job Launch Summary:")
        print(f"  Successful: {successful_jobs}")
        print(f"  Failed: {failed_jobs}")
        print(f"  Total: {len(cfg.products)}")

    else:
        # Single job launch (original behavior)
        print(f"Launching single job: {cfg.job_name}")
        launch_single_job(client, cfg)


if __name__ == "__main__":
    main()
