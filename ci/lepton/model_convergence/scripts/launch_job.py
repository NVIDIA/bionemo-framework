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
from lepton_utils import construct_env_var, construct_mount, validate_resource_shape
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import LeptonContainer
from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
from leptonai.api.v2.client import APIClient
from omegaconf import DictConfig, OmegaConf
from utils import register_resolvers, render_wrapper_string


# need this to sanitize config inputs (e.g. branch names for wandb)
register_resolvers()


def _resolve_scheduling_target(client, cfg: DictConfig):
    """Single-node-group version.

    Returns (chosen_node_group_obj, valid_node_ids_set, resource_shape).
    """
    # 1) Resolve the desired node group name as a plain string
    desired = None
    if getattr(cfg, "node_group", None):
        desired = str(cfg.node_group).strip()
    elif getattr(cfg, "node_group_name", None):
        desired = str(cfg.node_group_name).strip()
    else:
        raise SystemExit("Set `node_group` (or legacy `node_group_name`).")

    # 2) Get resource shape from config
    if not getattr(cfg, "resource_shape", None):
        raise SystemExit("Set `resource_shape` in your configuration.")
    resource_shape = str(cfg.resource_shape).strip()

    # 3) Validate that resource shape is compatible with node group
    validate_resource_shape(desired, resource_shape)

    # 4) Map available groups
    node_groups = client.nodegroup.list_all()
    node_group_map = {ng.metadata.name: ng for ng in node_groups}

    if desired not in node_group_map:
        available = ", ".join(sorted(node_group_map.keys()))
        raise SystemExit(f"Requested node group not found: {desired}\nAvailable: {available}")

    chosen = node_group_map[desired]

    # 5) Valid node IDs within that group
    valid_node_ids = {n.metadata.id_ for n in client.nodegroup.list_nodes(chosen.metadata.id_)}

    return chosen, valid_node_ids, resource_shape


def launch_single_job(client, cfg: DictConfig):
    """Launch a single job with the given configuration."""
    chosen_group, valid_node_ids, resource_shape = _resolve_scheduling_target(client, cfg)

    full_cfg_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    rendered = render_wrapper_string(cfg.script, full_cfg_json)
    command = ["bash", "-c", rendered]

    # env vars
    env_vars = []
    if getattr(cfg, "environment_variables", None):
        for env_var in cfg.environment_variables:
            env_vars.append(construct_env_var(env_var))

    # mounts
    mounts = []
    if getattr(cfg, "mounts", None):
        mounts = [construct_mount(path=m.path, mount_path=m.mount_path, from_=m.from_) for m in cfg.mounts]

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
        parallelism=1,
        envs=env_vars,
        image_pull_secrets=[cfg.container.registry_auth],
        mounts=mounts,
    )

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
