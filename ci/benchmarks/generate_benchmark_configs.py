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

"""Generate Lepton CI benchmark configs from a CSV benchmarking matrix.

Reads a CSV file defining (Hardware, Model, Variant, Precision) rows with parallelism
strategy columns and generates one Lepton CI YAML config per row. Each config contains
a products array with one entry per parallelism strategy.

Usage:
    # Generate from CI matrix (H100, BF16+FP8)
    python ci/benchmarks/generate_benchmark_configs.py --csv ci/benchmarks/benchmark_matrix.csv

    # Generate from Blackwell matrix (B200, all precisions, single-node)
    python ci/benchmarks/generate_benchmark_configs.py --csv ci/benchmarks/blackwell_matrix.csv

    # Dry run
    python ci/benchmarks/generate_benchmark_configs.py --csv ci/benchmarks/benchmark_matrix.csv --dry-run
"""

import argparse
import csv
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParallelismConfig:
    """A single parallelism strategy parsed from a CSV column header."""

    label: str  # e.g., "1n8g_cp1" or "4n32g_cp16"
    num_nodes: int
    num_devices: int  # GPUs per node
    cp: int
    tp: int  # Real TP for Evo2; informational for recipes
    pp: int  # Real PP for Evo2; always 1 for recipes


@dataclass
class BenchmarkRow:
    """One row of the benchmarking matrix CSV."""

    hardware: str
    model_family: str
    model_variant: str
    precision: str
    notes: str


# ---------------------------------------------------------------------------
# Hardware registry
# ---------------------------------------------------------------------------

HARDWARE_PROFILES = {
    "H100": {
        "node_group": "yo-bom-lepton-001",
        "gpu_type": "h100-sxm",
        "mount_from": "node-nfs:fs1",
    },
    "H200": {
        "node_group": "nv-int-multiteam-nebius-h200-01",
        "gpu_type": "h200",
        "mount_from": "node-nfs:lepton-shared-fs",
    },
    "B200": {
        # Placeholder — update when B200 cluster is available
        "node_group": "TBD-b200-node-group",
        "gpu_type": "b200",
        "mount_from": "TBD-b200-mount",
    },
}

# ---------------------------------------------------------------------------
# Precision registry
# ---------------------------------------------------------------------------

PRECISION_PROFILES = {
    "BF16": {
        "precision": "bf16",
        "fp8_enabled": False,
        "fp8_recipe": "",
        "fp8_format": "",
        "fp4_enabled": False,
        "fp4_recipe": "",
        "fp4_format": "",
    },
    "FP8": {
        "precision": "fp8",
        "fp8_enabled": True,
        "fp8_recipe": "transformer_engine.common.recipe.Float8BlockScaling",
        "fp8_format": "E4M3",
        "fp4_enabled": False,
        "fp4_recipe": "",
        "fp4_format": "",
    },
    "MXFP8": {
        "precision": "mxfp8",
        "fp8_enabled": True,
        "fp8_recipe": "transformer_engine.common.recipe.MXFP8BlockScaling",
        "fp8_format": "E4M3",
        "fp4_enabled": False,
        "fp4_recipe": "",
        "fp4_format": "",
    },
    "NVFP4": {
        "precision": "nvfp4",
        "fp8_enabled": False,
        "fp8_recipe": "",
        "fp8_format": "",
        "fp4_enabled": True,
        "fp4_recipe": "transformer_engine.common.recipe.NVFP4BlockScaling",
        "fp4_format": "E2M1",
    },
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_PROFILES = {
    ("ESM2", "3B"): {
        "recipe_subdir": "esm2_native_te",
        "model_type": "esm2",
        "model_tag": "nvidia/esm2_t36_3B_UR50D",
        "hydra_config": "L1_3B",
        "variant_label": "3b",
        "task_cmd_cp1": "train_fsdp2",
        "task_cmd_cp": "train_fsdp2_cp",
        "has_cp_script": True,
        "framework": "native",
        "num_train_steps": 500,
        "micro_batch_size": 16,
        "run_script_type": "esm2",
    },
    ("ESM2", "15B"): {
        "recipe_subdir": "esm2_native_te",
        "model_type": "esm2",
        "model_tag": "nvidia/esm2_t48_15B_UR50D",
        "hydra_config": "L1_15B_perf_test",
        "variant_label": "15b",
        "task_cmd_cp1": "train_fsdp2",
        "task_cmd_cp": "train_fsdp2_cp",
        "has_cp_script": True,
        "framework": "native",
        "num_train_steps": 500,
        "micro_batch_size": 8,
        "run_script_type": "esm2",
    },
    ("Llama3", "Llama-3.1-8B"): {
        "recipe_subdir": "llama3_native_te",
        "model_type": "llama3",
        "model_tag": "meta-llama/Llama-3.1-8B",
        "hydra_config": "L0_sanity",
        "variant_label": "8b",
        "task_cmd_cp1": "train_fsdp2",
        "task_cmd_cp": "train_fsdp2_cp",
        "has_cp_script": True,
        "framework": "native",
        "num_train_steps": 250,
        "micro_batch_size": 1,
        "run_script_type": "llama3",
    },
    ("Llama3", "OpenGenome2-7B"): {
        "recipe_subdir": "opengenome2_llama_native_te",
        "model_type": "opengenome2",
        "model_tag": "",  # Uses config_kwargs, not a single model_tag
        "hydra_config": "og2_7b_thd_gqa",
        "variant_label": "og2-7b",
        "task_cmd_cp1": "train_fsdp2",
        "task_cmd_cp": "train_fsdp2_cp",
        "has_cp_script": True,
        "framework": "native",
        "num_train_steps": 250,
        "micro_batch_size": 1,
        "run_script_type": "llama3",
    },
    ("CodonFM", "encodon_5b"): {
        "recipe_subdir": "codonfm_native_te",
        "model_type": "codonfm",
        "model_tag": "",  # Uses model_preset
        "hydra_config": "encodon_5b",
        "variant_label": "5b",
        "task_cmd_cp1": "train_fsdp2",
        "task_cmd_cp": "train_fsdp2_cp",
        "has_cp_script": False,  # No train_fsdp2_cp.py yet
        "framework": "native",
        "num_train_steps": 500,
        "micro_batch_size": 4,
        "run_script_type": "codonfm",
    },
    ("Evo2", "evo2/7b-1m"): {
        "recipe_subdir": "evo2",
        "model_type": "evo2",
        "model_tag": "",
        "hydra_config": "",
        "variant_label": "7b-1m",
        "task_cmd_cp1": "train_evo2",
        "task_cmd_cp": "train_evo2",
        "has_cp_script": True,
        "framework": "megatron",
        "num_train_steps": 600,
        "micro_batch_size": 8,
        "run_script_type": "evo2",
    },
}


# ---------------------------------------------------------------------------
# Run script templates
# ---------------------------------------------------------------------------


def _run_script_esm2() -> str:
    return textwrap.dedent("""\
        wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh;
        chmod +x init.sh;
        source init.sh;

        HYDRA_FULL_ERROR=1 torchrun \\
          --nnodes=$NNODES \\
          --nproc_per_node=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) \\
          --node_rank=$NODE_RANK \\
          --master_addr=$MASTER_ADDR \\
          --master_port=$MASTER_PORT \\
          ${task_cmd}.py \\
          --config-name ${config}.yaml \\
          +wandb_init_args.mode=${wandb_init_args.mode} \\
          wandb_init_args.project=${wandb_init_args.project} \\
          +wandb_init_args.group=${wandb_init_args.group} \\
          +wandb_init_args.job_type=${wandb_init_args.job_type} \\
          wandb_init_args.name=${wandb_name} \\
          config_name_or_path=${model_tag} \\
          num_train_steps=${num_train_steps} \\
          dataset.micro_batch_size=${micro_batch_size} \\
          dataset.load_dataset_kwargs.path=${load_dataset_kwargs_path} \\
          dataset.load_dataset_kwargs.streaming=${load_dataset_kwargs_streaming} \\
          cp_size=${cp_size} \\
          log_mfu=true \\
          checkpoint.ckpt_dir= \\
          checkpoint.save_final_model=false \\
          checkpoint.resume_from_checkpoint=false \\
          fp8_config.enabled=${fp8_enabled}
    """)


def _run_script_llama3() -> str:
    return textwrap.dedent("""\
        wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh;
        chmod +x init.sh;
        source init.sh;

        HYDRA_FULL_ERROR=1 torchrun \\
          --nnodes=$NNODES \\
          --nproc_per_node=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) \\
          --node_rank=$NODE_RANK \\
          --master_addr=$MASTER_ADDR \\
          --master_port=$MASTER_PORT \\
          ${task_cmd}.py \\
          --config-name ${config}.yaml \\
          +wandb.mode=${wandb_init_args.mode} \\
          +wandb.project=${wandb_init_args.project} \\
          +wandb.name=${wandb_name} \\
          num_train_steps=${num_train_steps} \\
          dataset.micro_batch_size=${micro_batch_size} \\
          use_meta_device=true \\
          log_mfu=true \\
          checkpoint.ckpt_dir= \\
          checkpoint.save_final_model=false \\
          checkpoint.resume_from_checkpoint=false \\
          fp8_config.enabled=${fp8_enabled}
    """)


def _run_script_codonfm() -> str:
    return textwrap.dedent("""\
        wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh;
        chmod +x init.sh;
        source init.sh;

        HYDRA_FULL_ERROR=1 torchrun \\
          --nnodes=$NNODES \\
          --nproc_per_node=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) \\
          --node_rank=$NODE_RANK \\
          --master_addr=$MASTER_ADDR \\
          --master_port=$MASTER_PORT \\
          ${task_cmd}.py \\
          --config-name ${config}.yaml \\
          +wandb_init_args.mode=${wandb_init_args.mode} \\
          wandb_init_args.project=${wandb_init_args.project} \\
          +wandb_init_args.group=${wandb_init_args.group} \\
          +wandb_init_args.job_type=${wandb_init_args.job_type} \\
          wandb_init_args.name=${wandb_name} \\
          num_train_steps=${num_train_steps} \\
          dataset.micro_batch_size=${micro_batch_size} \\
          log_mfu=true \\
          checkpoint.ckpt_dir= \\
          checkpoint.save_final_model=false \\
          checkpoint.resume_from_checkpoint=false \\
          fp8_config.enabled=${fp8_enabled}
    """)


def _run_script_evo2() -> str:
    return textwrap.dedent("""\
        wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh;
        chmod +x init.sh;
        source init.sh;

        WANDB_API_KEY=$BIONEMO_WANDB_API_KEY ${variant}_${model} \\
          -d /workspace/bionemo/sub-packages/bionemo-evo2/examples/configs/full_pretrain_shortphase_config.yaml \\
          --dataset-dir ${data_path} \\
          --fp8 --fp8-wgrad --activation-checkpoint-recompute-num-layers 5 \\
          --enable-preemption \\
          --ckpt-async-save \\
          --use-megatron-comm-overlap-llama3-8k \\
          --overlap-grad-reduce \\
          --eod-pad-in-loss-mask \\
          --seq-length=${seq_len} \\
          --seed 3735928559 \\
          --lr=0.00015 \\
          --wd=0.1 \\
          --min-lr=0.000015 \\
          --warmup-steps=5000 \\
          --tensor-parallel-size=${tp} \\
          --context-parallel-size=${cp} \\
          --pipeline-model-parallel-size=${pp} \\
          --workers 8 \\
          --num-nodes=${nodes} \\
          --devices=${gpus} \\
          --micro-batch-size=${batch_size} \\
          --model-size=${config_name} \\
          --max-steps=${max_steps} \\
          --early-stop-on-step ${stop_steps} \\
          --limit-val-batches=20 \\
          --log-every-n-steps=50 \\
          --val-check-interval=200 \\
          --use-subquadratic_ops \\
          --create-tflops-callback \\
          --create-tensorboard-logger \\
          --disable-checkpointing;
    """)


RUN_SCRIPT_TEMPLATES = {
    "esm2": _run_script_esm2,
    "llama3": _run_script_llama3,
    "codonfm": _run_script_codonfm,
    "evo2": _run_script_evo2,
}

# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------

_COLUMN_PATTERN = re.compile(
    r"(Single|Multi)\s+Node\s+(\d+)GPU\s+\((.+)\)",
    re.IGNORECASE,
)


def parse_parallelism_columns(headers: list[str]) -> list[ParallelismConfig]:
    """Parse parallelism configs from CSV column headers (columns 5+)."""
    configs = []
    for header in headers:
        m = _COLUMN_PATTERN.match(header.strip())
        if not m:
            continue
        total_gpus = int(m.group(2))
        params_str = m.group(3)

        # Parse CP=N; TP=N[; PP=N]
        params = {}
        for part in params_str.split(";"):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                params[key.strip().upper()] = int(val.strip())

        cp = params.get("CP", 1)
        tp = params.get("TP", 1)
        pp = params.get("PP", 1)
        num_devices = 8  # GPUs per node
        num_nodes = total_gpus // num_devices

        label = f"{num_nodes}n{total_gpus}g_cp{cp}"
        configs.append(
            ParallelismConfig(
                label=label,
                num_nodes=num_nodes,
                num_devices=num_devices,
                cp=cp,
                tp=tp,
                pp=pp,
            )
        )
    return configs


def parse_csv(csv_path: Path) -> tuple[list[BenchmarkRow], list[ParallelismConfig]]:
    """Parse the benchmark matrix CSV. Returns (rows, parallelism_configs)."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        headers = next(reader)

    parallelism_configs = parse_parallelism_columns(headers[5:])

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hardware = row["Hardware"].strip()
            if not hardware:
                continue
            rows.append(
                BenchmarkRow(
                    hardware=hardware,
                    model_family=row["Model Family"].strip(),
                    model_variant=row["Model Variant"].strip(),
                    precision=row["Precision"].strip(),
                    notes=row.get("Notes", "").strip(),
                )
            )
    return rows, parallelism_configs


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _sanitize(s: str) -> str:
    """Lowercase and replace non-alphanumeric chars with dashes."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def build_products(
    row: BenchmarkRow,
    model_profile: dict,
    hardware_profile: dict,
    parallelism_configs: list[ParallelismConfig],
) -> list[dict]:
    """Build the products array for a benchmark config."""
    products = []
    hw_label = _sanitize(row.hardware)
    variant_label = model_profile["variant_label"]
    precision_label = _sanitize(row.precision)
    gpu_type = hardware_profile["gpu_type"]

    for par in parallelism_configs:
        task_cmd = model_profile["task_cmd_cp1"] if par.cp == 1 else model_profile["task_cmd_cp"]

        # Check if CP script exists
        disabled = par.cp > 1 and not model_profile["has_cp_script"]

        product = {
            "config": model_profile["hydra_config"],
            "task_cmd": task_cmd,
            "num_nodes": par.num_nodes,
            "num_devices": par.num_devices,
            "cp_size": par.cp,
            "resource_shape": f"gpu.{par.num_devices}x{gpu_type}",
            "wandb_name": (
                f"bench__{model_profile['model_type']}_{variant_label}"
                f"__{precision_label}__{par.label}"
                "__${now:%Y%m%d-%H%M%S}__${gitsha:}"
            ),
            "job_name": f"bench-{hw_label}-{model_profile['model_type']}-{variant_label}-{precision_label}-{par.label}",
        }

        # Evo2 needs real TP/PP
        if model_profile["framework"] == "megatron":
            product["tp"] = par.tp
            product["pp"] = par.pp
            product["cp"] = par.cp

        if disabled:
            product["disabled"] = True

        products.append(product)

    return products


def build_config(
    row: BenchmarkRow,
    model_profile: dict,
    hardware_profile: dict,
    precision_profile: dict,
    products: list[dict],
) -> dict:
    """Build the full Lepton CI config dict for one CSV row."""
    config = {}

    # Cluster
    config["node_group"] = hardware_profile["node_group"]
    config["mount_from"] = hardware_profile["mount_from"]
    config["device_type"] = "gpu"
    config["gpu_type"] = hardware_profile["gpu_type"]

    # Recipe identifiers
    config["recipe_subdir"] = model_profile["recipe_subdir"]
    config["model_type"] = model_profile["model_type"]
    config["variant"] = "train"
    config["framework"] = model_profile["framework"]
    config["precision"] = precision_profile["precision"]
    config["te_enabled"] = True
    config["fp8_enabled"] = precision_profile["fp8_enabled"]
    if precision_profile["fp8_recipe"]:
        config["fp8_recipe"] = precision_profile["fp8_recipe"]
        config["fp8_format"] = precision_profile["fp8_format"]
    if precision_profile["fp4_enabled"]:
        config["fp4_enabled"] = True
        config["fp4_recipe"] = precision_profile["fp4_recipe"]
        config["fp4_format"] = precision_profile["fp4_format"]

    # Task defaults
    if model_profile["model_tag"]:
        config["model_tag"] = model_profile["model_tag"]
    config["num_train_steps"] = model_profile["num_train_steps"]
    config["micro_batch_size"] = model_profile["micro_batch_size"]
    config["log_mfu"] = True

    # ESM2-specific dataset config
    if model_profile["run_script_type"] == "esm2":
        config["load_dataset_kwargs_path"] = "nvidia/esm2_uniref_pretraining_data"
        config["load_dataset_kwargs_streaming"] = True

    # W&B
    config["total_gpus"] = "${multiply:${num_devices},${num_nodes}}"
    config["wandb_init_args"] = {
        "project": "perf_benchmarks__recipes__${sanitize:${branch}}",
        "group": "${model_type}__${task_cmd}__${total_gpus}gpus__${sanitize:${gpu_type}}",
        "job_type": "${recipe_subdir}",
        "name": None,
    }

    # Products
    config["products"] = products

    # Run script
    run_script_fn = RUN_SCRIPT_TEMPLATES.get(model_profile["run_script_type"])
    if run_script_fn:
        config["run_script"] = run_script_fn()

    # Expected output schema
    config["expected_output"] = {
        "description": f"Benchmark validation for {row.hardware} {row.model_family} {row.model_variant} {row.precision}",
        "tests": [
            {
                "criteria": {
                    "exit_code": 0,
                    "metrics": {
                        "train/step_time": {"operator": "gt", "value": 0},
                        "train/tokens_per_second_per_gpu": {"operator": "gt", "value": 0},
                        "train/mfu_percent": {"operator": "geq", "value": 0},
                        "train/tflops_per_gpu": {"operator": "geq", "value": 0},
                        "train/gpu_memory_allocated_max_gb": {"operator": "gt", "value": 0},
                    },
                }
            }
        ],
    }

    return config


def render_yaml(config: dict, row: BenchmarkRow) -> str:
    """Render a config dict to YAML string with header comments."""
    # Build header
    lines = [
        "# @package _global_",
        "# AUTO-GENERATED by ci/benchmarks/generate_benchmark_configs.py",
        f"# Source row: {row.hardware}, {row.model_family}, {row.model_variant}, {row.precision}",
        "# Regenerate: python ci/benchmarks/generate_benchmark_configs.py",
    ]
    if row.notes:
        lines.append(f"# NOTE: {row.notes}")
    lines.extend(
        [
            "defaults:",
            "  - /base",
            "  - _self_",
            "",
        ]
    )

    # Use a custom representer to handle interpolations and multiline strings
    class _Dumper(yaml.SafeDumper):
        pass

    def _str_representer(dumper, data):
        # Multiline strings get block scalar style
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        # OmegaConf interpolations get double-quoted
        if "${" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    _Dumper.add_representer(str, _str_representer)

    body = yaml.dump(config, Dumper=_Dumper, default_flow_style=False, sort_keys=False, width=120)
    lines.append(body)

    return "\n".join(lines)


def generate_configs(
    csv_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    model_filter: list[str] | None = None,
    validate_only: bool = False,
) -> None:
    """Main generation pipeline."""
    rows, parallelism_configs = parse_csv(csv_path)
    print(f"Parsed {len(rows)} rows with {len(parallelism_configs)} parallelism strategies from {csv_path}")

    if validate_only:
        for row in rows:
            key = (row.model_family, row.model_variant)
            if key not in MODEL_PROFILES:
                print(f"  WARNING: Unknown model {key}")
            if row.precision not in PRECISION_PROFILES:
                print(f"  WARNING: Unknown precision {row.precision}")
            if row.hardware not in HARDWARE_PROFILES:
                print(f"  WARNING: Unknown hardware {row.hardware}")
            if row.notes:
                print(f"  NOTE ({row.hardware} {row.model_family} {row.model_variant} {row.precision}): {row.notes}")
        return

    generated = 0
    for row in rows:
        key = (row.model_family, row.model_variant)
        if key not in MODEL_PROFILES:
            print(f"  SKIP: Unknown model {key}")
            continue
        if row.precision not in PRECISION_PROFILES:
            print(f"  SKIP: Unknown precision {row.precision}")
            continue
        if row.hardware not in HARDWARE_PROFILES:
            print(f"  SKIP: Unknown hardware {row.hardware}")
            continue

        model_profile = MODEL_PROFILES[key]
        hardware_profile = HARDWARE_PROFILES[row.hardware]
        precision_profile = PRECISION_PROFILES[row.precision]

        # Apply model filter
        if model_filter and model_profile["model_type"] not in model_filter:
            continue

        products = build_products(row, model_profile, hardware_profile, parallelism_configs)
        config = build_config(row, model_profile, hardware_profile, precision_profile, products)
        yaml_content = render_yaml(config, row)

        # Output path
        hw_label = _sanitize(row.hardware)
        variant_label = model_profile["variant_label"]
        precision_label = _sanitize(row.precision)
        model_dir = model_profile["model_type"]
        filename = f"{hw_label}_{model_profile['model_type']}_{variant_label}_{precision_label}.yaml"
        out_path = output_dir / model_dir / filename

        if dry_run:
            print(f"  [DRY RUN] Would write: {out_path} ({len(products)} products)")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_content)
            print(f"  Written: {out_path} ({len(products)} products)")
        generated += 1

    print(f"\n{'Would generate' if dry_run else 'Generated'} {generated} benchmark configs")


# ---------------------------------------------------------------------------
# Local config generation (Hydra overrides into recipe dirs)
# ---------------------------------------------------------------------------


def _build_local_base_config(row: BenchmarkRow, model_profile: dict, precision_profile: dict) -> dict:
    """Build a Hydra override config for local execution (CP=1 base)."""
    config = {}

    # Model identifier
    if model_profile["model_tag"]:
        config["config_name_or_path"] = model_profile["model_tag"]

    config["num_train_steps"] = model_profile["num_train_steps"]
    config["log_mfu"] = True

    config["dataset"] = {"micro_batch_size": model_profile["micro_batch_size"]}

    # Precision
    if precision_profile["fp8_enabled"]:
        config["fp8_config"] = {
            "enabled": True,
            "fp8_recipe": precision_profile["fp8_recipe"],
            "fp8_format": precision_profile["fp8_format"],
        }
    if precision_profile["fp4_enabled"]:
        config["fp4_config"] = {
            "enabled": True,
            "fp4_recipe": precision_profile["fp4_recipe"],
            "fp4_format": precision_profile["fp4_format"],
        }

    # Benchmarks don't checkpoint
    config["checkpoint"] = {"ckpt_dir": "", "save_final_model": False, "resume_from_checkpoint": False}

    # W&B
    hw_label = _sanitize(row.hardware)
    variant_label = model_profile["variant_label"]
    precision_label = precision_profile["precision"]
    wandb_key = "wandb_init_args" if model_profile["run_script_type"] == "esm2" else "wandb"
    config[wandb_key] = {"name": f"bench_{hw_label}_{variant_label}_{precision_label}", "mode": "online"}

    return config


def _render_local_yaml(config: dict, defaults_from: str, comment: str) -> str:
    """Render a local Hydra config with defaults and a comment header."""

    class _Dumper(yaml.SafeDumper):
        pass

    def _str_representer(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    _Dumper.add_representer(str, _str_representer)

    lines = [
        f"# AUTO-GENERATED benchmark config — {comment}",
        "# Regenerate: python ci/benchmarks/generate_benchmark_configs.py --mode local",
        "defaults:",
        f"  - {defaults_from}",
        "  - _self_",
        "",
    ]
    body = yaml.dump(config, Dumper=_Dumper, default_flow_style=False, sort_keys=False, width=120)
    lines.append(body)
    return "\n".join(lines)


def generate_local_configs(
    csv_path: Path,
    recipes_root: Path,
    *,
    dry_run: bool = False,
    model_filter: list[str] | None = None,
) -> list[dict]:
    """Generate Hydra override configs into each recipe's hydra_config/ dir.

    Returns a list of benchmark descriptors for RUN_BENCHMARKS.md generation.
    """
    rows, parallelism_configs = parse_csv(csv_path)
    print(f"Parsed {len(rows)} rows with {len(parallelism_configs)} parallelism strategies from {csv_path}")

    generated = 0
    benchmarks = []  # For documenting run commands

    # Group rows by (model_family, model_variant) to avoid duplicate base configs
    seen_bases = set()

    for row in rows:
        key = (row.model_family, row.model_variant)
        if key not in MODEL_PROFILES:
            print(f"  SKIP: Unknown model {key}")
            continue
        if row.precision not in PRECISION_PROFILES:
            print(f"  SKIP: Unknown precision {row.precision}")
            continue
        if row.hardware not in HARDWARE_PROFILES:
            print(f"  SKIP: Unknown hardware {row.hardware}")
            continue

        model_profile = MODEL_PROFILES[key]
        precision_profile = PRECISION_PROFILES[row.precision]

        if model_filter and model_profile["model_type"] not in model_filter:
            continue

        # Skip Evo2 — different framework, not a Hydra recipe
        if model_profile["framework"] == "megatron":
            print(f"  SKIP: {row.model_family}/{row.model_variant} (Megatron, not a Hydra recipe)")
            continue

        hw_label = _sanitize(row.hardware)
        variant_label = model_profile["variant_label"]
        precision_label = _sanitize(row.precision)
        recipe_dir = recipes_root / model_profile["recipe_subdir"]
        hydra_dir = recipe_dir / "hydra_config"

        base_name = f"bench_{hw_label}_{variant_label}_{precision_label}"
        base_key = (model_profile["recipe_subdir"], base_name)

        # Generate base config (CP=1)
        if base_key not in seen_bases:
            seen_bases.add(base_key)
            base_config = _build_local_base_config(row, model_profile, precision_profile)
            base_yaml = _render_local_yaml(
                base_config,
                "defaults",
                f"{row.hardware} {row.model_family} {row.model_variant} {row.precision}",
            )
            base_path = hydra_dir / f"{base_name}.yaml"

            if dry_run:
                print(f"  [DRY RUN] Would write: {base_path}")
            else:
                base_path.parent.mkdir(parents=True, exist_ok=True)
                base_path.write_text(base_yaml)
                print(f"  Written: {base_path}")
            generated += 1

            # Record CP=1 benchmark
            benchmarks.append(
                {
                    "recipe_subdir": model_profile["recipe_subdir"],
                    "config_name": base_name,
                    "script": model_profile["task_cmd_cp1"],
                    "cp": 1,
                    "description": f"{row.hardware} {row.model_family} {row.model_variant} {row.precision} CP=1",
                    "notes": row.notes,
                    "disabled": False,
                }
            )

        # Generate CP>1 override configs
        for par in parallelism_configs:
            if par.cp <= 1:
                continue
            if par.num_nodes > 1:
                continue  # Local mode is single-node only

            cp_name = f"{base_name}_cp{par.cp}"
            cp_config = {"cp_size": par.cp}
            disabled = not model_profile["has_cp_script"]

            cp_yaml = _render_local_yaml(
                cp_config,
                base_name,
                f"{row.hardware} {row.model_family} {row.model_variant} {row.precision} CP={par.cp}",
            )
            if disabled:
                cp_yaml = (
                    f"# WARNING: {model_profile['recipe_subdir']} does not have train_fsdp2_cp.py yet\n" + cp_yaml
                )

            cp_path = hydra_dir / f"{cp_name}.yaml"

            if dry_run:
                print(f"  [DRY RUN] Would write: {cp_path}{' (DISABLED)' if disabled else ''}")
            else:
                cp_path.parent.mkdir(parents=True, exist_ok=True)
                cp_path.write_text(cp_yaml)
                print(f"  Written: {cp_path}{' (DISABLED)' if disabled else ''}")
            generated += 1

            benchmarks.append(
                {
                    "recipe_subdir": model_profile["recipe_subdir"],
                    "config_name": cp_name,
                    "script": model_profile["task_cmd_cp"],
                    "cp": par.cp,
                    "description": f"{row.hardware} {row.model_family} {row.model_variant} {row.precision} CP={par.cp}",
                    "notes": row.notes,
                    "disabled": disabled,
                }
            )

    print(f"\n{'Would generate' if dry_run else 'Generated'} {generated} local Hydra configs")
    return benchmarks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for benchmark config generation."""
    parser = argparse.ArgumentParser(description="Generate benchmark configs from CSV matrix")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("ci/benchmarks/benchmark_matrix.csv"),
        help="Path to the benchmark matrix CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ci/lepton/perf_benchmarks/configs"),
        help="Output directory for generated Lepton YAML configs (lepton mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["lepton", "local"],
        default="lepton",
        help="Generation mode: 'lepton' for CI configs, 'local' for Hydra recipe configs",
    )
    parser.add_argument(
        "--recipes-root",
        type=Path,
        default=Path("bionemo-recipes/recipes"),
        help="Root directory of recipes (local mode)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated without writing")
    parser.add_argument(
        "--models", type=str, default=None, help="Comma-separated model types to generate (e.g., esm2,llama3)"
    )
    parser.add_argument("--validate-only", action="store_true", help="Parse CSV and print warnings only")
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    if args.mode == "lepton":
        generate_configs(
            csv_path=args.csv,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            model_filter=model_filter,
            validate_only=args.validate_only,
        )
    else:
        generate_local_configs(
            csv_path=args.csv,
            recipes_root=args.recipes_root,
            dry_run=args.dry_run,
            model_filter=model_filter,
        )


if __name__ == "__main__":
    main()
