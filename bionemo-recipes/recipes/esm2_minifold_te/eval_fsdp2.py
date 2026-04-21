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

"""Production evaluation entrypoint for real-weight MiniFold accuracy validation.

This script keeps the production FSDP2 checkpoint loading path but evaluates through
the plain MiniFold inference runtime after loading the TE checkpoint into a plain
state dict. That allows BF16 and FP8-native comparisons to share the exact same
checkpoint source and checkpoint loading path while reusing the existing native FP8
pair-precision implementation.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tarfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint import get_latest_checkpoint, load_checkpoint_fsdp2
from dataset import create_dataset
from distributed_config import DistributedConfig
from eval_accuracy_utils import (
    append_status_report,
    build_comparison_rows,
    compute_distogram_loss_per_sample,
    compute_distogram_metrics_per_sample,
    evaluate_acceptance,
    filter_state_dict_for_plain_runtime,
    protein_identifier,
    select_eval_stem,
    utc_now_iso,
    write_json,
)
from modeling_esm2_minifold_te import ESM2MiniFoldTE
from quantization import ComponentPrecisionConfig, resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parents[2]
DEFAULT_ARTIFACT_ROOT = Path("/scratch/claude_tasks/accuracy_validation")
DEFAULT_STATUS_REPORT = DEFAULT_ARTIFACT_ROOT / "status_report.md"
DEFAULT_NATIVE_EXT_PIN = DEFAULT_ARTIFACT_ROOT / "minifold_native_ext_pinned"
REPO_TRI_MUL_EXT_ROOT = RECIPE_DIR / "tri_mul_ext"
REPO_BMM_EXT_ROOT = RECIPE_DIR / "fp8_bmm_ext"
DEFAULT_CUTLASS_TAR_CANDIDATES = [
    Path("/scratch/claude_tasks/fp8_model_weights/vendor_backups/cutlass_2026-04-18.tar"),
    Path("/home/jomitchell/jomitchell_scratch/claude_tasks/fp8_model_weights/vendor_backups/cutlass_2026-04-18.tar"),
]
FEATURE_TENSOR_KEYS = {"input_ids", "attention_mask", "mask", "coords", "batch_of"}
RUN_METRIC_KEYS = [
    "loss",
    "distogram_acc",
    "contact_precision_8A",
    "contact_recall_8A",
    "lddt_from_distogram",
    "mean_distance_error",
]


def _abs_path(path_value: str | Path | None, *, relative_root: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return relative_root / path


def _safe_command_output(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_checkpoint_dir(args: DictConfig) -> tuple[Path, dict[str, Any]]:
    requested = Path(args.checkpoint.ckpt_dir)
    fallback = Path(args.checkpoint.fallback_ckpt_dir) if args.checkpoint.get("fallback_ckpt_dir") else None
    used_fallback = False
    resolved = requested

    if not resolved.exists():
        if fallback is None or not fallback.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: requested={requested} fallback={fallback}"
            )
        resolved = fallback
        used_fallback = True

    checkpoint_type = args.checkpoint.get("checkpoint_type", "fsdp2")
    latest_step = None
    latest_checkpoint_dir = None
    if checkpoint_type == "fsdp2":
        latest_checkpoint_dir, latest_step = get_latest_checkpoint(resolved / "train_fsdp2")
        if latest_checkpoint_dir is None:
            raise FileNotFoundError(f"No distributed checkpoint found under {resolved / 'train_fsdp2'}")

    return resolved, {
        "requested_ckpt_dir": str(requested),
        "resolved_ckpt_dir": str(resolved),
        "fallback_ckpt_dir": str(fallback) if fallback is not None else None,
        "used_fallback_ckpt_dir": used_fallback,
        "checkpoint_type": checkpoint_type,
        "latest_step": latest_step,
        "latest_checkpoint_dir": str(latest_checkpoint_dir) if latest_checkpoint_dir is not None else None,
    }


def _extract_cutlass_archive(tar_path: Path, pinned_root: Path) -> tuple[bool, str]:
    third_party_root = pinned_root / "third_party"
    cutlass_root = third_party_root / "cutlass"
    if (cutlass_root / "include" / "cute").exists():
        return True, str(cutlass_root)

    _ensure_dir(third_party_root)
    with tarfile.open(tar_path) as tar:
        tar.extractall(third_party_root)

    if (cutlass_root / "include" / "cute").exists():
        return True, str(cutlass_root)

    candidates = sorted(path.parent.parent for path in third_party_root.glob("*/include/cute"))
    if len(candidates) == 1:
        candidate = candidates[0]
        if candidate != cutlass_root:
            if cutlass_root.exists():
                shutil.rmtree(cutlass_root)
            shutil.move(str(candidate), str(cutlass_root))

    return (cutlass_root / "include" / "cute").exists(), str(cutlass_root)


def _patch_pinned_setup_to_drop_cutlass_source(setup_path: Path) -> bool:
    original = setup_path.read_text(encoding="utf-8")
    patched = original.replace('                "csrc/fc1_direct_cutlass.cu",\n', "")
    if patched == original:
        return False
    setup_path.write_text(patched, encoding="utf-8")
    return True


def _build_pinned_native_extension(artifact_root: Path, status_path: Path) -> dict[str, Any]:
    pinned_root = DEFAULT_NATIVE_EXT_PIN if artifact_root == DEFAULT_ARTIFACT_ROOT else artifact_root / "minifold_native_ext_pinned"
    repo_native_root = RECIPE_DIR / "minifold_native_ext"
    logs_dir = _ensure_dir(artifact_root / "logs")
    build_log_path = logs_dir / "minifold_native_ext_build.log"
    build_info_path = artifact_root / "artifacts" / "minifold_native_ext_build.json"
    _ensure_dir(artifact_root / "artifacts")

    if not pinned_root.exists():
        shutil.copytree(repo_native_root, pinned_root)

    cutlass_tarball = next((path for path in DEFAULT_CUTLASS_TAR_CANDIDATES if path.exists()), None)
    cutlass_ok = False
    cutlass_root = str(pinned_root / "third_party" / "cutlass")
    setup_patched = False
    if cutlass_tarball is not None:
        cutlass_ok, cutlass_root = _extract_cutlass_archive(cutlass_tarball, pinned_root)
    if not cutlass_ok:
        setup_patched = _patch_pinned_setup_to_drop_cutlass_source(pinned_root / "setup.py")

    so_candidates = sorted((pinned_root / "minifold_native_ext").glob("_C*.so"))
    build_info = None
    if build_info_path.exists():
        build_info = json.loads(build_info_path.read_text(encoding="utf-8"))

    if not so_candidates and build_info is not None:
        raise RuntimeError(
            f"Pinned native extension build metadata exists but compiled binary is missing: {build_info_path}"
        )

    if not so_candidates:
        build_cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", str(pinned_root)]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        build_log_path.write_text(result.stdout + "\n\nSTDERR\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            append_status_report(
                status_path,
                "Native Extension Build Failure",
                [
                    f"build command failed: {' '.join(build_cmd)}",
                    f"log: {build_log_path}",
                ],
            )
            raise RuntimeError(f"Pinned native extension build failed; see {build_log_path}")
        so_candidates = sorted((pinned_root / "minifold_native_ext").glob("_C*.so"))
        if not so_candidates:
            raise RuntimeError(f"Editable install completed but no compiled _C*.so found under {pinned_root}")

    build_info = {
        "pinned_root": str(pinned_root),
        "compiled_binary": str(so_candidates[0]),
        "cutlass_tarball": str(cutlass_tarball) if cutlass_tarball is not None else None,
        "cutlass_root": cutlass_root,
        "cutlass_available": cutlass_ok,
        "setup_patched_to_drop_cutlass": setup_patched,
        "timestamp_utc": utc_now_iso(),
    }
    write_json(build_info_path, build_info)
    return build_info


def _build_repo_bmm_extension(artifact_root: Path, status_path: Path) -> dict[str, Any]:
    logs_dir = _ensure_dir(artifact_root / "logs")
    build_log_path = logs_dir / "fp8_bmm_ext_build.log"
    build_info_path = artifact_root / "artifacts" / "fp8_bmm_ext_build.json"
    _ensure_dir(artifact_root / "artifacts")

    build_cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", str(REPO_BMM_EXT_ROOT)]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    build_log_path.write_text(result.stdout + "\n\nSTDERR\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        append_status_report(
            status_path,
            "FP8 BMM Extension Build Failure",
            [
                f"build command failed: {' '.join(build_cmd)}",
                f"log: {build_log_path}",
            ],
        )
        raise RuntimeError(f"Repo-local fp8_bmm_ext build failed; see {build_log_path}")

    so_candidates = sorted((REPO_BMM_EXT_ROOT / "bmm_ext").glob("_C*.so"))
    if not so_candidates:
        raise RuntimeError(f"Editable install completed but no compiled _C*.so found under {REPO_BMM_EXT_ROOT}")

    build_info = {
        "repo_root": str(REPO_BMM_EXT_ROOT),
        "compiled_binary": str(so_candidates[0]),
        "timestamp_utc": utc_now_iso(),
    }
    write_json(build_info_path, build_info)
    return build_info


def _build_repo_tri_mul_extension(artifact_root: Path, status_path: Path) -> dict[str, Any]:
    logs_dir = _ensure_dir(artifact_root / "logs")
    build_log_path = logs_dir / "tri_mul_ext_build.log"
    build_info_path = artifact_root / "artifacts" / "tri_mul_ext_build.json"
    _ensure_dir(artifact_root / "artifacts")

    build_cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", str(REPO_TRI_MUL_EXT_ROOT)]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    build_log_path.write_text(result.stdout + "\n\nSTDERR\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        append_status_report(
            status_path,
            "Tri Mul Extension Build Failure",
            [
                f"build command failed: {' '.join(build_cmd)}",
                f"log: {build_log_path}",
            ],
        )
        raise RuntimeError(f"Repo-local tri_mul_ext build failed; see {build_log_path}")

    so_candidates = sorted((REPO_TRI_MUL_EXT_ROOT / "tri_mul_ext").glob("_C*.so"))
    if not so_candidates:
        raise RuntimeError(f"Editable install completed but no compiled _C*.so found under {REPO_TRI_MUL_EXT_ROOT}")

    build_info = {
        "repo_root": str(REPO_TRI_MUL_EXT_ROOT),
        "compiled_binary": str(so_candidates[0]),
        "timestamp_utc": utc_now_iso(),
    }
    write_json(build_info_path, build_info)
    return build_info


def _load_compiled_module(module_name: str, so_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_plain_module(pair_precision: str, tri_impl: str, artifact_root: Path, status_path: Path):
    if tri_impl == "cublas_xbdnn":
        _build_repo_tri_mul_extension(artifact_root, status_path)
    if pair_precision in {"fp8_extreme", "fp8_hybrid", "fp8_native", "fp8_native_gold_packs"}:
        _build_repo_bmm_extension(artifact_root, status_path)

    import importlib
    import plain_minifold_infer as plain_infer

    plain_infer = importlib.reload(plain_infer)

    native_build_info = _build_pinned_native_extension(artifact_root, status_path)
    native_so_path = Path(native_build_info["compiled_binary"])
    plain_infer.minifold_native_raw = _load_compiled_module("minifold_native_ext._C", native_so_path)

    if pair_precision in {"fp8_native", "fp8_native_gold_packs"} and plain_infer.bmm_ext_raw is None:
        raise RuntimeError(
            "plain_minifold_infer.bmm_ext_raw is unavailable; fp8_native-style paths require the repo-local fp8_bmm_ext build"
        )
    return plain_infer, native_build_info


def _collect_environment_info(native_build_info: dict[str, Any] | None) -> dict[str, Any]:
    driver_version = _safe_command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    )
    return {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "container_name": os.environ.get("HOSTNAME"),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver_version": driver_version.splitlines()[0] if driver_version else None,
        "bionemo_git_branch": _safe_command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT),
        "bionemo_git_commit": _safe_command_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "native_extension": native_build_info,
    }


def _build_source_model(
    args: DictConfig,
    device: torch.device,
    block_precision: list[str | None],
    fp8_recipe,
    fp4_recipe,
    component_precision: ComponentPrecisionConfig,
) -> ESM2MiniFoldTE:
    return ESM2MiniFoldTE(
        esm_model_name=args.esm_model_name,
        c_s=args.model.c_s,
        c_z=args.model.c_z,
        num_blocks=args.model.num_blocks,
        no_bins=args.model.no_bins,
        use_structure_module=args.model.use_structure_module,
        block_precision=block_precision,
        fp8_recipe=fp8_recipe,
        fp4_recipe=fp4_recipe,
        component_precision=component_precision,
    ).to(device)


def _load_te_state_dict(
    args: DictConfig,
    dist_config: DistributedConfig,
    device: torch.device,
    resolved_ckpt_dir: Path,
    block_precision: list[str | None],
    fp8_recipe,
    fp4_recipe,
    component_precision: ComponentPrecisionConfig,
) -> dict[str, torch.Tensor]:
    checkpoint_type = args.checkpoint.get("checkpoint_type", "fsdp2")

    if checkpoint_type == "safetensors":
        return load_file(str(resolved_ckpt_dir / "model.safetensors"))

    if checkpoint_type != "fsdp2":
        raise ValueError(f"Unknown checkpoint_type: {checkpoint_type}")

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size,),
        mesh_dim_names=("dp",),
    )
    model = _build_source_model(args, device, block_precision, fp8_recipe, fp4_recipe, component_precision)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for block in model.fold.miniformer.blocks:
        fully_shard(block, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy_scheduler = get_linear_schedule_with_warmup(dummy_optimizer, num_warmup_steps=0, num_training_steps=1)
    load_checkpoint_fsdp2(
        model=model,
        optimizer=dummy_optimizer,
        scheduler=dummy_scheduler,
        ckpt_path=resolved_ckpt_dir / "train_fsdp2",
        dist_config=dist_config,
    )
    return get_model_state_dict(
        model=model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def _build_plain_runtime(
    args: DictConfig,
    device: torch.device,
    state_dict: dict[str, torch.Tensor],
    artifact_root: Path,
    status_path: Path,
):
    pair_precision = args.pair_precision
    linear_precision = args.linear_precision
    plain_infer, native_build_info = _prepare_plain_module(
        pair_precision,
        args.component_precision.tri_impl,
        artifact_root,
        status_path,
    )
    plain_infer.validate_fp8_extreme_configuration(
        pair_precision,
        linear_precision,
        args.component_precision.tri_impl,
    )
    bf16_native_rung = plain_infer.validate_bf16_native_configuration(
        pair_precision,
        linear_precision,
        args.component_precision.tri_impl,
        args.get("bf16_native_rung"),
    )

    model = plain_infer.PlainESM2MiniFold(
        esm_model_name=args.esm_model_name,
        c_s=args.model.c_s,
        c_z=args.model.c_z,
        num_blocks=args.model.num_blocks,
        no_bins=args.model.no_bins,
        tri_impl=args.component_precision.tri_impl,
        tri_einsum=args.component_precision.tri_einsum,
        pair_precision=pair_precision,
        linear_precision=linear_precision,
        bf16_native_rung=bf16_native_rung,
        hybrid_precision=OmegaConf.to_container(args.get("hybrid_precision"), resolve=True)
        if args.get("hybrid_precision") is not None
        else None,
    ).to(device=device, dtype=torch.bfloat16)

    plain_state_dict = filter_state_dict_for_plain_runtime(state_dict)
    load_result = model.load_state_dict(plain_state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Plain runtime state_dict mismatch. missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
        )

    plain_infer.configure_linear_precision(
        model,
        linear_precision,
        include_transition=pair_precision in plain_infer.FP8_BLOCK32_PAIR_PRECISIONS,
    )
    model.eval()
    return model, plain_infer, native_build_info


def _as_python(value: Any, index: int, default: Any = "") -> Any:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value[index].item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _mirror_eval_dataset_assets(args: DictConfig, artifact_root: Path) -> dict[str, str | None]:
    data_dir = _ensure_dir(artifact_root / "data")
    source_parquet = _abs_path(args.eval_dataset.parquet_path, relative_root=RECIPE_DIR)
    if source_parquet is None:
        raise ValueError("eval_dataset.parquet_path must be set for accuracy validation")
    if not source_parquet.exists():
        raise FileNotFoundError(f"Eval parquet not found: {source_parquet}")

    mirrored_parquet = data_dir / source_parquet.name
    if source_parquet.resolve() != mirrored_parquet.resolve():
        shutil.copy2(source_parquet, mirrored_parquet)

    manifest_path = source_parquet.parent / "eval_manifest.json"
    mirrored_manifest = data_dir / manifest_path.name if manifest_path.exists() else None
    if manifest_path.exists() and manifest_path.resolve() != mirrored_manifest.resolve():
        shutil.copy2(manifest_path, mirrored_manifest)

    pdb_ids_path = source_parquet.parent / "eval_pdb_ids.txt"
    mirrored_pdb_ids = data_dir / pdb_ids_path.name if pdb_ids_path.exists() else None
    if pdb_ids_path.exists() and pdb_ids_path.resolve() != mirrored_pdb_ids.resolve():
        shutil.copy2(pdb_ids_path, mirrored_pdb_ids)

    return {
        "source_parquet_path": str(source_parquet),
        "mirrored_parquet_path": str(mirrored_parquet),
        "source_manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "mirrored_manifest_path": str(mirrored_manifest) if mirrored_manifest is not None else None,
        "source_pdb_ids_path": str(pdb_ids_path) if pdb_ids_path.exists() else None,
        "mirrored_pdb_ids_path": str(mirrored_pdb_ids) if mirrored_pdb_ids is not None else None,
    }


def _create_local_eval_dataloader(args: DictConfig, mirrored_dataset_info: dict[str, str | None]) -> DataLoader:
    eval_kwargs = OmegaConf.to_container(args.eval_dataset, resolve=True)
    eval_kwargs["parquet_path"] = mirrored_dataset_info["mirrored_parquet_path"]
    dataset = create_dataset(**eval_kwargs)
    return DataLoader(
        dataset,
        batch_size=int(args.eval_dataset.micro_batch_size),
        shuffle=False,
        num_workers=int(args.eval_dataset.num_workers),
        pin_memory=True,
        drop_last=False,
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary = {}
    for key in RUN_METRIC_KEYS:
        summary[key] = float(sum(row[key] for row in rows) / len(rows)) if rows else 0.0
    return summary


def _write_run_markdown(payload: dict, markdown_path: Path) -> None:
    summary = payload["summary"]
    checkpoint = payload["checkpoint"]
    dataset = payload["dataset"]
    environment = payload["environment"]
    markdown = "\n".join(
        [
            f"# {payload['run_label']}",
            "",
            f"- Timestamp: {payload['timestamp_utc']}",
            f"- Runtime: {payload['runtime_impl']}",
            f"- Pair precision: {payload['config']['pair_precision']}",
            f"- Linear precision: {payload['config']['linear_precision']}",
            f"- Tri backend: {payload['config']['tri_impl']}",
            f"- Checkpoint dir: {checkpoint['resolved_ckpt_dir']}",
            f"- Checkpoint step: {checkpoint['latest_step']}",
            f"- Eval parquet: {dataset['mirrored_parquet_path']}",
            f"- Proteins evaluated: {dataset['protein_count']}",
            f"- Git commit: {environment.get('bionemo_git_commit')}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| loss | {summary['loss']:.6f} |",
            f"| lDDT | {summary['lddt_from_distogram']:.6f} |",
            f"| distogram_acc | {summary['distogram_acc']:.6f} |",
            f"| mean_distance_error | {summary['mean_distance_error']:.6f} |",
            f"| contact_precision_8A | {summary['contact_precision_8A']:.6f} |",
            f"| contact_recall_8A | {summary['contact_recall_8A']:.6f} |",
        ]
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")


def _write_comparison_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_counsel_report(
    report_path: Path,
    bf16_payload: dict,
    fp8_payload: dict,
    comparison_rows: list[dict[str, Any]],
    acceptance: dict[str, Any],
    comparison_csv_path: Path,
) -> None:
    fp8_label = fp8_payload["config"]["pair_precision"]
    outliers = acceptance["outliers"][:10]
    bands = acceptance["bands"]
    report_lines = [
        "# MiniFold FP8 Accuracy Validation",
        "",
        f"## Verdict: {acceptance['verdict']}",
        "",
        f"- Generated: {utc_now_iso()}",
        f"- BF16 artifact: {bf16_payload['artifacts']['json']}",
        f"- FP8 artifact: {fp8_payload['artifacts']['json']}",
        f"- Per-protein comparison CSV: {comparison_csv_path}",
        f"- BF16 checkpoint: {bf16_payload['checkpoint']['resolved_ckpt_dir']}",
        f"- BF16 checkpoint step: {bf16_payload['checkpoint']['latest_step']}",
        f"- FP8 checkpoint: {fp8_payload['checkpoint']['resolved_ckpt_dir']}",
        f"- FP8 checkpoint step: {fp8_payload['checkpoint']['latest_step']}",
        f"- Eval parquet: {bf16_payload['dataset']['mirrored_parquet_path']}",
        f"- Proteins compared: {acceptance['protein_count']}",
        "",
        "## Acceptance Bands",
        "",
        f"| Band | BF16 | {fp8_label} | Delta | Extra | Passed |",
        "|------|------|------------|-------|-------|--------|",
        (
            f"| lDDT | {bf16_payload['summary']['lddt_from_distogram']:.6f} | "
            f"{fp8_payload['summary']['lddt_from_distogram']:.6f} | "
            f"{bands['lddt']['delta']:+.6f} | Spearman={bands['lddt']['spearman']:.6f} | "
            f"{'yes' if bands['lddt']['passed'] else 'no'} |"
        ),
        (
            f"| distogram_acc | {bf16_payload['summary']['distogram_acc']:.6f} | "
            f"{fp8_payload['summary']['distogram_acc']:.6f} | "
            f"{bands['distogram_acc']['delta']:+.6f} | tolerance=0.005 | "
            f"{'yes' if bands['distogram_acc']['passed'] else 'no'} |"
        ),
        (
            f"| mean_distance_error | {bf16_payload['summary']['mean_distance_error']:.6f} | "
            f"{fp8_payload['summary']['mean_distance_error']:.6f} | "
            f"{bands['mean_distance_error']['delta']:+.6f} | "
            f"relative={bands['mean_distance_error']['relative_delta']:+.6%} | "
            f"{'yes' if bands['mean_distance_error']['passed'] else 'no'} |"
        ),
        (
            f"| catastrophic_outlier | n/a | n/a | n/a | "
            f"worst lDDT delta={bands['catastrophic_outlier']['worst_delta']:+.6f} | "
            f"{'yes' if bands['catastrophic_outlier']['passed'] else 'no'} |"
        ),
        "",
        "## Aggregate Metrics",
        "",
        f"| Metric | BF16 | {fp8_label} | Delta |",
        "|--------|------|------------|-------|",
    ]

    for key in RUN_METRIC_KEYS:
        report_lines.append(
            f"| {key} | {bf16_payload['summary'][key]:.6f} | {fp8_payload['summary'][key]:.6f} | "
            f"{fp8_payload['summary'][key] - bf16_payload['summary'][key]:+.6f} |"
        )

    report_lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Hostname: {bf16_payload['environment'].get('hostname')}",
            f"- GPU: {bf16_payload['environment'].get('gpu_name')}",
            f"- Driver: {bf16_payload['environment'].get('driver_version')}",
            f"- CUDA: {bf16_payload['environment'].get('cuda_version')}",
            f"- PyTorch: {bf16_payload['environment'].get('pytorch_version')}",
            f"- Git branch: {bf16_payload['environment'].get('bionemo_git_branch')}",
            f"- Git commit: {bf16_payload['environment'].get('bionemo_git_commit')}",
            f"- Native extension: {bf16_payload['environment'].get('native_extension', {}).get('compiled_binary')}",
            "",
            "## Outliers",
            "",
        ]
    )

    if not outliers:
        report_lines.append("- No proteins exceeded the catastrophic lDDT drop threshold.")
    else:
        report_lines.extend(
            [
                f"| Protein | Residues | BF16 lDDT | {fp8_label} lDDT | Delta |",
                "|---------|----------|-----------|-----------|-------|",
            ]
        )
        for row in outliers:
            report_lines.append(
                f"| {row['protein_id']} | {row['num_residues']} | {row['bf16_lddt']:.6f} | "
                f"{row['fp8_lddt']:.6f} | {row['lddt_delta']:+.6f} |"
            )

    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- BF16 JSON: {bf16_payload['artifacts']['json']}",
            f"- BF16 Markdown: {bf16_payload['artifacts']['markdown']}",
            f"- FP8 JSON: {fp8_payload['artifacts']['json']}",
            f"- FP8 Markdown: {fp8_payload['artifacts']['markdown']}",
            f"- Per-protein CSV: {comparison_csv_path}",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _comparison_csv_path_for_run(artifacts_dir: Path, run_stem: str) -> Path:
    if run_stem == "fp8_native_eval_metrics":
        return artifacts_dir / "per_protein_comparison.csv"
    suffix = run_stem.removesuffix("_eval_metrics")
    return artifacts_dir / f"per_protein_comparison_{suffix}.csv"


def _maybe_write_comparison_outputs(
    artifact_root: Path,
    report_path: Path | None,
    current_run_stem: str,
) -> dict[str, Any] | None:
    artifacts_dir = artifact_root / "artifacts"
    bf16_json = artifacts_dir / "bf16_baseline_eval_metrics.json"
    current_json = artifacts_dir / f"{current_run_stem}.json"
    if current_run_stem == "bf16_baseline_eval_metrics" or not bf16_json.exists() or not current_json.exists():
        return None

    bf16_payload = json.loads(bf16_json.read_text(encoding="utf-8"))
    fp8_payload = json.loads(current_json.read_text(encoding="utf-8"))
    comparison_rows = build_comparison_rows(bf16_payload, fp8_payload)
    comparison_csv_path = _comparison_csv_path_for_run(artifacts_dir, current_run_stem)
    _write_comparison_csv(comparison_rows, comparison_csv_path)
    acceptance = evaluate_acceptance(bf16_payload, fp8_payload, comparison_rows)

    if report_path is not None:
        _write_counsel_report(report_path, bf16_payload, fp8_payload, comparison_rows, acceptance, comparison_csv_path)

    return {
        "bf16_payload": bf16_payload,
        "fp8_payload": fp8_payload,
        "comparison_rows": comparison_rows,
        "acceptance": acceptance,
        "comparison_csv_path": str(comparison_csv_path),
        "report_path": str(report_path) if report_path is not None else None,
    }


def _evaluate_plain_runtime(model, plain_infer, dataloader: DataLoader, device: torch.device, args: DictConfig) -> list[dict[str, Any]]:
    use_bf16_autocast = args.pair_precision not in (
        plain_infer.PAIR_PRECISION_BF16_NATIVE,
        plain_infer.PAIR_PRECISION_FP8_EXTREME,
        plain_infer.PAIR_PRECISION_FP8_HYBRID,
        plain_infer.PAIR_PRECISION_FP8_NATIVE,
        plain_infer.PAIR_PRECISION_FP8_NATIVE_GOLD_PACKS,
    )
    eval_rows: list[dict[str, Any]] = []
    progress = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in FEATURE_TENSOR_KEYS and isinstance(value, torch.Tensor)
            }
            metadata = {key: value for key, value in batch.items() if key not in model_inputs}

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16_autocast else nullcontext()
            with autocast_ctx:
                r_dict = model(model_inputs, num_recycling=args.model.get("num_recycling", 0))

            losses = compute_distogram_loss_per_sample(
                preds=r_dict["preds"],
                coords=model_inputs["coords"],
                mask=model_inputs["mask"],
                no_bins=args.model.no_bins,
            )
            metrics = compute_distogram_metrics_per_sample(
                preds=r_dict["preds"].float(),
                coords=model_inputs["coords"],
                mask=model_inputs["mask"],
                no_bins=args.model.no_bins,
            )

            batch_lddt = metrics["lddt_from_distogram"].mean().item()
            progress.set_postfix({"lddt": f"{batch_lddt:.3f}"})

            for sample_idx in range(losses.shape[0]):
                pdb_id = str(_as_python(metadata.get("pdb_id"), sample_idx, default=f"sample_{batch_idx}_{sample_idx}"))
                chain_id = str(_as_python(metadata.get("chain_id"), sample_idx, default=""))
                num_residues = int(_as_python(metadata.get("num_residues"), sample_idx, default=0))
                eval_rows.append(
                    {
                        "protein_id": protein_identifier(pdb_id, chain_id),
                        "pdb_id": pdb_id,
                        "chain_id": chain_id,
                        "num_residues": num_residues,
                        "loss": float(losses[sample_idx].item()),
                        "distogram_acc": float(metrics["distogram_acc"][sample_idx].item()),
                        "contact_precision_8A": float(metrics["contact_precision_8A"][sample_idx].item()),
                        "contact_recall_8A": float(metrics["contact_recall_8A"][sample_idx].item()),
                        "lddt_from_distogram": float(metrics["lddt_from_distogram"][sample_idx].item()),
                        "mean_distance_error": float(metrics["mean_distance_error"][sample_idx].item()),
                    }
                )
    progress.close()
    return eval_rows


@hydra.main(config_path="hydra_config", config_name="eval", version_base="1.2")
def main(args: DictConfig) -> None:
    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    logging.getLogger("httpx").setLevel(logging.WARNING)

    artifact_root = _abs_path(args.get("artifact_root"), relative_root=REPO_ROOT) or DEFAULT_ARTIFACT_ROOT
    report_path = _abs_path(args.get("report_path"), relative_root=REPO_ROOT)
    status_path = artifact_root / "status_report.md"
    artifacts_dir = _ensure_dir(artifact_root / "artifacts")

    append_status_report(
        status_path,
        "Eval Startup",
        [
            f"pair_precision={args.pair_precision}",
            f"linear_precision={args.linear_precision}",
            f"artifact_root={artifact_root}",
            f"report_path={report_path}",
        ],
    )

    dist_config = DistributedConfig()
    logger.info("Initializing eval: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    resolved_ckpt_dir, checkpoint_info = _resolve_checkpoint_dir(args)
    append_status_report(
        status_path,
        "Checkpoint Resolution",
        [
            f"requested checkpoint dir: {checkpoint_info['requested_ckpt_dir']}",
            f"resolved checkpoint dir: {checkpoint_info['resolved_ckpt_dir']}",
            f"used fallback checkpoint dir: {checkpoint_info['used_fallback_ckpt_dir']}",
            f"checkpoint type: {checkpoint_info['checkpoint_type']}",
            f"latest checkpoint step: {checkpoint_info['latest_step']}",
        ],
    )

    block_precision = resolve_layer_precision(
        num_layers=args.model.num_blocks,
        fp8_enabled=args.fp8_config.enabled,
        fp4_enabled=args.fp4_config.enabled,
        fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
        fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
    )
    fp8_recipe = None
    fp4_recipe = None
    if args.fp8_config.enabled:
        from transformer_engine.common.recipe import Format

        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    if args.fp4_config.enabled:
        from transformer_engine.common.recipe import Format

        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
            fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
        )
    component_precision = ComponentPrecisionConfig(**OmegaConf.to_container(args.component_precision, resolve=True))

    state_dict = _load_te_state_dict(
        args,
        dist_config,
        device,
        resolved_ckpt_dir,
        block_precision,
        fp8_recipe,
        fp4_recipe,
        component_precision,
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if not dist_config.is_main_process():
        return

    mirrored_dataset_info = _mirror_eval_dataset_assets(args, artifact_root)
    append_status_report(
        status_path,
        "Eval Dataset Mirror",
        [
            f"source parquet: {mirrored_dataset_info['source_parquet_path']}",
            f"mirrored parquet: {mirrored_dataset_info['mirrored_parquet_path']}",
            f"manifest: {mirrored_dataset_info['mirrored_manifest_path']}",
            f"pdb ids: {mirrored_dataset_info['mirrored_pdb_ids_path']}",
        ],
    )

    model, plain_infer, native_build_info = _build_plain_runtime(
        args,
        torch.device("cuda:0"),
        state_dict,
        artifact_root,
        status_path,
    )
    environment_info = _collect_environment_info(native_build_info)
    append_status_report(
        status_path,
        "Plain Runtime Ready",
        [
            f"runtime implementation: plain_from_te_state_dict",
            f"native extension binary: {native_build_info['compiled_binary']}",
            f"git commit: {environment_info.get('bionemo_git_commit')}",
        ],
    )

    eval_dataloader = _create_local_eval_dataloader(args, mirrored_dataset_info)
    eval_rows = _evaluate_plain_runtime(model, plain_infer, eval_dataloader, torch.device("cuda:0"), args)
    summary = _summarize_rows(eval_rows)

    protein_ids_path = artifact_root / "data" / "eval_protein_ids_runtime.txt"
    protein_ids_path.write_text("\n".join(row["protein_id"] for row in eval_rows) + "\n", encoding="utf-8")

    run_stem = select_eval_stem(args.pair_precision, args.linear_precision)
    json_path = artifacts_dir / f"{run_stem}.json"
    markdown_path = artifacts_dir / f"{run_stem}.md"
    payload = {
        "run_label": run_stem,
        "runtime_impl": "plain_from_te_state_dict",
        "timestamp_utc": utc_now_iso(),
        "config": {
            "pair_precision": args.pair_precision,
            "linear_precision": args.linear_precision,
            "tri_impl": args.component_precision.tri_impl,
            "tri_einsum": args.component_precision.tri_einsum,
            "esm_model_name": args.esm_model_name,
            "num_blocks": int(args.model.num_blocks),
            "c_s": int(args.model.c_s),
            "c_z": int(args.model.c_z),
            "no_bins": int(args.model.no_bins),
            "num_recycling": int(args.model.get("num_recycling", 0)),
        },
        "checkpoint": checkpoint_info,
        "dataset": {
            **mirrored_dataset_info,
            "protein_count": len(eval_rows),
            "runtime_pdb_ids_path": str(protein_ids_path),
        },
        "environment": environment_info,
        "summary": summary,
        "per_protein": eval_rows,
        "artifacts": {
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
    }
    write_json(json_path, payload)
    _write_run_markdown(payload, markdown_path)

    comparison_output = _maybe_write_comparison_outputs(artifact_root, report_path, run_stem)
    append_status_report(
        status_path,
        "Artifacts Written",
        [
            f"run json: {json_path}",
            f"run markdown: {markdown_path}",
            f"comparison csv: {comparison_output['comparison_csv_path'] if comparison_output else 'pending second run'}",
            f"report path: {comparison_output['report_path'] if comparison_output else report_path}",
            f"comparison verdict: {comparison_output['acceptance']['verdict'] if comparison_output else 'pending second run'}",
        ],
    )

    run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    wandb.init(**args.wandb_init_args, config=run_config)
    wandb.log({f"eval/{key}": value for key, value in summary.items()})
    wandb.finish()

    logger.info("=== Evaluation Results ===")
    logger.info("Proteins evaluated: %d", len(eval_rows))
    for key, value in summary.items():
        logger.info("  eval/%s: %.6f", key, value)
    logger.info("Artifacts: %s", json_path)
    if comparison_output is not None:
        logger.info("Comparison verdict: %s", comparison_output["acceptance"]["verdict"])


if __name__ == "__main__":
    main()
