from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from dataset import create_dataset
from distributed_config import DistributedConfig
from eval_accuracy_utils import append_status_report, utc_now_iso, write_json
from eval_fsdp2 import _evaluate_plain_runtime, _summarize_rows
from plain_runtime_diagnostics import (
    build_mode_args,
    build_plain_runtime_from_args,
    cleanup_model,
    compose_eval_args,
    destroy_distributed_if_initialized,
    load_state_dict_for_eval,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _ladder_specs() -> list[dict]:
    return [
        {
            "rung": "L0",
            "label": "fp8_extreme",
            "pair_precision": "fp8_extreme",
            "linear_precision": "fp8",
            "tri_impl": "fp8_cublaslt",
            "hybrid_precision": None,
        },
        {
            "rung": "L1",
            "label": "native_layernorm",
            "pair_precision": "fp8_hybrid",
            "linear_precision": "fp8",
            "tri_impl": "fp8_cublaslt",
            "hybrid_precision": {
                "use_native_layernorm": True,
                "use_native_linear": False,
                "use_native_gate": False,
                "use_native_tri": False,
                "use_resident_fp8_residual": False,
            },
        },
        {
            "rung": "L2",
            "label": "native_layernorm_linear",
            "pair_precision": "fp8_hybrid",
            "linear_precision": "fp8",
            "tri_impl": "fp8_cublaslt",
            "hybrid_precision": {
                "use_native_layernorm": True,
                "use_native_linear": True,
                "use_native_gate": False,
                "use_native_tri": False,
                "use_resident_fp8_residual": False,
            },
        },
        {
            "rung": "L3",
            "label": "native_layernorm_linear_gate",
            "pair_precision": "fp8_hybrid",
            "linear_precision": "fp8",
            "tri_impl": "fp8_cublaslt",
            "hybrid_precision": {
                "use_native_layernorm": True,
                "use_native_linear": True,
                "use_native_gate": True,
                "use_native_tri": False,
                "use_resident_fp8_residual": False,
            },
        },
        {
            "rung": "L4",
            "label": "native_layernorm_linear_gate_tri",
            "pair_precision": "fp8_hybrid",
            "linear_precision": "fp8",
            "tri_impl": "fp8_cublaslt",
            "hybrid_precision": {
                "use_native_layernorm": True,
                "use_native_linear": True,
                "use_native_gate": True,
                "use_native_tri": True,
                "use_resident_fp8_residual": False,
            },
        },
        {
            "rung": "L5",
            "label": "native_layernorm_linear_gate_tri_residual",
            "pair_precision": "fp8_hybrid",
            "linear_precision": "fp8",
            "tri_impl": "fp8_cublaslt",
            "hybrid_precision": {
                "use_native_layernorm": True,
                "use_native_linear": True,
                "use_native_gate": True,
                "use_native_tri": True,
                "use_resident_fp8_residual": True,
            },
        },
        {
            "rung": "L6",
            "label": "fp8_native",
            "pair_precision": "fp8_native",
            "linear_precision": "fp8",
            "tri_impl": "fp8_cublaslt",
            "hybrid_precision": None,
        },
    ]


def _render_markdown(payload: dict) -> str:
    lines = [
        "# FP8 Hybrid Ladder",
        "",
        f"- Generated: {payload['generated_utc']}",
        f"- Smoke parquet: {payload['smoke_parquet']}",
        f"- Checkpoint: {payload['checkpoint']['resolved_ckpt_dir']}",
        "",
        "| Rung | Label | Pair Precision | lDDT | Distogram Acc | Loss | Mean Distance Error |",
        "|------|-------|----------------|------|---------------|------|---------------------|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['rung']} | {row['label']} | {row['pair_precision']} | {row['summary']['lddt_from_distogram']:.6f} | "
            f"{row['summary']['distogram_acc']:.6f} | {row['summary']['loss']:.6f} | "
            f"{row['summary']['mean_distance_error']:.6f} |"
        )
    first_bad_rung = payload["first_rung_below_half_l0"]
    lines.extend(["", f"- First rung below half of L0 lDDT: {first_bad_rung}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fp8_hybrid ablation ladder on the smoke eval set.")
    parser.add_argument("--config-name", default="eval_real_3B_fp8native")
    parser.add_argument("--artifact-root", type=Path, default=Path("/scratch/claude_tasks/accuracy_validation"))
    parser.add_argument(
        "--smoke-parquet",
        type=Path,
        default=Path("/scratch/claude_tasks/accuracy_validation/data/eval_structures_smoke_5.parquet"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()

    artifact_dir = cli.artifact_root / "artifacts" / "phase_a"
    json_path = cli.output_json or (artifact_dir / "hybrid_ladder_smoke.json")
    markdown_path = cli.output_markdown or (artifact_dir / "hybrid_ladder_smoke.md")
    status_path = cli.artifact_root / "status_report.md"

    base_args = compose_eval_args(cli.config_name, cli.overrides, artifact_root=cli.artifact_root)
    base_args.eval_dataset.parquet_path = str(cli.smoke_parquet)

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(dist_config.local_rank)

    state_dict, checkpoint_info = load_state_dict_for_eval(base_args, dist_config, device)
    destroy_distributed_if_initialized()

    dataset = create_dataset(**OmegaConf.to_container(base_args.eval_dataset, resolve=True))
    dataloader = DataLoader(
        dataset,
        batch_size=int(base_args.eval_dataset.micro_batch_size),
        shuffle=False,
        num_workers=int(base_args.eval_dataset.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    rows = []
    for spec in _ladder_specs():
        mode_args = build_mode_args(
            base_args,
            pair_precision=spec["pair_precision"],
            linear_precision=spec["linear_precision"],
            tri_impl=spec["tri_impl"],
            hybrid_precision=spec["hybrid_precision"],
        )
        model, plain_infer, native_build_info = build_plain_runtime_from_args(
            mode_args,
            torch.device("cuda:0"),
            state_dict,
            cli.artifact_root,
            status_path,
        )
        eval_rows = _evaluate_plain_runtime(model, plain_infer, dataloader, torch.device("cuda:0"), mode_args)
        summary = _summarize_rows(eval_rows)
        rows.append(
            {
                "rung": spec["rung"],
                "label": spec["label"],
                "pair_precision": spec["pair_precision"],
                "linear_precision": spec["linear_precision"],
                "tri_impl": spec["tri_impl"],
                "hybrid_precision": spec["hybrid_precision"],
                "summary": summary,
                "protein_count": len(eval_rows),
                "pair_stats": asdict(model.fold.miniformer.last_pair_precision_stats)
                if model.fold.miniformer.last_pair_precision_stats is not None
                else None,
                "native_extension": native_build_info,
            }
        )
        cleanup_model(model)

    l0_lddt = rows[0]["summary"]["lddt_from_distogram"]
    l0_half = 0.5 * l0_lddt
    first_bad_rung = next(
        (
            row["rung"]
            for row in rows[1:]
            if row["summary"]["lddt_from_distogram"] <= l0_half
        ),
        None,
    )

    payload = {
        "generated_utc": utc_now_iso(),
        "output_json": str(json_path),
        "output_markdown": str(markdown_path),
        "smoke_parquet": str(cli.smoke_parquet),
        "checkpoint": checkpoint_info,
        "rows": rows,
        "l0_half_lddt_threshold": l0_half,
        "first_rung_below_half_l0": first_bad_rung,
    }
    write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    append_status_report(
        status_path,
        "FP8 Hybrid Ladder",
        [
            f"smoke_parquet={cli.smoke_parquet}",
            f"output_json={json_path}",
            f"output_markdown={markdown_path}",
            f"first_rung_below_half_l0={first_bad_rung}",
        ],
    )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
