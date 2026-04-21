from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf
import torch

from distributed_config import DistributedConfig
from eval_accuracy_utils import append_status_report, utc_now_iso, write_json
from plain_runtime_diagnostics import (
    build_mode_args,
    build_plain_runtime_from_args,
    cleanup_model,
    compose_eval_args,
    destroy_distributed_if_initialized,
    extract_dataset_sample,
    load_state_dict_for_eval,
    run_plain_forward,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _default_linear_precision(pair_precision: str) -> str:
    return "fp8" if pair_precision in {"fp8_extreme", "fp8_hybrid", "fp8_native"} else "bf16"


def _default_tri_impl(pair_precision: str) -> str:
    return "fp8_cublaslt" if pair_precision in {"fp8_extreme", "fp8_hybrid", "fp8_native"} else "cublas_xbdnn"


def _make_hybrid_config(cli) -> dict[str, bool] | None:
    if cli.pair_precision != "fp8_hybrid":
        return None
    return {
        "use_native_layernorm": cli.use_native_layernorm,
        "use_native_linear": cli.use_native_linear,
        "use_native_gate": cli.use_native_gate,
        "use_native_tri": cli.use_native_tri,
        "use_resident_fp8_residual": cli.use_resident_fp8_residual,
    }


def _first_nonfinite_event(events: list[dict]) -> dict | None:
    for event in events:
        if event["nan_count"] > 0 or event["inf_count"] > 0:
            return event
    return None


def _magnitude_ratio(a: float, b: float) -> float:
    a = abs(a)
    b = abs(b)
    floor = 1e-12
    return max(a, b, floor) / max(min(a, b), floor)


def build_probe_diff(reference_payload: dict, candidate_payload: dict) -> dict:
    ref_events = {
        (event["block_idx"], event["op_name"]): event
        for event in reference_payload["trace"]["events"]
    }
    candidate_events = candidate_payload["trace"]["events"]
    rows = []
    first_nonfinite = None
    first_large_divergence = None
    shared_event_count = 0
    for candidate_event in candidate_events:
        key = (candidate_event["block_idx"], candidate_event["op_name"])
        ref_event = ref_events.get(key)
        if ref_event is None:
            continue
        shared_event_count += 1
        ratio = _magnitude_ratio(candidate_event["max_abs"], ref_event["max_abs"])
        row = {
            "block_idx": key[0],
            "op_name": key[1],
            "reference_mode_op_name": ref_event["mode_op_name"],
            "candidate_mode_op_name": candidate_event["mode_op_name"],
            "reference_max_abs": ref_event["max_abs"],
            "candidate_max_abs": candidate_event["max_abs"],
            "max_abs_ratio": ratio,
            "reference_nan_count": ref_event["nan_count"],
            "candidate_nan_count": candidate_event["nan_count"],
            "reference_inf_count": ref_event["inf_count"],
            "candidate_inf_count": candidate_event["inf_count"],
        }
        rows.append(row)
        if first_nonfinite is None and (candidate_event["nan_count"] > 0 or candidate_event["inf_count"] > 0):
            first_nonfinite = row
        if first_large_divergence is None and ratio >= 10.0:
            first_large_divergence = row
    return {
        "generated_utc": utc_now_iso(),
        "reference_trace": str(reference_payload["output"]),
        "candidate_trace": str(candidate_payload["output"]),
        "shared_event_count": shared_event_count,
        "first_nonfinite_event": first_nonfinite,
        "first_large_magnitude_divergence": first_large_divergence,
        "rows": rows,
    }


def render_probe_diff_markdown(diff_payload: dict) -> str:
    lines = [
        "# Activation Probe Diff",
        "",
        f"- Generated: {diff_payload['generated_utc']}",
        f"- Reference trace: {diff_payload['reference_trace']}",
        f"- Candidate trace: {diff_payload['candidate_trace']}",
        f"- Shared events: {diff_payload['shared_event_count']}",
        "",
    ]
    first_nonfinite = diff_payload["first_nonfinite_event"]
    if first_nonfinite is None:
        lines.append("- First nonfinite event: none")
    else:
        lines.append(
            "- First nonfinite event: "
            f"block {first_nonfinite['block_idx']} / {first_nonfinite['op_name']} "
            f"({first_nonfinite['candidate_mode_op_name']})"
        )
    first_large = diff_payload["first_large_magnitude_divergence"]
    if first_large is None:
        lines.append("- First >=10x magnitude divergence: none")
    else:
        lines.append(
            "- First >=10x magnitude divergence: "
            f"block {first_large['block_idx']} / {first_large['op_name']} "
            f"ratio={first_large['max_abs_ratio']:.3f}"
        )
    lines.extend(
        [
            "",
            "| Block | Canonical Op | Reference | Candidate | Max Abs Ratio | Candidate NaN | Candidate Inf |",
            "|-------|--------------|-----------|-----------|---------------|---------------|---------------|",
        ]
    )
    for row in diff_payload["rows"][:20]:
        lines.append(
            f"| {row['block_idx']} | {row['op_name']} | {row['reference_mode_op_name']} | "
            f"{row['candidate_mode_op_name']} | {row['max_abs_ratio']:.3f} | "
            f"{row['candidate_nan_count']} | {row['candidate_inf_count']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-sample activation probe on the plain MiniFold runtime.")
    parser.add_argument("--config-name", default="eval_real_3B_fp8native")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path, default=Path("/scratch/claude_tasks/accuracy_validation"))
    parser.add_argument("--pair-precision", default="fp8_native", choices=["bf16", "fp8_extreme", "fp8_hybrid", "fp8_native"])
    parser.add_argument("--linear-precision", default="")
    parser.add_argument("--tri-impl", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare-other", type=Path, default=None)
    parser.add_argument("--compare-report", type=Path, default=None)
    parser.add_argument("--use-native-layernorm", action="store_true")
    parser.add_argument("--use-native-linear", action="store_true")
    parser.add_argument("--use-native-gate", action="store_true")
    parser.add_argument("--use-native-tri", action="store_true")
    parser.add_argument("--use-resident-fp8-residual", action="store_true")
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()

    phase_dir = cli.artifact_root / "artifacts" / "phase_a"
    output_path = cli.output or (phase_dir / f"activation_probe_{cli.pair_precision}.json")
    compare_report = cli.compare_report or (phase_dir / "probe_diff_report.md")
    status_path = cli.artifact_root / "status_report.md"

    base_args = compose_eval_args(cli.config_name, cli.overrides, artifact_root=cli.artifact_root)
    mode_args = build_mode_args(
        base_args,
        pair_precision=cli.pair_precision,
        linear_precision=cli.linear_precision or _default_linear_precision(cli.pair_precision),
        tri_impl=cli.tri_impl or _default_tri_impl(cli.pair_precision),
        hybrid_precision=_make_hybrid_config(cli),
    )

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(dist_config.local_rank)

    state_dict, checkpoint_info = load_state_dict_for_eval(base_args, dist_config, device)
    destroy_distributed_if_initialized()
    sample_batch, sample_metadata = extract_dataset_sample(mode_args.eval_dataset, cli.sample_index)

    model, plain_infer, native_build_info = build_plain_runtime_from_args(
        mode_args,
        torch.device("cuda:0"),
        state_dict,
        cli.artifact_root,
        status_path,
    )
    probe = plain_infer.ActivationProbe(pair_precision_mode=mode_args.pair_precision)
    run_plain_forward(
        model,
        {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value for key, value in sample_batch.items()},
        mode_args,
        plain_infer,
        activation_probe=probe,
        dump_activation_stats=True,
    )

    payload = {
        "generated_utc": utc_now_iso(),
        "output": str(output_path),
        "config_name": cli.config_name,
        "overrides": cli.overrides,
        "config": {
            "pair_precision": mode_args.pair_precision,
            "linear_precision": mode_args.linear_precision,
            "tri_impl": mode_args.component_precision.tri_impl,
            "hybrid_precision": OmegaConf.to_container(mode_args.get("hybrid_precision"), resolve=True)
            if mode_args.get("hybrid_precision") is not None
            else None,
        },
        "checkpoint": checkpoint_info,
        "sample": sample_metadata,
        "native_extension": native_build_info,
        "trace": probe.to_dict(),
        "first_nonfinite_event": _first_nonfinite_event(probe.to_dict()["events"]),
    }
    write_json(output_path, payload)
    append_status_report(
        status_path,
        "Activation Probe",
        [
            f"pair_precision={mode_args.pair_precision}",
            f"linear_precision={mode_args.linear_precision}",
            f"tri_impl={mode_args.component_precision.tri_impl}",
            f"sample={sample_metadata['pdb_id']}:{sample_metadata['chain_id']}",
            f"output={output_path}",
            f"first_nonfinite_event={payload['first_nonfinite_event']}",
        ],
    )

    if cli.compare_other is not None:
        reference_payload = json.loads(cli.compare_other.read_text(encoding="utf-8"))
        diff_payload = build_probe_diff(reference_payload, payload)
        write_json(compare_report.with_suffix(".json"), diff_payload)
        compare_report.write_text(render_probe_diff_markdown(diff_payload), encoding="utf-8")

    cleanup_model(model)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
