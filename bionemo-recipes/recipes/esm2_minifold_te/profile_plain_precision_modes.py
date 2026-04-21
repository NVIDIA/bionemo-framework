import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf
import torch

from distributed_config import DistributedConfig
from plain_runtime_diagnostics import (
    build_mode_args,
    build_plain_runtime_from_args,
    cleanup_model,
    compose_eval_args,
    destroy_distributed_if_initialized,
    extract_dataset_sample,
    load_state_dict_for_eval,
    summarize_timings,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODE_PRESETS = {
    "bf16": {
        "pair_precision": "bf16",
        "linear_precision": "bf16",
        "tri_impl": "cublas_xbdnn",
    },
    "bf16_native_b4": {
        "pair_precision": "bf16_native",
        "linear_precision": "bf16_native",
        "tri_impl": "cublas_xbdnn",
        "bf16_native_rung": "B4",
    },
    "fp8_extreme": {
        "pair_precision": "fp8_extreme",
        "linear_precision": "fp8",
        "tri_impl": "fp8_cublaslt",
    },
    "fp8_native": {
        "pair_precision": "fp8_native",
        "linear_precision": "fp8",
        "tri_impl": "fp8_cublaslt",
    },
    "fp8_native_gold_packs": {
        "pair_precision": "fp8_native_gold_packs",
        "linear_precision": "fp8",
        "tri_impl": "fp8_cublaslt",
    },
    "fp8_native_mixed_tail_k0": {
        "pair_precision": "fp8_native_mixed_tail",
        "linear_precision": "fp8",
        "tri_impl": "fp8_cublaslt",
        "mixed_tail": {"tail_bf16_native_blocks": 0, "bf16_native_rung": "B3"},
    },
    "fp8_native_mixed_tail_k2": {
        "pair_precision": "fp8_native_mixed_tail",
        "linear_precision": "fp8",
        "tri_impl": "fp8_cublaslt",
        "mixed_tail": {"tail_bf16_native_blocks": 2, "bf16_native_rung": "B3"},
    },
    "fp8_native_mixed_tail_k4": {
        "pair_precision": "fp8_native_mixed_tail",
        "linear_precision": "fp8",
        "tri_impl": "fp8_cublaslt",
        "mixed_tail": {"tail_bf16_native_blocks": 4, "bf16_native_rung": "B3"},
    },
}

def _cuda_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _run_mode_benchmark(model, batch, mode_args, plain_infer, warmup_steps: int, measure_steps: int, device: torch.device):
    model_inputs = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    use_bf16_autocast = mode_args.pair_precision not in (
        plain_infer.PAIR_PRECISION_BF16_NATIVE,
        plain_infer.PAIR_PRECISION_FP8_EXTREME,
        plain_infer.PAIR_PRECISION_FP8_HYBRID,
        plain_infer.PAIR_PRECISION_FP8_NATIVE,
        plain_infer.PAIR_PRECISION_FP8_NATIVE_GOLD_PACKS,
        plain_infer.PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL,
    )

    with torch.no_grad():
        for _ in range(warmup_steps):
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16_autocast else nullcontext()
            with autocast_ctx:
                model(model_inputs, num_recycling=mode_args.model.get("num_recycling", 0))
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    times_ms = []
    with torch.no_grad():
        for _ in range(measure_steps):
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16_autocast else nullcontext()

            def _forward():
                with autocast_ctx:
                    model(model_inputs, num_recycling=mode_args.model.get("num_recycling", 0))

            times_ms.append(_cuda_ms(_forward))

    pair_stats = getattr(model.fold.miniformer, "last_pair_precision_stats", None)
    tokens = int(model_inputs["mask"].sum().item()) if "mask" in model_inputs else 0
    timing_summary = summarize_timings(times_ms)
    timing_summary["tokens_per_sec"] = tokens / (timing_summary["mean_ms"] / 1000.0) if tokens > 0 else 0.0
    timing_summary["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)

    return {
        "mode": mode_args.pair_precision,
        "linear_precision": mode_args.linear_precision,
        "tri_impl": mode_args.component_precision.tri_impl,
        "mixed_tail": OmegaConf.to_container(mode_args.mixed_tail, resolve=True)
        if getattr(mode_args, "mixed_tail", None) is not None
        else None,
        "tokens": tokens,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "timings_ms": times_ms,
        "summary": timing_summary,
        "pair_stats": asdict(pair_stats) if pair_stats is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmark the plain-runtime MiniFold precision modes on the same real checkpoint and sample."
    )
    parser.add_argument("--config-name", default="eval_real_3B_fp8native")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--artifact-root", type=Path, default=Path("/scratch/claude_tasks/accuracy_validation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["fp8_extreme", "fp8_native"],
        choices=sorted(MODE_PRESETS),
        help="Precision modes to benchmark on the same loaded checkpoint and sample.",
    )
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()

    base_args = compose_eval_args(cli.config_name, cli.overrides, artifact_root=cli.artifact_root)
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(dist_config.local_rank)

    state_dict, checkpoint_info = load_state_dict_for_eval(base_args, dist_config, device)
    sample_batch, sample_metadata = extract_dataset_sample(base_args.eval_dataset, cli.sample_index)

    mode_payloads = []
    for mode_name in cli.modes:
        if mode_name not in MODE_PRESETS:
            raise ValueError(f"Unsupported mode {mode_name!r}; expected one of {sorted(MODE_PRESETS)}")
        preset = MODE_PRESETS[mode_name]
        mode_args = build_mode_args(
            base_args,
            pair_precision=preset["pair_precision"],
            linear_precision=preset["linear_precision"],
            tri_impl=preset["tri_impl"],
            bf16_native_rung=preset.get("bf16_native_rung"),
            mixed_tail=preset.get("mixed_tail"),
        )
        logger.info("Benchmarking mode=%s pair_precision=%s tri_impl=%s", mode_name, mode_args.pair_precision, mode_args.component_precision.tri_impl)
        model, plain_infer, native_build_info = build_plain_runtime_from_args(
            mode_args,
            device,
            state_dict,
            cli.artifact_root,
            cli.artifact_root / "status_report.md",
        )
        mode_payload = _run_mode_benchmark(
            model,
            sample_batch,
            mode_args,
            plain_infer,
            cli.warmup_steps,
            cli.measure_steps,
            device,
        )
        mode_payload["native_extension"] = native_build_info
        mode_payloads.append(mode_payload)
        cleanup_model(model)

    payload = {
        "config_name": cli.config_name,
        "overrides": cli.overrides,
        "checkpoint": checkpoint_info,
        "sample": sample_metadata,
        "output": str(cli.output),
        "artifact_root": str(cli.artifact_root),
        "modes": mode_payloads,
    }

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))

    destroy_distributed_if_initialized()


if __name__ == "__main__":
    main()
