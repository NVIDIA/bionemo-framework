#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from omegaconf import OmegaConf

from plain_minifold_infer import FoldingTrunk


DEFAULT_CONFIG = Path(__file__).resolve().parent / "hydra_config" / "inference_single_config.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple single-config non-TE MiniFold folding inference benchmark.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--mbs", type=int)
    parser.add_argument("--tri_impl", choices=["bmm", "cublas_xbdnn"])
    parser.add_argument("--tri_einsum", choices=["off", "bf16"])
    parser.add_argument("--fp8_activations", action="store_true")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_blocks", type=int)
    parser.add_argument("--c_s", type=int)
    parser.add_argument("--c_z", type=int)
    parser.add_argument("--no_bins", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--num_recycling", type=int)
    parser.add_argument("--output", default="")
    return parser


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = OmegaConf.load(args.config)
    merged = {
        "seq_len": cfg.seq_len,
        "mbs": cfg.mbs,
        "tri_impl": cfg.tri_impl,
        "tri_einsum": cfg.tri_einsum,
        "fp8_activations": bool(cfg.fp8_activations),
        "warmup": cfg.warmup,
        "iters": cfg.iters,
        "seed": cfg.seed,
        "num_blocks": cfg.num_blocks,
        "c_s": cfg.c_s,
        "c_z": cfg.c_z,
        "no_bins": cfg.no_bins,
        "bins": cfg.bins,
        "num_recycling": cfg.num_recycling,
        "output": cfg.output,
        "config": args.config,
    }
    for key in merged:
        value = getattr(args, key, None)
        if value not in (None, ""):
            merged[key] = value
    return argparse.Namespace(**merged)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, device: torch.device) -> FoldingTrunk:
    model = FoldingTrunk(
        c_s=args.c_s,
        c_z=args.c_z,
        bins=args.bins,
        disto_bins=args.no_bins,
        num_layers=args.num_blocks,
        tri_impl=args.tri_impl,
        tri_einsum=args.tri_einsum,
        fp8_activations=args.fp8_activations,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def generate_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "s_s": torch.randn(args.mbs, args.seq_len, args.c_s, device=device, dtype=torch.bfloat16),
        "s_z": torch.randn(args.mbs, args.seq_len, args.seq_len, args.c_z, device=device, dtype=torch.bfloat16),
        "mask": torch.ones(args.mbs, args.seq_len, device=device, dtype=torch.bool),
    }


def forward_once(model: FoldingTrunk, batch: dict[str, torch.Tensor], num_recycling: int) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(batch["s_s"], batch["s_z"], batch["mask"], num_recycling=num_recycling)


def benchmark(model: FoldingTrunk, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):
        forward_once(model, batch, args.num_recycling)
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times_ms: list[float] = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward_once(model, batch, args.num_recycling)
        end.record()
        torch.cuda.synchronize(device)
        times_ms.append(float(start.elapsed_time(end)))

    median_ms = float(statistics.median(times_ms))
    result = {
        "median_ms": median_ms,
        "mean_ms": float(statistics.mean(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "proteins_per_sec": args.mbs / (median_ms / 1000.0),
        "unpadded_tokens_per_sec": (args.mbs * args.seq_len) / (median_ms / 1000.0),
        "peak_memory_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "peak_memory_reserved_gib": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
    }
    if model.miniformer.last_fp8_stats is not None:
        result["fp8_activation_storage_mib"] = model.miniformer.last_fp8_stats.quantized_bytes / (1024**2)
        result["fp8_scale_storage_mib"] = model.miniformer.last_fp8_stats.scale_bytes / (1024**2)
    return result


def render_markdown(args: argparse.Namespace, result: dict[str, float]) -> str:
    notes = "fp8_act" if args.fp8_activations else ""
    return "\n".join(
        [
            "| tri_impl | seq_len | mbs | median (ms) | prot/s | tok/s | peak mem (GiB) | notes |",
            "|----------|---------|-----|-------------|--------|-------|----------------|-------|",
            (
                f"| {args.tri_impl} | {args.seq_len} | {args.mbs} | {result['median_ms']:.2f} | "
                f"{result['proteins_per_sec']:.2f} | {result['unpadded_tokens_per_sec']:.2f} | "
                f"{result['peak_memory_allocated_gib']:.2f} | {notes} |"
            ),
        ]
    )


def main() -> None:
    raw_args = build_arg_parser().parse_args()
    args = load_config(raw_args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference_single_config.py")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    model = build_model(args, device)
    batch = generate_inputs(args, device)
    result = benchmark(model, batch, args, device)
    markdown = render_markdown(args, result)
    payload = {
        "config": {
            "config_file": args.config,
            "seq_len": args.seq_len,
            "mbs": args.mbs,
            "tri_impl": args.tri_impl,
            "tri_einsum": args.tri_einsum,
            "fp8_activations": args.fp8_activations,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "num_blocks": args.num_blocks,
            "c_s": args.c_s,
            "c_z": args.c_z,
            "no_bins": args.no_bins,
            "bins": args.bins,
            "num_recycling": args.num_recycling,
            "implementation": "plain_pytorch_single_config",
            "mode": "mock_backbone_only",
        },
        "result": result,
        "markdown_summary": markdown,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        out_path.with_suffix(".md").write_text(markdown + "\n")
        print(markdown)
        print(f"\nJSON: {out_path}")
        print(f"Markdown: {out_path.with_suffix('.md')}")
    else:
        print(json.dumps(payload, indent=2))
        print()
        print(markdown)


if __name__ == "__main__":
    main()
