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

"""Shared configuration, prompt generation, timing helpers, and reporting for benchmarks.

Both benchmark_hf.py and benchmark_vllm.py import from this module so that
the sweep grid, synthetic inputs, metric computation, and output format are
identical -- guaranteeing an apples-to-apples comparison.
"""

import argparse
import csv
import statistics
import time
from dataclasses import dataclass, fields

import torch
from transformers import PreTrainedTokenizerBase


DEFAULT_BATCH_SIZES = [1, 4, 8]
DEFAULT_PROMPT_LENS = [64, 256, 512]
DEFAULT_OUTPUT_LENS = [16, 64, 128]
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 5

STOCK_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Bright vixens jump; dozy fowl quack. "
)


@dataclass
class BenchmarkConfig:
    """Holds all parameters for a benchmark run."""

    model: str
    batch_sizes: list[int]
    prompt_lens: list[int]
    output_lens: list[int]
    warmup: int
    repeats: int
    csv_path: str | None


@dataclass
class BenchmarkResult:
    """One row of benchmark output."""

    batch_size: int
    prompt_len: int
    output_len: int
    e2e_ms: float
    ttft_ms: float
    tpot_ms: float
    throughput_tok_s: float


def build_prompts(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    prompt_length: int,
) -> tuple[list[str], torch.Tensor]:
    """Generate deterministic synthetic prompts of exactly *prompt_length* tokens.

    Returns:
        A tuple of (prompt_strings, input_ids_tensor).
        *prompt_strings* is a list[str] of length *batch_size* (for vLLM).
        *input_ids_tensor* is a (batch_size, prompt_length) int64 tensor (for HF).
        Both represent byte-identical inputs.
    """
    repeated = STOCK_TEXT * ((prompt_length // 10) + 2)
    token_ids = tokenizer.encode(repeated, add_special_tokens=False)[:prompt_length]
    prompt_str = tokenizer.decode(token_ids)

    prompt_strings = [prompt_str] * batch_size
    input_ids = torch.tensor([token_ids] * batch_size, dtype=torch.long)
    return prompt_strings, input_ids


def compute_metrics(
    e2e_seconds: float,
    ttft_seconds: float,
    batch_size: int,
    output_len: int,
) -> BenchmarkResult:
    """Derive TPOT and throughput from raw wall-clock timings."""
    e2e_ms = e2e_seconds * 1000.0
    ttft_ms = ttft_seconds * 1000.0
    tpot_ms = ((e2e_seconds - ttft_seconds) / max(output_len - 1, 1)) * 1000.0
    total_output_tokens = batch_size * output_len
    throughput = total_output_tokens / e2e_seconds if e2e_seconds > 0 else 0.0
    return BenchmarkResult(
        batch_size=batch_size,
        prompt_len=0,
        output_len=output_len,
        e2e_ms=e2e_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        throughput_tok_s=throughput,
    )


def median_timing(fn, repeats: int) -> float:
    """Run *fn* multiple times and return the median wall-clock duration in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


_HEADER = ["batch_size", "prompt_len", "output_len", "e2e_ms", "ttft_ms", "tpot_ms", "throughput_tok_s"]


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print a results table to stdout."""
    col_widths = [max(len(h), 12) for h in _HEADER]
    header_line = "  ".join(h.rjust(w) for h, w in zip(_HEADER, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for r in results:
        vals = [
            str(r.batch_size),
            str(r.prompt_len),
            str(r.output_len),
            f"{r.e2e_ms:.1f}",
            f"{r.ttft_ms:.1f}",
            f"{r.tpot_ms:.2f}",
            f"{r.throughput_tok_s:.1f}",
        ]
        print("  ".join(v.rjust(w) for v, w in zip(vals, col_widths)))


def write_csv(results: list[BenchmarkResult], path: str) -> None:
    """Write results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field.name for field in fields(BenchmarkResult)])
        for r in results:
            writer.writerow(
                [r.batch_size, r.prompt_len, r.output_len, r.e2e_ms, r.ttft_ms, r.tpot_ms, r.throughput_tok_s]
            )
    print(f"Results written to {path}")


def add_common_args(parser: argparse.ArgumentParser, default_model: str) -> None:
    """Register the shared CLI flags on *parser*."""
    parser.add_argument("--model", type=str, default=default_model, help="Model ID or checkpoint path.")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--prompt-lens",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PROMPT_LENS),
        help="Comma-separated prompt lengths (tokens).",
    )
    parser.add_argument(
        "--output-lens",
        type=str,
        default=",".join(str(x) for x in DEFAULT_OUTPUT_LENS),
        help="Comma-separated output lengths (tokens).",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations per grid point.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Timed iterations per grid point.")
    parser.add_argument("--csv", type=str, default=None, dest="csv_path", help="Optional CSV output path.")


def parse_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Convert parsed CLI args into a BenchmarkConfig."""
    return BenchmarkConfig(
        model=args.model,
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        prompt_lens=[int(x) for x in args.prompt_lens.split(",")],
        output_lens=[int(x) for x in args.output_lens.split(",")],
        warmup=args.warmup,
        repeats=args.repeats,
        csv_path=args.csv_path,
    )
