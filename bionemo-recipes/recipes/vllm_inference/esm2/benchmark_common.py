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

"""Shared configuration, sequence generation, timing helpers, and reporting for ESM2 benchmarks.

Both benchmark_hf.py and benchmark_vllm.py import from this module so that
the sweep grid, synthetic inputs, metric computation, and output format are
identical -- guaranteeing an apples-to-apples comparison.

ESM2 is an encoder-only (masked LM) model, so there is no autoregressive
generation.  The benchmark measures single-forward-pass latency and throughput
over a (batch_size x seq_len) grid.
"""

import argparse
import csv
import random
import statistics
import time
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase


MODEL_TAGS: dict[str, str] = {
    "8m": "esm2_t6_8M_UR50D",
    "3b": "esm2_t36_3B_UR50D",
    "15b": "esm2_t48_15B_UR50D",
}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
DEFAULT_SEQ_LENS = [64, 128, 256, 512, 1024]
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 5


@dataclass
class BenchmarkConfig:
    """Holds all parameters for a benchmark run."""

    tag: str
    batch_sizes: list[int]
    seq_lens: list[int]
    warmup: int
    repeats: int
    csv_path: str | None
    dtype: str
    export_dir: str
    force_export: bool


@dataclass
class BenchmarkResult:
    """One row of benchmark output."""

    batch_size: int
    seq_len: int
    e2e_ms: float
    throughput_tok_s: float
    throughput_seq_s: float


def ensure_exported(tag: str, export_dir: str, force: bool = False) -> str:
    """Export a facebook checkpoint to TE format if not already cached.

    Args:
        tag: The full ESM2 tag (e.g. ``esm2_t6_8M_UR50D``).
        export_dir: Parent directory for exported checkpoints.
        force: Re-export even if the directory already exists.

    Returns:
        Path to the exported TE checkpoint directory.
    """
    export_path = Path(export_dir)
    checkpoint_path = export_path / tag

    if checkpoint_path.exists() and not force:
        print(f"Using cached export: {checkpoint_path}")
        return str(checkpoint_path)

    print(f"Exporting {tag} from facebook/{tag} ...")
    export_path.mkdir(parents=True, exist_ok=True)

    from export import export_hf_checkpoint

    export_hf_checkpoint(tag, export_path)
    print(f"Export complete: {checkpoint_path}")
    return str(checkpoint_path)


def build_sequences(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    seq_len: int,
    seed: int = 42,
) -> tuple[list[str], torch.Tensor]:
    """Generate deterministic synthetic protein sequences targeting *seq_len* tokens.

    The ESM2 tokenizer adds <cls> and <eos> tokens, so the amino-acid string
    is ``seq_len - 2`` characters long to yield exactly *seq_len* tokens.

    Returns:
        A tuple of (sequence_strings, input_ids_tensor).
        *sequence_strings* is a list[str] of length *batch_size* (for vLLM).
        *input_ids_tensor* is a (batch_size, seq_len) int64 tensor (for HF).
    """
    rng = random.Random(seed)
    aa_len = max(seq_len - 2, 1)
    seq_str = "".join(rng.choices(AMINO_ACIDS, k=aa_len))

    encoded = tokenizer(seq_str, return_tensors="pt", padding=False, truncation=False)
    actual_len = encoded["input_ids"].shape[1]
    if actual_len != seq_len:
        seq_str = seq_str[: seq_str.__len__() - (actual_len - seq_len)]
        encoded = tokenizer(seq_str, return_tensors="pt", padding=False, truncation=False)

    sequence_strings = [seq_str] * batch_size
    input_ids = encoded["input_ids"].repeat(batch_size, 1)
    return sequence_strings, input_ids


def compute_metrics(
    e2e_seconds: float,
    batch_size: int,
    seq_len: int,
) -> BenchmarkResult:
    """Derive throughput from raw wall-clock timing."""
    e2e_ms = e2e_seconds * 1000.0
    total_tokens = batch_size * seq_len
    throughput_tok = total_tokens / e2e_seconds if e2e_seconds > 0 else 0.0
    throughput_seq = batch_size / e2e_seconds if e2e_seconds > 0 else 0.0
    return BenchmarkResult(
        batch_size=batch_size,
        seq_len=seq_len,
        e2e_ms=e2e_ms,
        throughput_tok_s=throughput_tok,
        throughput_seq_s=throughput_seq,
    )


def median_timing(fn, repeats: int) -> float:
    """Run *fn* multiple times and return the median wall-clock duration in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


_HEADER = ["batch_size", "seq_len", "e2e_ms", "throughput_tok_s", "throughput_seq_s"]


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print a results table to stdout."""
    col_widths = [max(len(h), 14) for h in _HEADER]
    header_line = "  ".join(h.rjust(w) for h, w in zip(_HEADER, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for r in results:
        vals = [
            str(r.batch_size),
            str(r.seq_len),
            f"{r.e2e_ms:.1f}",
            f"{r.throughput_tok_s:.1f}",
            f"{r.throughput_seq_s:.1f}",
        ]
        print("  ".join(v.rjust(w) for v, w in zip(vals, col_widths)))


def write_csv(results: list[BenchmarkResult], path: str) -> None:
    """Write results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field.name for field in fields(BenchmarkResult)])
        for r in results:
            writer.writerow([r.batch_size, r.seq_len, r.e2e_ms, r.throughput_tok_s, r.throughput_seq_s])
    print(f"Results written to {path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared CLI flags on *parser*."""
    parser.add_argument(
        "--tag",
        type=str,
        default="8m",
        choices=list(MODEL_TAGS.keys()),
        help="Model size tag (default: 8m).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEQ_LENS),
        help="Comma-separated sequence lengths (tokens).",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations per grid point.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Timed iterations per grid point.")
    parser.add_argument("--csv", type=str, default=None, dest="csv_path", help="Optional CSV output path.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype (default: bfloat16).")
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./benchmark_exports",
        help="Directory to cache exported TE checkpoints.",
    )
    parser.add_argument("--force-export", action="store_true", help="Re-export even if cached checkpoint exists.")


def parse_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Convert parsed CLI args into a BenchmarkConfig."""
    return BenchmarkConfig(
        tag=MODEL_TAGS[args.tag],
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        seq_lens=[int(x) for x in args.seq_lens.split(",")],
        warmup=args.warmup,
        repeats=args.repeats,
        csv_path=args.csv_path,
        dtype=args.dtype,
        export_dir=args.export_dir,
        force_export=args.force_export,
    )


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a CLI dtype string to a torch.dtype."""
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]
