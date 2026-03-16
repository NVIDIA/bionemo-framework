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

"""Benchmark vLLM pooling inference on a freshly exported ESM2 TE checkpoint.

The checkpoint is produced by export.py (facebook HF -> TE conversion).

Sweeps over a grid of (batch_size, seq_len) and reports end-to-end latency
and throughput.

Usage:
    python benchmark_vllm.py --tag 8m
    python benchmark_vllm.py --tag 3b --csv vllm_3b.csv
"""

import argparse
import itertools

from transformers import AutoTokenizer
from vllm import LLM

from benchmark_common import (
    add_common_args,
    build_sequences,
    compute_metrics,
    ensure_exported,
    median_timing,
    parse_config,
    print_results,
    write_csv,
)


def main() -> None:
    """Run the vLLM benchmark sweep."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    config = parse_config(parser.parse_args())

    checkpoint = ensure_exported(config.tag, config.export_dir, config.force_export)

    max_batched = max(config.batch_sizes) * max(config.seq_lens)

    print(f"Loading model: {checkpoint}  (dtype={config.dtype})")
    engine = LLM(
        model=checkpoint,
        runner="pooling",
        trust_remote_code=True,
        dtype=config.dtype,
        enforce_eager=True,
        max_num_batched_tokens=max_batched + 2,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    results = []
    grid = list(itertools.product(config.batch_sizes, config.seq_lens))

    for batch_size, seq_len in grid:
        label = f"batch={batch_size}  seq_len={seq_len}"
        print(f"\n[{label}]")

        sequences, _ = build_sequences(tokenizer, batch_size, seq_len)

        def _embed() -> None:
            engine.embed(sequences)

        for _ in range(config.warmup):
            _embed()

        e2e_s = median_timing(_embed, config.repeats)

        result = compute_metrics(e2e_s, batch_size, seq_len)
        results.append(result)
        print(
            f"  e2e={result.e2e_ms:.1f}ms  "
            f"throughput={result.throughput_tok_s:.1f} tok/s  "
            f"{result.throughput_seq_s:.1f} seq/s"
        )

    print("\n" + "=" * 60)
    print_results(results)
    if config.csv_path:
        write_csv(results, config.csv_path)


if __name__ == "__main__":
    main()
