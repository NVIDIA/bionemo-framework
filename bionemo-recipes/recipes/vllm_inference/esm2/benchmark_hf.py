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

"""Benchmark HuggingFace transformers inference on ESM2 models.

Supports two variants:
  --variant stock   Load the original facebook model (path 1).
  --variant te      Fresh-export facebook -> TE, then load via transformers (path 2).

Sweeps over a grid of (batch_size, seq_len) and reports end-to-end latency
and throughput.

Usage:
    python benchmark_hf.py --tag 8m --variant stock
    python benchmark_hf.py --tag 8m --variant te
    python benchmark_hf.py --tag 3b --variant te --csv te_3b.csv
"""

import argparse
import itertools
import statistics

import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_common import (
    add_common_args,
    build_sequences,
    compute_metrics,
    ensure_exported,
    median_timing,
    parse_config,
    print_results,
    resolve_dtype,
    sample_gpu_metrics,
    write_csv,
)


def main() -> None:
    """Run the HF benchmark sweep."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument(
        "--variant",
        type=str,
        default="stock",
        choices=["stock", "te"],
        help="'stock' = facebook model, 'te' = freshly exported TE model.",
    )
    args = parser.parse_args()
    config = parse_config(args)
    variant = args.variant
    dtype = resolve_dtype(config.dtype)

    if variant == "stock":
        model_id = f"facebook/{config.tag}"
        trust_remote_code = False
    else:
        model_id = ensure_exported(config.tag, config.export_dir, config.force_export)
        trust_remote_code = True

    print(f"Loading model: {model_id}  (variant={variant}, dtype={config.dtype})")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code, torch_dtype=dtype)
    model = model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    results = []
    grid = list(itertools.product(config.batch_sizes, config.seq_lens))

    for batch_size, seq_len in grid:
        label = f"batch={batch_size}  seq_len={seq_len}"
        print(f"\n[{label}]")

        _, input_ids = build_sequences(tokenizer, batch_size, seq_len)
        inputs = {"input_ids": input_ids.to("cuda"), "attention_mask": torch.ones_like(input_ids).to("cuda")}

        gpu_samples: list[tuple[float, float]] = []

        def _forward() -> None:
            with torch.no_grad():
                model(**inputs)
            torch.cuda.synchronize()

        def _forward_with_metrics() -> None:
            _forward()
            gpu_samples.append(sample_gpu_metrics())

        for _ in range(config.warmup):
            _forward()

        e2e_s = median_timing(_forward_with_metrics, config.repeats)
        avg_mem = statistics.mean(s[0] for s in gpu_samples)
        avg_util = statistics.mean(s[1] for s in gpu_samples)

        result = compute_metrics(e2e_s, batch_size, seq_len, avg_mem, avg_util)
        results.append(result)
        print(
            f"  e2e={result.e2e_ms:.1f}ms  "
            f"throughput={result.throughput_tok_s:.1f} tok/s  "
            f"{result.throughput_seq_s:.1f} seq/s  "
            f"gpu_mem={result.gpu_mem_mb:.0f}MB  "
            f"gpu_util={result.gpu_util_pct:.1f}%"
        )

    print("\n" + "=" * 60)
    print_results(results)
    if config.csv_path:
        write_csv(results, config.csv_path)

    if config.embeddings_path:
        print("\nCollecting sample embeddings ...")
        _, input_ids = build_sequences(tokenizer, config.batch_sizes[0], config.seq_lens[0])
        inputs = {"input_ids": input_ids.to("cuda"), "attention_mask": torch.ones_like(input_ids).to("cuda")}
        with torch.no_grad():
            out = model(**inputs)
        hidden = out.last_hidden_state
        vecs = hidden[:, -1, :]
        vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        torch.save(vecs.cpu(), config.embeddings_path)
        print(f"Embeddings saved to {config.embeddings_path}  shape={tuple(vecs.shape)}")


if __name__ == "__main__":
    main()
