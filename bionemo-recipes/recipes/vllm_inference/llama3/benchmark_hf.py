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

"""Benchmark HuggingFace native inference on the reference Llama-3 model.

Sweeps over a grid of (batch_size, prompt_length, output_length) and reports
end-to-end latency, time-to-first-token, time-per-output-token, and throughput.

Usage:
    python benchmark_hf.py
    python benchmark_hf.py --model meta-llama/Llama-3.2-1B-Instruct --csv hf_results.csv
"""

import argparse
import itertools

import torch
from benchmark_common import (
    add_common_args,
    build_prompts,
    compute_metrics,
    median_timing,
    parse_config,
    print_results,
    write_csv,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main() -> None:
    """Run the HF benchmark sweep."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_model=DEFAULT_MODEL)
    config = parse_config(parser.parse_args())

    print(f"Loading model: {config.model}")
    model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    grid = list(itertools.product(config.batch_sizes, config.prompt_lens, config.output_lens))

    for batch_size, prompt_len, output_len in grid:
        label = f"batch={batch_size}  prompt={prompt_len}  output={output_len}"
        print(f"\n[{label}]")

        _, input_ids = build_prompts(tokenizer, batch_size, prompt_len)
        input_ids = input_ids.to("cuda")

        def _generate(max_new: int) -> None:
            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=max_new, do_sample=False, use_cache=True)
            torch.cuda.synchronize()

        for _ in range(config.warmup):
            _generate(output_len)

        ttft_s = median_timing(lambda: _generate(1), config.repeats)
        e2e_s = median_timing(lambda: _generate(output_len), config.repeats)

        result = compute_metrics(e2e_s, ttft_s, batch_size, output_len)
        result.prompt_len = prompt_len
        results.append(result)
        print(
            f"  e2e={result.e2e_ms:.1f}ms  ttft={result.ttft_ms:.1f}ms  "
            f"tpot={result.tpot_ms:.2f}ms  throughput={result.throughput_tok_s:.1f} tok/s"
        )

    print("\n" + "=" * 60)
    print_results(results)
    if config.csv_path:
        write_csv(results, config.csv_path)


if __name__ == "__main__":
    main()
