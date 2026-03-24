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

"""Benchmark TransformerEngine inference on a one-way HF -> TE converted Llama-3 model.

Converts the HuggingFace model to TE format (with THD attention and TE-native
InferenceParams KV cache), then sweeps over a grid of (batch_size,
prompt_length, output_length) and reports end-to-end latency,
time-to-first-token, time-per-output-token, and throughput.

Uses HFCompatibleInferenceParams, a thin wrapper around TE's InferenceParams
that adds the get_seq_length() method required by HuggingFace's generate()
in transformers >= 5.0.

Usage:
    python benchmark_te.py
    python benchmark_te.py --model meta-llama/Llama-3.2-1B-Instruct --csv te_results.csv
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
from convert import convert_llama_hf_to_te
from transformer_engine.pytorch.attention import InferenceParams
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LEN = 4096


class HFCompatibleInferenceParams(InferenceParams):
    """Thin wrapper that adds HuggingFace Cache interface compatibility.

    HF's generate() (transformers >= 5.0) calls cache.get_seq_length()
    during _prefill to determine how many tokens are already cached.
    InferenceParams doesn't implement this method, so we read from TE's
    internal self.sequences OrderedDict (populated by pre_step() during
    each forward pass).
    """

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the cached sequence length from TE's internal tracking."""
        if not self.sequences:
            return 0
        return next(iter(self.sequences.values()))


def _make_inference_params(model, batch_size: int) -> HFCompatibleInferenceParams:
    """Allocate a fresh HF-compatible TE InferenceParams KV cache."""
    config = model.config
    params = HFCompatibleInferenceParams(
        max_batch_size=batch_size,
        max_sequence_length=MAX_SEQ_LEN,
        num_heads_kv=config.num_key_value_heads,
        head_dim_k=config.hidden_size // config.num_attention_heads,
        dtype=torch.bfloat16,
        qkv_format="thd",
        max_ctx_len=MAX_SEQ_LEN,
    )
    for layer_number in range(1, config.num_hidden_layers + 1):
        params.allocate_memory(layer_number)
    return params


def main() -> None:
    """Run the TE benchmark sweep."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_model=DEFAULT_MODEL)
    config = parse_config(parser.parse_args())

    print(f"Loading HF model: {config.model}")
    model_hf = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=torch.bfloat16)

    print("Converting HF -> TE (one-way, THD + InferenceParams KV cache)")
    model = convert_llama_hf_to_te(model_hf, attn_input_format="thd", self_attn_mask_type="padding_causal")
    del model_hf
    model = model.to("cuda").eval()

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
            past_kv = _make_inference_params(model, batch_size)
            with torch.no_grad():
                model.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    use_cache=True,
                    past_key_values=past_kv,
                )
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
