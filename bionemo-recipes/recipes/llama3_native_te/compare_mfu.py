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

"""Single-GPU MFU comparison: TE vs HF head-to-head.

Compares FLOPs counting methods and measures MFU for TE and HF models on a single GPU.
No distributed setup required.

Usage:
    cd bionemo-recipes/recipes/llama3_native_te
    python compare_mfu.py
    python compare_mfu.py --seq-len 2048 --batch-size 2
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from compare_mfu_common import (
    cleanup_model,
    compute_flops_first_principles,
    compute_flops_readme,
    count_flops_with_model,
    create_te_model_on_gpu,
    detect_gpu_peak_tflops,
    format_flops,
    measure_step_time,
    print_breakdown,
)
from modeling_llama_te import NVLlamaConfig


def main():
    """Run single-GPU MFU comparison: TE vs HF."""
    parser = argparse.ArgumentParser(description="Single-GPU MFU comparison: TE vs HF")
    parser.add_argument("--config-path", default="./model_configs/lingua-1B", help="Model config directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--peak-tflops", type=float, default=None, help="Override GPU peak bf16 TFLOPS")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup iterations before timing")
    parser.add_argument("--timed-steps", type=int, default=20, help="Timed iterations to average")
    args = parser.parse_args()

    # --- Load model config ---
    config_path = Path(args.config_path) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    b = args.batch_size
    s = args.seq_len
    h = config_dict["hidden_size"]
    num_layers = config_dict["num_hidden_layers"]
    vocab_size = config_dict["vocab_size"]
    n_kv_heads = config_dict["num_key_value_heads"]
    n_heads = config_dict["num_attention_heads"]
    head_dim = h // n_heads
    ffn_hidden_size = config_dict["intermediate_size"]

    # --- GPU detection ---
    if args.peak_tflops:
        peak_tflops = args.peak_tflops
        device_name = torch.cuda.get_device_name(0)
    else:
        peak_tflops, device_name = detect_gpu_peak_tflops()
        if peak_tflops is None:
            print(f"ERROR: Could not auto-detect GPU peak TFLOPS for: {device_name}")
            print("Use --peak-tflops to specify manually.")
            sys.exit(1)

    peak_flops_per_sec = peak_tflops * 1e12

    print(f"GPU: {device_name} (Peak: {peak_tflops:.1f} TFLOPS bf16)")
    print(
        f"Config: H={h}, L={num_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads},"
        f" head_dim={head_dim}, I={ffn_hidden_size}, V={vocab_size}"
    )
    print(f"Batch: B={b}, S={s}")
    print()

    # =========================================================================
    # Table 1: FLOPs Counting
    # =========================================================================
    total_flops_readme = compute_flops_readme(b, s, h, num_layers, vocab_size)
    total_flops_fp, breakdown, lm_head_fwd = compute_flops_first_principles(
        b, s, h, num_layers, n_kv_heads, head_dim, ffn_hidden_size, vocab_size
    )

    print("Counting FLOPs with HF model (meta device)...")
    hf_config = LlamaConfig.from_pretrained(args.config_path)
    hf_config._attn_implementation = "eager"
    with torch.device("meta"):
        hf_model_meta = LlamaForCausalLM(hf_config)
    meta_input_ids = torch.randint(0, vocab_size, (b, s), device="meta")
    total_flops_hf_counter = count_flops_with_model(hf_model_meta, meta_input_ids)
    del hf_model_meta
    print(f"  HF FlopCounter: {format_flops(total_flops_hf_counter)} (training)")

    # =========================================================================
    # Table 2: MFU — TE vs HF
    # =========================================================================
    input_ids = torch.randint(0, vocab_size, (b, s), device="cuda")

    # --- TE model ---
    print(f"\n[1/2] TE model (S={s})...")
    te_config = NVLlamaConfig.from_pretrained(
        args.config_path, dtype=torch.bfloat16, attn_input_format="bshd", self_attn_mask_type="causal"
    )
    te_model = create_te_model_on_gpu(te_config)
    te_model.train()
    print(f"Measuring TE step time ({args.warmup_steps} warmup + {args.timed_steps} timed)...")
    te_step_time = measure_step_time(te_model, input_ids, args.warmup_steps, args.timed_steps)
    model_params = sum(p.numel() for p in te_model.parameters())
    print(f"  TE step time: {te_step_time:.4f}s")
    cleanup_model(te_model)

    # --- HF model ---
    print(f"[2/2] HF model (S={s})...")
    hf_config_gpu = LlamaConfig.from_pretrained(args.config_path)
    hf_model = LlamaForCausalLM(hf_config_gpu).to(dtype=torch.bfloat16, device="cuda")
    hf_model.train()
    print(f"Measuring HF step time ({args.warmup_steps} warmup + {args.timed_steps} timed)...")
    hf_step_time = measure_step_time(hf_model, input_ids, args.warmup_steps, args.timed_steps)
    print(f"  HF step time: {hf_step_time:.4f}s")
    cleanup_model(hf_model)

    # =========================================================================
    # Print results
    # =========================================================================
    print()
    print("=" * 75)
    print(f"MFU Comparison: Lingua-1B (B={b}, S={s}, bf16)")
    print(f"GPU: {device_name} (Peak: {peak_tflops:.1f} TFLOPS bf16)")
    print("=" * 75)

    # --- Table 1 ---
    print()
    print("--- Table 1: FLOPs Counting (per training step) ---")
    hdr1 = f"{'Method':<24} {'FLOPs/step':>14}"
    print(hdr1)
    print("-" * len(hdr1))
    for name, flops in [
        ("README Formula", total_flops_readme),
        ("First Principles", total_flops_fp),
        ("FlopCounter (HF)", total_flops_hf_counter),
    ]:
        print(f"{name:<24} {format_flops(flops):>14}")

    # --- Table 2 ---
    print()
    print("--- Table 2: MFU ---")
    hdr2 = f"{'Model':<12} {'FLOPs/step':>14} {'Step (s)':>9} {'TFLOPS/s':>9} {'MFU':>7}"
    print(hdr2)
    print("-" * len(hdr2))

    for name, flops, step_time in [
        ("TE", total_flops_fp, te_step_time),
        ("HF", total_flops_fp, hf_step_time),
    ]:
        tflops = flops / step_time / 1e12
        mfu = flops / step_time / peak_flops_per_sec * 100
        print(f"{name:<12} {format_flops(flops):>14} {step_time:>8.3f}s {tflops:>8.2f} {mfu:>6.1f}%")

    print()
    print(f"TE vs HF speedup: {hf_step_time / te_step_time:.2f}x")

    # --- Breakdown ---
    print_breakdown(breakdown, lm_head_fwd, num_layers, total_flops_fp, model_params)


if __name__ == "__main__":
    main()
