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

"""Compare forward pass outputs between HF and vLLM across a grid of inputs.

Tests single-sequence inputs at various sequence lengths to isolate
whether the embedding mismatch depends on sequence length.

Usage:
    LOGNAME=user python debug_forward.py --checkpoint benchmark_exports/esm2_t36_3B_UR50D
"""

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_common import build_sequences


def _last_token_l2(hidden: torch.Tensor) -> torch.Tensor:
    """Extract last-token hidden state and L2-normalize."""
    vecs = hidden[:, -1, :]
    return vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def main() -> None:
    """Run grid of inputs through HF and vLLM sequentially, compare outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    seq_lens = [32, 64, 128, 256, 512]

    # ---- Phase 1: HF model ----
    print(f"Loading HF model from {args.checkpoint} (dtype={args.dtype}) ...")
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True, torch_dtype=dtype)
    model = model.to("cuda").eval()

    hf_embs = {}
    for seq_len in seq_lens:
        sequences, input_ids = build_sequences(tokenizer, 1, seq_len)
        with torch.no_grad():
            out = model(input_ids=input_ids.to("cuda"), attention_mask=torch.ones_like(input_ids).to("cuda"))
        hf_embs[seq_len] = _last_token_l2(out.last_hidden_state).cpu().float()

    del model
    torch.cuda.empty_cache()
    print("HF embeddings collected, model freed.\n")

    # ---- Phase 2: vLLM engine ----
    print(f"Loading vLLM engine from {args.checkpoint} ...")
    from vllm import LLM

    engine = LLM(
        model=args.checkpoint,
        runner="pooling",
        trust_remote_code=True,
        dtype=args.dtype,
        enforce_eager=True,
        max_num_batched_tokens=1026,
        max_num_seqs=1,
    )

    vllm_embs = {}
    for seq_len in seq_lens:
        sequences, _ = build_sequences(tokenizer, 1, seq_len)
        outputs = engine.embed(sequences)
        vllm_embs[seq_len] = torch.tensor(outputs[0].outputs.embedding).unsqueeze(0).float()

    del engine
    torch.cuda.empty_cache()
    print("vLLM embeddings collected, engine freed.\n")

    # ---- Phase 3: Compare ----
    print("--- HF vs vLLM engine (single sequence, by seq_len) ---")
    print(f"{'seq_len':>8} {'cosine':>12} {'max_diff':>12} {'mean_diff':>12}")
    print("-" * 50)

    for seq_len in seq_lens:
        h = hf_embs[seq_len]
        v = vllm_embs[seq_len]
        cos = torch.nn.functional.cosine_similarity(h, v, dim=-1).item()
        max_diff = (h - v).abs().max().item()
        mean_diff = (h - v).abs().mean().item()
        print(f"{seq_len:>8} {cos:>12.8f} {max_diff:>12.6e} {mean_diff:>12.6e}")


if __name__ == "__main__":
    main()
