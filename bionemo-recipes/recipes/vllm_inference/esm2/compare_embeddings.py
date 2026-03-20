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

"""Compare embedding tensors saved by benchmark_hf.py and benchmark_vllm.py.

Loads two .pt files produced with --embeddings-path and reports numerical
differences, cosine similarity, and a pass/fail verdict against a tolerance.

Usage:
    python compare_embeddings.py --hf hf_embeddings.pt --vllm vllm_embeddings.pt
    python compare_embeddings.py --hf hf_embeddings.pt --vllm vllm_embeddings.pt --atol 1e-3
"""

import argparse
import sys

import torch


def main() -> None:
    """Load two embedding .pt files and report numerical differences."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf", type=str, required=True, help="Path to HF embeddings .pt file.")
    parser.add_argument("--vllm", type=str, required=True, help="Path to vLLM embeddings .pt file.")
    parser.add_argument("--atol", type=float, default=2e-4, help="Absolute tolerance for allclose (default: 2e-4).")
    args = parser.parse_args()

    hf = torch.load(args.hf, map_location="cpu", weights_only=True).float()
    vllm = torch.load(args.vllm, map_location="cpu", weights_only=True).float()

    print(f"HF   shape: {tuple(hf.shape)}  dtype: {hf.dtype}")
    print(f"vLLM shape: {tuple(vllm.shape)}  dtype: {vllm.dtype}")

    if hf.shape != vllm.shape:
        print(f"\nFAIL: shape mismatch  HF={tuple(hf.shape)}  vLLM={tuple(vllm.shape)}")
        sys.exit(1)

    abs_diff = (hf - vllm).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"\nMax  absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    cosine = torch.nn.functional.cosine_similarity(hf, vllm, dim=-1)
    print("\nPer-sequence cosine similarity:")
    for i, c in enumerate(cosine):
        print(f"  seq {i}: {c.item():.8f}")
    print(f"  mean:  {cosine.mean().item():.8f}")

    passed = torch.allclose(hf, vllm, atol=args.atol, rtol=0)
    status = "PASS" if passed else "FAIL"
    print(f"\nallclose(atol={args.atol}, rtol=0): {status}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
