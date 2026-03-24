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

"""Compare HF-loaded model weights against raw checkpoint weights.

Loads the checkpoint via AutoModel.from_pretrained (HF path), then loads
the raw safetensors file directly, and compares every weight to find
discrepancies that would explain embedding mismatches.

Usage:
    python debug_weights.py --checkpoint benchmark_exports/esm2_t36_3B_UR50D
    python debug_weights.py --checkpoint benchmark_exports/esm2_t48_15B_UR50D --dtype float32
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModel


def main() -> None:
    """Load model via HF and compare against raw checkpoint weights."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to exported TE checkpoint.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    ckpt_path = Path(args.checkpoint)

    print(f"Loading HF model from {ckpt_path} (dtype={args.dtype}) ...")
    hf_model = AutoModel.from_pretrained(str(ckpt_path), trust_remote_code=True, torch_dtype=dtype)
    hf_model = hf_model.to("cuda").eval()

    hf_sd = {}
    for k, v in hf_model.state_dict().items():
        hf_sd[k] = v.cpu().float()
    print(f"HF state_dict: {len(hf_sd)} keys")

    del hf_model
    torch.cuda.empty_cache()

    print(f"\nLoading raw checkpoint from {ckpt_path} ...")
    safetensor_files = sorted(ckpt_path.glob("*.safetensors"))
    raw_sd = {}
    for sf in safetensor_files:
        raw_sd.update(load_file(str(sf)))

    # The checkpoint is saved from NVEsmForMaskedLM which wraps NVEsmModel
    # as self.model, so keys have a "model." prefix. HF's from_pretrained
    # strips this automatically. Do the same here for comparison, and also
    # skip lm_head keys which don't exist in the base model.
    ckpt_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("lm_head."):
            continue
        clean = k.removeprefix("model.")
        ckpt_sd[clean] = v.float()
    print(f"Raw checkpoint (after prefix strip): {len(ckpt_sd)} keys")

    ckpt_keys = set(ckpt_sd.keys())
    hf_keys = set(hf_sd.keys())

    only_ckpt = ckpt_keys - hf_keys
    only_hf = hf_keys - ckpt_keys
    common = ckpt_keys & hf_keys

    if only_ckpt:
        print(f"\nKeys only in checkpoint ({len(only_ckpt)}):")
        for k in sorted(only_ckpt):
            print(f"  {k}  shape={tuple(ckpt_sd[k].shape)}")

    if only_hf:
        print(f"\nKeys only in HF model ({len(only_hf)}):")
        for k in sorted(only_hf):
            print(f"  {k}  shape={tuple(hf_sd[k].shape)}")

    print(f"\nComparing {len(common)} common keys: HF model vs checkpoint ...")
    mismatches = []
    for k in sorted(common):
        h = hf_sd[k]
        r = ckpt_sd[k]
        if h.shape != r.shape:
            print(f"  SHAPE MISMATCH  {k}: HF={tuple(h.shape)} ckpt={tuple(r.shape)}")
            mismatches.append(k)
            continue
        if not torch.equal(h, r):
            max_diff = (h - r).abs().max().item()
            mean_diff = (h - r).abs().mean().item()
            print(f"  DIFFERS  {k}: max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  shape={tuple(h.shape)}")
            mismatches.append(k)

    if not mismatches:
        print("  ALL weights MATCH")
    else:
        print(f"  {len(mismatches)} weight(s) differ")

    print("\n" + "=" * 60)
    print("Simulating vLLM's model construction (from_config + load_state_dict) ...")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(ckpt_path), trust_remote_code=True)
    vllm_model = AutoModel.from_config(config, trust_remote_code=True, dtype=dtype)
    vllm_model = vllm_model.to("cuda").eval()

    # Load using the prefix-stripped checkpoint (matching from_config model keys)
    missing, unexpected = vllm_model.load_state_dict(ckpt_sd, strict=False)
    if missing:
        print(f"\n  Missing keys ({len(missing)}):")
        for k in sorted(missing):
            print(f"    {k}")
    if unexpected:
        print(f"\n  Unexpected keys ({len(unexpected)}):")
        for k in sorted(unexpected):
            print(f"    {k}")

    vllm_sd = {k: v.cpu().float() for k, v in vllm_model.state_dict().items()}
    print(f"\nfrom_config state_dict: {len(vllm_sd)} keys")

    common2 = set(hf_sd.keys()) & set(vllm_sd.keys())
    mismatches2 = []
    for k in sorted(common2):
        h = hf_sd[k]
        v = vllm_sd[k]
        if h.shape != v.shape:
            print(f"  SHAPE MISMATCH  {k}: HF={tuple(h.shape)} from_config={tuple(v.shape)}")
            mismatches2.append(k)
            continue
        if not torch.equal(h, v):
            max_diff = (h - v).abs().max().item()
            mean_diff = (h - v).abs().mean().item()
            print(f"  DIFFERS  {k}: max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  shape={tuple(h.shape)}")
            mismatches2.append(k)

    if not mismatches2:
        print("\n  ALL weights MATCH between from_pretrained and from_config+load_state_dict")
    else:
        print(f"\n  {len(mismatches2)} weight(s) differ")

    del vllm_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
