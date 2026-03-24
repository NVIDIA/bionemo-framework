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

"""Compare _extra_state buffers between HF and vLLM model initialization paths.

The TE model has 110 _extra_state buffers computed at init time (not saved
in the checkpoint). This script checks whether they differ between
from_pretrained (on CUDA) vs from_config (on meta device, then moved to CUDA).

Usage:
    LOGNAME=user python debug_extra_state.py --checkpoint benchmark_exports/esm2_t36_3B_UR50D
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel


def _collect_extra_state(model: torch.nn.Module) -> dict[str, dict]:
    """Collect _extra_state from all TE modules via get_extra_state()."""
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "get_extra_state"):
            try:
                extra = module.get_extra_state()
                if extra is not None:
                    result[name] = extra
            except Exception as e:
                result[name] = f"ERROR: {e}"
    return result


def _describe_extra_state(extra) -> str:
    """Summarize the content of an _extra_state entry."""
    if isinstance(extra, dict):
        parts = []
        for k, v in extra.items():
            if isinstance(v, torch.Tensor):
                parts.append(f"{k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            else:
                parts.append(f"{k}: {type(v).__name__}")
        return ", ".join(parts)
    if isinstance(extra, bytes):
        return f"bytes[{len(extra)}]"
    return str(type(extra).__name__)


def main() -> None:
    """Compare _extra_state between from_pretrained and from_config paths."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    ckpt_path = Path(args.checkpoint)

    # ---- Path 1: from_pretrained (HF) ----
    print(f"Loading HF model via from_pretrained (dtype={args.dtype}) ...")
    hf_model = AutoModel.from_pretrained(str(ckpt_path), trust_remote_code=True, torch_dtype=dtype)
    hf_model = hf_model.to("cuda").eval()
    hf_extra = _collect_extra_state(hf_model)
    print(f"  Collected _extra_state from {len(hf_extra)} modules")

    del hf_model
    torch.cuda.empty_cache()

    # ---- Path 2: from_config on meta device (vLLM-style) ----
    print("\nLoading model via from_config on meta device (vLLM-style) ...")
    config = AutoConfig.from_pretrained(str(ckpt_path), trust_remote_code=True)

    with torch.device("meta"):
        meta_model = AutoModel.from_config(config, trust_remote_code=True, dtype=dtype)

    # Move parameters from meta to CUDA (mimics vLLM's init_parameters)
    for name, param in meta_model.named_parameters():
        if param.device == torch.device("meta"):
            new_param = torch.nn.Parameter(torch.empty_like(param, device="cuda", dtype=dtype))
            parts = name.split(".")
            mod = meta_model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            setattr(mod, parts[-1], new_param)

    # Load checkpoint weights
    from safetensors.torch import load_file

    safetensor_files = sorted(ckpt_path.glob("*.safetensors"))
    raw_sd = {}
    for sf in safetensor_files:
        raw_sd.update(load_file(str(sf)))
    ckpt_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("lm_head."):
            continue
        ckpt_sd[k.removeprefix("model.")] = v
    meta_model.load_state_dict(ckpt_sd, strict=False)
    meta_model = meta_model.to("cuda").eval()

    meta_extra = _collect_extra_state(meta_model)
    print(f"  Collected _extra_state from {len(meta_extra)} modules")

    del meta_model
    torch.cuda.empty_cache()

    # ---- Compare ----
    all_keys = sorted(set(hf_extra.keys()) | set(meta_extra.keys()))
    print(f"\n--- Comparing _extra_state across {len(all_keys)} modules ---\n")

    mismatches = 0
    for key in all_keys:
        hf_val = hf_extra.get(key)
        meta_val = meta_extra.get(key)

        if hf_val is None:
            print(f"  ONLY IN META: {key}")
            mismatches += 1
            continue
        if meta_val is None:
            print(f"  ONLY IN HF:   {key}")
            mismatches += 1
            continue

        if isinstance(hf_val, str) or isinstance(meta_val, str):
            print(f"  ERROR: {key}: hf={hf_val}, meta={meta_val}")
            mismatches += 1
            continue

        if type(hf_val) is not type(meta_val):
            print(f"  TYPE MISMATCH: {key}: hf={type(hf_val).__name__}, meta={type(meta_val).__name__}")
            mismatches += 1
            continue

        if isinstance(hf_val, dict):
            for subkey in sorted(set(hf_val.keys()) | set(meta_val.keys())):
                hv = hf_val.get(subkey)
                mv = meta_val.get(subkey)
                if hv is None or mv is None:
                    print(f"  MISSING SUBKEY: {key}.{subkey}")
                    mismatches += 1
                elif isinstance(hv, torch.Tensor) and isinstance(mv, torch.Tensor):
                    if hv.shape != mv.shape:
                        print(f"  SHAPE MISMATCH: {key}.{subkey}: hf={tuple(hv.shape)} meta={tuple(mv.shape)}")
                        mismatches += 1
                    elif not torch.equal(hv.cpu().float(), mv.cpu().float()):
                        diff = (hv.cpu().float() - mv.cpu().float()).abs()
                        print(
                            f"  DIFFERS: {key}.{subkey}: max={diff.max().item():.6e} mean={diff.mean().item():.6e} shape={tuple(hv.shape)}"
                        )
                        mismatches += 1
                elif hv != mv:
                    print(f"  DIFFERS: {key}.{subkey}: hf={hv} meta={mv}")
                    mismatches += 1
        elif isinstance(hf_val, bytes):
            if hf_val != meta_val:
                print(f"  DIFFERS: {key}: hf=bytes[{len(hf_val)}] meta=bytes[{len(meta_val)}]")
                mismatches += 1

    if mismatches == 0:
        print("  ALL _extra_state entries MATCH")
    else:
        print(f"\n  {mismatches} _extra_state difference(s) found")

    # ---- Show structure of first _extra_state entry ----
    if hf_extra:
        first_key = next(iter(hf_extra))
        print(f"\nSample _extra_state structure ({first_key}):")
        print(f"  {_describe_extra_state(hf_extra[first_key])}")


if __name__ == "__main__":
    main()
