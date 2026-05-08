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

"""FSDP2 quantized model init — memory profiler version.

Stripped-down version of the TE fully_shard.py example with torch memory
profiling at each stage.  Designed to run on a small model so we can
clearly see the memory impact of quantized_model_init and share results
with the TE team.

Profiles:
  1. Meta-device model creation (zero GPU memory)
  2. FSDP2 sharding (still zero GPU memory)
  3. Parameter materialization (reset_parameters)
  4. Optimizer creation + master weight seeding
  5. Three training steps (forward + backward + optimizer.step)

Dumps a .pickle snapshot after 3 training steps that can be visualized at
https://pytorch.org/memory_viz

Usage::

    torchrun --nproc-per-node 2 fully_shard_memory_profile.py [--no-qinit] [--no-hpiv] [--snapshot-dir /tmp/snapshots]
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as f
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


# ── Configuration ────────────────────────────────────────────────────
HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
NUM_LAYERS = 3
SEQ_LEN = 32
BATCH_PER_RANK = 2
NUM_STEPS = 5
DTYPE = torch.float32


def dist_print(msg):
    """Print only on rank 0."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg)


def log_memory(tag: str):
    """Log GPU memory stats on rank 0."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[Memory: {tag}] allocated={alloc:.4f} GB, reserved={reserved:.4f} GB, peak={peak:.4f} GB")


def main():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-qinit", action="store_true", help="Disable quantized_model_init")
    parser.add_argument("--no-hpiv", action="store_true", help="Set preserve_high_precision_init_val=False")
    parser.add_argument("--snapshot-dir", type=str, default="/tmp/memory_snapshots", help="Where to save .pickle")
    parser.add_argument(
        "--snapshot-after-n-steps", type=int, default=3, help="Dump snapshot after N steps (default 3)"
    )
    args = parser.parse_args()

    use_qinit = not args.no_qinit
    use_hpiv = not args.no_hpiv and use_qinit  # HPIV only makes sense with qinit

    mode = "bf16" if not use_qinit else f"qinit_hpiv={'T' if use_hpiv else 'F'}"
    dist_print(f"\n{'=' * 60}")
    dist_print(f"Memory Profiler: fully_shard — mode={mode}")
    dist_print(f"  qinit={use_qinit}, hpiv={use_hpiv}")
    dist_print(f"  model: {NUM_LAYERS} layers, hidden={HIDDEN_SIZE}, ffn={FFN_HIDDEN_SIZE}")
    dist_print(f"  snapshot after {args.snapshot_after_n_steps} steps → {args.snapshot_dir}")
    dist_print(f"{'=' * 60}\n")

    # ── 1. Distributed setup ─────────────────────────────────────────
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Start memory profiler
    torch.cuda.memory._record_memory_history(max_entries=500000)
    dist_print("Memory profiler started — recording allocation history")
    log_memory("before_model_init")

    # ── 2. Create model on meta device ───────────────────────────────
    if use_qinit:
        with te.quantized_model_init(
            recipe=MXFP8BlockScaling(), enabled=True, preserve_high_precision_init_val=use_hpiv
        ):
            model = torch.nn.Sequential(
                *[
                    te.TransformerLayer(
                        HIDDEN_SIZE,
                        FFN_HIDDEN_SIZE,
                        NUM_ATTENTION_HEADS,
                        fuse_qkv_params=True,
                        params_dtype=DTYPE,
                        hidden_dropout=0.0,
                        attention_dropout=0.0,
                        device="meta",
                    )
                    for _ in range(NUM_LAYERS)
                ]
            )
    else:
        model = torch.nn.Sequential(
            *[
                te.TransformerLayer(
                    HIDDEN_SIZE,
                    FFN_HIDDEN_SIZE,
                    NUM_ATTENTION_HEADS,
                    fuse_qkv_params=True,
                    params_dtype=DTYPE,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                    device="meta",
                )
                for _ in range(NUM_LAYERS)
            ]
        )

    log_memory("after_model_init_meta")
    dist_print(f"Model created on meta device. use_qinit={use_qinit}")

    # ── 3. FSDP2 sharding ───────────────────────────────────────────
    mesh = DeviceMesh("cuda", list(range(world_size)))
    for child in model.children():
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)

    log_memory("after_fsdp_shard")
    dist_print("FSDP2 sharding applied.")

    # ── 4. Materialize parameters on GPU ─────────────────────────────
    for module in model.modules():
        if isinstance(module, TransformerEngineBaseModule):
            module.reset_parameters()

    log_memory("after_materialize")
    dist_print("Parameters materialized on GPU.")

    # Print parameter info
    if int(os.environ.get("RANK", "0")) == 0:
        for name, param in model.named_parameters():
            local = param._local_tensor if isinstance(param, DTensor) else param
            is_qt = isinstance(local, QuantizedTensor)
            print(
                f"  {name}: shape={list(param.shape)}, local_shape={list(local.shape)}, "
                f"dtype={local.dtype}, is_quantized={is_qt}"
            )

    # ── 5. Optimizer with FP32 master weights ────────────────────────
    optimizer = te.optimizers.FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    log_memory("after_optimizer_create")

    # ── 6. Seed master weights from high-precision init values ───────
    if use_qinit and use_hpiv:
        count = 0
        for name, param in model.named_parameters():
            optimizer.initialize_state(param, store_param_remainders=False)
            local = param._local_tensor if isinstance(param, DTensor) else param
            if isinstance(local, QuantizedTensor):
                hp_val = local.get_high_precision_init_val()
                if hp_val is not None:
                    optimizer.set_scaled_state(param, "master_param", hp_val.to(device=device, dtype=torch.float32))
                    local.clear_high_precision_init_val()
                    count += 1
        dist_print(f"Seeded {count} master weights from high-precision init values.")
    elif use_qinit:
        # Without HPIV, just initialize optimizer state normally
        for param in model.parameters():
            optimizer.initialize_state(param, store_param_remainders=False)
        dist_print("Optimizer state initialized (no HPIV — master weights from dequantized FP8).")

    log_memory("after_master_weight_seed")

    # ── 7. Training loop ─────────────────────────────────────────────
    x = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=DTYPE, device=device)
    target = torch.randn(SEQ_LEN, BATCH_PER_RANK, HIDDEN_SIZE, dtype=DTYPE, device=device)

    log_memory("before_training")

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)

        if use_qinit:
            with te.autocast(enabled=True, recipe=MXFP8BlockScaling()):
                output = model(x)
        else:
            output = model(x)

        loss = f.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        log_memory(f"after_step_{step}")
        dist_print(f"  Step {step}: loss = {loss.item():.6f}")

        # Dump snapshot after N steps
        if step == args.snapshot_after_n_steps - 1:
            if int(os.environ.get("RANK", "0")) == 0:
                snap_dir = Path(args.snapshot_dir) / mode
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_path = snap_dir / "memory_snapshot.pickle"
                torch.cuda.memory._dump_snapshot(str(snap_path))
                print(f"Memory snapshot saved to {snap_path}")
            torch.cuda.memory._record_memory_history(enabled=None)
            dist_print("Memory profiler stopped.")

    dist_print(f"\nTraining complete — mode={mode}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
