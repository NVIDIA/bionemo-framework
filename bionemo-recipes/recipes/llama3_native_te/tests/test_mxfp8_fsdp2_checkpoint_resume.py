#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Minimal reproduction of FSDP2 + MXFP8 checkpoint resume crash.

Bug: After fully_shard() wraps a model with quantized_model_init (MXFP8) params,
checkpoint resume via set_state_dict crashes with:
    RuntimeError: Attempted to access the data pointer on an invalid python storage.

Root cause: set_state_dict -> model.load_state_dict -> copy_() on MXFP8Tensor
re-quantizes, allocating new internal storage. FSDP2's reset_sharded_param
(post-load hook) then calls untyped_storage().data_ptr() on the invalidated
storage. PyTorch has a "# TODO: need to support tensor subclass" comment at
the crash site (_fsdp_param.py line 892).

Fix: Wrap the data_ptr() comparison in try/except RuntimeError. When it fails,
treat as same_local_tensor=False so _sharded_param_data gets re-recorded.

Run with: torchrun --nproc_per_node=2 test_mxfp8_fsdp2_checkpoint_resume.py
"""

import argparse
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import transformer_engine.pytorch as te
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.optimizers import FusedAdam


def apply_reset_sharded_param_fix():
    """Monkey-patch FSDPParam.reset_sharded_param to handle QuantizedTensor.

    After checkpoint load, copy_() on MXFP8Tensor re-quantizes which can
    invalidate the old untyped_storage, causing data_ptr() to crash.
    This wraps the comparison in try/except so reset_sharded_param can
    proceed normally (re-recording _sharded_param_data).
    """

    def _patched_reset_sharded_param(self):
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is not self.sharded_param:
            if torch.__future__.get_swap_module_params_on_conversion():
                raise AssertionError(
                    f"Expects swap_tensors to preserve object but got {new_param} instead of {self.sharded_param}"
                )
            self.sharded_param = new_param

        local_tensor = new_param._local_tensor
        if local_tensor.is_meta:
            return

        updated_local_tensor = False
        same_local_tensor = False

        if type(self._sharded_param_data) is torch.Tensor:
            try:
                same_local_tensor = (
                    self._sharded_param_data.untyped_storage().data_ptr() > 0
                    and self._sharded_param_data.untyped_storage().data_ptr()
                    == local_tensor.untyped_storage().data_ptr()
                )
            except RuntimeError:
                # QuantizedTensor (e.g. MXFP8Tensor) can have invalid storage
                # after copy_() re-quantization.
                same_local_tensor = False

        padded_sharded_size = self.padded_sharded_param_size
        shard_dim = self.fsdp_placement.dim
        length = local_tensor.size(shard_dim) if local_tensor.numel() > 0 else 0

        if local_tensor.size() != padded_sharded_size and not same_local_tensor:
            if shard_dim != 0:
                raise AssertionError(f"Shard({shard_dim}) requires even sharding: {local_tensor.size()=}")
            padded_local_tensor = local_tensor.new_zeros(padded_sharded_size)
            padded_local_tensor.narrow(dim=shard_dim, start=0, length=length).copy_(local_tensor)
            local_tensor = padded_local_tensor
            updated_local_tensor = True

        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.cpu().pin_memory()
            updated_local_tensor = True

        if not same_local_tensor:
            self._sharded_param_data = local_tensor.view(-1)

        if not isinstance(self.sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")

        if updated_local_tensor:
            self.sharded_param._local_tensor = local_tensor.narrow(dim=shard_dim, start=0, length=length)
            if not self.sharded_param._local_tensor.is_contiguous():
                raise AssertionError("Expected sharded_param._local_tensor to be contiguous")

        self._sharding_spec = self.sharded_param._spec

    FSDPParam.reset_sharded_param = _patched_reset_sharded_param


class SmallModel(nn.Module):
    """Tiny model using TE layers for testing."""

    def __init__(self, hidden=256, num_layers=2, recipe=None):
        super().__init__()
        self.layers = nn.ModuleList([te.Linear(hidden, hidden, bias=False) for _ in range(num_layers)])
        self.head = te.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def build_model_and_optimizer(device_mesh, recipe):
    """Build model with quantized_model_init, shard with FSDP2, create optimizer."""
    with te.quantized_model_init(recipe=recipe, enabled=True, preserve_high_precision_init_val=True):
        model = SmallModel(hidden=256, num_layers=2, recipe=recipe)

    for layer in model.layers:
        fully_shard(layer, mesh=device_mesh)
    fully_shard(model, mesh=device_mesh)

    optimizer = FusedAdam(model.parameters(), lr=1e-3, master_weights=True)

    # Initialize optimizer state
    for param in model.parameters():
        optimizer.initialize_state(param, store_param_remainders=False)

    return model, optimizer


def save_checkpoint(model, optimizer, ckpt_dir):
    """Save DCP checkpoint."""
    model_sd, optim_sd = get_state_dict(model, optimizer)
    # Filter _extra_state (not serializable for FP8)
    model_sd = {k: v for k, v in model_sd.items() if not k.endswith("_extra_state")}
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=ckpt_dir)


def load_checkpoint(model, optimizer, ckpt_dir):
    """Load DCP checkpoint using set_state_dict (the crash path)."""
    model_sd, optim_sd = get_state_dict(model, optimizer)
    model_sd = {k: v for k, v in model_sd.items() if not k.endswith("_extra_state")}
    state_dict = {"model": model_sd, "optim": optim_sd}
    dcp.load(state_dict, checkpoint_id=ckpt_dir)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict["optim"],
        options=StateDictOptions(strict=False),
    )


def run(apply_fix: bool):
    """Run the reproduction: save checkpoint, load it, run forward pass."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))

    recipe = MXFP8BlockScaling()

    if apply_fix:
        apply_reset_sharded_param_fix()
        if rank == 0:
            print("Applied reset_sharded_param fix")

    # Build model, do one forward pass, save checkpoint
    model, optimizer = build_model_and_optimizer(device_mesh, recipe)
    x = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    with te.autocast(enabled=True, recipe=recipe):
        out1 = model(x)
    loss1 = out1.sum()
    if rank == 0:
        print(f"Pre-save forward pass OK, loss={loss1.item():.4f}")

    with tempfile.TemporaryDirectory() as ckpt_dir:
        save_checkpoint(model, optimizer, ckpt_dir)
        dist.barrier()
        if rank == 0:
            print(f"Checkpoint saved to {ckpt_dir}")

        # Build fresh model, load checkpoint — this is where the crash happens
        model2, optimizer2 = build_model_and_optimizer(device_mesh, recipe)
        if rank == 0:
            print("Loading checkpoint (this crashes without the fix)...")

        load_checkpoint(model2, optimizer2, ckpt_dir)
        dist.barrier()
        if rank == 0:
            print("Checkpoint loaded successfully!")

        # Forward pass after resume — triggers lazy_init -> reset_sharded_param
        with te.autocast(enabled=True, recipe=recipe):
            out2 = model2(x)
        loss2 = out2.sum()
        if rank == 0:
            print(f"Post-load forward pass OK, loss={loss2.item():.4f}")
            print(f"Losses match: {torch.allclose(loss1, loss2)}")

    dist.destroy_process_group()
    if rank == 0:
        print("SUCCESS" if apply_fix else "SUCCESS (unexpected — bug may be fixed upstream)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Apply the reset_sharded_param monkey-patch fix")
    args = parser.parse_args()
    run(apply_fix=args.fix)
