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

"""Worker script for distributed DCP (Distributed Checkpoint) tests.

Launched by torchrun from BaseModelTest.test_dcp_output_parity / test_dcp_output_parity_fp8_init.
Verifies that a model sharded with FSDP2 produces identical outputs after a DCP save/load round-trip.
"""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import transformer_engine.pytorch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import set_seed


def _setup_sys_path():
    """Add model root and tests directory to sys.path so model/test imports work."""
    script_dir = Path(__file__).resolve().parent  # tests/common/
    tests_dir = script_dir.parent  # tests/
    model_root = tests_dir.parent  # model root (e.g., models/esm2/)
    for p in [str(model_root), str(tests_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_tester_class(tester_file, class_name):
    """Dynamically load a tester class from a file path."""
    # Ensure the tester file's directory tree is importable
    tester_dir = str(Path(tester_file).parent)
    tester_parent = str(Path(tester_file).parent.parent)
    for p in [tester_parent, tester_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location("_dcp_tester_module", tester_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _build_and_shard_model(tester, config, recipe, device, device_mesh):
    """Build a model (optionally with FP8 quantized_model_init), shard with FSDP2, and move to device."""
    model_class = tester.get_model_class()

    if recipe is not None:
        with transformer_engine.pytorch.quantized_model_init(recipe=recipe):
            model = model_class(config)
    else:
        model = model_class(config)

    # Shard each transformer layer, then the root model
    for layer in tester.get_layer_path(model):
        fully_shard(layer, mesh=device_mesh)
    fully_shard(model, mesh=device_mesh)

    model.to(device)
    return model


def _run_eval_forward(model, input_data, recipe):
    """Run an eval forward pass and return detached logits."""
    model.eval()
    with torch.no_grad():
        if recipe is not None:
            with transformer_engine.pytorch.autocast(recipe=recipe):
                outputs = model(**input_data)
        else:
            outputs = model(**input_data)
    return outputs.logits.detach().clone()


def run_dcp_output_parity(tester, fp8_recipe_name=None, seed=42):
    """Core DCP round-trip test: build → forward → save → rebuild → load → forward → compare."""
    from tests.common.fixtures import recipe_from_name

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,))

    # Resolve FP8 recipe
    recipe = recipe_from_name(fp8_recipe_name) if fp8_recipe_name else None

    # Build config
    set_seed(seed)
    config = tester.create_test_config(dtype=torch.bfloat16, attn_input_format="bshd")

    # Prepare input data
    input_data = tester.get_test_input_data("bshd", pad_to_multiple_of=32)

    # --- Model A: build, shard, forward ---
    set_seed(seed)
    model_a = _build_and_shard_model(tester, config, recipe, device, device_mesh)
    logits_a = _run_eval_forward(model_a, input_data, recipe)

    # --- DCP Save ---
    # Rank 0 creates temp dir, broadcast path to all ranks
    if rank == 0:
        tmp_dir = tempfile.mkdtemp(prefix="dcp_test_")
    else:
        tmp_dir = None
    tmp_dir_list = [tmp_dir]
    dist.broadcast_object_list(tmp_dir_list, src=0)
    tmp_dir = tmp_dir_list[0]

    checkpoint_path = os.path.join(tmp_dir, "checkpoint")

    state_dict_a = {"model": model_a.state_dict()}
    dcp.save(state_dict_a, checkpoint_id=checkpoint_path)

    dist.barrier()

    # Free model_a
    del model_a, state_dict_a
    torch.cuda.empty_cache()

    # --- Model B: build fresh, shard, load, forward ---
    set_seed(seed)
    model_b = _build_and_shard_model(tester, config, recipe, device, device_mesh)

    state_dict_b = {"model": model_b.state_dict()}
    dcp.load(state_dict_b, checkpoint_id=checkpoint_path)
    model_b.load_state_dict(state_dict_b["model"], strict=False)

    logits_b = _run_eval_forward(model_b, input_data, recipe)

    # --- Compare ---
    tolerances = tester.get_tolerances()
    torch.testing.assert_close(
        logits_a,
        logits_b,
        atol=tolerances.dcp_logits_atol,
        rtol=tolerances.dcp_logits_rtol,
        msg=lambda x: f"DCP round-trip logits mismatch: {x}",
    )

    # Cleanup
    del model_b, state_dict_b
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Rank {rank}] DCP output parity test PASSED (fp8_recipe={fp8_recipe_name})")


if __name__ == "__main__":
    _setup_sys_path()

    parser = argparse.ArgumentParser(description="DCP distributed test worker")
    parser.add_argument(
        "--tester-file", required=True, help="Absolute path to the test file containing the tester class"
    )
    parser.add_argument("--tester-class", required=True, help="Name of the tester class (e.g., TestESM2Model)")
    parser.add_argument("--fp8-recipe", default=None, help="FP8 recipe name (e.g., DelayedScaling)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")

    try:
        tester_cls = _load_tester_class(args.tester_file, args.tester_class)
        tester = tester_cls()
        run_dcp_output_parity(tester, fp8_recipe_name=args.fp8_recipe, seed=args.seed)
    finally:
        dist.destroy_process_group()
