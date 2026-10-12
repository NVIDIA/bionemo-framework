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

"""Replicated-projector tensor-parallel behavior with synthetic model doubles."""

from __future__ import annotations

import os
import socket

import pytest
import torch
from torch import nn

from nemotron_stitch.automodel.parallel import (
    check_replicated_inputs_once,
    configure_replicated_projector_tp,
    install_combined_moe_tp_dispatch,
    prepare_replicated_projector_output,
    replace_tp_linear,
    replicated_projector_tp_state,
)

_ATTRS = ("projection", "marker_delta")


class _ToyProjectorLm(nn.Module):
    """Replicated bridge feeding a vocabulary-sharded LM head."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 6))
        self.marker_delta = nn.Parameter(torch.zeros(1, 6))
        self.lm_head = nn.Linear(6, 10, bias=False)

    def forward(self, inputs):
        projected = self.projection(inputs)
        projected = prepare_replicated_projector_output(self, projected)
        return self.lm_head(projected)


def test_single_rank_configuration_is_inert():
    class SoloMesh:
        @staticmethod
        def size():
            return 1

    model = _ToyProjectorLm()
    configure_replicated_projector_tp(model, SoloMesh(), attrs=_ATTRS, reduce_gradient=True)
    state = replicated_projector_tp_state(model)
    assert state is not None
    assert state.reduce_gradient is True
    assert not (state.sync_pending or state.input_check_pending or state.gradient_check_pending)
    # Forward-path calls no-op on a size-1 mesh, hooks included.
    check_replicated_inputs_once(model, [("input_ids", torch.ones(2, 3, dtype=torch.long))])
    out = model(torch.randn(4, 5))
    out.sum().backward()


def test_meta_parameters_defer_the_initial_sync():
    class DualMesh:
        @staticmethod
        def size():
            return 2

    model = _ToyProjectorLm()
    with torch.device("meta"):
        meta_model = _ToyProjectorLm()
    configure_replicated_projector_tp(meta_model, DualMesh(), attrs=_ATTRS)
    state = replicated_projector_tp_state(meta_model)
    assert state.sync_pending and state.input_check_pending and state.gradient_check_pending
    # Materialized models synchronize eagerly at configure time — which needs a
    # process group, so assert only that the deferred path did not take it.
    del model


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tp_parity_worker(rank: int, port: int) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
    )
    dist.init_process_group("gloo")
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        torch.manual_seed(1234)
        model = _ToyProjectorLm()
        reference = _ToyProjectorLm()
        reference.load_state_dict(model.state_dict())
        inputs = torch.randn(7, 5)
        labels = torch.tensor([0, 7, -100, 4, 9, 2, 5])

        configure_replicated_projector_tp(model, mesh, attrs=_ATTRS)
        parallelize_module(
            model,
            mesh,
            {"lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)},
        )
        logits = model(inputs)
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
        loss.backward()

        reference_loss = torch.nn.functional.cross_entropy(reference(inputs), labels, ignore_index=-100)
        reference_loss.backward()

        torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-6)
        for parameter, reference_parameter in zip(
            model.projection.parameters(), reference.projection.parameters(), strict=True
        ):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def test_two_rank_tp_loss_and_replicated_projection_gradient_parity():
    torch.multiprocessing.start_processes(
        _tp_parity_worker, args=(_free_port(),), nprocs=2, start_method="spawn", join=True
    )


def _tp_partial_gradient_worker(rank: int, port: int) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
    )
    dist.init_process_group("gloo")
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        torch.manual_seed(1234)
        model = _ToyProjectorLm()
        reference = _ToyProjectorLm()
        reference.load_state_dict(model.state_dict())
        configure_replicated_projector_tp(model, mesh, attrs=_ATTRS, reduce_gradient=True)
        inputs = torch.randn(7, 5)

        check_replicated_inputs_once(model, [("test input", inputs)])
        state = replicated_projector_tp_state(model)
        assert not state.input_check_pending

        mismatched = inputs.clone()
        mismatched[0, 0] += rank
        state.input_check_pending = True
        with pytest.raises(RuntimeError, match="values diverged across TP ranks"):
            check_replicated_inputs_once(model, [("mismatched test input", mismatched)])
        state.input_check_pending = False

        for base_scale in (1.0, 3.0):
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            local_scale = base_scale + rank
            total_scale = 2 * base_scale + 1

            projected = model.projection(inputs)
            projected = prepare_replicated_projector_output(model, projected)
            (projected * local_scale).sum().backward()
            (reference.projection(inputs) * total_scale).sum().backward()

            for parameter, reference_parameter in zip(
                model.projection.parameters(), reference.projection.parameters(), strict=True
            ):
                torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def test_two_rank_partial_tp_gradients_are_summed_on_every_backward():
    torch.multiprocessing.start_processes(
        _tp_partial_gradient_worker, args=(_free_port(),), nprocs=2, start_method="spawn", join=True
    )


def test_replace_tp_linear_without_inventing_bias():
    class TeStyleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(7, 5))

        @property
        def bias(self):
            return torch.empty(0)

    parent = nn.Module()
    parent.proj = TeStyleLinear()
    original_weight = parent.proj.weight

    replace_tp_linear(parent, "proj")

    assert isinstance(parent.proj, nn.Linear)
    assert parent.proj.weight is original_weight
    assert parent.proj.bias is None

    # Idempotent on an already-torch linear.
    replace_tp_linear(parent, "proj")
    assert parent.proj.weight is original_weight


def test_install_combined_moe_tp_dispatch_applies_tp_then_generic_ep(monkeypatch):
    moe_parallelizer = pytest.importorskip(
        "nemo_automodel.components.moe.parallelizer",
        reason="AutoModel is not installed",
    )

    calls = []

    def original(model, world_mesh, moe_mesh, *args, **kwargs):
        calls.append((model, world_mesh, moe_mesh, args, kwargs))
        return "parallelized"

    class Model:
        pass

    class TpMesh:
        @staticmethod
        def size():
            return 2

    world_mesh = {"tp": TpMesh()}
    model = Model()
    applied = []
    monkeypatch.setattr(moe_parallelizer, "parallelize_model", original)

    assert install_combined_moe_tp_dispatch(model_cls=Model, apply_tp=lambda m, mesh: applied.append((m, mesh)))
    # Re-installing over the wrapper is a no-op.
    assert install_combined_moe_tp_dispatch(model_cls=Model, apply_tp=lambda m, mesh: None)

    assert (
        moe_parallelizer.parallelize_model(model, world_mesh, "moe-mesh", tp_axis_name="tp", dp_axis_names=("dp",))
        == "parallelized"
    )
    assert applied == [(model, world_mesh["tp"])]
    assert calls == [(model, world_mesh, "moe-mesh", (), {"tp_axis_name": None, "dp_axis_names": ("dp",)})]

    # Other model classes bypass the TP pre-pass entirely.
    applied.clear()
    calls.clear()
    assert moe_parallelizer.parallelize_model(object(), world_mesh, "moe-mesh", tp_axis_name="tp") == "parallelized"
    assert applied == []
    assert calls[0][4] == {"tp_axis_name": "tp"}
