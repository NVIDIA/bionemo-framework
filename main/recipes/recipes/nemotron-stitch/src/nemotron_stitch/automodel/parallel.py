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

"""Replicated projector under tensor parallelism, and TP×EP composition.

A module-owned projector sits outside the LM's sharded
architecture plan: FSDP2 shards it, TP leaves it replicated, and AutoModel's
per-architecture plans do not know it exists. This module owns the protocol
that keeps those replicas bit-identical — one broadcast after materialization,
a one-shot input-divergence check before the first TP compute, and a gradient
boundary that either validates replica agreement once or sums partial
contributions on every backward.

Two framework workarounds live here, both against AutoModel
9af3b45a8fefd817e9ff562e8e8780d54e8cf68e; see docs/upstream-gaps.md:

- U-16, ``replace_tp_linear``: the stock DTensor TP plan cannot shard TE-style
  custom linears, so they are swapped for torch.nn.Linear holding the *same*
  parameter objects first. Delete when AutoModel's plan handles them.
- U-17, ``install_combined_moe_tp_dispatch``: AutoModel routes every EP job through
  its generic custom-MoE parallelizer, bypassing the per-architecture
  strategy registry, and that entry point rejects TP unconditionally. The
  installed wrapper applies the consumer's TP plan first, then re-enters the
  generic path with TP disabled. Delete when AutoModel exposes a TP×EP
  composition hook.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

_STATE_ATTR = "_mm_replicated_projector_tp"


@dataclass
class ReplicatedProjectorTP:
    """Bookkeeping for a projector replicated across a TP group.

    ``attrs`` names the model attributes holding the replicated parameters —
    the projector module(s) and any extra parameters (e.g. a marker embedding
    delta). Pending flags arm one-shot validations; ``reduce_gradient``
    switches the gradient boundary from validate-once to sum-every-backward
    (the combined TP+EP route leaves rank-local partials at the boundary).
    """

    tp_mesh: Any
    attrs: tuple[str, ...]
    reduce_gradient: bool
    sync_pending: bool = False
    input_check_pending: bool = False
    gradient_check_pending: bool = False

    def parameters(self, model) -> list[torch.nn.Parameter]:
        parameters: list[torch.nn.Parameter] = []
        for attr in self.attrs:
            target = getattr(model, attr)
            if isinstance(target, torch.nn.Parameter):
                parameters.append(target)
            else:
                parameters.extend(target.parameters())
        return parameters


def replicated_projector_tp_state(model) -> ReplicatedProjectorTP | None:
    return getattr(model, _STATE_ATTR, None)


def configure_replicated_projector_tp(
    model,
    tp_mesh: Any,
    *,
    attrs: tuple[str, ...],
    reduce_gradient: bool = False,
) -> None:
    """Replicate the named parameters across TP and arm the boundary checks."""
    state = ReplicatedProjectorTP(
        tp_mesh=tp_mesh,
        attrs=tuple(attrs),
        reduce_gradient=bool(reduce_gradient),
    )
    setattr(model, _STATE_ATTR, state)
    if tp_mesh.size() <= 1:
        return
    state.sync_pending = True
    state.input_check_pending = True
    state.gradient_check_pending = True
    if not any(parameter.is_meta for parameter in state.parameters(model)):
        synchronize_replicated_projector_tp(model)


def synchronize_replicated_projector_tp(model) -> None:
    """Broadcast the materialized projector parameters within each TP group once."""
    state = replicated_projector_tp_state(model)
    if state is None or not state.sync_pending:
        return

    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    group = state.tp_mesh.get_group()
    source_rank = dist.get_global_rank(group, 0)
    with torch.no_grad():
        for parameter in state.parameters(model):
            local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            if local_parameter.is_meta:
                raise RuntimeError("the replicated projector was not materialized before its first TP forward")
            dist.broadcast(local_parameter, src=source_rank, group=group)
    state.sync_pending = False


def _check_replicated_input(state: ReplicatedProjectorTP, name: str, value: torch.Tensor) -> None:
    import torch.distributed as dist

    if value.ndim > 8:
        raise RuntimeError(f"replicated projector {name} has unsupported rank {value.ndim}")

    group = state.tp_mesh.get_group()
    metadata = torch.full((9,), -1, dtype=torch.int64, device=value.device)
    metadata[0] = value.ndim
    if value.ndim:
        metadata[1 : value.ndim + 1] = torch.tensor(
            value.shape,
            dtype=torch.int64,
            device=value.device,
        )
    peer_metadata = [torch.empty_like(metadata) for _ in range(state.tp_mesh.size())]
    dist.all_gather(peer_metadata, metadata, group=group)
    if any(not torch.equal(metadata, peer) for peer in peer_metadata):
        observed = [tuple(int(item) for item in peer[1 : int(peer[0]) + 1].tolist()) for peer in peer_metadata]
        raise RuntimeError(f"replicated projector {name} received different TP shapes: {observed}")

    reference = value.detach().clone()
    source_rank = dist.get_global_rank(group, 0)
    dist.broadcast(reference, src=source_rank, group=group)
    mismatch = torch.tensor(
        0 if torch.equal(value.detach(), reference) else 1,
        dtype=torch.int64,
        device=value.device,
    )
    dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=group)
    if mismatch.item():
        raise RuntimeError(f"replicated projector {name} values diverged across TP ranks")


def check_replicated_input(model, name: str, value: torch.Tensor) -> None:
    """Fail when one projector input differs within the TP group.

    No-op before ``configure_replicated_projector_tp`` runs or on a size-1
    mesh; the one-shot pending flag is managed by
    ``check_replicated_inputs_once`` (the forward path) — direct callers
    (diagnostics, tests) check unconditionally.
    """
    state = replicated_projector_tp_state(model)
    if state is None or state.tp_mesh.size() <= 1:
        return
    _check_replicated_input(state, name, value)


def check_replicated_inputs_once(model, named_values: list[tuple[str, torch.Tensor]]) -> None:
    """Fail before the first TP compute when projector inputs differ within the group."""
    state = replicated_projector_tp_state(model)
    if state is None or not state.input_check_pending:
        return
    for name, value in named_values:
        _check_replicated_input(state, name, value)
    state.input_check_pending = False


def _check_replicated_gradient(
    state: ReplicatedProjectorTP | None,
    gradient: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    import torch.distributed as dist

    if state is None or state.tp_mesh.size() <= 1:
        return gradient
    group = state.tp_mesh.get_group()
    check_pending = state.gradient_check_pending
    if check_pending:
        shape = torch.tensor(gradient.shape, dtype=torch.int64, device=gradient.device)
        peer_shapes = [torch.empty_like(shape) for _ in range(state.tp_mesh.size())]
        dist.all_gather(peer_shapes, shape, group=group)
        if any(not torch.equal(shape, peer_shape) for peer_shape in peer_shapes):
            observed = [tuple(int(value) for value in peer.tolist()) for peer in peer_shapes]
            raise RuntimeError(f"replicated projector received different TP gradient shapes: {observed}")

    if state.reduce_gradient:
        # Tensor hooks must not mutate their input; clone even when the
        # incoming gradient is already contiguous.
        reduced = gradient.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
        state.gradient_check_pending = False
        return reduced

    if check_pending:
        reference = gradient.detach().clone()
        source_rank = dist.get_global_rank(group, 0)
        dist.broadcast(reference, src=source_rank, group=group)
        if gradient.numel():
            difference = (gradient.detach() - reference).abs().max().float()
            magnitude = reference.abs().max().float()
        else:
            difference = torch.zeros((), device=gradient.device, dtype=torch.float32)
            magnitude = difference.clone()
        stats = torch.stack((difference, magnitude))
        dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=group)
        if stats[0] > atol + rtol * stats[1]:
            raise RuntimeError(
                "replicated projector gradients diverged across TP ranks: "
                f"max_abs_diff={stats[0].item():.6g}, reference_max={stats[1].item():.6g}"
            )
        state.gradient_check_pending = False
    return gradient


def prepare_replicated_projector_output(
    model,
    tensor: torch.Tensor,
    *,
    gradient_atol: float = 0.0,
    gradient_rtol: float = 0.0,
) -> torch.Tensor:
    """Attach the TP gradient boundary when backward synchronization is needed."""
    state = replicated_projector_tp_state(model)
    should_reduce = state.reduce_gradient if state is not None else False
    should_check = state.gradient_check_pending if state is not None else False
    if tensor.requires_grad and (should_reduce or should_check):
        # Resolve the state at backward time, matching the pre-extraction
        # semantics of reading the model's TP attributes live.
        tensor.register_hook(
            lambda gradient: _check_replicated_gradient(
                replicated_projector_tp_state(model),
                gradient,
                atol=gradient_atol,
                rtol=gradient_rtol,
            )
        )
    return tensor


def replace_tp_linear(parent: Any, name: str) -> None:
    """Replace a TE linear with an equivalent torch linear before DTensor TP."""
    from torch import nn

    module = getattr(parent, name)
    if isinstance(module, nn.Linear):
        return
    parameters = dict(module.named_parameters(recurse=False))
    weight = parameters.get("weight")
    if weight is None or weight.ndim != 2:
        raise TypeError(f"{name} is not a tensor-parallelizable linear module")
    bias = parameters.get("bias")
    replacement = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    replacement.weight = weight
    if bias is not None:
        replacement.bias = bias
    replacement.train(module.training)
    setattr(parent, name, replacement)


def install_combined_moe_tp_dispatch(
    *,
    model_cls: type,
    apply_tp: Callable[[Any, Any], None],
) -> bool:
    """Compose a consumer's TP plan with AutoModel's EP/FSDP implementation.

    ``apply_tp(model, tp_mesh)`` runs the architecture's TP portion (linear
    normalization, projector replication, parallelize_module calls) before the
    otherwise-compatible generic EP/FSDP path runs with TP disabled.
    Idempotent; returns False when AutoModel is not installed.
    """
    try:
        from nemo_automodel.components.moe import parallelizer as moe_parallelizer
    except ModuleNotFoundError:
        return False

    original = moe_parallelizer.parallelize_model
    if getattr(original, "_mm_combined_tp_ep", False):
        return True

    def parallelize_model(model, world_mesh, moe_mesh, *args, **kwargs):
        tp_axis_name = kwargs.get("tp_axis_name")
        tp_enabled = tp_axis_name is not None and world_mesh[tp_axis_name].size() > 1
        if not (isinstance(model, model_cls) and tp_enabled):
            return original(model, world_mesh, moe_mesh, *args, **kwargs)

        apply_tp(model, world_mesh[tp_axis_name])
        ep_kwargs = dict(kwargs)
        ep_kwargs["tp_axis_name"] = None
        return original(model, world_mesh, moe_mesh, *args, **ep_kwargs)

    # Marker attributes on the patched function (patch-provenance audit);
    # setattr keeps function-attribute writes visible to type checkers.
    setattr(parallelize_model, "_mm_combined_tp_ep", True)
    setattr(parallelize_model, "_mm_original", original)
    moe_parallelizer.parallelize_model = parallelize_model
    return True
