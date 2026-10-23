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

"""Projector ABC, build_projector, and the projector zoo.

The zoo is exactly what the two consumers run today: ``mlp2x_gelu`` and
``mlp2x_gelu_norm`` adds output normalization to a two-layer projector.
LLaVA's ``linear`` kind is deliberately absent: no consumer has one, and this
package does not keep speculative code (AGENTS.md). Add it with its first
caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor


class Projector(nn.Module, ABC):
    """Contract implemented by every encoder→LM projector.

    Features in (``mm_hidden_size`` wide), soft tokens out (``output_size``
    wide). Geometry follows the flat index contract: flat
    ``[N, mm_hidden_size]`` → ``[N, output_size]`` is canonical; the dense
    ``[B, T, ...]`` spelling is accepted where the kind supports it.
    """

    mm_hidden_size: int
    output_size: int

    @abstractmethod
    def reset_parameters(self) -> None:
        """Initialize trainable state, including meta-device construction."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return soft tokens, ``[N, output_size]`` or ``[B, T, output_size]``."""


ProjectorFactory = Callable[..., Projector]
_BUILTIN_KINDS = frozenset({"perceiver3d", "mlp2x_gelu", "mlp2x_gelu_norm"})
_PROJECTOR_FACTORIES: dict[str, ProjectorFactory] = {}


def register_projector(kind: str, factory: ProjectorFactory) -> ProjectorFactory:
    """Register an application-owned projector factory under ``kind``.

    The factory receives the projector config fields as keyword arguments,
    plus the language model's authoritative ``output_size``. Registration is
    process-local and must happen before constructing a model or loading a
    projector artifact in each process that does so.
    """
    if not isinstance(kind, str) or not kind or any(character.isspace() for character in kind):
        raise ValueError("projector kind must be a nonempty, whitespace-free string")
    if kind in _BUILTIN_KINDS:
        raise ValueError(f"cannot replace built-in projector kind: {kind!r}")
    if not callable(factory):
        raise TypeError("projector factory must be callable")
    registered = _PROJECTOR_FACTORIES.get(kind)
    if registered is not None and registered is not factory:
        raise ValueError(f"projector kind is already registered: {kind!r}")
    _PROJECTOR_FACTORIES[kind] = factory
    return factory


class Mlp2xGeluProjector(Projector):
    """Project a precomputed encoder sequence without resampling.

    Every op is position-local, so flat ``[N,F]`` and dense ``[B,T,F]``
    features are the same computation.
    """

    def __init__(self, mm_hidden_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.mm_hidden_size = int(mm_hidden_size)
        self.output_size = int(output_size)
        self.network = nn.Sequential(
            nn.LayerNorm(mm_hidden_size),
            nn.Linear(mm_hidden_size, hidden_size, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def reset_parameters(self) -> None:
        for module in self.network.modules():
            if isinstance(module, nn.Linear | nn.LayerNorm):
                module.reset_parameters()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim not in (2, 3) or features.shape[-1] != self.mm_hidden_size:
            raise ValueError(
                f"token projector expected [N,{self.mm_hidden_size}] or [B,T,{self.mm_hidden_size}], "
                f"got {tuple(features.shape)}"
            )
        return self.network(features)


class _PerceiverBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.latent_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        update, _ = self.cross_attention(
            self.latent_norm(latents), self.context_norm(context), self.context_norm(context), need_weights=False
        )
        latents = latents + update
        update, _ = self.self_attention(
            self.self_norm(latents), self.self_norm(latents), self.self_norm(latents), need_weights=False
        )
        latents = latents + update
        return latents + self.feed_forward(self.ff_norm(latents))


class Perceiver3DProjector(Projector):
    """Map a channels-first 3D feature grid to a fixed soft-token sequence."""

    def __init__(
        self,
        mm_hidden_size: int,
        hidden_size: int,
        output_size: int,
        num_tokens: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.mm_hidden_size = int(mm_hidden_size)
        self.output_size = int(output_size)
        self.num_tokens = int(num_tokens)
        self.input_norm = nn.LayerNorm(mm_hidden_size)
        self.input_projection = nn.Linear(mm_hidden_size, hidden_size, bias=False)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(3, hidden_size, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.latents = nn.Parameter(torch.empty(num_tokens, hidden_size))
        self.blocks = nn.ModuleList(_PerceiverBlock(hidden_size, num_heads, dropout) for _ in range(depth))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.Linear | nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()

    @staticmethod
    def coordinates(shape: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        axes = [torch.linspace(-1, 1, steps=size, device=device, dtype=dtype) for size in shape]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(f"expected 3D features [B,C,D,H,W], got {tuple(features.shape)}")
        if features.shape[1] != self.mm_hidden_size:
            raise ValueError(f"expected {self.mm_hidden_size} channels, got {features.shape[1]}")
        batch, _, depth, height, width = features.shape
        context = features.flatten(2).transpose(1, 2)
        context = self.input_projection(self.input_norm(context))
        coordinates = self.coordinates((depth, height, width), device=context.device, dtype=context.dtype)
        context = context + self.coordinate_projection(coordinates).unsqueeze(0)
        latents = self.latents.to(dtype=context.dtype).unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            latents = block(latents, context)
        return self.output_projection(self.output_norm(latents))


def _invalid_trunc_normal_mask(sample: torch.Tensor) -> torch.Tensor:
    return (~torch.isfinite(sample)) | (sample == -2.0) | (sample == 2.0)


def _trunc_normal_parameter_(parameter: torch.Tensor, *, std: float) -> None:
    target = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    if target.is_meta:
        nn.init.trunc_normal_(target, std=std)
        return
    with torch.no_grad():
        sample = torch.empty_like(target, dtype=torch.float32, device=target.device)
        nn.init.trunc_normal_(sample, std=std)
        invalid = _invalid_trunc_normal_mask(sample)
        while invalid.any():
            refill = torch.empty(
                int(invalid.sum().item()),
                dtype=sample.dtype,
                device=sample.device,
            )
            nn.init.trunc_normal_(refill, std=std)
            refill_invalid = _invalid_trunc_normal_mask(refill)
            if refill_invalid.any():
                valid_refill = ~refill_invalid
                sample[invalid] = torch.where(valid_refill, refill, sample[invalid])
            else:
                sample[invalid] = refill
            invalid = _invalid_trunc_normal_mask(sample)
        target.copy_(sample.to(dtype=target.dtype))


class Mlp2xGeluNormProjector(Projector):
    """Linear-GELU-linear bridge with output normalization.

    DTensor/meta initialization and an initial BF16 cast are supported.
    The final LayerNorm keeps the bridge
    output bounded during Stage 1 and prevents non-finite gradients when the
    bridge is continued in Stage 2.
    """

    def __init__(self, mm_hidden_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.mm_hidden_size = int(mm_hidden_size)
        self.output_size = int(output_size)
        self.fc1 = nn.Linear(mm_hidden_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.reset_parameters()
        self.to(torch.bfloat16)

    def reset_parameters(self) -> None:
        _trunc_normal_parameter_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        _trunc_normal_parameter_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)
        # Eager construction initializes the LayerNorm itself, but the
        # meta->to_empty->reset flow (AutoModel materialization) leaves it as
        # uninitialized without an explicit reset.
        self.norm.reset_parameters()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(self.act(self.fc1(features))))


def build_projector(config: dict[str, Any], output_size: int) -> Projector:
    kind = str(config["kind"])
    if kind == "perceiver3d":
        return Perceiver3DProjector(
            mm_hidden_size=int(config["mm_hidden_size"]),
            hidden_size=int(config.get("hidden_size", 1024)),
            output_size=output_size,
            num_tokens=int(config["num_tokens"]),
            depth=int(config.get("depth", 2)),
            num_heads=int(config.get("num_heads", 16)),
            dropout=float(config.get("dropout", 0.0)),
        )
    if kind == "mlp2x_gelu":
        return Mlp2xGeluProjector(
            mm_hidden_size=int(config["mm_hidden_size"]),
            hidden_size=int(config.get("hidden_size", output_size)),
            output_size=output_size,
        )
    if kind == "mlp2x_gelu_norm":
        return Mlp2xGeluNormProjector(
            mm_hidden_size=int(config["mm_hidden_size"]),
            hidden_size=int(config["hidden_size"]),
            output_size=output_size,
        )
    factory = _PROJECTOR_FACTORIES.get(kind)
    if factory is not None:
        kwargs = {key: value for key, value in config.items() if key not in {"kind", "name"}}
        kwargs["output_size"] = output_size
        projector = factory(**kwargs)
        if not isinstance(projector, Projector):
            raise TypeError(f"projector factory for {kind!r} returned {type(projector).__name__}, expected Projector")
        if projector.output_size != output_size:
            raise ValueError(f"projector {kind!r} declares output_size={projector.output_size}, expected {output_size}")
        return projector
    raise ValueError(f"unknown projector kind: {kind!r}")
