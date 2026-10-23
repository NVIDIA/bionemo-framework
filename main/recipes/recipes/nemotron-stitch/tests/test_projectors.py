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

"""Projector construction, initialization, and output parity."""

from __future__ import annotations

import pytest
import torch

from nemotron_stitch.projector import (
    Mlp2xGeluNormProjector,
    Mlp2xGeluProjector,
    Perceiver3DProjector,
    Projector,
    build_projector,
    register_projector,
)


def test_perceiver3d_shape_determinism_and_gradient():
    torch.manual_seed(7)
    projector = Perceiver3DProjector(
        mm_hidden_size=8, hidden_size=16, output_size=24, num_tokens=5, depth=2, num_heads=4
    )
    features = torch.randn(2, 8, 3, 4, 2, requires_grad=True)
    first = projector(features)
    second = projector(features)
    assert first.shape == (2, 5, 24)
    torch.testing.assert_close(first, second)
    first.square().mean().backward()
    assert features.grad is not None
    assert projector.latents.grad is not None


def test_perceiver3d_validates_channels():
    projector = Perceiver3DProjector(
        mm_hidden_size=8, hidden_size=16, output_size=24, num_tokens=5, depth=1, num_heads=4
    )
    with pytest.raises(ValueError, match="expected 8"):
        projector(torch.zeros(1, 7, 2, 2, 2))


def test_coordinates_span_the_unit_cube():
    coords = Perceiver3DProjector.coordinates((2, 3, 4), device=torch.device("cpu"), dtype=torch.float32)
    assert coords.shape == (24, 3)
    torch.testing.assert_close(coords.amin(dim=0), torch.tensor([-1.0, -1.0, -1.0]))
    torch.testing.assert_close(coords.amax(dim=0), torch.tensor([1.0, 1.0, 1.0]))


def test_mlp2x_gelu_projects_token_sequences():
    projector = Mlp2xGeluProjector(mm_hidden_size=8, hidden_size=16, output_size=24)
    assert projector(torch.randn(2, 5, 8)).shape == (2, 5, 24)


def test_mlp2x_gelu_norm_projects_in_bf16():
    # Construction casts this projector to BF16.
    projector = Mlp2xGeluNormProjector(mm_hidden_size=8, hidden_size=16, output_size=24)
    out = projector(torch.randn(2, 5, 8, dtype=torch.bfloat16))
    assert out.shape == (2, 5, 24)
    assert out.dtype == torch.bfloat16
    assert projector.fc1.in_features == projector.mm_hidden_size


def test_build_projector_kinds():
    assert isinstance(
        build_projector({"kind": "mlp2x_gelu", "mm_hidden_size": 8}, output_size=24),
        Mlp2xGeluProjector,
    )
    assert isinstance(
        build_projector({"kind": "mlp2x_gelu_norm", "mm_hidden_size": 8, "hidden_size": 16}, output_size=24),
        Mlp2xGeluNormProjector,
    )
    assert isinstance(
        build_projector(
            {
                "kind": "perceiver3d",
                "mm_hidden_size": 8,
                "hidden_size": 16,
                "num_tokens": 5,
                "depth": 1,
                "num_heads": 4,
            },
            output_size=24,
        ),
        Perceiver3DProjector,
    )


def test_build_projector_rejects_unknown_kind():
    # "linear" is deliberately absent until a consumer needs it (AGENTS.md).
    with pytest.raises(ValueError, match="unknown projector kind"):
        build_projector({"kind": "linear", "mm_hidden_size": 8}, output_size=24)


def test_application_can_register_an_external_projector():
    class ExternalProjector(Projector):
        def __init__(self, mm_hidden_size: int, output_size: int, *, scale: float):
            super().__init__()
            self.mm_hidden_size = mm_hidden_size
            self.output_size = output_size
            self.scale = scale
            self.weight = torch.nn.Parameter(torch.empty(mm_hidden_size, output_size))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            torch.nn.init.ones_(self.weight)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features @ self.weight * self.scale

    register_projector("test_external", ExternalProjector)
    register_projector("test_external", ExternalProjector)  # identical registration is idempotent
    projector = build_projector(
        {"name": "tokens", "kind": "test_external", "mm_hidden_size": 3, "scale": 0.5},
        output_size=7,
    )

    assert isinstance(projector, ExternalProjector)
    assert projector.output_size == 7
    output = projector(torch.ones(2, 3))
    assert output.shape == (2, 7)
    torch.testing.assert_close(output, torch.full((2, 7), 1.5))


def test_external_projector_registration_fails_closed():
    class First(Projector):
        def reset_parameters(self) -> None:
            pass

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features

    class Second(First):
        pass

    register_projector("test_conflict", First)
    with pytest.raises(ValueError, match="already registered"):
        register_projector("test_conflict", Second)
    with pytest.raises(ValueError, match="built-in"):
        register_projector("mlp2x_gelu", First)


def test_external_projector_factory_must_return_projector():
    register_projector("test_invalid", lambda **_kwargs: torch.nn.Identity())
    with pytest.raises(TypeError, match="expected Projector"):
        build_projector({"kind": "test_invalid"}, output_size=7)


# ---------------------------------------------------------------------------
# Mlp2xGeluNormProjector — init hardening and reset semantics
# ---------------------------------------------------------------------------


from nemotron_stitch.projector import projectors as _projectors_module  # noqa: E402


def _legacy_fp32_temp_init(parameter: torch.nn.Parameter, *, std: float, sampler) -> None:
    with torch.no_grad():
        sample = torch.empty_like(parameter, dtype=torch.float32, device=parameter.device)
        sampler(sample, std=std)
        parameter.copy_(sample.to(dtype=parameter.dtype))


def _has_absolute_truncation_endpoint(values: torch.Tensor) -> torch.Tensor:
    return (values == -2.0) | (values == 2.0)


def test_init_hardening_resamples_only_invalid_values(monkeypatch):
    valid_edge = 0.0400390625
    expected_edge = torch.tensor(valid_edge, dtype=torch.bfloat16).float().item()
    first_draw = torch.tensor(
        [-2.0, 2.0, float("nan"), float("inf"), float("-inf"), valid_edge, -0.0390625],
        dtype=torch.float32,
    )
    refill_values = iter((0.015625, -0.0234375, 0.03125, -0.03515625, 0.02734375))

    def injected_sampler(tensor: torch.Tensor, *, std: float) -> torch.Tensor:
        del std
        tensor.fill_(0.0078125)
        if injected_sampler.calls == 0:
            tensor.view(-1)[: first_draw.numel()].copy_(first_draw)
        else:
            tensor.fill_(next(refill_values))
        injected_sampler.calls += 1
        return tensor

    injected_sampler.calls = 0
    leaked = torch.nn.Parameter(torch.empty(64, dtype=torch.bfloat16))
    _legacy_fp32_temp_init(leaked, std=0.02, sampler=injected_sampler)
    leaked_values = leaked.float()
    leaked_invalid = (~torch.isfinite(leaked_values)) | _has_absolute_truncation_endpoint(leaked_values)
    assert leaked_invalid.any()

    injected_sampler.calls = 0
    monkeypatch.setattr(_projectors_module.nn.init, "trunc_normal_", injected_sampler)
    hardened = torch.nn.Parameter(torch.empty(64, dtype=torch.bfloat16))
    _projectors_module._trunc_normal_parameter_(hardened, std=0.02)

    hardened_values = hardened.float()
    hardened_invalid = (~torch.isfinite(hardened_values)) | _has_absolute_truncation_endpoint(hardened_values)
    assert not hardened_invalid.any()
    assert (hardened_values == expected_edge).any()
    assert (hardened_values == torch.tensor(-0.0390625, dtype=torch.bfloat16).float()).any()
    assert injected_sampler.calls == 2


def test_bf16_reinit_has_zero_absolute_endpoints_at_scale():
    num_seeds = 12
    values_per_seed = 1_048_576
    endpoint_hits = 0
    finite_hits = 0
    total_sum = 0.0
    total_sumsq = 0.0
    max_abs = 0.0

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        parameter = torch.nn.Parameter(torch.empty(values_per_seed, dtype=torch.bfloat16))
        _projectors_module._trunc_normal_parameter_(parameter, std=0.02)
        values = parameter.float()
        endpoint_hits += int((values == -2.0).sum().item() + (values == 2.0).sum().item())
        finite_hits += int(torch.isfinite(values).sum().item())
        total_sum += float(values.sum().item())
        total_sumsq += float(values.square().sum().item())
        max_abs = max(max_abs, float(values.abs().max().item()))

    total_values = num_seeds * values_per_seed
    mean = total_sum / total_values
    variance = (total_sumsq / total_values) - (mean * mean)
    std = variance**0.5

    assert total_values == 12_582_912
    assert finite_hits == total_values
    assert endpoint_hits == 0
    assert abs(mean) < 5e-4
    assert 0.019 <= std <= 0.021
    assert max_abs < 0.15


def test_init_keeps_bias_zero_and_layernorm_identity():
    projector = Mlp2xGeluNormProjector(mm_hidden_size=64, hidden_size=128, output_size=32)

    assert torch.count_nonzero(projector.fc1.bias).item() == 0
    assert torch.count_nonzero(projector.fc2.bias).item() == 0
    torch.testing.assert_close(projector.norm.weight.float(), torch.ones_like(projector.norm.weight.float()))
    torch.testing.assert_close(projector.norm.bias.float(), torch.zeros_like(projector.norm.bias.float()))


def test_reset_preserves_meta_parameters():
    projector = Mlp2xGeluNormProjector(mm_hidden_size=32, hidden_size=32, output_size=16).to(device="meta")

    projector.reset_parameters()

    assert projector.fc1.weight.is_meta
    assert projector.fc2.weight.is_meta


def test_reset_resamples_a_sharded_dtensor(tmp_path, monkeypatch):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    if dist.is_initialized():
        pytest.skip("requires an isolated process group")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'process-group'}",
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        parameter = distribute_tensor(
            torch.empty(8, 4, dtype=torch.bfloat16),
            mesh,
            [Shard(0)],
        )

        def injected_sampler(tensor: torch.Tensor, *, std: float) -> torch.Tensor:
            del std
            tensor.fill_(-2.0 if injected_sampler.calls == 0 else 0.015625)
            injected_sampler.calls += 1
            return tensor

        injected_sampler.calls = 0
        monkeypatch.setattr(_projectors_module.nn.init, "trunc_normal_", injected_sampler)

        _projectors_module._trunc_normal_parameter_(parameter, std=0.02)

        expected = torch.full_like(parameter.to_local(), 0.015625)
        torch.testing.assert_close(parameter.to_local(), expected)
        assert injected_sampler.calls == 2
    finally:
        dist.destroy_process_group()


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only init/forward kernels")


@_CUDA
def test_projection_shape_and_dtype():
    projector = Mlp2xGeluNormProjector(mm_hidden_size=1024, hidden_size=4096, output_size=2688).cuda()
    x = torch.randn(10, 1024, dtype=torch.bfloat16, device="cuda")
    y = projector(x)
    assert y.shape == (10, 2688)
    assert y.dtype == torch.bfloat16


@_CUDA
def test_projection_grads_flow():
    projector = Mlp2xGeluNormProjector(mm_hidden_size=1024, hidden_size=4096, output_size=2688).cuda()
    x = torch.randn(10, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = projector(x)
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters())


@_CUDA
def test_projection_layernorm_keeps_norm_bounded():
    """LayerNorm at the end is load-bearing — without it, post-projection norm
    grows unbounded with input scale."""
    projector = Mlp2xGeluNormProjector(mm_hidden_size=1024, hidden_size=4096, output_size=2688).cuda()
    x_small = torch.randn(100, 1024, dtype=torch.bfloat16, device="cuda")
    x_large = x_small * 100.0
    y_small = projector(x_small).float()
    y_large = projector(x_large).float()
    # LayerNorm normalizes per-row; per-row std should be ~1 regardless of input scale.
    assert y_small.std(dim=-1).mean().item() < 5.0
    assert y_large.std(dim=-1).mean().item() < 5.0
