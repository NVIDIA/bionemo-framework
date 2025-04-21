# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from einops import rearrange
from nemo.collections.llm.gpt.model.hyena import HyenaNVTestConfig, HyenaTestConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer_specs import hyena_stack_spec_no_te
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_mixer import HyenaMixer

from bionemo.testing import megatron_parallel_state_utils


@pytest.fixture(params=[torch.bfloat16, torch.float32])
def hyena_test_config(request) -> HyenaTestConfig:
    config = HyenaTestConfig()
    config.params_dtype = request.param
    return config


@pytest.fixture(params=[torch.bfloat16, torch.float32])
def hyena_nv_test_config(request) -> HyenaNVTestConfig:
    config = HyenaNVTestConfig()
    config.params_dtype = request.param
    return config


@pytest.fixture
def hyena_config() -> HyenaConfig:
    config = HyenaConfig()
    config.num_groups_hyena = 4096
    config.num_groups_hyena_short = 256
    config.num_groups_hyena_medium = 256
    return config


@pytest.fixture(
    params=[
        (1, 1),  # (TP=1, CP=1)
        (2, 1),  # (TP=2, CP=1)
        (1, 2),  # (TP=1, CP=2)
        # (2, 2),  # Uncomment if you have 4 GPUs
    ]
)
def parallel_config(request):
    """Return tuple of (tensor_parallel_size, context_parallel_size)"""
    tp_size, cp_size = request.param
    # Calculate required GPU count
    # required_gpus = tp_size * cp_size

    # Skip this configuration if not enough GPUs
    # available_gpus = torch.cuda.device_count()
    # if required_gpus > available_gpus:
    #     pytest.skip(
    #         f"Skipping test with TP={tp_size}, CP={cp_size} - requires {required_gpus} GPUs, but only {available_gpus} available"
    #     )
    return request.param


@pytest.fixture
def mixer(hyena_test_config: HyenaTestConfig, hyena_config: HyenaConfig, parallel_config):
    """Create a HyenaMixer instance for testing with standard config"""
    tp_size, cp_size = parallel_config
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=tp_size * cp_size, tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    ):
        # Create necessary submodules - use the mixer submodules like in the regular mixer fixture
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

        yield HyenaMixer(
            transformer_config=hyena_test_config,
            hyena_config=hyena_config,
            max_sequence_length=512,
            submodules=submodules,
            layer_number=1,
            operator_type="hyena_short_conv",
            use_b2b_causal_conv1d=True,
        )


@pytest.fixture
def nv_mixer(hyena_nv_test_config: HyenaNVTestConfig, hyena_config: HyenaConfig, parallel_config):
    """Create a HyenaMixer instance for testing with NV config"""
    tp_size, cp_size = parallel_config
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=tp_size * cp_size, tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    ):
        # Create necessary submodules - use the mixer submodules like in the regular mixer fixture
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

        yield HyenaMixer(
            transformer_config=hyena_nv_test_config,
            hyena_config=hyena_config,
            max_sequence_length=512,
            submodules=submodules,
            layer_number=1,
            operator_type="hyena_short_conv",
            use_b2b_causal_conv1d=True,
        )


def b2b_torch_forward(mixer: HyenaMixer, features: torch.Tensor, _proj_use_cp: bool = False):
    features = mixer.hyena_proj_conv(features, _use_cp=_proj_use_cp)  # [B, D, L]
    x1, x2, v = rearrange(features, "b (g dg p) l -> b (g dg) p l", p=3, g=mixer.num_groups_per_tp_rank).unbind(dim=2)
    z = mixer.mixer(x1, x2, v)
    return z


def test_b2b_causal_conv1d(mixer: HyenaMixer, parallel_config):
    """Test the B2B causal conv1d layer"""
    tp_size, cp_size = parallel_config

    assert hasattr(mixer, "hyena_proj_conv")
    assert hasattr(mixer, "mixer")

    # Scale input features based on TP size
    hidden_size_per_tp = mixer.hidden_size // mixer.model_parallel_size

    # Choose whether to use CP based on the parallel configuration
    use_cp = cp_size > 1

    # For CP, we need to adjust sequence length in some cases
    seq_len = 512
    if use_cp:
        # When using CP, make sure sequence length is divisible by 2*CP for zigzag splitting
        seq_len = ((seq_len + (2 * cp_size - 1)) // (2 * cp_size)) * (2 * cp_size)

    input_features = torch.rand(
        (2, hidden_size_per_tp * 3, seq_len),
        dtype=mixer.transformer_config.params_dtype,
        device=mixer.hyena_proj_conv.short_conv_weight.device,
    )

    output_features_b2b_torch = b2b_torch_forward(mixer, input_features, _proj_use_cp=use_cp)

    assert hasattr(mixer, "b2b_kernel")
    output_features_b2b = mixer.b2b_kernel(input_features, _use_cp=use_cp)

    # Compare with stored expected output using parametrized tolerance
    assert torch.allclose(output_features_b2b, output_features_b2b_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.skip(reason="NV config (with conv bias) is not supported by b2b CUDA kernel yet")
def test_nv_b2b_causal_conv1d(nv_mixer: HyenaMixer, parallel_config):
    """Test the B2B causal conv1d layer with NV config"""
    tp_size, cp_size = parallel_config

    assert hasattr(nv_mixer, "hyena_proj_conv")
    assert hasattr(nv_mixer, "mixer")

    # Scale input features based on TP size
    hidden_size_per_tp = nv_mixer.hidden_size // nv_mixer.model_parallel_size

    # Choose whether to use CP based on the parallel configuration
    use_cp = cp_size > 1

    # For CP, we need to adjust sequence length in some cases
    seq_len = 512
    if use_cp:
        # When using CP, make sure sequence length is divisible by 2*CP for zigzag splitting
        seq_len = ((seq_len + (2 * cp_size - 1)) // (2 * cp_size)) * (2 * cp_size)

    input_features = torch.rand(
        (2, hidden_size_per_tp * 3, seq_len),
        dtype=nv_mixer.transformer_config.params_dtype,
        device=nv_mixer.hyena_proj_conv.short_conv_weight.device,
    )

    output_features_b2b_torch = b2b_torch_forward(nv_mixer, input_features, _proj_use_cp=use_cp)

    assert hasattr(nv_mixer, "b2b_kernel")
    output_features_b2b = nv_mixer.b2b_kernel(input_features, _use_cp=use_cp)

    # Compare with stored expected output using parametrized tolerance
    assert torch.allclose(output_features_b2b, output_features_b2b_torch, rtol=1e-2, atol=1e-2)


def analyze_gradient_stats(grad_tensor, name="Gradient"):
    """Analyze gradient statistics and return a dictionary of metrics."""
    if grad_tensor is None:
        print(f"Warning: {name} tensor is None")
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

    return {
        "mean": grad_tensor.mean().item(),
        "std": grad_tensor.std().item(),
        "max": grad_tensor.max().item(),
        "min": grad_tensor.min().item(),
    }


def analyze_boundary_gradients(grad_tensor, chunk_size, window=10):
    """Analyze gradients at boundaries and return a list of boundary statistics."""
    if grad_tensor is None:
        print("Warning: Gradient tensor is None, can't analyze boundaries")
        return []

    boundaries = []

    for i in range(1, 4):  # 3 boundaries for 4 chunks with CP=2
        boundary = i * chunk_size

        # Skip if boundary is out of range
        if boundary >= grad_tensor.shape[-1] or boundary < window:
            continue

        region = grad_tensor[..., boundary - window : boundary + window]
        before = grad_tensor[..., boundary - window : boundary]
        after = grad_tensor[..., boundary + 1 : boundary + window + 1]

        # Check for gradient jumps
        jump = torch.norm(grad_tensor[..., boundary] - grad_tensor[..., boundary - 1]).item()

        boundaries.append(
            {
                "boundary": boundary,
                "region_mean": region.mean().item(),
                "region_std": region.std().item(),
                "before_mean": before.mean().item(),
                "after_mean": after.mean().item(),
                "jump": jump,
            }
        )

    return boundaries


def analyze_gradient_flow(grad_tensor, boundaries, window=20):
    """Analyze gradient flow across boundaries."""
    if grad_tensor is None:
        print("Warning: Gradient tensor is None, can't analyze flow")
        return []

    flow_stats = []

    for b in boundaries:
        boundary = b["boundary"]

        # Skip if boundary is out of range
        if boundary >= grad_tensor.shape[-1] or boundary < window:
            continue

        region = grad_tensor[..., boundary - window : boundary + window]

        # Calculate correlation
        def calc_correlation(tensor, offset=1):
            if tensor is None:
                return 0.0
            x1 = tensor[..., :-offset].flatten()
            x2 = tensor[..., offset:].flatten()
            # Remove NaN values
            mask = ~(torch.isnan(x1) | torch.isnan(x2))
            if mask.sum() == 0:
                return float("nan")
            x1, x2 = x1[mask], x2[mask]
            # Normalize
            x1 = (x1 - x1.mean()) / (x1.std() + 1e-8)
            x2 = (x2 - x2.mean()) / (x2.std() + 1e-8)
            return (x1 * x2).mean().item()

        correlation = calc_correlation(region)

        # Check for discontinuity at boundary
        boundary_idx = window  # middle of our extracted region
        jump = torch.norm(region[..., boundary_idx] - region[..., boundary_idx - 1]).item()

        flow_stats.append({"boundary": boundary, "correlation": correlation, "jump": jump})

    return flow_stats


def compare_gradients(grad1, grad2, name1="CP=1", name2="CP=2", rtol=1e-2, atol=1e-2):
    """Compare two gradient tensors and print detailed analysis."""
    if grad1 is None or grad2 is None:
        print(f"Warning: Cannot compare gradients, one is None ({name1}: {grad1 is None}, {name2}: {grad2 is None})")
        return False, 0.0

    # Check if gradients are close
    grad_close = torch.allclose(grad1, grad2, rtol=rtol, atol=atol)

    # Calculate difference statistics
    grad_diff = torch.abs(grad1 - grad2)
    max_diff = torch.max(grad_diff).item()
    mean_diff = torch.mean(grad_diff).item()

    # Calculate relative difference
    grad_distribution_diff = torch.norm(grad1 - grad2).item() / (torch.norm(grad1).item() + 1e-10)

    print(f"\nGradient Comparison ({name1} vs {name2}):")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Relative difference: {grad_distribution_diff:.6f} ({grad_distribution_diff * 100:.2f}%)")

    # If gradients are not close, analyze the differences
    if not grad_close:
        # Find largest differences
        flat_indices = torch.topk(grad_diff.view(-1), 5).indices
        print(f"\nTop 5 gradient differences ({name1} vs {name2}):")

        for idx in flat_indices:
            idx_item = idx.item()
            # Convert flat index to 3D position
            total_elems_per_batch = grad_diff.shape[1] * grad_diff.shape[2]
            batch_idx = int(idx_item // total_elems_per_batch)
            remainder = int(idx_item % total_elems_per_batch)
            channel_idx = int(remainder // grad_diff.shape[2])
            seq_idx = int(remainder % grad_diff.shape[2])

            # Get the values at this position - properly handle indexing
            val1 = grad1[batch_idx, channel_idx, seq_idx].item()
            val2 = grad2[batch_idx, channel_idx, seq_idx].item()
            diff_val = grad_diff[batch_idx, channel_idx, seq_idx].item()

            print(
                f"  [b={batch_idx}, c={channel_idx}, seq={seq_idx}]: "
                f"{name1}={val1:.6f}, {name2}={val2:.6f}, diff={diff_val:.6f}"
            )

    return grad_close, grad_distribution_diff


def test_baseline_backward(mixer: HyenaMixer, parallel_config):
    """Test that the baseline PyTorch implementation gives consistent results between CP=1 and CP=2.

    This test specifically targets the gradient consistency of the PyTorch implementation
    when run with different CP settings. The goal is to establish whether there are any
    baseline inconsistencies that could explain the gradient differences observed when
    comparing PyTorch and CUDA implementations.
    """
    tp_size, cp_size = parallel_config

    # Skip if we're not in CP=2 setting
    if cp_size != 2:
        pytest.skip("This test only runs with CP=2 to compare with CP=1 reference results")

    # Scale input features based on TP size
    hidden_size_per_tp = mixer.hidden_size // mixer.model_parallel_size

    # Create sequence length adjusted for CP=2 (ensure divisible by 2*CP)
    seq_len_cp2 = 512
    seq_len_cp2 = ((seq_len_cp2 + (2 * 2 - 1)) // (2 * 2)) * (2 * 2)

    # Create input features for both CP settings with matching data
    torch.manual_seed(42)  # Ensure reproducible randomization

    input_features_cp2 = torch.rand(
        (2, hidden_size_per_tp * 3, seq_len_cp2),
        dtype=mixer.transformer_config.params_dtype,
        device=mixer.hyena_proj_conv.short_conv_weight.device,
        requires_grad=True,
    )

    # Clone the input to ensure identical starting point
    input_features_cp1 = input_features_cp2.clone().detach().requires_grad_(True)

    # Run the CP=2 forward and backward pass
    output_cp2 = b2b_torch_forward(mixer, input_features_cp2, _proj_use_cp=True)
    loss_cp2 = output_cp2.sum()
    loss_cp2.backward()
    grad_cp2 = input_features_cp2.grad

    # Store key statistics about CP=2 results
    cp2_stats = analyze_gradient_stats(grad_cp2, "CP=2 gradient")

    # Now analyze the gradients at CP boundaries for CP=2
    chunk_size = seq_len_cp2 // 4  # For CP=2 with zigzag, we have 4 chunks
    cp2_boundaries = analyze_boundary_gradients(grad_cp2, chunk_size)

    # Analyze gradient flow across boundaries
    cp2_flow = analyze_gradient_flow(grad_cp2, cp2_boundaries)

    # Clear gradients for next test
    input_features_cp2.grad = None

    # Print the CP=2 analysis
    print("\n==== CP=2 Baseline Results ====")
    print(f"Forward output: mean={output_cp2.mean().item():.6f}, std={output_cp2.std().item():.6f}")
    print(f"Gradient: mean={cp2_stats['mean']:.6f}, std={cp2_stats['std']:.6f}")
    print(f"Gradient range: [{cp2_stats['min']:.6f}, {cp2_stats['max']:.6f}]")

    print("\nCP=2 Boundary Analysis:")
    for b in cp2_boundaries:
        print(f"Boundary at {b['boundary']}: jump={b['jump']:.6f}")
        print(f"  Before mean: {b['before_mean']:.6f}, After mean: {b['after_mean']:.6f}")

    print("\nCP=2 Gradient Flow Analysis:")
    for f in cp2_flow:
        print(f"Boundary at {f['boundary']}: correlation={f['correlation']:.6f}, jump={f['jump']:.6f}")

    # ==== Now run with CP=1 for comparison ====

    # Temporarily wrap in a context that mocks CP=1
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=tp_size * 1,  # Force CP=1
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,  # Explicit CP=1
    ):
        # Forward and backward pass with CP=1
        output_cp1 = b2b_torch_forward(mixer, input_features_cp1, _proj_use_cp=False)  # No CP in this path
        loss_cp1 = output_cp1.sum()
        loss_cp1.backward()
        grad_cp1 = input_features_cp1.grad

    # Store key statistics about CP=1 results
    cp1_stats = analyze_gradient_stats(grad_cp1, "CP=1 gradient")

    # Now analyze the same boundary points in CP=1 results
    cp1_boundaries = analyze_boundary_gradients(grad_cp1, chunk_size)

    # Analyze gradient flow across boundaries
    cp1_flow = analyze_gradient_flow(grad_cp1, cp1_boundaries)

    # Print the CP=1 analysis
    print("\n==== CP=1 Reference Results ====")
    print(f"Forward output: mean={output_cp1.mean().item():.6f}, std={output_cp1.std().item():.6f}")
    print(f"Gradient: mean={cp1_stats['mean']:.6f}, std={cp1_stats['std']:.6f}")
    print(f"Gradient range: [{cp1_stats['min']:.6f}, {cp1_stats['max']:.6f}]")

    print("\nCP=1 Boundary Analysis (same positions):")
    for b in cp1_boundaries:
        print(f"Position {b['boundary']}: jump={b['jump']:.6f}")
        print(f"  Before mean: {b['before_mean']:.6f}, After mean: {b['after_mean']:.6f}")

    print("\nCP=1 Gradient Flow Analysis:")
    for f in cp1_flow:
        print(f"Boundary at {f['boundary']}: correlation={f['correlation']:.6f}, jump={f['jump']:.6f}")

    # Compare results between CP=1 and CP=2
    print("\n==== Comparison CP=2 vs CP=1 ====")
    print(f"Forward mean diff: {abs(output_cp2.mean().item() - output_cp1.mean().item()):.6f}")

    # Compare boundary jumps
    print("\nBoundary Jump Comparison:")
    for i, (b2, b1) in enumerate(zip(cp2_boundaries, cp1_boundaries)):
        if b2 and b1:  # Make sure both boundaries exist
            pos = b2["boundary"]
            print(
                f"Position {pos}: CP=2 jump={b2['jump']:.6f}, CP=1 jump={b1['jump']:.6f}, "
                f"Ratio: {b2['jump'] / max(b1['jump'], 1e-10):.2f}x"
            )

    # Compare gradient flow
    print("\nGradient Flow Comparison:")
    for i, (f2, f1) in enumerate(zip(cp2_flow, cp1_flow)):
        if f2 and f1:  # Make sure both flow stats exist
            pos = f2["boundary"]
            print(
                f"Position {pos}: CP=2 correlation={f2['correlation']:.6f}, CP=1 correlation={f1['correlation']:.6f}"
            )

    # Validate results - forward outputs should be similar
    forward_tol = 1e-4
    assert torch.allclose(output_cp1, output_cp2, rtol=forward_tol, atol=forward_tol), (
        f"Forward outputs differ between CP=1 and CP=2: {torch.max(torch.abs(output_cp1 - output_cp2)):.6f} max diff"
    )

    # Compare gradient distributions
    grad_close, grad_dist_diff = compare_gradients(grad_cp1, grad_cp2, "CP=1", "CP=2", rtol=1e-2, atol=1e-2)

    # Final assertion checks if gradient distributions are similar enough
    assert grad_dist_diff < 0.1, (
        f"Gradient distributions differ significantly between CP=1 and CP=2: {grad_dist_diff * 100:.2f}% difference"
    )


def test_b2b_causal_conv1d_backward(mixer: HyenaMixer, parallel_config):
    """Test the backward pass of B2B causal conv1d layer against PyTorch implementation"""
    tp_size, cp_size = parallel_config

    assert hasattr(mixer, "hyena_proj_conv")
    assert hasattr(mixer, "mixer")
    assert hasattr(mixer, "b2b_kernel")

    # Scale input features based on TP size
    hidden_size_per_tp = mixer.hidden_size // mixer.model_parallel_size

    # Use CP based on the parallel configuration
    use_cp = cp_size > 1

    # For CP, we need to adjust sequence length in some cases
    seq_len = 512
    if use_cp:
        # When using CP, make sure sequence length is divisible by 2*CP for zigzag splitting
        seq_len = ((seq_len + (2 * cp_size - 1)) // (2 * cp_size)) * (2 * cp_size)

    chunk_size = seq_len // (2 * cp_size) if use_cp else seq_len

    # Create input features that require gradients
    input_features_torch = torch.rand(
        (2, hidden_size_per_tp * 3, seq_len),
        dtype=mixer.transformer_config.params_dtype,
        device=mixer.hyena_proj_conv.short_conv_weight.device,
        requires_grad=True,
    )
    input_features_kernel = input_features_torch.clone().detach().requires_grad_(True)

    if use_cp:
        # Debug CP boundary regions in input
        print("\nAnalyzing input features at CP boundaries:")
        for i in range(1, 2 * cp_size):
            boundary = i * chunk_size
            region = input_features_torch[..., boundary - 5 : boundary + 5]
            print(f"\nCP Boundary {i} (pos {boundary}) input stats:")
            print(f"Mean: {region.mean():.6f}, Std: {region.std():.6f}")
            print(f"Max: {region.max():.6f}, Min: {region.min():.6f}")

    # Forward pass with PyTorch implementation
    output_torch = b2b_torch_forward(mixer, input_features_torch, _proj_use_cp=use_cp)

    # Forward pass with CUDA kernel implementation
    output_kernel = mixer.b2b_kernel(input_features_kernel, _use_cp=use_cp)

    # Verify forward pass outputs match
    forward_match = torch.allclose(output_kernel, output_torch, rtol=1e-2, atol=1e-2)
    if not forward_match:
        # Print max difference in forward pass
        max_diff = torch.max(torch.abs(output_kernel - output_torch))
        print(f"Forward pass max difference: {max_diff}")

        # Analyze differences at chunk boundaries
        if use_cp and cp_size > 1:
            for i in range(1, 2 * cp_size):
                boundary = i * chunk_size
                window = 10
                region_torch = output_torch[..., boundary - window : boundary + window]
                region_kernel = output_kernel[..., boundary - window : boundary + window]

                print(f"\nForward pass at boundary {i} (pos {boundary}):")
                boundary_diff = torch.abs(region_torch - region_kernel)
                print(f"Max diff: {torch.max(boundary_diff):.6f}")
                print(f"Mean diff: {torch.mean(boundary_diff):.6f}")

    assert forward_match, "Forward pass outputs don't match"

    # Create a dummy loss for backpropagation (sum of outputs)
    loss_torch = output_torch.sum()
    loss_kernel = output_kernel.sum()

    # Backward pass
    loss_torch.backward()
    loss_kernel.backward()

    # Verify the gradients match
    assert input_features_torch.grad is not None, "PyTorch implementation didn't produce gradients"
    assert input_features_kernel.grad is not None, "CUDA kernel implementation didn't produce gradients"

    # Get gradient tensors
    grad_torch = input_features_torch.grad
    grad_kernel = input_features_kernel.grad

    # Store key statistics about gradients
    torch_stats = analyze_gradient_stats(grad_torch, "PyTorch gradient")
    kernel_stats = analyze_gradient_stats(grad_kernel, "CUDA kernel gradient")

    # Print gradient statistics
    print("\n==== Gradient Statistics ====")
    print(f"PyTorch grad: mean={torch_stats['mean']:.6f}, std={torch_stats['std']:.6f}")
    print(f"PyTorch grad range: [{torch_stats['min']:.6f}, {torch_stats['max']:.6f}]")
    print(f"CUDA kernel grad: mean={kernel_stats['mean']:.6f}, std={kernel_stats['std']:.6f}")
    print(f"CUDA kernel grad range: [{kernel_stats['min']:.6f}, {kernel_stats['max']:.6f}]")

    # Analyze boundaries for both implementations
    if use_cp and cp_size > 1:
        # Analyze boundary gradients
        torch_boundaries = analyze_boundary_gradients(grad_torch, chunk_size)
        kernel_boundaries = analyze_boundary_gradients(grad_kernel, chunk_size)

        # Analyze gradient flow
        torch_flow = analyze_gradient_flow(grad_torch, torch_boundaries)
        kernel_flow = analyze_gradient_flow(grad_kernel, kernel_boundaries)

        # Print boundary analysis
        print("\n==== Boundary Gradient Analysis ====")
        for i, (b_torch, b_kernel) in enumerate(zip(torch_boundaries, kernel_boundaries)):
            if b_torch and b_kernel:
                pos = b_torch["boundary"]
                print(f"\nBoundary at position {pos}:")
                print(f"  PyTorch jump: {b_torch['jump']:.6f}")
                print(f"  CUDA kernel jump: {b_kernel['jump']:.6f}")
                print(f"  Jump ratio: {b_torch['jump'] / max(b_kernel['jump'], 1e-10):.2f}x")
                print(f"  PyTorch before/after: {b_torch['before_mean']:.6f}/{b_torch['after_mean']:.6f}")
                print(f"  CUDA kernel before/after: {b_kernel['before_mean']:.6f}/{b_kernel['after_mean']:.6f}")

        # Print flow analysis
        print("\n==== Gradient Flow Analysis ====")
        for i, (f_torch, f_kernel) in enumerate(zip(torch_flow, kernel_flow)):
            if f_torch and f_kernel:
                pos = f_torch["boundary"]
                print(f"\nBoundary at position {pos}:")
                print(f"  PyTorch correlation: {f_torch['correlation']:.6f}")
                print(f"  CUDA kernel correlation: {f_kernel['correlation']:.6f}")
                print(f"  PyTorch jump: {f_torch['jump']:.6f}")
                print(f"  CUDA kernel jump: {f_kernel['jump']:.6f}")

    # Compare gradients using dedicated function
    grad_close, grad_dist_diff = compare_gradients(
        grad_torch, grad_kernel, "PyTorch", "CUDA kernel", rtol=1e-2, atol=1e-2
    )

    # Use relaxed tolerances for CP > 1
    rtol = 1e-2
    atol = 1e-2
    if use_cp and cp_size > 1:
        rtol = 1e-1
        atol = 1e-1

    # Final assertion with appropriate tolerance
    assert torch.allclose(
        grad_torch,
        grad_kernel,
        rtol=rtol,
        atol=atol,
    ), f"Gradients do not match! Difference: {grad_dist_diff * 100:.2f}%"
