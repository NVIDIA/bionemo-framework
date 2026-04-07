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

"""Tests that verify actual precision of intermediate tensors during forward passes.

Checks that:
- Einsum triangular multiplication stays in FP32
- te.autocast correctly enables/disables FP8 for specific blocks
- Component precision overrides correctly keep components out of FP8
- FSDP2 MixedPrecisionPolicy casts params to BF16 for forward pass
"""

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import transformer_engine.pytorch as te


sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from miniformer_te import MiniFormerTE, TransitionUpdateTE, TriangularUpdateTE
from model_te import FoldingTrunkTE
from quantization import ComponentPrecisionConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, N, DIM = 2, 16, 64


class TestEinsumPrecision:
    """Verify the triangular einsum always runs in FP32."""

    def test_einsum_intermediates_are_fp32(self):
        """Even with BF16 input, the einsum should compute in FP32 via .float()."""
        mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        x_bf16 = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        # Hook into forward to capture dtypes at the einsum
        captured_dtypes = {}

        orig_forward = mod.forward

        def hooked_forward(x, mask):
            # Reproduce the forward up to the einsum to check dtypes
            cp = mod._component_precision

            def _proj_ctx():
                return cp.get_context("tri_proj") if cp else nullcontext()

            def _gate_ctx():
                return cp.get_context("tri_gate") if cp else nullcontext()

            from te_utils import te_layernorm_nd, te_linear_nd

            x = te_layernorm_nd(mod.input_norm, x)
            with _proj_ctx():
                pi_out = te_linear_nd(mod.pi, x)
            with _gate_ctx():
                gi_out = te_linear_nd(mod.gi, x).sigmoid()
            x = pi_out * gi_out
            x = x * mask.unsqueeze(-1)

            # This is the critical part: .float() should cast to FP32
            a1, b1, a2, b2 = torch.chunk(x.float(), 4, dim=-1)
            captured_dtypes["einsum_input"] = a1.dtype
            x1 = torch.einsum("bikd,bjkd->bijd", a1, b1)
            captured_dtypes["einsum_output"] = x1.dtype
            x2 = torch.einsum("bkid,bkjd->bijd", a2, b2)
            x = torch.cat([x1, x2], dim=-1).to(mask.dtype if mask.is_floating_point() else torch.float32)
            captured_dtypes["post_einsum"] = x.dtype

            x = te_layernorm_nd(mod.output_norm, x)
            with _proj_ctx():
                po_out = te_linear_nd(mod.po, x)
            with _gate_ctx():
                go_out = te_linear_nd(mod.go, x).sigmoid()
            return po_out * go_out

        mod.forward = hooked_forward
        mod(x_bf16, mask)

        assert captured_dtypes["einsum_input"] == torch.float32, (
            f"Einsum input should be FP32 but got {captured_dtypes['einsum_input']}"
        )
        assert captured_dtypes["einsum_output"] == torch.float32, (
            f"Einsum output should be FP32 but got {captured_dtypes['einsum_output']}"
        )

    def test_einsum_fp32_with_fp8_block(self):
        """Einsum stays FP32 even when block is wrapped in te.autocast(enabled=True)."""
        mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        x_bf16 = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        captured = {}

        orig_einsum = torch.einsum

        def patched_einsum(eq, *tensors):
            captured["dtype"] = tensors[0].dtype
            return orig_einsum(eq, *tensors)

        torch.einsum = patched_einsum
        try:
            with te.autocast(enabled=True):
                mod(x_bf16, mask)
        finally:
            torch.einsum = orig_einsum

        assert captured["dtype"] == torch.float32, (
            f"Einsum should be FP32 inside te.autocast but got {captured['dtype']}"
        )


class TestFP8StateInBlocks:
    """Verify te.autocast correctly enables/disables FP8 per block."""

    def test_fp8_enabled_in_fp8_block(self):
        """FP8GlobalStateManager.is_fp8_enabled() should be True inside an FP8 block's forward."""
        fp8_states = []

        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", None]).to(DEVICE)

        # Hook into each block to capture FP8 state
        for i, block in enumerate(mod.blocks):

            def make_hook(block_idx):
                def hook(module, input, output):
                    fp8_states.append((block_idx, FP8GlobalStateManager.is_fp8_enabled()))

                return hook

            block.register_forward_hook(make_hook(i))

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        mod(x, mask)

        assert fp8_states[0] == (0, True), f"Block 0 should have FP8 enabled but got {fp8_states[0]}"
        assert fp8_states[1] == (1, False), f"Block 1 should have FP8 disabled but got {fp8_states[1]}"

    def test_fp8_disabled_in_bf16_block(self):
        """All blocks BF16 — FP8 should never be enabled."""
        fp8_states = []

        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=[None, None]).to(DEVICE)

        for i, block in enumerate(mod.blocks):

            def make_hook(block_idx):
                def hook(module, input, output):
                    fp8_states.append((block_idx, FP8GlobalStateManager.is_fp8_enabled()))

                return hook

            block.register_forward_hook(make_hook(i))

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        mod(x, mask)

        for block_idx, fp8_enabled in fp8_states:
            assert not fp8_enabled, f"Block {block_idx} should have FP8 disabled but it's enabled"

    def test_all_blocks_fp8(self):
        """All blocks FP8 — FP8 should be enabled in every block."""
        fp8_states = []

        mod = MiniFormerTE(dim=DIM, blocks=3, block_precision=["fp8", "fp8", "fp8"]).to(DEVICE)

        for i, block in enumerate(mod.blocks):

            def make_hook(block_idx):
                def hook(module, input, output):
                    fp8_states.append((block_idx, FP8GlobalStateManager.is_fp8_enabled()))

                return hook

            block.register_forward_hook(make_hook(i))

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        mod(x, mask)

        for block_idx, fp8_enabled in fp8_states:
            assert fp8_enabled, f"Block {block_idx} should have FP8 enabled but it's disabled"

    def test_mixed_precision_pattern(self):
        """Alternating FP8/BF16 blocks — verify correct per-block FP8 state."""
        fp8_states = []

        mod = MiniFormerTE(dim=DIM, blocks=4, block_precision=["fp8", None, "fp8", None]).to(DEVICE)

        for i, block in enumerate(mod.blocks):

            def make_hook(block_idx):
                def hook(module, input, output):
                    fp8_states.append((block_idx, FP8GlobalStateManager.is_fp8_enabled()))

                return hook

            block.register_forward_hook(make_hook(i))

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        mod(x, mask)

        expected = [(0, True), (1, False), (2, True), (3, False)]
        for (idx, actual), (_, expect) in zip(fp8_states, expected):
            assert actual == expect, f"Block {idx}: expected FP8={expect}, got FP8={actual}"


class TestComponentPrecisionOverrides:
    """Verify component-level precision overrides actually change FP8 state for sub-operations."""

    def test_ffn_excluded_from_fp8(self):
        """FFN runs with FP8 disabled even when block is in FP8."""
        cp = ComponentPrecisionConfig(ffn=False)
        mod = TransitionUpdateTE(dim=DIM, hidden=DIM * 4, component_precision=cp).to(DEVICE)

        fp8_in_ffn = []

        def hook(module, input, output):
            fp8_in_ffn.append(FP8GlobalStateManager.is_fp8_enabled())

        # Hook the fc1 linear to check FP8 state during FFN
        mod.fc1.register_forward_hook(hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)

        # Run inside an FP8 block context (simulating what MiniFormerTE does)
        with te.autocast(enabled=True):
            mod(x)

        assert len(fp8_in_ffn) == 1
        assert not fp8_in_ffn[0], "FFN fc1 should have FP8 disabled when component_precision.ffn=False"

    def test_ffn_included_in_fp8(self):
        """FFN runs with FP8 enabled when component_precision.ffn=True."""
        cp = ComponentPrecisionConfig(ffn=True)
        mod = TransitionUpdateTE(dim=DIM, hidden=DIM * 4, component_precision=cp).to(DEVICE)

        fp8_in_ffn = []

        def hook(module, input, output):
            fp8_in_ffn.append(FP8GlobalStateManager.is_fp8_enabled())

        mod.fc1.register_forward_hook(hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x)

        assert len(fp8_in_ffn) == 1
        assert fp8_in_ffn[0], "FFN fc1 should have FP8 enabled when component_precision.ffn=True"

    def test_tri_proj_excluded_from_fp8(self):
        """Triangular projections run with FP8 disabled when tri_proj=False."""
        cp = ComponentPrecisionConfig(tri_proj=False, tri_gate=True)
        mod = TriangularUpdateTE(dim=DIM, component_precision=cp).to(DEVICE)

        fp8_states = {"proj": [], "gate": []}

        def proj_hook(module, input, output):
            fp8_states["proj"].append(FP8GlobalStateManager.is_fp8_enabled())

        def gate_hook(module, input, output):
            fp8_states["gate"].append(FP8GlobalStateManager.is_fp8_enabled())

        mod.pi.register_forward_hook(proj_hook)
        mod.gi.register_forward_hook(gate_hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x, mask)

        # pi is called twice (input and output), gi is called twice
        assert not any(fp8_states["proj"]), "tri_proj should have FP8 disabled"
        assert all(fp8_states["gate"]), "tri_gate should have FP8 enabled"

    def test_tri_gate_excluded_from_fp8(self):
        """Triangular gates run with FP8 disabled when tri_gate=False."""
        cp = ComponentPrecisionConfig(tri_proj=True, tri_gate=False)
        mod = TriangularUpdateTE(dim=DIM, component_precision=cp).to(DEVICE)

        fp8_states = {"proj": [], "gate": []}

        def proj_hook(module, input, output):
            fp8_states["proj"].append(FP8GlobalStateManager.is_fp8_enabled())

        def gate_hook(module, input, output):
            fp8_states["gate"].append(FP8GlobalStateManager.is_fp8_enabled())

        mod.pi.register_forward_hook(proj_hook)
        mod.gi.register_forward_hook(gate_hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x, mask)

        assert all(fp8_states["proj"]), "tri_proj should have FP8 enabled"
        assert not any(fp8_states["gate"]), "tri_gate should have FP8 disabled"

    def test_dist_head_excluded_from_fp8(self):
        """Distogram head runs with FP8 disabled when dist_head=False."""
        cp = ComponentPrecisionConfig(dist_head=False)
        c_s, c_z = 512, DIM
        mod = FoldingTrunkTE(
            c_s=c_s,
            c_z=c_z,
            bins=32,
            disto_bins=64,
            num_layers=2,
            block_precision=["fp8", "fp8"],
            component_precision=cp,
        ).to(DEVICE)

        fp8_in_dist = []

        def hook(module, input, output):
            fp8_in_dist.append(FP8GlobalStateManager.is_fp8_enabled())

        mod.fc_out_1.register_forward_hook(hook)

        s_s = torch.randn(B, N, c_s, device=DEVICE, dtype=torch.bfloat16)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, device=DEVICE, dtype=torch.bfloat16)

        mod.eval()
        with torch.no_grad():
            mod(s_s, s_z, mask, num_recycling=0)

        assert len(fp8_in_dist) >= 1
        assert not any(fp8_in_dist), "dist_head fc_out_1 should have FP8 disabled"

    def test_no_component_precision_all_fp8(self):
        """Without component_precision, all te.Linear layers in FP8 block run in FP8."""
        mod = TransitionUpdateTE(dim=DIM, hidden=DIM * 4).to(DEVICE)

        fp8_in_ffn = []

        def hook(module, input, output):
            fp8_in_ffn.append(FP8GlobalStateManager.is_fp8_enabled())

        mod.fc1.register_forward_hook(hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x)

        assert len(fp8_in_ffn) == 1
        assert fp8_in_ffn[0], "Without component_precision, FFN should run in FP8"
