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

"""Tests for ESM2-MiniFold TE model components.

Tests model instantiation, forward pass shapes, and gradient flow.
Uses small model dimensions for fast testing.
"""

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from heads_te import PerResidueLDDTCaPredictorTE
from miniformer_te import BlockTE, MiniFormerTE, TransitionUpdateTE, TriangularUpdateTE
from model_te import FoldingTrunkTE, PairToSequenceTE, SequenceToPairTE
from precision_config import FoldingHeadPrecisionConfig
from structure_te import MLPTE, AngleResnetTE, AttentionTE
from te_utils import te_linear_nd


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, N, DIM = 2, 16, 64
HIDDEN = 256


# ===========================================================================
# TE Module Shape Tests
# ===========================================================================


class TestTransitionUpdateTE:
    def test_forward_shape(self):
        mod = TransitionUpdateTE(dim=DIM, hidden=HIDDEN).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        out = mod(x)
        assert out.shape == (B, N, N, DIM)

    def test_gradient_flow(self):
        mod = TransitionUpdateTE(dim=DIM, hidden=HIDDEN).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None


class TestTriangularUpdateTE:
    def test_forward_shape(self):
        mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)


class TestBlockTE:
    def test_forward_shape(self):
        mod = BlockTE(dim=DIM).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)


class TestMiniFormerTE:
    def test_forward_shape(self):
        mod = MiniFormerTE(dim=DIM, blocks=2).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)


class TestSequenceToPairTE:
    def test_forward_shape(self):
        seq_dim, inner, pair_dim = 512, 32, DIM
        mod = SequenceToPairTE(seq_dim, inner, pair_dim).to(DEVICE)
        x = torch.randn(B, N, seq_dim, device=DEVICE)
        out = mod(x)
        assert out.shape == (B, N, N, pair_dim)


class TestPairToSequenceTE:
    def test_forward_shape(self):
        c_z, c_s, c_s_out = DIM, 512, 512
        mod = PairToSequenceTE(c_z=c_z, c_s=c_s, c_s_out=c_s_out).to(DEVICE)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE)
        s_s = torch.randn(B, N, c_s, device=DEVICE)
        pair_mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(s_z, s_s, pair_mask)
        assert out.shape == (B, N, c_s_out)


class TestFoldingTrunkTE:
    def test_forward_shape(self):
        c_s, c_z = 512, DIM
        mod = FoldingTrunkTE(c_s=c_s, c_z=c_z, bins=32, disto_bins=64, num_layers=2).to(DEVICE)
        s_s = torch.randn(B, N, c_s, device=DEVICE)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)

        mod.eval()
        with torch.no_grad():
            preds, sz = mod(s_s, s_z, mask, num_recycling=0)

        assert preds.shape == (B, N, N, 64)
        assert sz.shape[:3] == (B, N, N)


class TestAttentionTE:
    def test_forward_shape(self):
        dim, heads, head_width = 512, 8, 64
        mod = AttentionTE(dim, heads, head_width).to(DEVICE)
        x = torch.randn(B, N, dim, device=DEVICE)
        bias = torch.randn(B, heads, N, N, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)
        out = mod(x, bias, mask)
        assert out.shape == (B, N, dim)


class TestMLPTE:
    def test_forward_shape(self):
        mod = MLPTE(512, 512).to(DEVICE)
        x = torch.randn(B, N, 512, device=DEVICE)
        out = mod(x)
        assert out.shape == (B, N, 512)


class TestAngleResnetTE:
    def test_forward_shape(self):
        mod = AngleResnetTE(512, DIM, no_blocks=2, no_angles=7, epsilon=1e-5).to(DEVICE)
        s = torch.randn(B, N, 512, device=DEVICE)
        s_init = torch.randn(B, N, 512, device=DEVICE)
        unnorm, norm = mod(s, s_init)
        assert unnorm.shape == (B, N, 7, 2)
        assert norm.shape == (B, N, 7, 2)


class TestPerResidueLDDTCaPredictorTE:
    def test_forward_shape(self):
        mod = PerResidueLDDTCaPredictorTE(50, 512, DIM).to(DEVICE)
        x = torch.randn(B, N, 512, device=DEVICE)
        out = mod(x)
        assert out.shape == (B, N, 50)


# ===========================================================================
# Precision Config Tests
# ===========================================================================


class TestPrecisionConfig:
    def test_disabled_by_default(self):
        config = FoldingHeadPrecisionConfig()
        assert not config.enabled
        assert config.get_enabled_groups() == []

    def test_selective_enable(self):
        config = FoldingHeadPrecisionConfig(enabled=True, ffn=True, tri_proj=True)
        assert config.is_enabled("ffn")
        assert config.is_enabled("tri_proj")
        assert not config.is_enabled("tri_gate")
        assert set(config.get_enabled_groups()) == {"ffn", "tri_proj"}

    def test_summary(self):
        config = FoldingHeadPrecisionConfig(enabled=True, ffn=True)
        assert "ffn" in config.summary()


# ===========================================================================
# te_utils Tests
# ===========================================================================


class TestTeUtils:
    def test_te_linear_nd_2d(self):
        import transformer_engine.pytorch as te

        linear = te.Linear(DIM, HIDDEN).to(DEVICE)
        x = torch.randn(B * N, DIM, device=DEVICE)
        out = te_linear_nd(linear, x)
        assert out.shape == (B * N, HIDDEN)

    def test_te_linear_nd_3d(self):
        import transformer_engine.pytorch as te

        linear = te.Linear(DIM, HIDDEN).to(DEVICE)
        x = torch.randn(B, N, DIM, device=DEVICE)
        out = te_linear_nd(linear, x)
        assert out.shape == (B, N, HIDDEN)

    def test_te_linear_nd_4d(self):
        import transformer_engine.pytorch as te

        linear = te.Linear(DIM, HIDDEN).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        out = te_linear_nd(linear, x)
        assert out.shape == (B, N, N, HIDDEN)
