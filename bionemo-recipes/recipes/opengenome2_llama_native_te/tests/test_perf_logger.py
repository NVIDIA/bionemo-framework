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

"""Tests for the non-attention + Σ(Lᵢ²) attention FLOP formula.

Non-attention FLOPs are tracked per real token; attention FLOPs are tracked as
coeff * Σ(Lᵢ²) over per-doc real lengths. These tests lock in the formula and its
invariants so future drift between sibling recipes is caught immediately.
"""

import torch

from perf_logger import (
    _attn_work_from_batch,
    _compute_attn_flop_coeff,
    _compute_non_attn_per_token_flops,
)


def _llama_cfg():
    """Llama-like OG2 config used by the split-formula tests."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # GQA
        "intermediate_size": 14336,
        "vocab_size": 256,  # OG2's nucleotide vocab
    }


class TestFlopSplitAndAttention:
    """Verify the non-attn + Σ(Lᵢ²) attention formula is correctly computed."""

    def test_bshd_shape_synthesis(self):
        """BSHD batch (no cu_seq_lens) synthesizes Σ(Lᵢ²) = B·S² from input_ids shape."""
        b, s = 2, 512
        batch = {"input_ids": torch.zeros(b, s, dtype=torch.long)}
        sigma_l_sq = _attn_work_from_batch(batch, torch.device("cpu")).item()
        assert sigma_l_sq == b * s * s

    def test_thd_single_doc_matches_bshd(self):
        """cu_seq_lens_q=[0, S] (synthetic-single-doc) reproduces BSHD's Σ(Lᵢ²)=S²."""
        s = 512
        bshd = {"input_ids": torch.zeros(1, s, dtype=torch.long)}
        thd = {
            "input_ids": torch.zeros(1, s, dtype=torch.long),
            "cu_seq_lens_q": torch.tensor([0, s], dtype=torch.int32),
        }
        assert _attn_work_from_batch(bshd, torch.device("cpu")).item() == s * s
        assert _attn_work_from_batch(thd, torch.device("cpu")).item() == s * s

    def test_thd_multi_doc_uses_squared_sum(self):
        """Multi-doc pack computes Σ(Lᵢ²), not (ΣLᵢ)² — the whole point of the fix."""
        cu = torch.tensor([0, 3, 8, 15], dtype=torch.int32)
        batch = {"input_ids": torch.zeros(1, 15, dtype=torch.long), "cu_seq_lens_q": cu}
        work = _attn_work_from_batch(batch, torch.device("cpu")).item()
        assert work == 3**2 + 5**2 + 7**2  # 83 real QK pairs per layer
        assert work < 15 * 15  # old formula would have said 225

    def test_cp_size_divides_attention_only(self):
        """cp_size divides the attention term only; non-attention stays untouched."""
        cfg = _llama_cfg()
        non_attn_per_token = _compute_non_attn_per_token_flops(cfg)
        coeff = _compute_attn_flop_coeff(cfg)
        num_tokens, attn_work = 100, 10_000
        non_attn = non_attn_per_token * num_tokens
        flops_cp1 = non_attn + (coeff * attn_work) // 1
        flops_cp4 = non_attn + (coeff * attn_work) // 4
        assert flops_cp1 - non_attn == coeff * attn_work
        assert flops_cp4 - non_attn == (coeff * attn_work) // 4

    def test_unpadded_preferred_over_padded(self):
        """When both cu_seq_lens_q and cu_seq_lens_q_padded are present, _q wins."""
        batch = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "cu_seq_lens_q": torch.tensor([0, 5, 11], dtype=torch.int32),
            "cu_seq_lens_q_padded": torch.tensor([0, 8, 16], dtype=torch.int32),
        }
        work = _attn_work_from_batch(batch, torch.device("cpu")).item()
        assert work == 5**2 + 6**2  # 61 (unpadded doc lens 5 and 6)
        assert work != 8**2 + 8**2  # 128 (padded slot lens 8 and 8)

    def test_padded_fallback_when_unpadded_absent(self):
        """If only cu_seq_lens_q_padded is present, it is used as a fallback."""
        batch = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "cu_seq_lens_q_padded": torch.tensor([0, 8, 16], dtype=torch.int32),
        }
        assert _attn_work_from_batch(batch, torch.device("cpu")).item() == 8**2 + 8**2

    def test_bshd_cp_correction(self):
        """BSHD with CP: per-rank shape (B, S/cp) → helper must return global B*S².

        ContextParallelDataLoaderWrapper pre-splits the sequence so each rank's
        input_ids.shape is (B, S/cp), not (B, S). The helper returns a GLOBAL
        quantity (the caller divides by cp_size), so the BSHD synthesis branch
        must multiply per-rank shape² by cp_size² to recover global B*S².
        """
        batch = {"input_ids": torch.zeros(1, 16, dtype=torch.long)}
        assert _attn_work_from_batch(batch, torch.device("cpu"), cp_size=1).item() == 1 * 16 * 16
        assert _attn_work_from_batch(batch, torch.device("cpu"), cp_size=8).item() == 128 * 128
        thd = {
            "input_ids": torch.zeros(1, 64, dtype=torch.long),
            "cu_seq_lens_q": torch.tensor([0, 3, 8, 15], dtype=torch.int32),
        }
        assert _attn_work_from_batch(thd, torch.device("cpu"), cp_size=1).item() == 3**2 + 5**2 + 7**2
        assert _attn_work_from_batch(thd, torch.device("cpu"), cp_size=8).item() == 3**2 + 5**2 + 7**2

    def test_include_padding_thd(self):
        """THD include_padding=True uses cu_seq_lens_q_padded; False uses cu_seq_lens_q."""
        batch = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "cu_seq_lens_q": torch.tensor([0, 5, 11], dtype=torch.int32),
            "cu_seq_lens_q_padded": torch.tensor([0, 8, 16], dtype=torch.int32),
        }
        dev = torch.device("cpu")
        assert _attn_work_from_batch(batch, dev, cp_size=1, include_padding=False).item() == 5**2 + 6**2
        assert _attn_work_from_batch(batch, dev, cp_size=1, include_padding=True).item() == 8**2 + 8**2

    def test_include_padding_bshd_with_attention_mask(self):
        """BSHD include_padding=False uses attention_mask; True uses full shape."""
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.int64)
        batch = {"input_ids": torch.zeros(2, 8, dtype=torch.long), "attention_mask": mask}
        dev = torch.device("cpu")
        assert _attn_work_from_batch(batch, dev, cp_size=1, include_padding=False).item() == 5**2 + 3**2
        assert _attn_work_from_batch(batch, dev, cp_size=1, include_padding=True).item() == 2 * 8 * 8
