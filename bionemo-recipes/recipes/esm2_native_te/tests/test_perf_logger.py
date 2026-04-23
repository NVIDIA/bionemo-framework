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

"""Tests for ESM-2's PerfLogger: FLOP formula split + grad-acc accumulator pattern.

ESM-2 was previously the odd-one-out: its PerfLogger read num_tokens from the *last*
micro-batch at log time, so any future gradient accumulation would have undercounted
FLOPs by 1/grad_acc_steps. This retrofit introduces the ``log_micro_step`` /
``log_step`` split shared with the other MFU-tracking recipes (llama3, og2, codonfm)
and fixes attention-FLOP overcounting on packed (THD) batches.
"""

from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf
from transformers.modeling_outputs import MaskedLMOutput

from distributed_config import DistributedConfig
from perf_logger import (
    PerfLogger,
    _attn_work_from_batch,
    _compute_attn_flop_coeff,
    _compute_non_attn_per_token_flops,
)


ESM2_VOCAB = 33


def _make_args(logging_frequency=1, num_train_steps=100, log_mfu=False, max_seq_length=128):
    """Create a minimal args config for PerfLogger."""
    return OmegaConf.create(
        {
            "logger": {"frequency": logging_frequency},
            "wandb_init_args": {"project": "test", "mode": "disabled"},
            "num_train_steps": num_train_steps,
            "quant_stats_config": {"enabled": False},
            "log_mfu": log_mfu,
            "dataset": {"max_seq_length": max_seq_length},
        }
    )


def _make_batch(seq_len=128, device="cuda:0"):
    """Create a minimal batch dict."""
    return {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long, device=device),
        "labels": torch.ones(1, seq_len, dtype=torch.long, device=device),
    }


def _make_outputs(loss_value, seq_len=128, device="cuda:0"):
    """Create MaskedLMOutput with loss + logits."""
    logits = torch.randn(1, seq_len, ESM2_VOCAB, device=device)
    return MaskedLMOutput(loss=torch.tensor(loss_value, device=device), logits=logits)


@pytest.fixture
def mock_wandb():
    with mock.patch("perf_logger.wandb") as mocked:
        mocked.init.return_value = mock.MagicMock()
        yield mocked


@pytest.fixture
def mock_tqdm():
    with mock.patch("perf_logger.tqdm") as mocked:
        yield mocked


def _esm_cfg():
    """ESM-2-like MLM encoder config (MHA, no GQA, gelu MLP)."""
    return {
        "model_type": "esm",  # not in _GATED_MLP_MODEL_TYPES → 2-proj MLP
        "hidden_size": 1280,
        "num_hidden_layers": 33,
        "num_attention_heads": 20,
        "intermediate_size": 5120,
        "vocab_size": ESM2_VOCAB,
    }


class TestFlopSplitAndAttention:
    """Verify the non-attn + Σ(Lᵢ²) attention formula is correctly computed for ESM-2."""

    def test_bshd_shape_synthesis(self):
        """BSHD batch (no cu_seq_lens) synthesizes Σ(Lᵢ²) = B·S² from input_ids shape."""
        b, s = 2, 512
        batch = {"input_ids": torch.zeros(b, s, dtype=torch.long)}
        sigma_l_sq = _attn_work_from_batch(batch, torch.device("cpu")).item()
        assert sigma_l_sq == b * s * s

    def test_thd_single_doc_matches_bshd(self):
        """cu_seq_lens_q=[0, S] reproduces BSHD's Σ(Lᵢ²)=S²."""
        s = 512
        bshd = {"input_ids": torch.zeros(1, s, dtype=torch.long)}
        thd = {
            "input_ids": torch.zeros(1, s, dtype=torch.long),
            "cu_seq_lens_q": torch.tensor([0, s], dtype=torch.int32),
        }
        assert _attn_work_from_batch(bshd, torch.device("cpu")).item() == s * s
        assert _attn_work_from_batch(thd, torch.device("cpu")).item() == s * s

    def test_thd_multi_doc_uses_squared_sum(self):
        """Multi-doc pack computes Σ(Lᵢ²), not (ΣLᵢ)²."""
        cu = torch.tensor([0, 3, 8, 15], dtype=torch.int32)
        batch = {"input_ids": torch.zeros(1, 15, dtype=torch.long), "cu_seq_lens_q": cu}
        work = _attn_work_from_batch(batch, torch.device("cpu")).item()
        assert work == 3**2 + 5**2 + 7**2
        assert work < 15 * 15

    def test_cp_size_divides_attention_only(self):
        """cp_size divides the attention term only; non-attn untouched."""
        cfg = _esm_cfg()
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
        assert work == 5**2 + 6**2
        assert work != 8**2 + 8**2

    def test_padded_fallback_when_unpadded_absent(self):
        """If only cu_seq_lens_q_padded is present, it is used as a fallback."""
        batch = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "cu_seq_lens_q_padded": torch.tensor([0, 8, 16], dtype=torch.int32),
        }
        assert _attn_work_from_batch(batch, torch.device("cpu")).item() == 8**2 + 8**2

    def test_bshd_cp_correction(self):
        """BSHD with CP: per-rank shape (B, S/cp) → helper must return global B*S²."""
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


class TestGradAccAccumulation:
    """Lock in ESM-2's new log_micro_step/log_step split under gradient accumulation.

    Before this retrofit ESM-2 read num_tokens from only the last micro-batch of an
    optimizer step, so with grad_acc_steps > 1 it would have reported 1/grad_acc the
    true FLOP count. The new accumulator pattern sums across micro-batches.
    """

    def test_num_tokens_accumulates_across_grad_acc(self, mock_wandb, mock_tqdm):
        """4 micro-batches of seq_len=128 → num_tokens = 4*128 at log boundary."""
        dist_config = DistributedConfig()
        args = _make_args(logging_frequency=1, max_seq_length=128)
        perf_logger = PerfLogger(dist_config, args)
        device = perf_logger._device

        # One optimizer step with 4 micro-batches of shape (1, 128).
        for _ in range(4):
            batch = _make_batch(seq_len=128, device=device)
            outputs = _make_outputs(1.0, seq_len=128, device=device)
            perf_logger.log_micro_step(step=1, batch=batch, outputs=outputs)

        assert perf_logger.grad_acc_step_count == 4
        assert perf_logger.num_tokens == 4 * 128  # 4 micro-batches * 128 tokens each
        # running_loss should sum 4 losses of 1.0 each
        assert perf_logger.running_loss.item() == pytest.approx(4.0)

    def test_attn_work_accumulates_across_grad_acc(self, mock_wandb, mock_tqdm):
        """_attn_work_accum sums Σ(Lᵢ²) over all micro-batches when log_mfu=True."""
        dist_config = DistributedConfig()
        args = _make_args(logging_frequency=1, log_mfu=True, max_seq_length=128)
        perf_logger = PerfLogger(dist_config, args, model_config_dict=_esm_cfg())
        device = perf_logger._device

        # 3 micro-batches of shape (2, 64) → each batch has Σ(Lᵢ²) = 2 * 64² = 8192
        for _ in range(3):
            batch = {
                "input_ids": torch.ones(2, 64, dtype=torch.long, device=device),
                "labels": torch.ones(2, 64, dtype=torch.long, device=device),
            }
            outputs = _make_outputs(1.0, seq_len=64, device=device)
            # Perplexity expects (B, S, V) logits
            outputs.logits = torch.randn(2, 64, ESM2_VOCAB, device=device)
            perf_logger.log_micro_step(step=1, batch=batch, outputs=outputs)

        # Accumulator should hold 3 * 2 * 64² = 24576
        assert perf_logger._attn_work_accum.item() == 3 * 2 * 64 * 64

    def test_reset_on_log_boundary(self, mock_wandb, mock_tqdm):
        """Calling log_step on a logging-boundary step drains all accumulators."""
        dist_config = DistributedConfig()
        args = _make_args(logging_frequency=1, log_mfu=True, max_seq_length=128)
        perf_logger = PerfLogger(dist_config, args, model_config_dict=_esm_cfg())
        device = perf_logger._device

        batch = _make_batch(seq_len=128, device=device)
        outputs = _make_outputs(1.0, seq_len=128, device=device)
        perf_logger.log_micro_step(step=1, batch=batch, outputs=outputs)
        perf_logger.log_step(step=1, grad_norm=torch.tensor(1.0, device=device), lr=1e-4)

        assert perf_logger.grad_acc_step_count == 0
        assert perf_logger.num_tokens == 0
        assert perf_logger.num_unpadded_tokens.item() == 0
        assert perf_logger._attn_work_accum.item() == 0
        assert perf_logger.running_loss.item() == pytest.approx(0.0)
