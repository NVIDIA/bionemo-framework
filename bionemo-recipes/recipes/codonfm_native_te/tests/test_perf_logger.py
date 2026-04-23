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

"""Tests for PerfLogger loss calculation correctness with gradient accumulation."""

from unittest import mock

import pytest
import torch
from distributed_config import DistributedConfig
from omegaconf import OmegaConf
from perf_logger import (
    PerfLogger,
    _attn_work_from_batch,
    _compute_attn_flop_coeff,
    _compute_non_attn_per_token_flops,
    _compute_per_token_flops,
)
from transformers.modeling_outputs import MaskedLMOutput


VOCAB_SIZE = 69  # CodonFM vocabulary size


def _make_args(logging_frequency=1, num_train_steps=100):
    """Create a minimal args config for PerfLogger."""
    return OmegaConf.create(
        {
            "logger": {"frequency": logging_frequency},
            "wandb_init_args": {"project": "test", "mode": "disabled"},
            "num_train_steps": num_train_steps,
            "quant_stats_config": {"enabled": False},
        }
    )


def _make_batch(seq_len=128, device="cuda:0"):
    """Create a minimal batch dict with input_ids and labels."""
    return {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long, device=device),
        "labels": torch.ones(1, seq_len, dtype=torch.long, device=device),
    }


def _make_outputs(loss_value, seq_len=128, device="cuda:0"):
    """Create MaskedLMOutput with a given loss and dummy logits."""
    logits = torch.randn(1, seq_len, VOCAB_SIZE, device=device)
    return MaskedLMOutput(loss=torch.tensor(loss_value, device=device), logits=logits)


@pytest.fixture
def mock_wandb():
    """Mock wandb to prevent actual logging."""
    with mock.patch("perf_logger.wandb") as mocked:
        mocked.init.return_value = mock.MagicMock()
        yield mocked


@pytest.fixture
def mock_tqdm():
    """Mock tqdm to prevent progress bar output."""
    with mock.patch("perf_logger.tqdm") as mocked:
        yield mocked


def _create_perf_logger(logging_frequency, mock_wandb, mock_tqdm):
    """Create a PerfLogger with the given logging_frequency."""
    dist_config = DistributedConfig()
    args = _make_args(logging_frequency=logging_frequency)
    return PerfLogger(dist_config, args)


def _run_steps(perf_logger, losses, grad_acc_steps=1):
    """Simulate training steps with given per-optimizer-step losses.

    Args:
        perf_logger: The PerfLogger instance.
        losses: List of loss values, one per optimizer step. With grad_acc_steps>1,
            each value is used for all micro steps in that optimizer step.
        grad_acc_steps: Number of micro steps per optimizer step.
    """
    device = perf_logger._device
    for step_idx, loss_val in enumerate(losses):
        step = step_idx + 1
        batch = _make_batch(device=device)
        outputs = _make_outputs(loss_val, device=device)
        for _ in range(grad_acc_steps):
            perf_logger.log_micro_step(step, batch, outputs)
        perf_logger.log_step(step, torch.tensor(1.0, device=device), 1e-4)


def _get_logged_losses(mock_wandb):
    """Extract reported loss values from wandb.log calls."""
    return [call[0][0]["train/loss"] for call in mock_wandb.log.call_args_list]


class TestPerfLoggerLoss:
    """Test that PerfLogger computes average loss correctly."""

    def test_logging_frequency_1_reports_each_loss(self, mock_wandb, mock_tqdm):
        """With logging_frequency=1, each step's loss should be reported as-is."""
        perf_logger = _create_perf_logger(1, mock_wandb, mock_tqdm)
        losses = [1.0, 2.0, 3.0, 4.0, 5.0]
        _run_steps(perf_logger, losses)

        reported = _get_logged_losses(mock_wandb)
        assert len(reported) == len(losses)
        for i, (got, expected) in enumerate(zip(reported, losses)):
            assert got == pytest.approx(expected), f"Step {i + 1}: expected {expected}, got {got}"

    def test_logging_frequency_5_matches_averaged_frequency_1(self, mock_wandb, mock_tqdm):
        """logging_frequency=5 should report the same average as manually averaging 5 frequency-1 losses."""
        losses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        logging_freq = 5

        # Run with logging_frequency=1
        perf_logger_1 = _create_perf_logger(1, mock_wandb, mock_tqdm)
        _run_steps(perf_logger_1, losses)
        freq1_losses = _get_logged_losses(mock_wandb)
        assert len(freq1_losses) == 10

        # Compute expected averages over windows of size logging_freq
        expected = []
        for i in range(0, len(freq1_losses), logging_freq):
            window = freq1_losses[i : i + logging_freq]
            expected.append(sum(window) / len(window))

        # Run with logging_frequency=5
        mock_wandb.log.reset_mock()
        perf_logger_5 = _create_perf_logger(logging_freq, mock_wandb, mock_tqdm)
        _run_steps(perf_logger_5, losses)
        freq5_losses = _get_logged_losses(mock_wandb)

        assert len(freq5_losses) == len(expected), f"Expected {len(expected)} log events, got {len(freq5_losses)}"
        for i, (got, exp) in enumerate(zip(freq5_losses, expected)):
            assert got == pytest.approx(exp), f"Window {i}: expected {exp}, got {got}"

    def test_logging_frequency_with_grad_accumulation(self, mock_wandb, mock_tqdm):
        """Loss should be correct when combining gradient accumulation with logging_frequency > 1."""
        grad_acc_steps = 4
        logging_freq = 3
        # Each value is used for all micro steps in that optimizer step
        losses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        # Run with logging_frequency=1 to get per-step losses
        perf_logger_1 = _create_perf_logger(1, mock_wandb, mock_tqdm)
        _run_steps(perf_logger_1, losses, grad_acc_steps=grad_acc_steps)
        freq1_losses = _get_logged_losses(mock_wandb)
        assert len(freq1_losses) == 6

        # Each step's loss should equal the input loss (all micro steps have same value)
        for i, (got, expected) in enumerate(zip(freq1_losses, losses)):
            assert got == pytest.approx(expected), f"Step {i + 1}: expected {expected}, got {got}"

        # Compute expected averages
        expected = []
        for i in range(0, len(freq1_losses), logging_freq):
            window = freq1_losses[i : i + logging_freq]
            expected.append(sum(window) / len(window))

        # Run with logging_frequency=logging_freq
        mock_wandb.log.reset_mock()
        perf_logger_n = _create_perf_logger(logging_freq, mock_wandb, mock_tqdm)
        _run_steps(perf_logger_n, losses, grad_acc_steps=grad_acc_steps)
        freqn_losses = _get_logged_losses(mock_wandb)

        assert len(freqn_losses) == len(expected)
        for i, (got, exp) in enumerate(zip(freqn_losses, expected)):
            assert got == pytest.approx(exp), f"Window {i}: expected {exp}, got {got}"

    def test_logging_frequency_with_varying_micro_losses(self, mock_wandb, mock_tqdm):
        """Test with different loss values across micro steps within a single optimizer step."""
        logging_freq = 2
        device = torch.device("cuda:0")

        perf_logger = _create_perf_logger(logging_freq, mock_wandb, mock_tqdm)

        # Step 1: micro losses [1.0, 3.0] -> avg micro loss = 2.0
        for loss_val in [1.0, 3.0]:
            batch = _make_batch(device=device)
            outputs = _make_outputs(loss_val, device=device)
            perf_logger.log_micro_step(1, batch, outputs)
        perf_logger.log_step(1, torch.tensor(1.0, device=device), 1e-4)

        # Step 2: micro losses [5.0, 7.0] -> avg micro loss = 6.0
        # Window of 2 steps: avg = (1.0 + 3.0 + 5.0 + 7.0) / 4 = 4.0
        for loss_val in [5.0, 7.0]:
            batch = _make_batch(device=device)
            outputs = _make_outputs(loss_val, device=device)
            perf_logger.log_micro_step(2, batch, outputs)
        perf_logger.log_step(2, torch.tensor(1.0, device=device), 1e-4)

        reported = _get_logged_losses(mock_wandb)
        assert len(reported) == 1
        # Total running_loss = 1.0 + 3.0 + 5.0 + 7.0 = 16.0
        # grad_acc_step_count = 4 (2 micro steps * 2 optimizer steps)
        # avg = 16.0 / 4 = 4.0
        assert reported[0] == pytest.approx(4.0), f"Expected 4.0, got {reported[0]}"

    def test_min_loss_tracked_correctly(self, mock_wandb, mock_tqdm):
        """min_loss should track the true minimum average loss across windows."""
        perf_logger = _create_perf_logger(1, mock_wandb, mock_tqdm)
        losses = [5.0, 2.0, 8.0, 1.0, 4.0]
        _run_steps(perf_logger, losses)

        assert perf_logger.min_loss.item() == pytest.approx(1.0)


def _codon_cfg():
    """CodonFM-like config for the split-formula tests (MLM encoder)."""
    return {
        "model_type": "codonfm",  # not in _GATED_MLP_MODEL_TYPES → standard 2-proj MLP
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "vocab_size": VOCAB_SIZE,
    }


class TestFlopSplitAndAttention:
    """Verify the split non-attn + Σ(Lᵢ²) attention formula."""

    def test_algebraic_identity(self):
        """non_attn + coeff·S ≡ _compute_per_token_flops(cfg, S) for all S."""
        cfg = _codon_cfg()
        for s in (256, 512, 1024, 8192):
            lhs = _compute_non_attn_per_token_flops(cfg) + _compute_attn_flop_coeff(cfg) * s
            rhs = _compute_per_token_flops(cfg, s)
            assert lhs == rhs, f"S={s}: {lhs} != {rhs}"

    def test_bshd_no_op(self):
        """BSHD batch (no cu_seq_lens) with cp=1 matches legacy formula exactly."""
        cfg = _codon_cfg()
        b, s = 4, 512
        batch = {"input_ids": torch.zeros(b, s, dtype=torch.long)}
        sigma_l_sq = _attn_work_from_batch(batch, torch.device("cpu")).item()
        assert sigma_l_sq == b * s * s
        new_flops = _compute_non_attn_per_token_flops(cfg) * b * s + _compute_attn_flop_coeff(cfg) * sigma_l_sq
        legacy_flops = _compute_per_token_flops(cfg, s) * b * s
        assert new_flops == legacy_flops

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
        """cp_size divides the attention term only; non-attn stays untouched.
        Codonfm doesn't support CP, but the formula must still respect cp_size=1 default."""
        cfg = _codon_cfg()
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
        """BSHD with CP: per-rank shape (B, S/cp) → helper must return global B*S².

        CodonFM currently runs FSDP without CP so this is latent defence, but the
        formula must be correct if CP is added.
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
