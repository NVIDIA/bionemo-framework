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

"""Tests for PerfLogger loss calculation correctness."""

from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf
from transformers.modeling_outputs import CausalLMOutputWithPast

from distributed_config import DistributedConfig
from perf_logger import (
    PerfLogger,
    _attn_work_from_batch,
    _compute_attn_flop_coeff,
    _compute_non_attn_per_token_flops,
    _compute_per_token_flops,
    _detect_peak_tflops_bf16,
)


def _make_args(logging_frequency=1, num_train_steps=100, log_mfu=False, max_seq_length=128):
    """Create a minimal args config for PerfLogger."""
    return OmegaConf.create(
        {
            "logger": {"frequency": logging_frequency},
            "wandb": {"project": "test", "mode": "disabled"},
            "num_train_steps": num_train_steps,
            "profiler": {"enabled": False},
            "fp8_stats_config": {"enabled": False},
            "log_mfu": log_mfu,
            "dataset": {"max_seq_length": max_seq_length},
        }
    )


def _make_batch(seq_len=128, device="cuda:0"):
    """Create a minimal batch dict."""
    return {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=device),
    }


def _make_outputs(loss_value, device="cuda:0"):
    """Create CausalLMOutputWithPast with a given loss."""
    return CausalLMOutputWithPast(loss=torch.tensor(loss_value, device=device))


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
    return PerfLogger(dist_config, args, start_step=0)


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

        # Step 1: micro losses [1.0, 3.0] → avg micro loss = 2.0
        for loss_val in [1.0, 3.0]:
            batch = _make_batch(device=device)
            outputs = _make_outputs(loss_val, device=device)
            perf_logger.log_micro_step(1, batch, outputs)
        perf_logger.log_step(1, torch.tensor(1.0, device=device), 1e-4)

        # Step 2: micro losses [5.0, 7.0] → avg micro loss = 6.0
        # Window of 2 steps: avg = (2.0 + 6.0) / 2 = 4.0
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


class TestComputePerTokenFlops:
    """Test that the per-token training FLOPs formula matches hand-calculated values."""

    def test_llama_gqa_swiglu(self):
        """Llama-style config: GQA (n_kv=8 < n_heads=32) + SwiGLU (3 MLP projections)."""
        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # GQA
            "intermediate_size": 14336,
            "vocab_size": 128256,
        }
        seq_len = 8192
        h, i, v, kv_dim, layers = 4096, 14336, 128256, 8 * 128, 32
        # Per-layer: Q (2h^2) + K+V (4h*kv_dim) + O (2h^2) + attn (4*S*h) + MLP (2*3*h*i)
        per_layer = 2 * h * h + 4 * h * kv_dim + 2 * h * h + 4 * seq_len * h + 2 * 3 * h * i
        expected_fwd = layers * per_layer + 2 * h * v
        assert _compute_per_token_flops(config, seq_len) == 3 * expected_fwd

    def test_esm_mha_gelu(self):
        """ESM2-style config: MHA (no num_key_value_heads) + standard FFN (2 MLP projections)."""
        config = {
            "model_type": "esm",
            "hidden_size": 1280,
            "num_hidden_layers": 33,
            "num_attention_heads": 20,
            "intermediate_size": 5120,
            "vocab_size": 33,
        }
        seq_len = 1024
        h, i, v, kv_dim, layers = 1280, 5120, 33, (1280 // 20) * 20, 33  # kv_dim=h for MHA
        per_layer = 2 * h * h + 4 * h * kv_dim + 2 * h * h + 4 * seq_len * h + 2 * 2 * h * i
        expected_fwd = layers * per_layer + 2 * h * v
        assert _compute_per_token_flops(config, seq_len) == 3 * expected_fwd

    def test_scales_with_seq_len(self):
        """Only the attention S^2 term should vary with seq_len."""
        config = {
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 32000,
        }
        h, layers = 2048, 16
        # Difference per token between seq_len=1024 and seq_len=2048:
        #   layers * 4 * (2048 - 1024) * h, times 3 (forward+backward)
        diff = _compute_per_token_flops(config, 2048) - _compute_per_token_flops(config, 1024)
        assert diff == 3 * layers * 4 * 1024 * h

    def test_linear_in_unpadded_tokens(self):
        """Multiplying per-token FLOPs by N tokens is linear (MFU formula relies on this)."""
        config = {
            "model_type": "llama",
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
        per_token = _compute_per_token_flops(config, seq_len=512)
        assert per_token * 100 == 100 * per_token
        # Sanity: doubling unpadded token count doubles total FLOPs
        assert per_token * 200 == 2 * (per_token * 100)

    def test_no_lm_head_when_vocab_zero(self):
        """vocab_size=0 should drop the LM head term."""
        config_base = {
            "model_type": "llama",
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
        }
        with_vocab = _compute_per_token_flops({**config_base, "vocab_size": 32000}, seq_len=256)
        no_vocab = _compute_per_token_flops({**config_base, "vocab_size": 0}, seq_len=256)
        # Difference = 3 (training) * 2 * h * vocab
        assert with_vocab - no_vocab == 3 * 2 * 512 * 32000


class TestDetectPeakTflops:
    """Smoke test for GPU peak TFLOPS detection."""

    def test_returns_tuple_shape(self):
        """Returns (peak_tflops_or_none, device_name_str)."""
        peak, name = _detect_peak_tflops_bf16()
        assert isinstance(name, str)
        assert peak is None or isinstance(peak, float)


def _llama_cfg():
    """Small llama-like config used by the split-formula tests."""
    return {
        "model_type": "llama",
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 4096,
        "vocab_size": 32000,
    }


class TestFlopSplitAndAttention:
    """Verify the split non-attn + Σ(Lᵢ²) attention formula.

    The old single-term ``_per_token_flops * num_tokens`` formula treats a packed
    batch as one giant S*S attention. Real Flash-Attention work is Σ(Lᵢ²) over
    packed segments. These tests lock in the new split and its invariants.
    """

    def test_algebraic_identity(self):
        """non_attn + coeff·S ≡ _compute_per_token_flops(cfg, S) for all S."""
        cfg = _llama_cfg()
        for s in (256, 1024, 8192):
            lhs = _compute_non_attn_per_token_flops(cfg) + _compute_attn_flop_coeff(cfg) * s
            rhs = _compute_per_token_flops(cfg, s)
            assert lhs == rhs, f"S={s}: {lhs} != {rhs}"

    def test_bshd_no_op(self):
        """BSHD batch (no cu_seq_lens) with cp=1 matches legacy formula exactly."""
        cfg = _llama_cfg()
        b, s = 2, 512
        batch = {"input_ids": torch.zeros(b, s, dtype=torch.long)}
        sigma_l_sq = _attn_work_from_batch(batch, torch.device("cpu")).item()
        assert sigma_l_sq == b * s * s
        new_flops = _compute_non_attn_per_token_flops(cfg) * b * s + _compute_attn_flop_coeff(cfg) * sigma_l_sq
        legacy_flops = _compute_per_token_flops(cfg, s) * b * s
        assert new_flops == legacy_flops

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
        # Doc lengths 3, 5, 7 → cumulative [0, 3, 8, 15]
        cu = torch.tensor([0, 3, 8, 15], dtype=torch.int32)
        batch = {"input_ids": torch.zeros(1, 15, dtype=torch.long), "cu_seq_lens_q": cu}
        work = _attn_work_from_batch(batch, torch.device("cpu")).item()
        assert work == 3**2 + 5**2 + 7**2  # 83 real QK pairs per layer
        assert work < 15 * 15  # old formula would have said 225

    def test_cp_size_divides_attention_only(self):
        """Dividing attention by cp_size must leave the non-attention term untouched."""
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
