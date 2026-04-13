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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: bionemo-recipes/recipes/llama3_native_te/tests/test_flops.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Tests for the flops.py FLOPs counting and MFU module."""

import sys
from pathlib import Path

import pytest


# Add parent directory so we can import flops
sys.path.insert(0, str(Path(__file__).parent.parent))

from flops import (
    MFUTracker,
    ModelFLOPsConfig,
    compute_flops_analytical,
    compute_flops_hyena,
    compute_flops_simplified,
    estimate_cp_comm_bytes,
    from_hf_config,
)


# ============================================================================
# Test configs matching real models
# ============================================================================

LLAMA_1B_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 25,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 6144,
    "vocab_size": 128256,
    "model_type": "llama",
    "hidden_act": "silu",
}

ESM2_8M_CONFIG = {
    "hidden_size": 320,
    "num_hidden_layers": 6,
    "num_attention_heads": 20,
    "intermediate_size": 1280,
    "vocab_size": 33,
    "model_type": "nv_esm",
    "hidden_act": "gelu",
}

CODONFM_1B_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 18,
    "num_attention_heads": 16,
    "intermediate_size": 8192,
}


# ============================================================================
# Config auto-detection
# ============================================================================


class TestFromHfConfig:
    """Test auto-detection of model architecture from config dicts."""

    def test_llama_detects_gqa_and_swiglu(self):
        cfg = from_hf_config(LLAMA_1B_CONFIG)
        assert cfg.num_kv_heads == 8
        assert cfg.num_mlp_projections == 3
        assert cfg.head_dim == 128

    def test_esm2_detects_mha_and_standard_ffn(self):
        cfg = from_hf_config(ESM2_8M_CONFIG)
        assert cfg.num_kv_heads == 20
        assert cfg.num_mlp_projections == 2

    def test_codonfm_defaults_to_mha_and_2_proj(self):
        cfg = from_hf_config(CODONFM_1B_CONFIG)
        assert cfg.num_kv_heads == 16
        assert cfg.num_mlp_projections == 2

    def test_missing_vocab_defaults_to_no_lm_head(self):
        cfg = from_hf_config(
            {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": 512}
        )
        assert cfg.vocab_size == 0
        assert cfg.has_lm_head is False

    def test_overrides_take_precedence(self):
        cfg = from_hf_config(ESM2_8M_CONFIG, num_mlp_projections=3, vocab_size=0, has_lm_head=False)
        assert cfg.num_mlp_projections == 3
        assert cfg.vocab_size == 0
        assert cfg.has_lm_head is False


# ============================================================================
# Analytical FLOPs formula
# ============================================================================


class TestComputeFlopsAnalytical:
    """Test the first-principles analytical FLOPs formula."""

    def test_training_is_3x_forward(self):
        cfg = from_hf_config(LLAMA_1B_CONFIG)
        total, breakdown, lm_head = compute_flops_analytical(cfg, 1, 4096)
        forward = cfg.num_hidden_layers * sum(breakdown.values()) + lm_head
        assert total == 3 * forward

    def test_swiglu_has_3_mlp_projections(self):
        cfg = from_hf_config(LLAMA_1B_CONFIG)
        _, breakdown, _ = compute_flops_analytical(cfg, 1, 1024)
        assert "Gate projection" in breakdown
        assert "Up projection" in breakdown
        assert "Down projection" in breakdown

    def test_standard_ffn_has_2_mlp_projections(self):
        cfg = from_hf_config(ESM2_8M_CONFIG)
        _, breakdown, _ = compute_flops_analytical(cfg, 1, 1024)
        assert "Gate projection" not in breakdown
        assert "Up projection" in breakdown
        assert "Down projection" in breakdown

    def test_no_lm_head_when_vocab_zero(self):
        cfg = from_hf_config(CODONFM_1B_CONFIG)
        _, _, lm_head = compute_flops_analytical(cfg, 1, 1024)
        assert lm_head == 0

    def test_flops_scale_linearly_with_batch(self):
        cfg = from_hf_config(LLAMA_1B_CONFIG)
        flops_b1, _, _ = compute_flops_analytical(cfg, 1, 1024)
        flops_b4, _, _ = compute_flops_analytical(cfg, 4, 1024)
        assert flops_b4 == 4 * flops_b1

    def test_known_value_llama_lingua_1b(self):
        """Golden value: validated against PyTorch FlopCounterMode and README formula."""
        cfg = from_hf_config(LLAMA_1B_CONFIG)
        total, _, _ = compute_flops_analytical(cfg, 1, 4096)
        assert total == 47_687_021_887_488


# ============================================================================
# Simplified formula
# ============================================================================


class TestComputeFlopsSimplified:
    """Test the simplified README formula and its relationship to analytical."""

    def test_matches_analytical_when_mha_and_i_equals_4h(self):
        """ESM2 has MHA + I=4H + 2 projections: same assumptions as simplified formula."""
        cfg = from_hf_config(ESM2_8M_CONFIG)
        analytical, _, _ = compute_flops_analytical(cfg, 1, 1024)
        simplified = compute_flops_simplified(1, 1024, cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size)
        assert analytical == simplified

    def test_differs_when_gqa_or_swiglu(self):
        """GQA + SwiGLU breaks the simplified formula's assumptions."""
        cfg_dict = {
            **LLAMA_1B_CONFIG,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
        }
        cfg = from_hf_config(cfg_dict)
        analytical, _, _ = compute_flops_analytical(cfg, 1, 4096)
        simplified = compute_flops_simplified(1, 4096, cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size)
        assert analytical != simplified


# ============================================================================
# Hyena formula
# ============================================================================


class TestComputeFlopsHyena:
    """Test the Hyena (Evo2) FLOPs formula."""

    @pytest.fixture()
    def hyena_config(self):
        return ModelFLOPsConfig(
            hidden_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=4096,
            num_mlp_projections=3,
            vocab_size=512,
            has_lm_head=True,
        )

    def test_scales_subquadratically(self, hyena_config):
        """Hyena uses O(S log S) convolution, not O(S^2) attention."""
        flops_1k = compute_flops_hyena(hyena_config, 1, 1024, hyena_layer_counts={"S": 0, "D": 0, "H": 4, "A": 0})
        flops_2k = compute_flops_hyena(hyena_config, 1, 2048, hyena_layer_counts={"S": 0, "D": 0, "H": 4, "A": 0})
        ratio = flops_2k / flops_1k
        assert ratio < 3.0  # S^2 would give ~4x; Hyena should be well below

    def test_hybrid_attention_adds_quadratic_cost(self, hyena_config):
        """Adding standard attention layers increases FLOPs due to S^2 term."""
        all_hyena = compute_flops_hyena(hyena_config, 1, 4096, hyena_layer_counts={"S": 0, "D": 0, "H": 4, "A": 0})
        with_attn = compute_flops_hyena(hyena_config, 1, 4096, hyena_layer_counts={"S": 0, "D": 0, "H": 2, "A": 2})
        assert with_attn > all_hyena


# ============================================================================
# MFUTracker
# ============================================================================


class TestMFUTracker:
    """Test the MFUTracker class used by training scripts."""

    def test_from_config_dict(self):
        tracker = MFUTracker.from_config_dict(LLAMA_1B_CONFIG, batch_size=1, seq_len=4096, peak_tflops=155.0)
        assert tracker.total_flops == 47_687_021_887_488
        assert tracker.per_gpu_flops == tracker.total_flops

    def test_multi_gpu_divides_flops(self):
        single = MFUTracker.from_config_dict(
            LLAMA_1B_CONFIG, batch_size=1, seq_len=4096, num_gpus=1, peak_tflops=155.0
        )
        multi = MFUTracker.from_config_dict(LLAMA_1B_CONFIG, batch_size=1, seq_len=4096, num_gpus=2, peak_tflops=155.0)
        assert multi.per_gpu_flops == single.total_flops // 2

    def test_compute_mfu_correctness(self):
        tracker = MFUTracker.from_config_dict(LLAMA_1B_CONFIG, batch_size=1, seq_len=4096, peak_tflops=155.0)
        result = tracker.compute_mfu(step_time=0.5)
        expected_tflops = tracker.per_gpu_flops / 0.5 / 1e12
        expected_mfu = expected_tflops / 155.0 * 100
        assert abs(result["mfu"] - expected_mfu) < 0.01
        assert abs(result["tflops_per_gpu"] - expected_tflops) < 0.01

    def test_mfu_inversely_proportional_to_step_time(self):
        tracker = MFUTracker.from_config_dict(LLAMA_1B_CONFIG, batch_size=1, seq_len=4096, peak_tflops=155.0)
        fast = tracker.compute_mfu(step_time=0.5)
        slow = tracker.compute_mfu(step_time=1.0)
        assert abs(fast["mfu"] - 2 * slow["mfu"]) < 0.01

    def test_all_formula_options(self):
        for formula in ["analytical", "simplified", "hyena"]:
            if formula == "hyena":
                cfg = ModelFLOPsConfig(
                    hidden_size=1024,
                    num_hidden_layers=4,
                    num_attention_heads=8,
                    num_kv_heads=8,
                    head_dim=128,
                    intermediate_size=4096,
                    num_mlp_projections=3,
                    vocab_size=512,
                    has_lm_head=True,
                )
                tracker = MFUTracker(cfg, batch_size=1, seq_len=4096, peak_tflops=155.0, formula=formula)
            else:
                tracker = MFUTracker.from_config_dict(
                    LLAMA_1B_CONFIG, batch_size=1, seq_len=4096, peak_tflops=155.0, formula=formula
                )
            assert tracker.total_flops > 0

    def test_invalid_formula_raises(self):
        with pytest.raises(ValueError, match="Unknown formula"):
            MFUTracker.from_config_dict(LLAMA_1B_CONFIG, batch_size=1, seq_len=4096, peak_tflops=155.0, formula="bad")

    def test_cp_communication_estimate(self):
        tracker = MFUTracker.from_config_dict(
            LLAMA_1B_CONFIG, batch_size=1, seq_len=16384, num_gpus=2, parallelism={"dp": 1, "cp": 2}, peak_tflops=155.0
        )
        assert tracker.comm_bytes > 0
        overhead = tracker.estimate_comm_overhead(step_time=1.0, measured_bw_gbps=6.6)
        assert overhead["estimated_comm_time"] > 0
        assert 0 < overhead["comm_pct"] < 100

    def test_no_comm_single_gpu(self):
        tracker = MFUTracker.from_config_dict(LLAMA_1B_CONFIG, batch_size=1, seq_len=4096, peak_tflops=155.0)
        assert tracker.comm_bytes == 0


# ============================================================================
# Communication estimation
# ============================================================================


class TestCPCommEstimation:
    """Test CP ring attention communication byte estimates."""

    def test_zero_without_cp(self):
        assert estimate_cp_comm_bytes(1, 4096, 25, 8, 128, cp_size=1) == 0

    def test_scales_linearly_with_seq_len(self):
        comm_4k = estimate_cp_comm_bytes(1, 4096, 25, 8, 128, cp_size=2)
        comm_8k = estimate_cp_comm_bytes(1, 8192, 25, 8, 128, cp_size=2)
        assert comm_8k == 2 * comm_4k

    def test_scales_linearly_with_batch(self):
        comm_b1 = estimate_cp_comm_bytes(1, 4096, 25, 8, 128, cp_size=2)
        comm_b4 = estimate_cp_comm_bytes(4, 4096, 25, 8, 128, cp_size=2)
        assert comm_b4 == 4 * comm_b1

    def test_known_value_lingua_1b(self):
        """Golden value for lingua-1B at S=4096, CP=2."""
        assert estimate_cp_comm_bytes(1, 4096, 25, 8, 128, cp_size=2) == 419_430_400
