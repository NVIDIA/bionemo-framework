# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf
import torch


RECIPE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RECIPE_DIR))

PLAIN_SPEC = importlib.util.spec_from_file_location("plain_minifold_infer", RECIPE_DIR / "plain_minifold_infer.py")
PLAIN_MODULE = importlib.util.module_from_spec(PLAIN_SPEC)
assert PLAIN_SPEC is not None and PLAIN_SPEC.loader is not None
sys.modules[PLAIN_SPEC.name] = PLAIN_MODULE
PLAIN_SPEC.loader.exec_module(PLAIN_MODULE)

SINGLE_BLOCK_SPEC = importlib.util.spec_from_file_location(
    "run_single_block_equivalence", RECIPE_DIR / "run_single_block_equivalence.py"
)
SINGLE_BLOCK_MODULE = importlib.util.module_from_spec(SINGLE_BLOCK_SPEC)
assert SINGLE_BLOCK_SPEC is not None and SINGLE_BLOCK_SPEC.loader is not None
sys.modules[SINGLE_BLOCK_SPEC.name] = SINGLE_BLOCK_MODULE
SINGLE_BLOCK_SPEC.loader.exec_module(SINGLE_BLOCK_MODULE)


def test_activation_probe_records_quantized_snapshot():
    probe = PLAIN_MODULE.ActivationProbe(pair_precision_mode="fp8_extreme", retain_tensors=True)
    tensor = PLAIN_MODULE.QuantizedPairTensor.from_tensor(torch.randn(1, 2, 2, 8, dtype=torch.bfloat16))
    probe.record(0, "tri_out", tensor, mode_op_name="tri_quantized_out")

    assert len(probe.events) == 1
    event = probe.events[0]
    assert event.op_name == "tri_out"
    assert event.mode_op_name == "tri_quantized_out"
    assert event.tensor_format == "quantized_fp8"
    assert event.snapshot_index == 0
    assert probe.tensor_snapshots[0].shape == tensor.payload.shape


def test_miniformer_dump_activation_stats_populates_probe():
    model = PLAIN_MODULE.MiniFormer(dim=32, blocks=1, pair_precision="bf16", linear_precision="bf16").to(dtype=torch.bfloat16)
    x = torch.randn(1, 4, 4, 32, dtype=torch.bfloat16)
    mask = torch.ones(1, 4, 4, dtype=torch.bfloat16)

    y = model(x, mask, dump_activation_stats=True)

    assert y.shape == x.shape
    assert model.last_activation_probe is not None
    op_names = [event.op_name for event in model.last_activation_probe.events]
    assert "block_input" in op_names
    assert "block_output" in op_names


def test_eval_real_3b_fp8native_defaults_to_scaled_eval_and_hybrid_config():
    cfg = OmegaConf.load(RECIPE_DIR / "hydra_config" / "eval_real_3B_fp8native.yaml")

    assert cfg.eval_dataset.parquet_path == "/scratch/pdb_eval_data/eval_structures.parquet"
    assert cfg.hybrid_precision.use_native_layernorm is False
    assert cfg.hybrid_precision.use_native_linear is False
    assert cfg.hybrid_precision.use_native_gate is False
    assert cfg.hybrid_precision.use_native_tri is False
    assert cfg.hybrid_precision.use_resident_fp8_residual is False
    assert cfg.mixed_tail.tail_bf16_native_blocks == 0
    assert cfg.mixed_tail.bf16_native_rung == "B3"


def test_scale_summary_flags_zero_scale_collapse():
    scale = torch.zeros(1, 2, 2, 4, dtype=torch.float32)

    summary = SINGLE_BLOCK_MODULE._scale_summary(scale)

    assert summary is not None
    assert summary["numel"] == 16
    assert summary["zero_count"] == 16
    assert summary["all_zero"] is True
    assert summary["finite_unique_count"] == 1
    assert summary["scale_unique_ratio"] == 1.0 / 16.0


def test_find_first_bad_stage_prefers_payload_nonfinite():
    rows = [
        {
            "rung": "S1",
            "stage_name": "tri_input_norm",
            "native_mode_op_name": "native_mxfp8_layernorm_quantized",
            "native": {
                "nan_count": 0,
                "inf_count": 0,
                "scale": {
                    "nonfinite_count": 0,
                    "all_zero": False,
                    "constant_finite_value": False,
                    "finite_unique_count": 4,
                },
            },
            "reference": {"scale": {"finite_unique_count": 4}},
        },
        {
            "rung": "S2",
            "stage_name": "tri_gated",
            "native_mode_op_name": "native_gate_sigmoid_mul_quantized",
            "native": {
                "nan_count": 8,
                "inf_count": 0,
                "scale": {
                    "nonfinite_count": 0,
                    "all_zero": True,
                    "constant_finite_value": True,
                    "finite_unique_count": 1,
                },
            },
            "reference": {"scale": {"finite_unique_count": 8}},
        },
    ]

    culprit = SINGLE_BLOCK_MODULE._find_first_bad_stage(rows)

    assert culprit is not None
    assert culprit["rung"] == "S2"
    assert culprit["stage_name"] == "tri_gated"
    assert culprit["reason"] == "native payload produced nonfinite values"


def test_find_first_bad_tri_subboundary_prefers_pack_exact_mismatch():
    rows = [
        {
            "subboundary": "pack",
            "component": "a1",
            "native_mode_op_name": "pack_block32_to_mxfp8_fused_debug",
            "native_payload": {"nan_count": 0, "inf_count": 0},
            "native_scale": {"nonfinite_count": 0, "all_zero": False, "zero_count": 0},
            "reference_scale": {"all_zero": False},
            "payload_comparison": {"exact_match": False, "allclose": True, "rel_l2_error": 0.0},
            "scale_comparison": {"exact_match": True, "rel_l2_error": 0.0},
        },
        {
            "subboundary": "gemm",
            "component": "x1",
            "native_mode_op_name": "tri_mul_pair_from_packed_debug",
            "native_payload": {"nan_count": 0, "inf_count": 0},
            "native_scale": None,
            "reference_scale": None,
            "payload_comparison": {"exact_match": False, "allclose": True, "rel_l2_error": 1e-5},
            "scale_comparison": None,
        },
    ]

    culprit = SINGLE_BLOCK_MODULE._find_first_bad_tri_subboundary(rows)

    assert culprit is not None
    assert culprit["subboundary"] == "pack"
    assert culprit["component"] == "a1"
    assert culprit["reason"] == "native payload does not exactly match the Python gold path"


def test_find_first_bad_tri_subboundary_flags_gemm_divergence():
    rows = [
        {
            "subboundary": "pack",
            "component": "a1",
            "native_mode_op_name": "pack_block32_to_mxfp8_fused_debug",
            "native_payload": {"nan_count": 0, "inf_count": 0},
            "native_scale": {"nonfinite_count": 0, "all_zero": False, "zero_count": 0},
            "reference_scale": {"all_zero": False},
            "payload_comparison": {"exact_match": True, "allclose": True, "rel_l2_error": 0.0},
            "scale_comparison": {"exact_match": True, "rel_l2_error": 0.0},
        },
        {
            "subboundary": "gemm",
            "component": "x1",
            "native_mode_op_name": "tri_mul_pair_from_packed_debug",
            "native_payload": {"nan_count": 0, "inf_count": 0},
            "native_scale": None,
            "reference_scale": None,
            "payload_comparison": {"exact_match": False, "allclose": False, "rel_l2_error": 0.25},
            "scale_comparison": None,
        },
    ]

    culprit = SINGLE_BLOCK_MODULE._find_first_bad_tri_subboundary(rows)

    assert culprit is not None
    assert culprit["subboundary"] == "gemm"
    assert culprit["component"] == "x1"
    assert culprit["reason"] == "native GEMM output diverged from the raw FP8 reference"
