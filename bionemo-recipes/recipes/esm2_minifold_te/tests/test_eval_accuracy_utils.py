# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_accuracy_utils import (
    build_comparison_rows,
    compute_distogram_loss_per_sample,
    compute_distogram_metrics_per_sample,
    evaluate_acceptance,
    filter_state_dict_for_plain_runtime,
    select_eval_stem,
)
from train_fsdp2 import compute_distogram_loss, compute_distogram_metrics


def test_filter_state_dict_for_plain_runtime_drops_te_extra_state():
    state_dict = {
        "fc.weight": torch.randn(2, 2),
        "fc.bias": torch.randn(2),
        "fc._extra_state": torch.tensor([1]),
    }
    filtered = filter_state_dict_for_plain_runtime(state_dict)
    assert "fc._extra_state" not in filtered
    assert "fc.weight" in filtered
    assert "fc.bias" in filtered


def test_compute_distogram_loss_per_sample_matches_aggregate_mean():
    torch.manual_seed(7)
    preds = torch.randn(2, 8, 8, 16)
    coords = torch.randn(2, 8, 3)
    mask = torch.ones(2, 8)

    per_sample = compute_distogram_loss_per_sample(preds, coords, mask, no_bins=16)
    aggregate = compute_distogram_loss(preds, coords, mask, no_bins=16)

    assert torch.allclose(per_sample.mean(), aggregate, atol=1e-5)


def test_compute_distogram_metrics_per_sample_match_aggregate_mean():
    torch.manual_seed(11)
    preds = torch.randn(3, 10, 10, 32)
    coords = torch.randn(3, 10, 3)
    mask = torch.ones(3, 10)

    per_sample = compute_distogram_metrics_per_sample(preds, coords, mask, no_bins=32)
    aggregate = compute_distogram_metrics(preds, coords, mask, no_bins=32)

    for key, value in aggregate.items():
        assert torch.allclose(per_sample[key].mean(), value, atol=1e-5), key


def test_select_eval_stem_uses_expected_artifact_names():
    assert select_eval_stem("bf16", "bf16") == "bf16_baseline_eval_metrics"
    assert select_eval_stem("fp8_native", "fp8") == "fp8_native_eval_metrics"
    assert select_eval_stem("fp8_storage", "bf16") == "fp8_storage_bf16_eval_metrics"


def test_evaluate_acceptance_marks_clean_pass():
    bf16_payload = {
        "summary": {
            "loss": 1.0,
            "distogram_acc": 0.8,
            "contact_precision_8A": 0.7,
            "contact_recall_8A": 0.6,
            "lddt_from_distogram": 0.5,
            "mean_distance_error": 2.0,
        },
        "per_protein": [
            {
                "protein_id": "A",
                "pdb_id": "A",
                "chain_id": "",
                "num_residues": 100,
                "loss": 1.0,
                "distogram_acc": 0.8,
                "contact_precision_8A": 0.7,
                "contact_recall_8A": 0.6,
                "lddt_from_distogram": 0.50,
                "mean_distance_error": 2.00,
            },
            {
                "protein_id": "B",
                "pdb_id": "B",
                "chain_id": "",
                "num_residues": 120,
                "loss": 1.2,
                "distogram_acc": 0.82,
                "contact_precision_8A": 0.72,
                "contact_recall_8A": 0.61,
                "lddt_from_distogram": 0.55,
                "mean_distance_error": 1.90,
            },
        ],
    }
    fp8_payload = {
        "summary": {
            "loss": 1.01,
            "distogram_acc": 0.802,
            "contact_precision_8A": 0.701,
            "contact_recall_8A": 0.602,
            "lddt_from_distogram": 0.503,
            "mean_distance_error": 2.01,
        },
        "per_protein": [
            {
                "protein_id": "A",
                "pdb_id": "A",
                "chain_id": "",
                "num_residues": 100,
                "loss": 1.01,
                "distogram_acc": 0.801,
                "contact_precision_8A": 0.701,
                "contact_recall_8A": 0.601,
                "lddt_from_distogram": 0.502,
                "mean_distance_error": 2.01,
            },
            {
                "protein_id": "B",
                "pdb_id": "B",
                "chain_id": "",
                "num_residues": 120,
                "loss": 1.22,
                "distogram_acc": 0.803,
                "contact_precision_8A": 0.702,
                "contact_recall_8A": 0.603,
                "lddt_from_distogram": 0.551,
                "mean_distance_error": 1.93,
            },
        ],
    }

    rows = build_comparison_rows(bf16_payload, fp8_payload)
    acceptance = evaluate_acceptance(bf16_payload, fp8_payload, rows)

    assert acceptance["verdict"] == "PASS"
    assert acceptance["failed_bands"] == []
