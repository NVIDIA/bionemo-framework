from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_status_report(status_path: str | Path, title: str, lines: list[str]) -> None:
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = "\n".join(f"- {line}" for line in lines) if lines else "- no details recorded"
    with status_path.open("a", encoding="utf-8") as f:
        f.write(f"## {timestamp} — {title}\n{body}\n\n")


def filter_state_dict_for_plain_runtime(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if not key.endswith("._extra_state")}


def protein_identifier(pdb_id: str | None, chain_id: str | None) -> str:
    pdb_id = (pdb_id or "").strip()
    chain_id = (chain_id or "").strip()
    return f"{pdb_id}:{chain_id}" if chain_id else pdb_id


def compute_distogram_loss_per_sample(
    preds: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    no_bins: int = 64,
    max_dist: float = 25.0,
) -> torch.Tensor:
    dists = torch.cdist(coords, coords)
    boundaries = torch.linspace(2, max_dist, no_bins - 1, device=dists.device)
    labels = (dists.unsqueeze(-1) > boundaries).sum(dim=-1)
    errors = -torch.gather(F.log_softmax(preds, dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    pair_mask = mask[:, None, :] * mask[:, :, None]
    eye = torch.eye(mask.shape[1], device=mask.device, dtype=pair_mask.dtype).unsqueeze(0)
    pair_mask = pair_mask * (1 - eye)
    denom = pair_mask.sum(dim=(1, 2)).clamp(min=1e-5)
    return (errors * pair_mask).sum(dim=(1, 2)) / denom


def compute_distogram_metrics_per_sample(
    preds: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    no_bins: int = 64,
    max_dist: float = 25.0,
    contact_threshold: float = 8.0,
) -> dict[str, torch.Tensor]:
    true_dists = torch.cdist(coords, coords)
    boundaries = torch.linspace(2, max_dist, no_bins - 1, device=preds.device)
    bin_centers = torch.cat(
        [
            torch.tensor([1.0], device=preds.device),
            (boundaries[:-1] + boundaries[1:]) / 2,
            torch.tensor([max_dist + 2.0], device=preds.device),
        ]
    )

    true_bins = (true_dists.unsqueeze(-1) > boundaries).sum(dim=-1)
    pred_bins = preds.argmax(dim=-1)
    pred_probs = F.softmax(preds, dim=-1)
    pred_dists = (pred_probs * bin_centers).sum(dim=-1)

    pair_mask = mask[:, None, :] * mask[:, :, None]
    eye = torch.eye(mask.shape[1], device=mask.device, dtype=pair_mask.dtype).unsqueeze(0)
    pair_mask = pair_mask * (1 - eye)
    n_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1)

    correct = (pred_bins == true_bins).float() * pair_mask
    distogram_acc = correct.sum(dim=(1, 2)) / n_pairs

    true_contacts = (true_dists < contact_threshold).float() * pair_mask
    pred_contacts = (pred_dists < contact_threshold).float() * pair_mask
    tp = (true_contacts * pred_contacts).sum(dim=(1, 2))
    contact_precision = tp / pred_contacts.sum(dim=(1, 2)).clamp(min=1)
    contact_recall = tp / true_contacts.sum(dim=(1, 2)).clamp(min=1)

    dist_error = torch.abs(pred_dists - true_dists)
    lddt_score = (
        (dist_error < 0.5).float()
        + (dist_error < 1.0).float()
        + (dist_error < 2.0).float()
        + (dist_error < 4.0).float()
    ) * 0.25
    lddt_mask = pair_mask * (true_dists < 15.0).float()
    lddt_denom = lddt_mask.sum(dim=(1, 2)).clamp(min=1)
    lddt_from_distogram = (lddt_score * lddt_mask).sum(dim=(1, 2)) / lddt_denom
    mean_distance_error = (dist_error * lddt_mask).sum(dim=(1, 2)) / lddt_denom

    return {
        "distogram_acc": distogram_acc,
        "contact_precision_8A": contact_precision,
        "contact_recall_8A": contact_recall,
        "lddt_from_distogram": lddt_from_distogram,
        "mean_distance_error": mean_distance_error,
    }


def select_eval_stem(pair_precision: str, linear_precision: str) -> str:
    if pair_precision == "bf16" and linear_precision == "bf16":
        return "bf16_baseline_eval_metrics"
    if pair_precision == "fp8_native" and linear_precision == "fp8":
        return "fp8_native_eval_metrics"
    if pair_precision == "fp8_native_gold_packs" and linear_precision == "fp8":
        return "fp8_native_gold_packs_eval_metrics"
    return f"{pair_precision}_{linear_precision}_eval_metrics".replace("/", "_").replace(" ", "_")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def spearman_rank_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError(f"Spearman inputs must have same length, got {len(xs)} and {len(ys)}")
    if len(xs) < 2:
        return 1.0
    rx = torch.tensor(_rankdata(xs), dtype=torch.float64)
    ry = torch.tensor(_rankdata(ys), dtype=torch.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.linalg.vector_norm(rx) * torch.linalg.vector_norm(ry)
    if denom.item() == 0:
        return 1.0 if torch.equal(rx, ry) else 0.0
    return float((rx * ry).sum().item() / denom.item())


def build_comparison_rows(bf16_payload: dict, fp8_payload: dict) -> list[dict]:
    bf16_by_id = {row["protein_id"]: row for row in bf16_payload["per_protein"]}
    fp8_by_id = {row["protein_id"]: row for row in fp8_payload["per_protein"]}
    shared_ids = sorted(set(bf16_by_id) & set(fp8_by_id))
    rows = []
    for protein_id in shared_ids:
        bf16_row = bf16_by_id[protein_id]
        fp8_row = fp8_by_id[protein_id]
        rows.append(
            {
                "protein_id": protein_id,
                "pdb_id": bf16_row.get("pdb_id", ""),
                "chain_id": bf16_row.get("chain_id", ""),
                "num_residues": bf16_row.get("num_residues"),
                "bf16_loss": bf16_row["loss"],
                "fp8_loss": fp8_row["loss"],
                "loss_delta": fp8_row["loss"] - bf16_row["loss"],
                "bf16_lddt": bf16_row["lddt_from_distogram"],
                "fp8_lddt": fp8_row["lddt_from_distogram"],
                "lddt_delta": fp8_row["lddt_from_distogram"] - bf16_row["lddt_from_distogram"],
                "bf16_distogram_acc": bf16_row["distogram_acc"],
                "fp8_distogram_acc": fp8_row["distogram_acc"],
                "distogram_acc_delta": fp8_row["distogram_acc"] - bf16_row["distogram_acc"],
                "bf16_mean_distance_error": bf16_row["mean_distance_error"],
                "fp8_mean_distance_error": fp8_row["mean_distance_error"],
                "mean_distance_error_delta": fp8_row["mean_distance_error"] - bf16_row["mean_distance_error"],
                "bf16_contact_precision_8A": bf16_row["contact_precision_8A"],
                "fp8_contact_precision_8A": fp8_row["contact_precision_8A"],
                "contact_precision_8A_delta": fp8_row["contact_precision_8A"] - bf16_row["contact_precision_8A"],
                "bf16_contact_recall_8A": bf16_row["contact_recall_8A"],
                "fp8_contact_recall_8A": fp8_row["contact_recall_8A"],
                "contact_recall_8A_delta": fp8_row["contact_recall_8A"] - bf16_row["contact_recall_8A"],
            }
        )
    return rows


def evaluate_acceptance(bf16_payload: dict, fp8_payload: dict, comparison_rows: list[dict]) -> dict:
    bf16_summary = bf16_payload["summary"]
    fp8_summary = fp8_payload["summary"]

    lddt_corr = spearman_rank_correlation(
        [row["bf16_lddt"] for row in comparison_rows],
        [row["fp8_lddt"] for row in comparison_rows],
    )
    max_lddt_drop = min((row["lddt_delta"] for row in comparison_rows), default=0.0)

    mean_distance_ref = bf16_summary["mean_distance_error"]
    mean_distance_delta = fp8_summary["mean_distance_error"] - mean_distance_ref
    if abs(mean_distance_ref) < 1e-12:
        mean_distance_rel = 0.0 if abs(mean_distance_delta) < 1e-12 else float("inf")
    else:
        mean_distance_rel = mean_distance_delta / mean_distance_ref

    bands = {
        "lddt": {
            "passed": abs(fp8_summary["lddt_from_distogram"] - bf16_summary["lddt_from_distogram"]) <= 0.01
            and lddt_corr >= 0.95,
            "delta": fp8_summary["lddt_from_distogram"] - bf16_summary["lddt_from_distogram"],
            "spearman": lddt_corr,
        },
        "distogram_acc": {
            "passed": abs(fp8_summary["distogram_acc"] - bf16_summary["distogram_acc"]) <= 0.005,
            "delta": fp8_summary["distogram_acc"] - bf16_summary["distogram_acc"],
        },
        "mean_distance_error": {
            "passed": abs(mean_distance_rel) <= 0.01,
            "delta": mean_distance_delta,
            "relative_delta": mean_distance_rel,
        },
        "catastrophic_outlier": {
            "passed": max_lddt_drop >= -0.05,
            "worst_delta": max_lddt_drop,
        },
    }

    failed = [name for name, details in bands.items() if not details["passed"]]
    if not failed:
        verdict = "PASS"
    elif len(failed) == 1:
        verdict = "PARTIAL PASS"
    else:
        verdict = "FAIL"

    return {
        "bands": bands,
        "failed_bands": failed,
        "verdict": verdict,
        "protein_count": len(comparison_rows),
        "outliers": sorted(
            (row for row in comparison_rows if row["lddt_delta"] < -0.05),
            key=lambda row: row["lddt_delta"],
        ),
    }
