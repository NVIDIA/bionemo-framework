from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

import torch

from distributed_config import DistributedConfig
from eval_accuracy_utils import append_status_report, utc_now_iso, write_json
from plain_runtime_diagnostics import (
    build_mode_args,
    build_plain_runtime_from_args,
    cleanup_model,
    compose_eval_args,
    destroy_distributed_if_initialized,
    extract_dataset_sample,
    load_state_dict_for_eval,
    prepare_miniformer_input,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _variant_specs(include_hybrid: bool, hybrid_config: dict[str, bool] | None) -> list[dict]:
    specs = [
        {"name": "bf16", "pair_precision": "bf16", "linear_precision": "bf16", "tri_impl": "cublas_xbdnn", "hybrid_precision": None},
        {"name": "fp8_extreme", "pair_precision": "fp8_extreme", "linear_precision": "fp8", "tri_impl": "fp8_cublaslt", "hybrid_precision": None},
        {"name": "fp8_native", "pair_precision": "fp8_native", "linear_precision": "fp8", "tri_impl": "fp8_cublaslt", "hybrid_precision": None},
    ]
    if include_hybrid:
        specs.append(
            {
                "name": "fp8_hybrid",
                "pair_precision": "fp8_hybrid",
                "linear_precision": "fp8",
                "tri_impl": "fp8_cublaslt",
                "hybrid_precision": hybrid_config or {},
            }
        )
    return specs


def _make_hybrid_config(cli) -> dict[str, bool]:
    return {
        "use_native_layernorm": cli.use_native_layernorm,
        "use_native_linear": cli.use_native_linear,
        "use_native_gate": cli.use_native_gate,
        "use_native_tri": cli.use_native_tri,
        "use_resident_fp8_residual": cli.use_resident_fp8_residual,
    }


def _tensor_from_event(probe, event_by_op: dict[str, object], op_name: str) -> torch.Tensor:
    event = event_by_op[op_name]
    if event.snapshot_index is None:
        raise ValueError(f"Event {op_name} does not retain a tensor snapshot")
    return probe.tensor_snapshots[event.snapshot_index].to(torch.float32)


def _compare_variant_to_bf16(reference_probe, variant_probe) -> list[dict]:
    ref_by_op = {event.op_name: event for event in reference_probe.events}
    variant_by_op = {event.op_name: event for event in variant_probe.events}
    shared_ops = [op_name for op_name in ref_by_op if op_name in variant_by_op]
    rows = []
    for op_name in shared_ops:
        ref_tensor = _tensor_from_event(reference_probe, ref_by_op, op_name)
        variant_tensor = _tensor_from_event(variant_probe, variant_by_op, op_name)
        diff = variant_tensor - ref_tensor
        diff_norm = torch.linalg.vector_norm(diff)
        ref_norm = torch.linalg.vector_norm(ref_tensor)
        rows.append(
            {
                "op_name": op_name,
                "reference_mode_op_name": ref_by_op[op_name].mode_op_name,
                "variant_mode_op_name": variant_by_op[op_name].mode_op_name,
                "reference_max_abs": ref_by_op[op_name].max_abs,
                "variant_max_abs": variant_by_op[op_name].max_abs,
                "max_abs_diff": float(diff.abs().amax().item()),
                "rel_l2_error": float(diff_norm.item() / max(ref_norm.item(), 1e-12)),
                "variant_nan_count": variant_by_op[op_name].nan_count,
                "variant_inf_count": variant_by_op[op_name].inf_count,
            }
        )
    return rows


def _find_native_culprit(fp8_extreme_rows: list[dict], fp8_native_rows: list[dict]) -> dict | None:
    extreme_by_op = {row["op_name"]: row for row in fp8_extreme_rows}
    for row in fp8_native_rows:
        extreme_row = extreme_by_op.get(row["op_name"])
        if extreme_row is None:
            continue
        native_error = row["rel_l2_error"]
        extreme_error = extreme_row["rel_l2_error"]
        if native_error >= 10.0 * max(extreme_error, 1e-12):
            return {
                "op_name": row["op_name"],
                "native_mode_op_name": row["variant_mode_op_name"],
                "extreme_mode_op_name": extreme_row["variant_mode_op_name"],
                "native_rel_l2_error": native_error,
                "extreme_rel_l2_error": extreme_error,
                "error_ratio": native_error / max(extreme_error, 1e-12),
            }
    return None


def _dequantize_tensor(tensor) -> torch.Tensor:
    if hasattr(tensor, "dequantize"):
        try:
            return tensor.dequantize(dtype=torch.float32)
        except TypeError:
            return tensor.dequantize().to(torch.float32)
    return tensor.detach().to(torch.float32)


def _scale_summary(scale: torch.Tensor | None) -> dict | None:
    if scale is None:
        return None
    scale_f32 = scale.detach().to(torch.float32)
    numel = int(scale_f32.numel())
    if numel == 0:
        return {
            "shape": tuple(scale_f32.shape),
            "numel": 0,
            "zero_count": 0,
            "nonfinite_count": 0,
            "finite_count": 0,
            "max_abs": None,
            "mean_abs": None,
            "finite_unique_count": 0,
            "scale_unique_ratio": 0.0,
            "all_zero": False,
            "constant_finite_value": False,
        }
    finite_mask = torch.isfinite(scale_f32)
    finite_count = int(finite_mask.sum().item())
    nonfinite_count = numel - finite_count
    zero_count = int((scale_f32 == 0).sum().item())
    finite_values = scale_f32[finite_mask]
    if finite_count > 0:
        finite_values_cpu = finite_values.detach().cpu()
        finite_unique_count = int(torch.unique(finite_values_cpu).numel())
        max_abs = float(finite_values.abs().amax().item())
        mean_abs = float(finite_values.abs().mean().item())
    else:
        finite_unique_count = 0
        max_abs = None
        mean_abs = None
    return {
        "shape": tuple(scale_f32.shape),
        "numel": numel,
        "zero_count": zero_count,
        "nonfinite_count": nonfinite_count,
        "finite_count": finite_count,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "finite_unique_count": finite_unique_count,
        "scale_unique_ratio": float(finite_unique_count / max(finite_count, 1)),
        "all_zero": zero_count == numel and numel > 0,
        "constant_finite_value": finite_count > 0 and finite_unique_count == 1,
    }


def _tensor_summary(tensor) -> dict:
    dequantized = _dequantize_tensor(tensor)
    if hasattr(tensor, "payload") and hasattr(tensor, "scale"):
        payload_shape = tuple(tensor.payload.shape)
        scale = tensor.scale
        scale_shape = tuple(scale.shape)
        tensor_format = "mxfp8_block32"
    else:
        payload_shape = None
        scale = None
        scale_shape = None
        tensor_format = str(dequantized.dtype).replace("torch.", "")

    abs_tensor = dequantized.abs()
    nan_count = int(torch.isnan(dequantized).sum().item())
    inf_count = int(torch.isinf(dequantized).sum().item())
    return {
        "tensor_format": tensor_format,
        "shape": tuple(dequantized.shape),
        "payload_shape": payload_shape,
        "scale_shape": scale_shape,
        "max_abs": float(abs_tensor.amax().item()) if abs_tensor.numel() > 0 else 0.0,
        "mean_abs": float(abs_tensor.mean().item()) if abs_tensor.numel() > 0 else 0.0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "scale": _scale_summary(scale),
    }


def _rel_l2_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    diff_norm = torch.linalg.vector_norm(candidate - reference)
    ref_norm = torch.linalg.vector_norm(reference)
    return float(diff_norm.item() / max(ref_norm.item(), 1e-12))


def _tensor_comparison(reference, candidate) -> dict:
    reference_tensor = _dequantize_tensor(reference)
    candidate_tensor = _dequantize_tensor(candidate)
    comparison = {
        "payload_max_abs_diff": float((candidate_tensor - reference_tensor).abs().amax().item()),
        "payload_rel_l2_error": _rel_l2_error(candidate_tensor, reference_tensor),
    }
    reference_scale = getattr(reference, "scale", None)
    candidate_scale = getattr(candidate, "scale", None)
    if isinstance(reference_scale, torch.Tensor) and isinstance(candidate_scale, torch.Tensor):
        reference_scale_f32 = reference_scale.detach().to(torch.float32)
        candidate_scale_f32 = candidate_scale.detach().to(torch.float32)
        comparison["scale_max_abs_diff"] = float((candidate_scale_f32 - reference_scale_f32).abs().amax().item())
        comparison["scale_rel_l2_error"] = _rel_l2_error(candidate_scale_f32, reference_scale_f32)
    else:
        comparison["scale_max_abs_diff"] = None
        comparison["scale_rel_l2_error"] = None
    return comparison


def _make_stage_row(
    *,
    rung: str,
    stage_name: str,
    stage_input,
    reference_output,
    native_output,
    reference_mode_op_name: str,
    native_mode_op_name: str,
) -> dict:
    return {
        "rung": rung,
        "stage_name": stage_name,
        "reference_mode_op_name": reference_mode_op_name,
        "native_mode_op_name": native_mode_op_name,
        "input": _tensor_summary(stage_input),
        "reference": _tensor_summary(reference_output),
        "native": _tensor_summary(native_output),
        "comparison": _tensor_comparison(reference_output, native_output),
    }


def _find_first_bad_stage(scale_audit_rows: list[dict]) -> dict | None:
    for row in scale_audit_rows:
        native_summary = row["native"]
        native_scale = native_summary.get("scale")
        if native_summary["nan_count"] > 0 or native_summary["inf_count"] > 0:
            return {
                "rung": row["rung"],
                "stage_name": row["stage_name"],
                "native_mode_op_name": row["native_mode_op_name"],
                "reason": "native payload produced nonfinite values",
            }
        if native_scale is None:
            continue
        if native_scale["nonfinite_count"] > 0:
            return {
                "rung": row["rung"],
                "stage_name": row["stage_name"],
                "native_mode_op_name": row["native_mode_op_name"],
                "reason": "native scale tensor produced nonfinite values",
            }
        if native_scale["all_zero"]:
            return {
                "rung": row["rung"],
                "stage_name": row["stage_name"],
                "native_mode_op_name": row["native_mode_op_name"],
                "reason": "native scale tensor collapsed to all zeros",
            }
        if native_scale["constant_finite_value"] and row["reference"]["scale"] is not None:
            reference_scale = row["reference"]["scale"]
            if reference_scale["finite_unique_count"] > 1:
                return {
                    "rung": row["rung"],
                    "stage_name": row["stage_name"],
                    "native_mode_op_name": row["native_mode_op_name"],
                    "reason": "native scale tensor collapsed to a single repeated finite value",
                }
    return None


def _build_tri_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    mask_bool = mask.to(torch.bool)
    if mask_bool.dim() == 2:
        return (mask_bool[:, :, None] & mask_bool[:, None, :]).contiguous()
    if mask_bool.dim() == 3:
        return mask_bool.contiguous()
    raise ValueError(f"Unsupported mask rank for tri path: {mask_bool.dim()}")


def _compare_raw_tensors(reference: torch.Tensor, candidate: torch.Tensor, *, rtol: float = 1e-3, atol: float = 1e-3) -> dict:
    reference_f32 = _dequantize_tensor(reference)
    candidate_f32 = _dequantize_tensor(candidate)
    return {
        "exact_match": bool(reference.dtype == candidate.dtype and torch.equal(reference, candidate)),
        "allclose": bool(torch.allclose(candidate_f32, reference_f32, rtol=rtol, atol=atol, equal_nan=False)),
        "max_abs_diff": float((candidate_f32 - reference_f32).abs().amax().item()),
        "rel_l2_error": _rel_l2_error(candidate_f32, reference_f32),
    }


def _make_tri_bisect_row(
    *,
    subboundary: str,
    component: str,
    reference_mode_op_name: str,
    native_mode_op_name: str,
    reference_payload: torch.Tensor,
    native_payload: torch.Tensor,
    reference_scale: torch.Tensor | None = None,
    native_scale: torch.Tensor | None = None,
    payload_rtol: float = 1e-3,
    payload_atol: float = 1e-3,
) -> dict:
    row = {
        "subboundary": subboundary,
        "component": component,
        "reference_mode_op_name": reference_mode_op_name,
        "native_mode_op_name": native_mode_op_name,
        "reference_payload": _tensor_summary(reference_payload),
        "native_payload": _tensor_summary(native_payload),
        "payload_comparison": _compare_raw_tensors(reference_payload, native_payload, rtol=payload_rtol, atol=payload_atol),
        "reference_scale": _scale_summary(reference_scale),
        "native_scale": _scale_summary(native_scale),
        "scale_comparison": _compare_raw_tensors(reference_scale, native_scale) if reference_scale is not None and native_scale is not None else None,
    }
    return row


def _find_first_bad_tri_subboundary(rows: list[dict]) -> dict | None:
    for row in rows:
        native_payload = row["native_payload"]
        native_scale = row["native_scale"]
        payload_cmp = row["payload_comparison"]
        scale_cmp = row["scale_comparison"]
        reference_scale = row["reference_scale"]

        if native_payload["nan_count"] > 0 or native_payload["inf_count"] > 0:
            return {
                "subboundary": row["subboundary"],
                "component": row["component"],
                "native_mode_op_name": row["native_mode_op_name"],
                "reason": "native payload produced nonfinite values",
            }
        if native_scale is not None and native_scale["nonfinite_count"] > 0:
            return {
                "subboundary": row["subboundary"],
                "component": row["component"],
                "native_mode_op_name": row["native_mode_op_name"],
                "reason": "native scale tensor produced nonfinite values",
            }
        if native_scale is not None and reference_scale is not None and native_scale["all_zero"] and not reference_scale["all_zero"]:
            return {
                "subboundary": row["subboundary"],
                "component": row["component"],
                "native_mode_op_name": row["native_mode_op_name"],
                "reason": "native scale tensor collapsed to all zeros",
            }
        if row["subboundary"] in {"pack", "repack"}:
            if not payload_cmp["exact_match"]:
                return {
                    "subboundary": row["subboundary"],
                    "component": row["component"],
                    "native_mode_op_name": row["native_mode_op_name"],
                    "reason": "native payload does not exactly match the Python gold path",
                }
            if scale_cmp is not None and not scale_cmp["exact_match"]:
                return {
                    "subboundary": row["subboundary"],
                    "component": row["component"],
                    "native_mode_op_name": row["native_mode_op_name"],
                    "reason": "native scale tensor does not exactly match the Python gold path",
                }
        if row["subboundary"] == "gemm" and not payload_cmp["allclose"]:
            return {
                "subboundary": row["subboundary"],
                "component": row["component"],
                "native_mode_op_name": row["native_mode_op_name"],
                "reason": "native GEMM output diverged from the raw FP8 reference",
            }
    return None


def _run_scale_audit(block, block_input: torch.Tensor, pair_mask: torch.Tensor, plain_infer) -> dict:
    stage_rows = []
    block_input_q = plain_infer.Mxfp8PairTensor.from_tensor(block_input.clone(), scale_dtype=torch.float32)
    stage0 = {
        "rung": "S0",
        "stage_name": "block_input",
        "reference_mode_op_name": "block_input",
        "native_mode_op_name": "block_input",
        "input": _tensor_summary(block_input_q),
        "reference": _tensor_summary(block_input_q),
        "native": _tensor_summary(block_input_q),
        "comparison": {
            "payload_max_abs_diff": 0.0,
            "payload_rel_l2_error": 0.0,
            "scale_max_abs_diff": 0.0,
            "scale_rel_l2_error": 0.0,
        },
    }
    stage_rows.append(stage0)

    stage1_extreme = plain_infer.mxfp8_layernorm_quantized(block.triangular.input_norm, block_input_q)
    stage1_native = plain_infer.native_mxfp8_layernorm_quantized(block.triangular.input_norm, block_input_q)
    stage_rows.append(
        _make_stage_row(
            rung="S1",
            stage_name="tri_input_norm",
            stage_input=block_input_q,
            reference_output=stage1_extreme,
            native_output=stage1_native,
            reference_mode_op_name="mxfp8_layernorm_quantized",
            native_mode_op_name="native_mxfp8_layernorm_quantized",
        )
    )

    gate_input = stage1_extreme
    stage2_pi = plain_infer.mxfp8_linear_forward_quantized(block.triangular.pi, gate_input)
    stage2_gi = plain_infer.mxfp8_linear_forward_quantized(block.triangular.gi, gate_input)
    stage2_extreme = plain_infer.mxfp8_mul_quantized(stage2_pi, plain_infer.mxfp8_sigmoid_quantized(stage2_gi))
    stage2_native = plain_infer.native_gate_sigmoid_mul_quantized(
        block.triangular.pi,
        block.triangular.gi,
        gate_input,
        residual=None,
    )
    stage_rows.append(
        _make_stage_row(
            rung="S2",
            stage_name="tri_gated",
            stage_input=gate_input,
            reference_output=stage2_extreme,
            native_output=stage2_native,
            reference_mode_op_name="mxfp8_gate_decomposition",
            native_mode_op_name="native_gate_sigmoid_mul_quantized",
        )
    )

    tri_input = stage2_extreme
    stage3_extreme = plain_infer.mxfp8_tri_mul_fp8_cublaslt_quantized(
        tri_input,
        out_dtype=torch.float16,
        mask=pair_mask,
    )
    stage3_native = plain_infer.native_tri_mul_from_block32_quantized(tri_input, mask=pair_mask)
    stage_rows.append(
        _make_stage_row(
            rung="S3",
            stage_name="tri_mul_out",
            stage_input=tri_input,
            reference_output=stage3_extreme,
            native_output=stage3_native,
            reference_mode_op_name="mxfp8_tri_mul_fp8_cublaslt_quantized",
            native_mode_op_name="native_tri_mul_from_block32_quantized",
        )
    )

    stage4_extreme = stage3_extreme
    stage4_native = plain_infer.native_tri_mul_from_input_quantized(
        block.triangular.input_norm,
        block.triangular.pi,
        block.triangular.gi,
        block_input_q,
        mask=pair_mask,
    )
    stage_rows.append(
        _make_stage_row(
            rung="S4",
            stage_name="tri_native_fused_from_input",
            stage_input=block_input_q,
            reference_output=stage4_extreme,
            native_output=stage4_native,
            reference_mode_op_name="decomposed_extreme_tri_path",
            native_mode_op_name="native_tri_mul_from_input_quantized",
        )
    )

    return {
        "rows": stage_rows,
        "first_bad_stage": _find_first_bad_stage(stage_rows),
    }


def _render_scale_audit_markdown(payload: dict) -> str:
    lines = [
        "# Single-Block Scale Audit",
        "",
        f"- Generated: {payload['generated_utc']}",
        f"- Sample: {payload['sample']['pdb_id']}:{payload['sample']['chain_id']}",
        f"- Block index: {payload['block_index']}",
        f"- Checkpoint: {payload['checkpoint']['resolved_ckpt_dir']}",
    ]
    first_bad_stage = payload["first_bad_stage"]
    if first_bad_stage is None:
        lines.append("- First bad native stage: none")
    else:
        lines.append(
            "- First bad native stage: "
            f"{first_bad_stage['rung']} {first_bad_stage['stage_name']} "
            f"({first_bad_stage['native_mode_op_name']}) because {first_bad_stage['reason']}"
        )
    lines.extend(
        [
            "",
            "| Rung | Stage | Reference | Native | Native NaN | Native Inf | Native Scale Zero | Native Scale Nonfinite | Native Scale Unique Ratio | Payload Rel L2 | Scale Rel L2 |",
            "|------|-------|-----------|--------|------------|------------|-------------------|------------------------|---------------------------|----------------|--------------|",
        ]
    )
    for row in payload["rows"]:
        native_scale = row["native"]["scale"] or {}
        lines.append(
            f"| {row['rung']} | {row['stage_name']} | {row['reference_mode_op_name']} | {row['native_mode_op_name']} | "
            f"{row['native']['nan_count']} | {row['native']['inf_count']} | "
            f"{native_scale.get('zero_count', 'n/a')} | {native_scale.get('nonfinite_count', 'n/a')} | "
            f"{native_scale.get('scale_unique_ratio', 'n/a')} | "
            f"{row['comparison']['payload_rel_l2_error']:.6e} | "
            f"{row['comparison']['scale_rel_l2_error'] if row['comparison']['scale_rel_l2_error'] is not None else 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def _run_native_tri_bisect(block, block_input: torch.Tensor, pair_mask: torch.Tensor, plain_infer) -> dict:
    if plain_infer.minifold_native_raw is None:
        raise RuntimeError("Native extension is unavailable; cannot run native tri bisect")
    required_debug_ops = (
        "pack_block32_to_mxfp8_fused_debug",
        "tri_mul_pair_from_packed_debug",
        "tri_pair_to_block32_carrier_debug",
    )
    for op_name in required_debug_ops:
        if not hasattr(plain_infer.minifold_native_raw, op_name):
            raise RuntimeError(f"Native extension is missing debug op {op_name}")
    if plain_infer.bmm_ext_raw is None:
        raise RuntimeError("FP8 BMM raw extension is unavailable; cannot run native tri bisect")

    tri_mask = _build_tri_mask(pair_mask)
    block_input_q = plain_infer.Mxfp8PairTensor.from_tensor(block_input.clone(), scale_dtype=torch.float32)
    stage1_extreme = plain_infer.mxfp8_layernorm_quantized(block.triangular.input_norm, block_input_q)
    stage2_pi = plain_infer.mxfp8_linear_forward_quantized(block.triangular.pi, stage1_extreme)
    stage2_gi = plain_infer.mxfp8_linear_forward_quantized(block.triangular.gi, stage1_extreme)
    tri_input = plain_infer.mxfp8_mul_quantized(stage2_pi, plain_infer.mxfp8_sigmoid_quantized(stage2_gi))

    gold_a1, gold_a1_scale = plain_infer.fp8_pack_block32_carrier_to_mxfp8_lhs(
        tri_input.payload,
        tri_input.scale,
        mask=tri_mask,
        channel_group=0,
        transpose=False,
    )
    gold_b1, gold_b1_scale = plain_infer.fp8_pack_block32_carrier_to_mxfp8_lhs(
        tri_input.payload,
        tri_input.scale,
        mask=tri_mask,
        channel_group=1,
        transpose=False,
    )
    gold_a2_t, gold_a2_t_scale = plain_infer.fp8_pack_block32_carrier_to_mxfp8_lhs(
        tri_input.payload,
        tri_input.scale,
        mask=tri_mask,
        channel_group=2,
        transpose=True,
    )
    gold_b2_rhs, gold_b2_rhs_scale = plain_infer.fp8_pack_block32_carrier_to_mxfp8_rhs(
        tri_input.payload,
        tri_input.scale,
        mask=tri_mask,
        channel_group=3,
    )
    native_pack = plain_infer.minifold_native_raw.pack_block32_to_mxfp8_fused_debug(
        tri_input.payload.contiguous(),
        tri_input.scale.contiguous(),
        tri_mask,
    )
    pack_rows = [
        _make_tri_bisect_row(
            subboundary="pack",
            component="a1",
            reference_mode_op_name="fp8_pack_block32_carrier_to_mxfp8_lhs(group=0, transpose=False)",
            native_mode_op_name="pack_block32_to_mxfp8_fused_debug",
            reference_payload=gold_a1,
            native_payload=native_pack[0],
            reference_scale=gold_a1_scale,
            native_scale=native_pack[1],
        ),
        _make_tri_bisect_row(
            subboundary="pack",
            component="b1",
            reference_mode_op_name="fp8_pack_block32_carrier_to_mxfp8_lhs(group=1, transpose=False)",
            native_mode_op_name="pack_block32_to_mxfp8_fused_debug",
            reference_payload=gold_b1,
            native_payload=native_pack[2],
            reference_scale=gold_b1_scale,
            native_scale=native_pack[3],
        ),
        _make_tri_bisect_row(
            subboundary="pack",
            component="a2_t",
            reference_mode_op_name="fp8_pack_block32_carrier_to_mxfp8_lhs(group=2, transpose=True)",
            native_mode_op_name="pack_block32_to_mxfp8_fused_debug",
            reference_payload=gold_a2_t,
            native_payload=native_pack[4],
            reference_scale=gold_a2_t_scale,
            native_scale=native_pack[5],
        ),
        _make_tri_bisect_row(
            subboundary="pack",
            component="b2_rhs",
            reference_mode_op_name="fp8_pack_block32_carrier_to_mxfp8_rhs(group=3)",
            native_mode_op_name="pack_block32_to_mxfp8_fused_debug",
            reference_payload=gold_b2_rhs,
            native_payload=native_pack[6],
            reference_scale=gold_b2_rhs_scale,
            native_scale=native_pack[7],
        ),
    ]

    gold_x1, gold_x2 = plain_infer.bmm_ext_raw.mxfp8_cublaslt_tri_mul_pair(
        gold_a1.contiguous(),
        gold_b1.contiguous(),
        gold_a2_t.contiguous(),
        gold_b2_rhs.contiguous(),
        gold_a1_scale.contiguous(),
        gold_b1_scale.contiguous(),
        gold_a2_t_scale.contiguous(),
        gold_b2_rhs_scale.contiguous(),
        "float16",
    )
    native_x1, native_x2 = plain_infer.minifold_native_raw.tri_mul_pair_from_packed_debug(
        gold_a1.contiguous(),
        gold_b1.contiguous(),
        gold_a2_t.contiguous(),
        gold_b2_rhs.contiguous(),
        gold_a1_scale.contiguous(),
        gold_b1_scale.contiguous(),
        gold_a2_t_scale.contiguous(),
        gold_b2_rhs_scale.contiguous(),
        "float16",
    )
    gemm_rows = [
        _make_tri_bisect_row(
            subboundary="gemm",
            component="x1",
            reference_mode_op_name="bmm_ext_raw.mxfp8_cublaslt_tri_mul_pair",
            native_mode_op_name="tri_mul_pair_from_packed_debug",
            reference_payload=gold_x1,
            native_payload=native_x1,
            payload_rtol=1e-3,
            payload_atol=1e-3,
        ),
        _make_tri_bisect_row(
            subboundary="gemm",
            component="x2",
            reference_mode_op_name="bmm_ext_raw.mxfp8_cublaslt_tri_mul_pair",
            native_mode_op_name="tri_mul_pair_from_packed_debug",
            reference_payload=gold_x2,
            native_payload=native_x2,
            payload_rtol=1e-3,
            payload_atol=1e-3,
        ),
    ]

    batch = int(tri_input.payload.shape[0])
    gold_repack_payload, gold_repack_scale = plain_infer.fp8_tri_outputs_to_block32_carrier(
        gold_x1, gold_x2, batch=batch, scale_dtype=tri_input.scale.dtype
    )
    native_repack_payload, native_repack_scale = plain_infer.minifold_native_raw.tri_pair_to_block32_carrier_debug(
        gold_x1.contiguous(),
        gold_x2.contiguous(),
        batch,
    )
    repack_rows = [
        _make_tri_bisect_row(
            subboundary="repack",
            component="carrier",
            reference_mode_op_name="fp8_tri_outputs_to_block32_carrier",
            native_mode_op_name="tri_pair_to_block32_carrier_debug",
            reference_payload=gold_repack_payload,
            native_payload=native_repack_payload,
            reference_scale=gold_repack_scale,
            native_scale=native_repack_scale,
        )
    ]
    rows = pack_rows + gemm_rows + repack_rows
    return {
        "rows": rows,
        "first_bad_subboundary": _find_first_bad_tri_subboundary(rows),
    }


def _render_tri_bisect_markdown(payload: dict) -> str:
    lines = [
        "# Native Tri Bisect",
        "",
        f"- Generated: {payload['generated_utc']}",
        f"- Sample: {payload['sample']['pdb_id']}:{payload['sample']['chain_id']}",
        f"- Block index: {payload['block_index']}",
        f"- Checkpoint: {payload['checkpoint']['resolved_ckpt_dir']}",
    ]
    first_bad = payload["first_bad_subboundary"]
    if first_bad is None:
        lines.append("- First bad tri sub-boundary: none")
    else:
        lines.append(
            "- First bad tri sub-boundary: "
            f"{first_bad['subboundary']} {first_bad['component']} "
            f"({first_bad['native_mode_op_name']}) because {first_bad['reason']}"
        )
    lines.extend(
        [
            "",
            "| Boundary | Component | Reference | Native | Payload Exact | Payload Allclose | Payload Rel L2 | Native NaN | Native Inf | Scale Exact | Scale Rel L2 | Native Scale Zero | Native Scale Nonfinite |",
            "|----------|-----------|-----------|--------|---------------|------------------|----------------|------------|------------|-------------|--------------|-------------------|------------------------|",
        ]
    )
    for row in payload["rows"]:
        native_scale = row["native_scale"] or {}
        scale_cmp = row["scale_comparison"] or {}
        lines.append(
            f"| {row['subboundary']} | {row['component']} | {row['reference_mode_op_name']} | {row['native_mode_op_name']} | "
            f"{row['payload_comparison']['exact_match']} | {row['payload_comparison']['allclose']} | "
            f"{row['payload_comparison']['rel_l2_error']:.6e} | {row['native_payload']['nan_count']} | {row['native_payload']['inf_count']} | "
            f"{scale_cmp.get('exact_match', 'n/a')} | {scale_cmp.get('rel_l2_error', 'n/a')} | "
            f"{native_scale.get('zero_count', 'n/a')} | {native_scale.get('nonfinite_count', 'n/a')} |"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Single-Block Equivalence",
        "",
        f"- Generated: {payload['generated_utc']}",
        f"- Sample: {payload['sample']['pdb_id']}:{payload['sample']['chain_id']}",
        f"- Block index: {payload['block_index']}",
        f"- Checkpoint: {payload['checkpoint']['resolved_ckpt_dir']}",
        "",
    ]
    culprit = payload["native_vs_extreme_culprit"]
    if culprit is None:
        lines.append("- First `fp8_native >= 10x fp8_extreme` error point: none")
    else:
        lines.append(
            "- First `fp8_native >= 10x fp8_extreme` error point: "
            f"{culprit['op_name']} ({culprit['native_mode_op_name']}) "
            f"ratio={culprit['error_ratio']:.3f}"
        )
    for variant_name, rows in payload["comparisons"].items():
        lines.extend(
            [
                "",
                f"## {variant_name} vs bf16",
                "",
                "| Op | Reference | Variant | Max Abs Diff | Rel L2 Error | NaN | Inf |",
                "|----|-----------|---------|--------------|--------------|-----|-----|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['op_name']} | {row['reference_mode_op_name']} | {row['variant_mode_op_name']} | "
                f"{row['max_abs_diff']:.6e} | {row['rel_l2_error']:.6e} | "
                f"{row['variant_nan_count']} | {row['variant_inf_count']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a single Pairformer block across bf16, fp8_extreme, and fp8_native.")
    parser.add_argument("--config-name", default="eval_real_3B_fp8native")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path, default=Path("/scratch/claude_tasks/accuracy_validation"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--scale-audit-json", type=Path, default=None)
    parser.add_argument("--scale-audit-markdown", type=Path, default=None)
    parser.add_argument("--include-hybrid", action="store_true")
    parser.add_argument("--use-native-layernorm", action="store_true")
    parser.add_argument("--use-native-linear", action="store_true")
    parser.add_argument("--use-native-gate", action="store_true")
    parser.add_argument("--use-native-tri", action="store_true")
    parser.add_argument("--use-resident-fp8-residual", action="store_true")
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()

    phase_a_dir = cli.artifact_root / "artifacts" / "phase_a"
    phase_b_dir = cli.artifact_root / "artifacts" / "phase_b"
    json_path = cli.output_json or (phase_a_dir / "single_block_equivalence.json")
    markdown_path = cli.output_markdown or (phase_a_dir / "single_block_equivalence.md")
    scale_audit_json_path = cli.scale_audit_json or (phase_a_dir / "single_block_scale_audit.json")
    scale_audit_markdown_path = cli.scale_audit_markdown or (phase_a_dir / "single_block_scale_audit.md")
    tri_bisect_json_path = phase_b_dir / "native_tri_bisect.json"
    tri_bisect_markdown_path = phase_b_dir / "native_tri_bisect.md"
    status_path = cli.artifact_root / "status_report.md"

    base_args = compose_eval_args(cli.config_name, cli.overrides, artifact_root=cli.artifact_root)
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(dist_config.local_rank)

    state_dict, checkpoint_info = load_state_dict_for_eval(base_args, dist_config, device)
    destroy_distributed_if_initialized()
    sample_batch, sample_metadata = extract_dataset_sample(base_args.eval_dataset, cli.sample_index)
    sample_batch = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value for key, value in sample_batch.items()}

    hybrid_config = _make_hybrid_config(cli)
    variant_specs = _variant_specs(cli.include_hybrid, hybrid_config)

    reference_spec = variant_specs[0]
    reference_args = build_mode_args(
        base_args,
        pair_precision=reference_spec["pair_precision"],
        linear_precision=reference_spec["linear_precision"],
        tri_impl=reference_spec["tri_impl"],
        hybrid_precision=reference_spec["hybrid_precision"],
    )
    reference_model, reference_plain_infer, _ = build_plain_runtime_from_args(
        reference_args,
        torch.device("cuda:0"),
        state_dict,
        cli.artifact_root,
        status_path,
    )
    if cli.block_index < 0 or cli.block_index >= len(reference_model.fold.miniformer.blocks):
        raise IndexError(f"block_index {cli.block_index} out of range for {len(reference_model.fold.miniformer.blocks)} blocks")

    block_input, pair_mask = prepare_miniformer_input(reference_model, sample_batch)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for idx in range(cli.block_index):
            block_input = reference_model.fold.miniformer.blocks[idx](block_input, pair_mask, block_idx=idx)
    block_input = block_input.detach().to(torch.bfloat16)
    variant_payloads = {}
    variant_probes = {}
    scale_audit = None
    tri_bisect = None
    for spec in variant_specs:
        mode_args = build_mode_args(
            base_args,
            pair_precision=spec["pair_precision"],
            linear_precision=spec["linear_precision"],
            tri_impl=spec["tri_impl"],
            hybrid_precision=spec["hybrid_precision"],
        )
        model, plain_infer, native_build_info = build_plain_runtime_from_args(
            mode_args,
            torch.device("cuda:0"),
            state_dict,
            cli.artifact_root,
            status_path,
        )
        block = model.fold.miniformer.blocks[cli.block_index]
        probe = plain_infer.ActivationProbe(pair_precision_mode=mode_args.pair_precision, retain_tensors=True)
        if spec["pair_precision"] == "bf16":
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                block(block_input.clone(), pair_mask, block_idx=cli.block_index, activation_probe=probe)
            pair_stats = None
        else:
            stats = plain_infer.FP8ActivationStats(pair_precision_mode=mode_args.pair_precision)
            quantized_input = plain_infer.Mxfp8PairTensor.from_tensor(block_input.clone(), scale_dtype=torch.float32)
            with torch.no_grad():
                block(
                    quantized_input,
                    pair_mask,
                    stats=stats,
                    block_idx=cli.block_index,
                    activation_probe=probe,
                )
            pair_stats = asdict(stats)
        variant_probes[spec["name"]] = probe
        variant_payloads[spec["name"]] = {
            "pair_precision": spec["pair_precision"],
            "linear_precision": spec["linear_precision"],
            "tri_impl": spec["tri_impl"],
            "hybrid_precision": spec["hybrid_precision"],
            "event_count": len(probe.events),
            "events": probe.to_dict()["events"],
            "pair_stats": pair_stats,
            "native_extension": native_build_info,
        }
        if spec["name"] == "fp8_extreme":
            scale_audit = _run_scale_audit(model.fold.miniformer.blocks[cli.block_index], block_input, pair_mask, plain_infer)
            tri_bisect = _run_native_tri_bisect(model.fold.miniformer.blocks[cli.block_index], block_input, pair_mask, plain_infer)
        cleanup_model(model)

    if scale_audit is None:
        raise RuntimeError("Scale audit was not generated; fp8_extreme variant did not run")
    if tri_bisect is None:
        raise RuntimeError("Native tri bisect was not generated; fp8_extreme variant did not run")

    bf16_probe = variant_probes["bf16"]
    comparisons = {
        name: _compare_variant_to_bf16(bf16_probe, probe)
        for name, probe in variant_probes.items()
        if name != "bf16"
    }

    payload = {
        "generated_utc": utc_now_iso(),
        "output_json": str(json_path),
        "output_markdown": str(markdown_path),
        "checkpoint": checkpoint_info,
        "sample": sample_metadata,
        "block_index": cli.block_index,
        "variants": variant_payloads,
        "comparisons": comparisons,
        "native_vs_extreme_culprit": _find_native_culprit(
            comparisons.get("fp8_extreme", []),
            comparisons.get("fp8_native", []),
        ),
    }
    scale_audit_payload = {
        "generated_utc": payload["generated_utc"],
        "output_json": str(scale_audit_json_path),
        "output_markdown": str(scale_audit_markdown_path),
        "checkpoint": checkpoint_info,
        "sample": sample_metadata,
        "block_index": cli.block_index,
        **scale_audit,
    }
    tri_bisect_payload = {
        "generated_utc": payload["generated_utc"],
        "output_json": str(tri_bisect_json_path),
        "output_markdown": str(tri_bisect_markdown_path),
        "checkpoint": checkpoint_info,
        "sample": sample_metadata,
        "block_index": cli.block_index,
        **tri_bisect,
    }
    write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    write_json(scale_audit_json_path, scale_audit_payload)
    scale_audit_markdown_path.write_text(_render_scale_audit_markdown(scale_audit_payload), encoding="utf-8")
    write_json(tri_bisect_json_path, tri_bisect_payload)
    tri_bisect_markdown_path.write_text(_render_tri_bisect_markdown(tri_bisect_payload), encoding="utf-8")
    append_status_report(
        status_path,
        "Single-Block Equivalence",
        [
            f"block_index={cli.block_index}",
            f"sample={sample_metadata['pdb_id']}:{sample_metadata['chain_id']}",
            f"output_json={json_path}",
            f"output_markdown={markdown_path}",
            f"scale_audit_json={scale_audit_json_path}",
            f"scale_audit_markdown={scale_audit_markdown_path}",
            f"tri_bisect_json={tri_bisect_json_path}",
            f"tri_bisect_markdown={tri_bisect_markdown_path}",
            f"first_bad_stage={scale_audit_payload['first_bad_stage']}",
            f"first_bad_tri_subboundary={tri_bisect_payload['first_bad_subboundary']}",
            f"native_vs_extreme_culprit={payload['native_vs_extreme_culprit']}",
        ],
    )

    cleanup_model(reference_model)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
