from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

try:
    from . import _C
except ImportError:
    _C = None


_LINEAR_BLOCK32_RETRYABLE_ERRORS = (
    "cublasLtMatmulAlgoGetHeuristic",
    "cublasLtMatmul(",
)
_LINEAR_BLOCK32_CAPABILITY_CACHE: dict[tuple[Any, ...], tuple[bool, bool]] = {}
_LINEAR_BLOCK32_CACHE_LOCK = threading.Lock()


def _extension_unavailable(name: str) -> RuntimeError:
    return RuntimeError(f"minifold_native_ext._C is not available for {name}")


def _new_scale(
    ref: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return torch.empty(shape, device=ref.device, dtype=torch.float32)


def _new_flat_swizzled_scale(ref: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    padded_rows = ((int(rows) + 127) // 128) * 128
    padded_groups = ((int(groups) + 3) // 4) * 4
    return torch.empty((1, padded_rows, padded_groups), device=ref.device, dtype=torch.float8_e8m0fnu)


def _linear_payload_and_scale(
    a: torch.Tensor,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = a.new_empty((a.shape[0], a.shape[1], out_features))
    scale = _new_scale(a, (a.shape[0], a.shape[1], out_features // 32))
    return payload, scale


def _linear_payload_scale_and_swizzled(
    a: torch.Tensor,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload, scale = _linear_payload_and_scale(a, out_features)
    scale_swizzled = _new_flat_swizzled_scale(a, int(a.shape[0]) * int(a.shape[1]), out_features // 32)
    return payload, scale, scale_swizzled


def _tri_payload_and_scale(
    payload: torch.Tensor,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_payload = payload.new_empty((*payload.shape[:-1], out_features))
    out_scale = _new_scale(payload, (*payload.shape[:-1], out_features // 32))
    return out_payload, out_scale


def _tri_payload_scale_and_swizzled(
    payload: torch.Tensor,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_payload, out_scale = _tri_payload_and_scale(payload, out_features)
    rows = 1
    for dim in payload.shape[:-1]:
        rows *= int(dim)
    out_swizzled = _new_flat_swizzled_scale(payload, rows, out_features // 32)
    return out_payload, out_scale, out_swizzled


_SUPPORTED_OUT_DTYPE_NAMES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _dtype_from_name(out_dtype: str) -> torch.dtype:
    try:
        return _SUPPORTED_OUT_DTYPE_NAMES[out_dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported out_dtype {out_dtype!r}") from exc


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def _tensor_debug_metadata(tensor: Optional[torch.Tensor]) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": int(tensor.storage_offset()),
        "storage_size": int(tensor.untyped_storage().size()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _linear_block32_debug_path() -> Path | None:
    raw_path = os.getenv("MINIFOLD_NATIVE_LINEAR_DEBUG_PATH")
    if not raw_path:
        return None
    return Path(raw_path)


def _linear_block32_debug_label() -> str:
    return os.getenv("MINIFOLD_NATIVE_LINEAR_DEBUG_LABEL", "")


def _log_linear_block32_debug(record: dict[str, Any]) -> None:
    debug_path = _linear_block32_debug_path()
    if debug_path is None:
        return
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": _linear_block32_debug_label(),
        **record,
    }
    with debug_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _linear_block32_retry_key(
    a: torch.Tensor,
    b_t: torch.Tensor,
    *,
    out_dtype: str,
    bias: Optional[torch.Tensor],
    apply_relu: bool,
    direct_fp8_output: bool,
    fuse_bias_epilogue: bool,
    residual_payload: Optional[torch.Tensor],
    residual_scale: Optional[torch.Tensor],
) -> tuple[Any, ...]:
    return (
        int(a.device.index if a.device.index is not None else -1),
        tuple(a.shape),
        tuple(b_t.shape),
        str(a.dtype),
        str(b_t.dtype),
        out_dtype,
        bool(bias is not None),
        bool(apply_relu),
        bool(direct_fp8_output),
        bool(fuse_bias_epilogue),
        bool(residual_payload is not None and residual_scale is not None),
    )


def _linear_block32_should_retry(exc: RuntimeError) -> bool:
    exc_text = str(exc)
    return any(fragment in exc_text for fragment in _LINEAR_BLOCK32_RETRYABLE_ERRORS)


def _linear_block32_attempts(
    *,
    direct_fp8_output: bool,
    fuse_bias_epilogue: bool,
) -> list[tuple[bool, bool]]:
    attempts = [(direct_fp8_output, fuse_bias_epilogue)]
    if direct_fp8_output:
        attempts.append((False, fuse_bias_epilogue))
    if fuse_bias_epilogue:
        attempts.append((False, False))

    deduped: list[tuple[bool, bool]] = []
    for attempt in attempts:
        if attempt not in deduped:
            deduped.append(attempt)
    return deduped


def _linear_block32_transition_linear_dims(
    a: torch.Tensor,
    b_t: torch.Tensor,
) -> bool:
    if a.dim() != 3 or b_t.dim() != 3:
        return False
    in_features = int(a.shape[2])
    out_features = int(b_t.shape[1])
    return (in_features, out_features) in ((128, 512), (512, 128))


def _linear_block32_proactive_attempts(
    a: torch.Tensor,
    b_t: torch.Tensor,
    *,
    bias: Optional[torch.Tensor],
    out_dtype: str,
    direct_fp8_output: bool,
    fuse_bias_epilogue: bool,
) -> tuple[list[tuple[bool, bool]], str | None]:
    attempts = _linear_block32_attempts(
        direct_fp8_output=direct_fp8_output,
        fuse_bias_epilogue=fuse_bias_epilogue,
    )
    if bias is None or out_dtype != "bfloat16":
        return attempts, None
    if not (direct_fp8_output or fuse_bias_epilogue):
        return attempts, None
    if not _linear_block32_transition_linear_dims(a, b_t):
        return attempts, None

    # The transition FC1/FC2 optimized cuBLASLt epilogues currently fail at runtime
    # and always recover via the plain (direct_fp8_output=False, fuse_bias_epilogue=False) path.
    return [(False, False)], "transition_linear_optimized_epilogue_disabled"


def _call_linear_block32_backend(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: str,
    apply_relu: bool,
    direct_fp8_output: bool,
    fuse_bias_epilogue: bool,
    residual_payload: Optional[torch.Tensor],
    residual_scale: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("linear_block32_fused")

    cache_key = _linear_block32_retry_key(
        a,
        b_t,
        out_dtype=out_dtype,
        bias=bias,
        apply_relu=apply_relu,
        direct_fp8_output=direct_fp8_output,
        fuse_bias_epilogue=fuse_bias_epilogue,
        residual_payload=residual_payload,
        residual_scale=residual_scale,
    )
    attempts, proactive_demote_reason = _linear_block32_proactive_attempts(
        a,
        b_t,
        bias=bias,
        out_dtype=out_dtype,
        direct_fp8_output=direct_fp8_output,
        fuse_bias_epilogue=fuse_bias_epilogue,
    )
    with _LINEAR_BLOCK32_CACHE_LOCK:
        cached_attempt = _LINEAR_BLOCK32_CAPABILITY_CACHE.get(cache_key)
    cache_hit = cached_attempt is not None
    if cached_attempt is not None:
        attempts = [cached_attempt, *[attempt for attempt in attempts if attempt != cached_attempt]]

    debug_record = {
        "cache_hit": cache_hit,
        "requested_flags": {
            "direct_fp8_output": bool(direct_fp8_output),
            "fuse_bias_epilogue": bool(fuse_bias_epilogue),
            "apply_relu": bool(apply_relu),
        },
        "proactive_demote_reason": proactive_demote_reason,
        "tensors": {
            "a": _tensor_debug_metadata(a),
            "b_t": _tensor_debug_metadata(b_t),
            "a_scale_swizzled": _tensor_debug_metadata(a_scale_swizzled),
            "b_scale_swizzled": _tensor_debug_metadata(b_scale_swizzled),
            "bias": _tensor_debug_metadata(bias),
            "residual_payload": _tensor_debug_metadata(residual_payload),
            "residual_scale": _tensor_debug_metadata(residual_scale),
        },
        "attempts": [],
    }

    last_error: RuntimeError | None = None
    for current_direct, current_fused in attempts:
        try:
            result = _C.linear_block32_fused(
                a,
                b_t,
                a_scale_swizzled,
                b_scale_swizzled,
                bias,
                out_dtype,
                apply_relu,
                current_direct,
                current_fused,
                residual_payload,
                residual_scale,
            )
        except RuntimeError as exc:
            debug_record["attempts"].append(
                {
                    "direct_fp8_output": bool(current_direct),
                    "fuse_bias_epilogue": bool(current_fused),
                    "outcome": "error",
                    "error": str(exc),
                }
            )
            last_error = exc
            if not _linear_block32_should_retry(exc):
                debug_record["final_outcome"] = "fatal_error"
                _log_linear_block32_debug(debug_record)
                raise
            continue

        with _LINEAR_BLOCK32_CACHE_LOCK:
            _LINEAR_BLOCK32_CAPABILITY_CACHE[cache_key] = (bool(current_direct), bool(current_fused))
        debug_record["attempts"].append(
            {
                "direct_fp8_output": bool(current_direct),
                "fuse_bias_epilogue": bool(current_fused),
                "outcome": "success",
            }
        )
        debug_record["resolved_flags"] = {
            "direct_fp8_output": bool(current_direct),
            "fuse_bias_epilogue": bool(current_fused),
        }
        debug_record["final_outcome"] = "success"
        _log_linear_block32_debug(debug_record)
        return result

    debug_record["final_outcome"] = "retry_exhausted"
    _log_linear_block32_debug(debug_record)
    if last_error is None:
        raise RuntimeError("linear_block32_fused exhausted retry attempts without a recorded error")
    raise last_error


@torch.library.custom_op("minifold_native_ext::linear_block32_fused", mutates_args=(), device_types="cuda")
def _linear_block32_fused_op(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _call_linear_block32_backend(
        a,
        b_t,
        a_scale_swizzled,
        b_scale_swizzled,
        bias,
        out_dtype,
        apply_relu,
        direct_fp8_output,
        fuse_bias_epilogue,
        residual_payload,
        residual_scale,
    )


@_linear_block32_fused_op.register_fake
def _linear_block32_fused_fake(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _linear_payload_and_scale(a, int(b_t.shape[1]))


@torch.library.custom_op("minifold_native_ext::linear_block32_fused_with_swizzled_scale", mutates_args=(), device_types="cuda")
def _linear_block32_fused_with_swizzled_scale_op(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("linear_block32_fused_with_swizzled_scale")
    return _C.linear_block32_fused_with_swizzled_scale(
        a,
        b_t,
        a_scale_swizzled,
        b_scale_swizzled,
        bias,
        out_dtype,
        apply_relu,
        direct_fp8_output,
        fuse_bias_epilogue,
        residual_payload,
        residual_scale,
    )


@_linear_block32_fused_with_swizzled_scale_op.register_fake
def _linear_block32_fused_with_swizzled_scale_fake(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _linear_payload_scale_and_swizzled(a, int(b_t.shape[1]))


@torch.library.custom_op("minifold_native_ext::linear_block32_raw_debug", mutates_args=(), device_types="cuda")
def _linear_block32_raw_debug_op(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _C is None:
        raise _extension_unavailable("linear_block32_raw_debug")
    return _C.linear_block32_raw_debug(
        a,
        b_t,
        a_scale_swizzled,
        b_scale_swizzled,
        bias,
        out_dtype,
        apply_relu,
        direct_fp8_output,
        fuse_bias_epilogue,
        residual_payload,
        residual_scale,
    )


@_linear_block32_raw_debug_op.register_fake
def _linear_block32_raw_debug_fake(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty((a.shape[0], a.shape[1], b_t.shape[1]), device=a.device, dtype=_dtype_from_name(out_dtype))


@torch.library.custom_op(
    "minifold_native_ext::transition_norm_fc1_block32_fused",
    mutates_args=(),
    device_types="cuda",
)
def _transition_norm_fc1_block32_fused_op(
    payload: torch.Tensor,
    scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    b_cutlass_col: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("transition_norm_fc1_block32_fused")
    return _C.transition_norm_fc1_block32_fused(
        payload,
        scale,
        norm_weight,
        norm_bias,
        float(norm_eps),
        b_cutlass_col,
        b_scale_swizzled,
        bias,
    )


@_transition_norm_fc1_block32_fused_op.register_fake
def _transition_norm_fc1_block32_fused_fake(
    payload: torch.Tensor,
    scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    b_cutlass_col: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _tri_payload_and_scale(payload, int(b_cutlass_col.shape[2]))


@torch.library.custom_op("minifold_native_ext::transition_norm_fc1_bf16_fused", mutates_args=(), device_types="cuda")
def _transition_norm_fc1_bf16_fused_op(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _C is None:
        raise _extension_unavailable("transition_norm_fc1_bf16_fused")
    return _C.transition_norm_fc1_bf16_fused(input_tensor, norm_weight, norm_bias, float(norm_eps), fc1_weight, fc1_bias)


@_transition_norm_fc1_bf16_fused_op.register_fake
def _transition_norm_fc1_bf16_fused_fake(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return input_tensor.new_empty((input_tensor.shape[0], input_tensor.shape[1], fc1_weight.shape[0]))


@torch.library.custom_op("minifold_native_ext::transition_fc2_residual_bf16_fused", mutates_args=(), device_types="cuda")
def _transition_fc2_residual_bf16_fused_op(
    input_tensor: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    residual: torch.Tensor,
) -> torch.Tensor:
    if _C is None:
        raise _extension_unavailable("transition_fc2_residual_bf16_fused")
    return _C.transition_fc2_residual_bf16_fused(input_tensor, fc2_weight, fc2_bias, residual)


@_transition_fc2_residual_bf16_fused_op.register_fake
def _transition_fc2_residual_bf16_fused_fake(
    input_tensor: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    residual: torch.Tensor,
) -> torch.Tensor:
    return residual.new_empty(residual.shape)


@torch.library.custom_op("minifold_native_ext::gate_sigmoid_mul_block32_fused", mutates_args=(), device_types="cuda")
def _gate_sigmoid_mul_block32_fused_op(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("gate_sigmoid_mul_block32_fused")
    if rhs_b_t is None or rhs_scale_swizzled is None:
        raise ValueError("rhs_b_t and rhs_scale_swizzled are required")
    return _C.gate_sigmoid_mul_block32_fused(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        out_dtype,
        residual_payload,
        residual_scale,
    )


@_gate_sigmoid_mul_block32_fused_op.register_fake
def _gate_sigmoid_mul_block32_fused_fake(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _linear_payload_and_scale(a, int(lhs_b_t.shape[1]))


@torch.library.custom_op(
    "minifold_native_ext::gate_sigmoid_mul_block32_fused_with_swizzled_scale",
    mutates_args=(),
    device_types="cuda",
)
def _gate_sigmoid_mul_block32_fused_with_swizzled_scale_op(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("gate_sigmoid_mul_block32_fused_with_swizzled_scale")
    if rhs_b_t is None or rhs_scale_swizzled is None:
        raise ValueError("rhs_b_t and rhs_scale_swizzled are required")
    return _C.gate_sigmoid_mul_block32_fused_with_swizzled_scale(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        out_dtype,
        residual_payload,
        residual_scale,
    )


@_gate_sigmoid_mul_block32_fused_with_swizzled_scale_op.register_fake
def _gate_sigmoid_mul_block32_fused_with_swizzled_scale_fake(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _linear_payload_scale_and_swizzled(a, int(lhs_b_t.shape[1]))


@torch.library.custom_op("minifold_native_ext::gate_sigmoid_mul_block32_raw_debug", mutates_args=(), device_types="cuda")
def _gate_sigmoid_mul_block32_raw_debug_op(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _C is None:
        raise _extension_unavailable("gate_sigmoid_mul_block32_raw_debug")
    if rhs_b_t is None or rhs_scale_swizzled is None:
        raise ValueError("rhs_b_t and rhs_scale_swizzled are required")
    return _C.gate_sigmoid_mul_block32_raw_debug(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        out_dtype,
        residual_payload,
        residual_scale,
    )


@_gate_sigmoid_mul_block32_raw_debug_op.register_fake
def _gate_sigmoid_mul_block32_raw_debug_fake(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty((a.shape[0], a.shape[1], lhs_b_t.shape[1]), device=a.device, dtype=_dtype_from_name(out_dtype))


@torch.library.custom_op("minifold_native_ext::tri_mul_pair_from_block32_carrier", mutates_args=(), device_types="cuda")
def _tri_mul_pair_from_block32_carrier_op(
    payload: torch.Tensor,
    scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("tri_mul_pair_from_block32_carrier")
    return _C.tri_mul_pair_from_block32_carrier(payload, scale, mask, out_dtype)


@_tri_mul_pair_from_block32_carrier_op.register_fake
def _tri_mul_pair_from_block32_carrier_fake(
    payload: torch.Tensor,
    scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    return _tri_payload_and_scale(payload, int(payload.shape[-1] // 2))


@torch.library.custom_op("minifold_native_ext::pack_block32_to_mxfp8_fused_debug", mutates_args=(), device_types="cuda")
def _pack_block32_to_mxfp8_fused_debug_op(
    payload: torch.Tensor,
    scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("pack_block32_to_mxfp8_fused_debug")
    return _C.pack_block32_to_mxfp8_fused_debug(payload, scale, mask)


@_pack_block32_to_mxfp8_fused_debug_op.register_fake
def _pack_block32_to_mxfp8_fused_debug_fake(
    payload: torch.Tensor,
    scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = payload.shape[0]
    n = payload.shape[1]
    padded_rows = ((n + 127) // 128) * 128
    padded_cols = (((n // 32) + 3) // 4) * 4
    payload_shape = (batch * 32, n, n)
    scale_shape = (batch * 32, padded_rows, padded_cols)
    a1 = payload.new_empty(payload_shape)
    b1 = payload.new_empty(payload_shape)
    a2_t = payload.new_empty(payload_shape)
    b2_rhs = payload.new_empty(payload_shape)
    a1_scale = torch.empty(scale_shape, device=payload.device, dtype=torch.float8_e8m0fnu)
    b1_scale = torch.empty(scale_shape, device=payload.device, dtype=torch.float8_e8m0fnu)
    a2_t_scale = torch.empty(scale_shape, device=payload.device, dtype=torch.float8_e8m0fnu)
    b2_rhs_scale = torch.empty(scale_shape, device=payload.device, dtype=torch.float8_e8m0fnu)
    return a1, a1_scale, b1, b1_scale, a2_t, a2_t_scale, b2_rhs, b2_rhs_scale


@torch.library.custom_op("minifold_native_ext::tri_mul_pair_from_packed_debug", mutates_args=(), device_types="cuda")
def _tri_mul_pair_from_packed_debug_op(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2_t: torch.Tensor,
    b2_rhs: torch.Tensor,
    a1_scale_swizzled: torch.Tensor,
    b1_scale_swizzled: torch.Tensor,
    a2_t_scale_swizzled: torch.Tensor,
    b2_rhs_scale_swizzled: torch.Tensor,
    out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("tri_mul_pair_from_packed_debug")
    return _C.tri_mul_pair_from_packed_debug(
        a1,
        b1,
        a2_t,
        b2_rhs,
        a1_scale_swizzled,
        b1_scale_swizzled,
        a2_t_scale_swizzled,
        b2_rhs_scale_swizzled,
        out_dtype,
    )


@_tri_mul_pair_from_packed_debug_op.register_fake
def _tri_mul_pair_from_packed_debug_fake(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2_t: torch.Tensor,
    b2_rhs: torch.Tensor,
    a1_scale_swizzled: torch.Tensor,
    b1_scale_swizzled: torch.Tensor,
    a2_t_scale_swizzled: torch.Tensor,
    b2_rhs_scale_swizzled: torch.Tensor,
    out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = _dtype_from_name(out_dtype)
    return torch.empty(a1.shape, device=a1.device, dtype=dtype), torch.empty(a2_t.shape, device=a2_t.device, dtype=dtype)


@torch.library.custom_op("minifold_native_ext::tri_pair_to_block32_carrier_debug", mutates_args=(), device_types="cuda")
def _tri_pair_to_block32_carrier_debug_op(
    x1: torch.Tensor,
    x2: torch.Tensor,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("tri_pair_to_block32_carrier_debug")
    return _C.tri_pair_to_block32_carrier_debug(x1, x2, int(batch))


@_tri_pair_to_block32_carrier_debug_op.register_fake
def _tri_pair_to_block32_carrier_debug_fake(
    x1: torch.Tensor,
    x2: torch.Tensor,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_chunk = x1.shape[0] // int(batch)
    n = x1.shape[1]
    channels = d_chunk * 2
    payload = torch.empty((int(batch), n, n, channels), device=x1.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((int(batch), n, n, channels // 32), device=x1.device, dtype=torch.float32)
    return payload, scale


@torch.library.custom_op("minifold_native_ext::tri_gate_block32_fused", mutates_args=(), device_types="cuda")
def _tri_gate_block32_fused_op(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("tri_gate_block32_fused")
    if rhs_b_t is None or rhs_scale_swizzled is None or mask is None:
        raise ValueError("rhs_b_t, rhs_scale_swizzled, and mask are required")
    return _C.tri_gate_block32_fused(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        mask,
        tri_out_dtype,
    )


@_tri_gate_block32_fused_op.register_fake
def _tri_gate_block32_fused_fake(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    return _linear_payload_and_scale(a, int(lhs_b_t.shape[1] // 2))


@torch.library.custom_op(
    "minifold_native_ext::tri_input_norm_gate_block32_fused",
    mutates_args=(),
    device_types="cuda",
)
def _tri_input_norm_gate_block32_fused_op(
    payload: torch.Tensor,
    scale: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    input_norm_eps: float,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("tri_input_norm_gate_block32_fused")
    if rhs_b_t is None or rhs_scale_swizzled is None or mask is None:
        raise ValueError("rhs_b_t, rhs_scale_swizzled, and mask are required")
    return _C.tri_input_norm_gate_block32_fused(
        payload,
        scale,
        input_norm_weight,
        input_norm_bias,
        float(input_norm_eps),
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        mask,
        tri_out_dtype,
    )


@_tri_input_norm_gate_block32_fused_op.register_fake
def _tri_input_norm_gate_block32_fused_fake(
    payload: torch.Tensor,
    scale: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    input_norm_eps: float,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    return _tri_payload_and_scale(payload, int(lhs_b_t.shape[1] // 2))


@torch.library.custom_op(
    "minifold_native_ext::tri_gate_layernorm_block32_fused",
    mutates_args=(),
    device_types="cuda",
)
def _tri_gate_layernorm_block32_fused_op(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    output_norm_weight: torch.Tensor | None = None,
    output_norm_bias: torch.Tensor | None = None,
    output_norm_eps: float = 1e-5,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("tri_gate_layernorm_block32_fused")
    if (
        rhs_b_t is None
        or rhs_scale_swizzled is None
        or mask is None
        or output_norm_weight is None
        or output_norm_bias is None
    ):
        raise ValueError("rhs_b_t, rhs_scale_swizzled, mask, and output norm tensors are required")
    return _C.tri_gate_layernorm_block32_fused(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        mask,
        output_norm_weight,
        output_norm_bias,
        float(output_norm_eps),
        tri_out_dtype,
    )


@_tri_gate_layernorm_block32_fused_op.register_fake
def _tri_gate_layernorm_block32_fused_fake(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    output_norm_weight: torch.Tensor | None = None,
    output_norm_bias: torch.Tensor | None = None,
    output_norm_eps: float = 1e-5,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    return _linear_payload_and_scale(a, int(lhs_b_t.shape[1] // 2))


@torch.library.custom_op("minifold_native_ext::relu_block32", mutates_args=(), device_types="cuda")
def _relu_block32_op(payload: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("relu_block32")
    return _C.relu_block32(payload, scale)


@_relu_block32_op.register_fake
def _relu_block32_fake(payload: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return payload.new_empty(payload.shape), _new_scale(payload, scale.shape)


@torch.library.custom_op("minifold_native_ext::add_block32", mutates_args=(), device_types="cuda")
def _add_block32_op(
    lhs_payload: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_payload: torch.Tensor,
    rhs_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("add_block32")
    return _C.add_block32(lhs_payload, lhs_scale, rhs_payload, rhs_scale)


@_add_block32_op.register_fake
def _add_block32_fake(
    lhs_payload: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_payload: torch.Tensor,
    rhs_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return lhs_payload.new_empty(lhs_payload.shape), _new_scale(lhs_payload, lhs_scale.shape)


@torch.library.custom_op("minifold_native_ext::layernorm_block32", mutates_args=(), device_types="cuda")
def _layernorm_block32_op(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("layernorm_block32")
    return _C.layernorm_block32(payload, scale, weight, bias, float(eps))


@_layernorm_block32_op.register_fake
def _layernorm_block32_fake(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return payload.new_empty(payload.shape), _new_scale(payload, scale.shape)


@torch.library.custom_op("minifold_native_ext::layernorm_block32_with_swizzled_scale", mutates_args=(), device_types="cuda")
def _layernorm_block32_with_swizzled_scale_op(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("layernorm_block32_with_swizzled_scale")
    return _C.layernorm_block32_with_swizzled_scale(payload, scale, weight, bias, float(eps))


@_layernorm_block32_with_swizzled_scale_op.register_fake
def _layernorm_block32_with_swizzled_scale_fake(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_payload = payload.new_empty(payload.shape)
    out_scale = _new_scale(payload, scale.shape)
    rows = 1
    for dim in payload.shape[:-1]:
        rows *= int(dim)
    out_swizzled = _new_flat_swizzled_scale(payload, rows, int(payload.shape[-1]) // 32)
    return out_payload, out_scale, out_swizzled


def linear_block32_fused(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        return _call_linear_block32_backend(
            a,
            b_t,
            a_scale_swizzled,
            b_scale_swizzled,
            bias,
            out_dtype,
            apply_relu,
            direct_fp8_output,
            fuse_bias_epilogue,
            residual_payload,
            residual_scale,
        )
    return _linear_block32_fused_op(
        a,
        b_t,
        a_scale_swizzled,
        b_scale_swizzled,
        bias,
        out_dtype,
        apply_relu,
        direct_fp8_output,
        fuse_bias_epilogue,
        residual_payload,
        residual_scale,
    )


def linear_block32_fused_with_swizzled_scale(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("linear_block32_fused_with_swizzled_scale")
        return _C.linear_block32_fused_with_swizzled_scale(
            a,
            b_t,
            a_scale_swizzled,
            b_scale_swizzled,
            bias,
            out_dtype,
            apply_relu,
            direct_fp8_output,
            fuse_bias_epilogue,
            residual_payload,
            residual_scale,
        )
    return _linear_block32_fused_with_swizzled_scale_op(
        a,
        b_t,
        a_scale_swizzled,
        b_scale_swizzled,
        bias,
        out_dtype,
        apply_relu,
        direct_fp8_output,
        fuse_bias_epilogue,
        residual_payload,
        residual_scale,
    )


def linear_block32_raw_debug(
    a: torch.Tensor,
    b_t: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("linear_block32_raw_debug")
        return _C.linear_block32_raw_debug(
            a,
            b_t,
            a_scale_swizzled,
            b_scale_swizzled,
            bias,
            out_dtype,
            apply_relu,
            direct_fp8_output,
            fuse_bias_epilogue,
            residual_payload,
            residual_scale,
        )
    return _linear_block32_raw_debug_op(
        a,
        b_t,
        a_scale_swizzled,
        b_scale_swizzled,
        bias,
        out_dtype,
        apply_relu,
        direct_fp8_output,
        fuse_bias_epilogue,
        residual_payload,
        residual_scale,
    )


def linear_block32_fc1_direct(
    a: torch.Tensor,
    b_cutlass_col: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise _extension_unavailable("linear_block32_fc1_direct")
    return _C.linear_block32_fc1_direct(a, b_cutlass_col, a_scale_swizzled, b_scale_swizzled, bias)


def transition_norm_fc1_block32_fused(
    payload: torch.Tensor,
    scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    b_cutlass_col: torch.Tensor,
    b_scale_swizzled: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("transition_norm_fc1_block32_fused")
        return _C.transition_norm_fc1_block32_fused(
            payload,
            scale,
            norm_weight,
            norm_bias,
            float(norm_eps),
            b_cutlass_col,
            b_scale_swizzled,
            bias,
        )
    return _transition_norm_fc1_block32_fused_op(
        payload,
        scale,
        norm_weight,
        norm_bias,
        norm_eps,
        b_cutlass_col,
        b_scale_swizzled,
        bias,
    )


def transition_norm_fc1_bf16_fused(
    input_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("transition_norm_fc1_bf16_fused")
        return _C.transition_norm_fc1_bf16_fused(input_tensor, norm_weight, norm_bias, float(norm_eps), fc1_weight, fc1_bias)
    return _transition_norm_fc1_bf16_fused_op(input_tensor, norm_weight, norm_bias, norm_eps, fc1_weight, fc1_bias)


def transition_fc2_residual_bf16_fused(
    input_tensor: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    residual: torch.Tensor,
) -> torch.Tensor:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("transition_fc2_residual_bf16_fused")
        return _C.transition_fc2_residual_bf16_fused(input_tensor, fc2_weight, fc2_bias, residual)
    return _transition_fc2_residual_bf16_fused_op(input_tensor, fc2_weight, fc2_bias, residual)


def gate_sigmoid_mul_block32_fused(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("gate_sigmoid_mul_block32_fused")
        if rhs_b_t is None or rhs_scale_swizzled is None:
            raise ValueError("rhs_b_t and rhs_scale_swizzled are required")
        return _C.gate_sigmoid_mul_block32_fused(
            a,
            a_scale_swizzled,
            lhs_b_t,
            lhs_scale_swizzled,
            lhs_bias,
            rhs_b_t,
            rhs_scale_swizzled,
            rhs_bias,
            out_dtype,
            residual_payload,
            residual_scale,
        )
    return _gate_sigmoid_mul_block32_fused_op(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        out_dtype,
        residual_payload,
        residual_scale,
    )


def gate_sigmoid_mul_block32_fused_with_swizzled_scale(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("gate_sigmoid_mul_block32_fused_with_swizzled_scale")
        if rhs_b_t is None or rhs_scale_swizzled is None:
            raise ValueError("rhs_b_t and rhs_scale_swizzled are required")
        return _C.gate_sigmoid_mul_block32_fused_with_swizzled_scale(
            a,
            a_scale_swizzled,
            lhs_b_t,
            lhs_scale_swizzled,
            lhs_bias,
            rhs_b_t,
            rhs_scale_swizzled,
            rhs_bias,
            out_dtype,
            residual_payload,
            residual_scale,
        )
    return _gate_sigmoid_mul_block32_fused_with_swizzled_scale_op(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        out_dtype,
        residual_payload,
        residual_scale,
    )


def gate_sigmoid_mul_block32_raw_debug(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    out_dtype: str = "bfloat16",
    residual_payload: Optional[torch.Tensor] = None,
    residual_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("gate_sigmoid_mul_block32_raw_debug")
        if rhs_b_t is None or rhs_scale_swizzled is None:
            raise ValueError("rhs_b_t and rhs_scale_swizzled are required")
        return _C.gate_sigmoid_mul_block32_raw_debug(
            a,
            a_scale_swizzled,
            lhs_b_t,
            lhs_scale_swizzled,
            lhs_bias,
            rhs_b_t,
            rhs_scale_swizzled,
            rhs_bias,
            out_dtype,
            residual_payload,
            residual_scale,
        )
    return _gate_sigmoid_mul_block32_raw_debug_op(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        out_dtype,
        residual_payload,
        residual_scale,
    )


def tri_mul_pair_from_block32_carrier(
    payload: torch.Tensor,
    scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("tri_mul_pair_from_block32_carrier")
        return _C.tri_mul_pair_from_block32_carrier(payload, scale, mask, out_dtype)
    return _tri_mul_pair_from_block32_carrier_op(payload, scale, mask, out_dtype)


def pack_block32_to_mxfp8_fused_debug(
    payload: torch.Tensor,
    scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("pack_block32_to_mxfp8_fused_debug")
        return _C.pack_block32_to_mxfp8_fused_debug(payload, scale, mask)
    return _pack_block32_to_mxfp8_fused_debug_op(payload, scale, mask)


def tri_mul_pair_from_packed_debug(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2_t: torch.Tensor,
    b2_rhs: torch.Tensor,
    a1_scale_swizzled: torch.Tensor,
    b1_scale_swizzled: torch.Tensor,
    a2_t_scale_swizzled: torch.Tensor,
    b2_rhs_scale_swizzled: torch.Tensor,
    out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("tri_mul_pair_from_packed_debug")
        return _C.tri_mul_pair_from_packed_debug(
            a1,
            b1,
            a2_t,
            b2_rhs,
            a1_scale_swizzled,
            b1_scale_swizzled,
            a2_t_scale_swizzled,
            b2_rhs_scale_swizzled,
            out_dtype,
        )
    return _tri_mul_pair_from_packed_debug_op(
        a1,
        b1,
        a2_t,
        b2_rhs,
        a1_scale_swizzled,
        b1_scale_swizzled,
        a2_t_scale_swizzled,
        b2_rhs_scale_swizzled,
        out_dtype,
    )


def tri_pair_to_block32_carrier_debug(
    x1: torch.Tensor,
    x2: torch.Tensor,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("tri_pair_to_block32_carrier_debug")
        return _C.tri_pair_to_block32_carrier_debug(x1, x2, int(batch))
    return _tri_pair_to_block32_carrier_debug_op(x1, x2, int(batch))


def tri_gate_block32_fused(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("tri_gate_block32_fused")
        if rhs_b_t is None or rhs_scale_swizzled is None or mask is None:
            raise ValueError("rhs_b_t, rhs_scale_swizzled, and mask are required")
        return _C.tri_gate_block32_fused(
            a,
            a_scale_swizzled,
            lhs_b_t,
            lhs_scale_swizzled,
            lhs_bias,
            rhs_b_t,
            rhs_scale_swizzled,
            rhs_bias,
            mask,
            tri_out_dtype,
        )
    return _tri_gate_block32_fused_op(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        mask,
        tri_out_dtype,
    )


def tri_input_norm_gate_block32_fused(
    payload: torch.Tensor,
    scale: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    input_norm_eps: float,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("tri_input_norm_gate_block32_fused")
        if rhs_b_t is None or rhs_scale_swizzled is None or mask is None:
            raise ValueError("rhs_b_t, rhs_scale_swizzled, and mask are required")
        return _C.tri_input_norm_gate_block32_fused(
            payload,
            scale,
            input_norm_weight,
            input_norm_bias,
            float(input_norm_eps),
            lhs_b_t,
            lhs_scale_swizzled,
            lhs_bias,
            rhs_b_t,
            rhs_scale_swizzled,
            rhs_bias,
            mask,
            tri_out_dtype,
        )
    return _tri_input_norm_gate_block32_fused_op(
        payload,
        scale,
        input_norm_weight,
        input_norm_bias,
        input_norm_eps,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        mask,
        tri_out_dtype,
    )


def tri_gate_layernorm_block32_fused(
    a: torch.Tensor,
    a_scale_swizzled: torch.Tensor,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    output_norm_weight: torch.Tensor | None = None,
    output_norm_bias: torch.Tensor | None = None,
    output_norm_eps: float = 1e-5,
    tri_out_dtype: str = "float16",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("tri_gate_layernorm_block32_fused")
        if (
            rhs_b_t is None
            or rhs_scale_swizzled is None
            or mask is None
            or output_norm_weight is None
            or output_norm_bias is None
        ):
            raise ValueError("rhs_b_t, rhs_scale_swizzled, mask, and output norm tensors are required")
        return _C.tri_gate_layernorm_block32_fused(
            a,
            a_scale_swizzled,
            lhs_b_t,
            lhs_scale_swizzled,
            lhs_bias,
            rhs_b_t,
            rhs_scale_swizzled,
            rhs_bias,
            mask,
            output_norm_weight,
            output_norm_bias,
            float(output_norm_eps),
            tri_out_dtype,
        )
    return _tri_gate_layernorm_block32_fused_op(
        a,
        a_scale_swizzled,
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        mask,
        output_norm_weight,
        output_norm_bias,
        output_norm_eps,
        tri_out_dtype,
    )


def relu_block32(payload: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("relu_block32")
        return _C.relu_block32(payload, scale)
    return _relu_block32_op(payload, scale)


def add_block32(
    lhs_payload: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_payload: torch.Tensor,
    rhs_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("add_block32")
        return _C.add_block32(lhs_payload, lhs_scale, rhs_payload, rhs_scale)
    return _add_block32_op(lhs_payload, lhs_scale, rhs_payload, rhs_scale)


def layernorm_block32(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("layernorm_block32")
        return _C.layernorm_block32(payload, scale, weight, bias, float(eps))
    return _layernorm_block32_op(payload, scale, weight, bias, eps)


def layernorm_block32_with_swizzled_scale(
    payload: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_torch_compiling():
        if _C is None:
            raise _extension_unavailable("layernorm_block32_with_swizzled_scale")
        return _C.layernorm_block32_with_swizzled_scale(payload, scale, weight, bias, float(eps))
    return _layernorm_block32_with_swizzled_scale_op(payload, scale, weight, bias, eps)


def _debug_gate_sigmoid_mul_pack_to_mxfp8_reference(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_bias: torch.Tensor,
    rhs_bias: torch.Tensor,
    mask: torch.Tensor,
):
    if _C is None:
        raise _extension_unavailable("_debug_gate_sigmoid_mul_pack_to_mxfp8_reference")
    return _C._debug_gate_sigmoid_mul_pack_to_mxfp8_reference(lhs, rhs, lhs_bias, rhs_bias, mask)


def _debug_gate_sigmoid_mul_pack_to_mxfp8_warp(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_bias: torch.Tensor,
    rhs_bias: torch.Tensor,
    mask: torch.Tensor,
):
    if _C is None:
        raise _extension_unavailable("_debug_gate_sigmoid_mul_pack_to_mxfp8_warp")
    return _C._debug_gate_sigmoid_mul_pack_to_mxfp8_warp(lhs, rhs, lhs_bias, rhs_bias, mask)


def _debug_tri_input_norm_gate_block32_reference_stages(
    payload: torch.Tensor,
    scale: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    input_norm_eps: float,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    tri_out_dtype: str = "float16",
):
    if _C is None:
        raise _extension_unavailable("_debug_tri_input_norm_gate_block32_reference_stages")
    if rhs_b_t is None or rhs_scale_swizzled is None or mask is None:
        raise ValueError("rhs_b_t, rhs_scale_swizzled, and mask are required")
    return _C._debug_tri_input_norm_gate_block32_reference_stages(
        payload,
        scale,
        input_norm_weight,
        input_norm_bias,
        float(input_norm_eps),
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        mask,
        tri_out_dtype,
    )


def _debug_tri_input_norm_gate_block32_warp_stages(
    payload: torch.Tensor,
    scale: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    input_norm_eps: float,
    lhs_b_t: torch.Tensor,
    lhs_scale_swizzled: torch.Tensor,
    lhs_bias: Optional[torch.Tensor] = None,
    rhs_b_t: torch.Tensor | None = None,
    rhs_scale_swizzled: torch.Tensor | None = None,
    rhs_bias: Optional[torch.Tensor] = None,
    mask: torch.Tensor | None = None,
    tri_out_dtype: str = "float16",
):
    if _C is None:
        raise _extension_unavailable("_debug_tri_input_norm_gate_block32_warp_stages")
    if rhs_b_t is None or rhs_scale_swizzled is None or mask is None:
        raise ValueError("rhs_b_t, rhs_scale_swizzled, and mask are required")
    return _C._debug_tri_input_norm_gate_block32_warp_stages(
        payload,
        scale,
        input_norm_weight,
        input_norm_bias,
        float(input_norm_eps),
        lhs_b_t,
        lhs_scale_swizzled,
        lhs_bias,
        rhs_b_t,
        rhs_scale_swizzled,
        rhs_bias,
        mask,
        tri_out_dtype,
    )
