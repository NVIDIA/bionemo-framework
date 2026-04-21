from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from fp8_extreme_ops import (
    fp8_binary_rows,
    fp8_binary_block32,
    fp8_layernorm_rows,
    fp8_layernorm_block32,
    fp8_mask_mul_rows,
    fp8_mask_mul_block32,
    fp8_pack_block32_carrier_to_mxfp8_lhs,
    fp8_pack_block32_carrier_to_mxfp8_rhs,
    fp8_pack_carrier_to_mxfp8_lhs,
    fp8_pack_carrier_to_mxfp8_rhs,
    fp8_requantize_block32,
    fp8_requantize_rows,
    fp8_tri_outputs_to_block32_carrier,
    fp8_tri_outputs_to_carrier,
    fp8_unary_block32,
    fp8_unary_rows,
    _swizzle_mxfp8_scale_rowwise,
)


TRI_MUL_EXT_ROOT = Path(__file__).resolve().parent / "tri_mul_ext"
FP8_BMM_EXT_ROOT = Path(__file__).resolve().parent / "fp8_bmm_ext"
MINIFOLD_NATIVE_EXT_ROOT = Path(__file__).resolve().parent / "minifold_native_ext"
if str(TRI_MUL_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRI_MUL_EXT_ROOT))
if str(FP8_BMM_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(FP8_BMM_EXT_ROOT))
if str(MINIFOLD_NATIVE_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(MINIFOLD_NATIVE_EXT_ROOT))

from tri_mul_ext import tri_mul_xbdnn_cublas

try:
    from bmm_ext import mxfp8_cublaslt_tri_mul_xbdnn, mxfp8_cublaslt_tri_mul_xbdnn_inference
except ImportError:
    mxfp8_cublaslt_tri_mul_xbdnn = None
    mxfp8_cublaslt_tri_mul_xbdnn_inference = None

try:
    _raw_ext_path = next((FP8_BMM_EXT_ROOT / "bmm_ext").glob("_C*.so"))
    _raw_ext_spec = importlib.util.spec_from_file_location("bmm_ext._C", _raw_ext_path)
    if _raw_ext_spec is None or _raw_ext_spec.loader is None:
        raise ImportError(f"Could not create module spec for {_raw_ext_path}")
    bmm_ext_raw = importlib.util.module_from_spec(_raw_ext_spec)
    _raw_ext_spec.loader.exec_module(bmm_ext_raw)
except Exception:
    bmm_ext_raw = None

try:
    _native_ext_path = next((MINIFOLD_NATIVE_EXT_ROOT / "minifold_native_ext").glob("_C*.so"))
    _native_ext_spec = importlib.util.spec_from_file_location("minifold_native_ext._C", _native_ext_path)
    if _native_ext_spec is None or _native_ext_spec.loader is None:
        raise ImportError(f"Could not create module spec for {_native_ext_path}")
    minifold_native_raw = importlib.util.module_from_spec(_native_ext_spec)
    _native_ext_spec.loader.exec_module(minifold_native_raw)
except Exception:
    minifold_native_raw = None


PAIR_PRECISION_BF16 = "bf16"
PAIR_PRECISION_BF16_NATIVE = "bf16_native"
PAIR_PRECISION_FP8_STORAGE = "fp8_storage"
PAIR_PRECISION_FP8_RESIDENT = "fp8_resident"
PAIR_PRECISION_FP8_EXTREME = "fp8_extreme"
PAIR_PRECISION_FP8_HYBRID = "fp8_hybrid"
PAIR_PRECISION_FP8_NATIVE = "fp8_native"
PAIR_PRECISION_FP8_NATIVE_GOLD_PACKS = "fp8_native_gold_packs"
PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL = "fp8_native_mixed_tail"
SUPPORTED_PAIR_PRECISION = (
    PAIR_PRECISION_BF16,
    PAIR_PRECISION_BF16_NATIVE,
    PAIR_PRECISION_FP8_STORAGE,
    PAIR_PRECISION_FP8_RESIDENT,
    PAIR_PRECISION_FP8_EXTREME,
    PAIR_PRECISION_FP8_HYBRID,
    PAIR_PRECISION_FP8_NATIVE,
    PAIR_PRECISION_FP8_NATIVE_GOLD_PACKS,
    PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL,
)
FP8_BLOCK32_PAIR_PRECISIONS = (
    PAIR_PRECISION_FP8_EXTREME,
    PAIR_PRECISION_FP8_HYBRID,
    PAIR_PRECISION_FP8_NATIVE,
    PAIR_PRECISION_FP8_NATIVE_GOLD_PACKS,
    PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL,
)
LINEAR_PRECISION_BF16 = "bf16"
LINEAR_PRECISION_BF16_NATIVE = "bf16_native"
LINEAR_PRECISION_FP8 = "fp8"
SUPPORTED_LINEAR_PRECISION = (
    LINEAR_PRECISION_BF16,
    LINEAR_PRECISION_BF16_NATIVE,
    LINEAR_PRECISION_FP8,
)
SUPPORTED_TRI_IMPLS = ("bmm", "cublas_xbdnn", "fp8_cublaslt", "fp8_grouped")
BF16_NATIVE_RUNG_B2 = "B2"
BF16_NATIVE_RUNG_B3 = "B3"
BF16_NATIVE_RUNG_B4 = "B4"
BF16_NATIVE_RUNG_B5 = "B5"
SUPPORTED_BF16_NATIVE_RUNGS = (
    BF16_NATIVE_RUNG_B2,
    BF16_NATIVE_RUNG_B3,
    BF16_NATIVE_RUNG_B4,
    BF16_NATIVE_RUNG_B5,
)
_FP8_MAX = 448.0
_FP8_LINEAR_MAX_ROWS = 1 << 22
_NATIVE_FC1_DIRECT_MAX_ROWS = 1 << 16


def resolve_pair_precision(pair_precision: str | None = None, fp8_activations: bool | None = None) -> str:
    if pair_precision is None:
        return PAIR_PRECISION_FP8_STORAGE if fp8_activations else PAIR_PRECISION_BF16
    if pair_precision not in SUPPORTED_PAIR_PRECISION:
        raise ValueError(f"Unsupported pair_precision {pair_precision!r}; expected one of {SUPPORTED_PAIR_PRECISION!r}")
    return pair_precision


def resolve_linear_precision(linear_precision: str | None = None) -> str:
    if linear_precision is None:
        return LINEAR_PRECISION_BF16
    if linear_precision not in SUPPORTED_LINEAR_PRECISION:
        raise ValueError(f"Unsupported linear_precision {linear_precision!r}; expected one of {SUPPORTED_LINEAR_PRECISION!r}")
    return linear_precision


def resolve_bf16_native_rung(bf16_native_rung: str | None = None) -> str | None:
    if bf16_native_rung is None:
        return None
    rung = bf16_native_rung.upper()
    if rung not in SUPPORTED_BF16_NATIVE_RUNGS:
        raise ValueError(f"Unsupported bf16_native_rung {bf16_native_rung!r}; expected one of {SUPPORTED_BF16_NATIVE_RUNGS!r}")
    return rung


def tri_mul_bmm_bdnn(a: torch.Tensor, b: torch.Tensor, k_dim: int) -> torch.Tensor:
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError("a and b must have shape (B, D, N, N)")
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")

    bsz, d, n1, n2 = a.shape
    a3 = a.reshape(bsz * d, n1, n2)
    b3 = b.reshape(bsz * d, n1, n2)
    if k_dim == 2:
        out = torch.bmm(a3, b3.transpose(1, 2))
    elif k_dim == 1:
        out = torch.bmm(a3.transpose(1, 2), b3)
    else:
        raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")
    return out.reshape(bsz, d, n1, n2).permute(0, 2, 3, 1)


def tri_mul_xbdnn(x_bdnn: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    out_dtype = x_bdnn.dtype if out_dtype is None else out_dtype
    return tri_mul_xbdnn_cublas(x_bdnn, out_dtype=out_dtype)


def _validate_tri_mul_bdnn(x_bdnn: torch.Tensor, impl: str) -> tuple[int, int, int, int]:
    if x_bdnn.dim() != 4:
        raise ValueError("x_bdnn must have shape (B, D, N, N)")
    if x_bdnn.shape[1] % 4 != 0:
        raise ValueError(f"{impl} requires channel dimension divisible by 4, got {x_bdnn.shape[1]}")
    bsz, d, n1, n2 = x_bdnn.shape
    if n1 != n2:
        raise ValueError(f"{impl} expects square sequence dimensions, got {(n1, n2)}")
    if n1 % 32 != 0:
        raise ValueError(f"{impl} requires sequence length divisible by 32, got {n1}")
    return bsz, d, n1, n2


def tri_mul_fp8_cublaslt(x_bdnn: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    if mxfp8_cublaslt_tri_mul_xbdnn is None:
        raise RuntimeError("fp8_cublaslt was requested but bmm_ext is not importable")
    bsz, _, n1, _ = _validate_tri_mul_bdnn(x_bdnn, "fp8_cublaslt")
    out_dtype = x_bdnn.dtype if out_dtype is None else out_dtype
    if not torch.is_grad_enabled() and mxfp8_cublaslt_tri_mul_xbdnn_inference is not None:
        x = mxfp8_cublaslt_tri_mul_xbdnn_inference(x_bdnn, out_dtype=out_dtype)
    else:
        x = mxfp8_cublaslt_tri_mul_xbdnn(x_bdnn, out_dtype=out_dtype)
    d_out = x.shape[1]
    return x.reshape(bsz, d_out, n1, n1).permute(0, 2, 3, 1)


def tri_mul_fp8_grouped(x_bdnn: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    try:
        from te_utils import tri_mul_fp8_grouped_bdnn
    except ImportError as exc:
        raise RuntimeError("fp8_grouped requires te_utils / TransformerEngine support in this environment") from exc
    return tri_mul_fp8_grouped_bdnn(x_bdnn, out_dtype=x_bdnn.dtype if out_dtype is None else out_dtype)


def quantize_to_fp8(x_bf16: torch.Tensor, scale_dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _FP8_MAX).to(scale_dtype)
    x_fp8 = (x_bf16 / scale.to(x_bf16.dtype)).to(torch.float8_e4m3fn)
    return x_fp8, scale


def dequantize_from_fp8(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    compute_dtype = torch.float32 if scale.dtype == torch.float32 else torch.bfloat16
    return x_fp8.to(compute_dtype) * scale.to(compute_dtype)


@dataclass(frozen=True)
class QuantizedPairTensor:
    payload: torch.Tensor
    scale: torch.Tensor
    logical_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, scale_dtype: torch.dtype = torch.bfloat16) -> "QuantizedPairTensor":
        compute_dtype = torch.float32 if scale_dtype == torch.float32 else torch.bfloat16
        x_fp8, scale = quantize_to_fp8(tensor.to(compute_dtype), scale_dtype=scale_dtype)
        return cls(payload=x_fp8, scale=scale, logical_dtype=tensor.dtype)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        target_dtype = self.logical_dtype if dtype is None else dtype
        return dequantize_from_fp8(self.payload, self.scale).to(target_dtype)

    @property
    def shape(self) -> torch.Size:
        return self.payload.shape

    @property
    def device(self) -> torch.device:
        return self.payload.device

    @property
    def quantized_bytes(self) -> int:
        return self.payload.numel() * self.payload.element_size()

    @property
    def scale_bytes(self) -> int:
        return self.scale.numel() * self.scale.element_size()


@dataclass(frozen=True)
class Mxfp8PairTensor:
    payload: torch.Tensor
    scale: torch.Tensor
    logical_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, scale_dtype: torch.dtype = torch.float32) -> "Mxfp8PairTensor":
        payload, scale = fp8_requantize_block32(tensor.to(torch.float32), scale_dtype=scale_dtype)
        return cls(payload=payload, scale=scale, logical_dtype=tensor.dtype)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        target_dtype = self.logical_dtype if dtype is None else dtype
        groups = self.scale.shape[-1]
        expanded_scale = self.scale.to(torch.float32).unsqueeze(-1).expand(*self.scale.shape, 32).reshape(*self.payload.shape[:-1], groups * 32)
        return (self.payload.to(torch.float32) * expanded_scale).to(target_dtype)

    @property
    def shape(self) -> torch.Size:
        return self.payload.shape

    @property
    def device(self) -> torch.device:
        return self.payload.device

    @property
    def quantized_bytes(self) -> int:
        return self.payload.numel() * self.payload.element_size()

    @property
    def scale_bytes(self) -> int:
        return self.scale.numel() * self.scale.element_size()


@dataclass(frozen=True)
class HybridPrecisionConfig:
    use_native_layernorm: bool = False
    use_native_linear: bool = False
    use_native_gate: bool = False
    use_native_tri: bool = False
    use_resident_fp8_residual: bool = False


def resolve_hybrid_precision_config(
    hybrid_precision: HybridPrecisionConfig | dict[str, bool] | None = None,
) -> HybridPrecisionConfig:
    if hybrid_precision is None:
        return HybridPrecisionConfig()
    if isinstance(hybrid_precision, HybridPrecisionConfig):
        return HybridPrecisionConfig(**asdict(hybrid_precision))
    if isinstance(hybrid_precision, dict):
        return HybridPrecisionConfig(**hybrid_precision)
    raise TypeError(
        "hybrid_precision must be a HybridPrecisionConfig, a dict of booleans, or None; "
        f"got {type(hybrid_precision)!r}"
    )


@dataclass(frozen=True)
class MixedTailConfig:
    tail_bf16_native_blocks: int = 0
    bf16_native_rung: str = BF16_NATIVE_RUNG_B3


def resolve_mixed_tail_config(
    mixed_tail: MixedTailConfig | dict[str, object] | None = None,
) -> MixedTailConfig:
    if mixed_tail is None:
        return MixedTailConfig()
    if isinstance(mixed_tail, MixedTailConfig):
        raw = asdict(mixed_tail)
    elif isinstance(mixed_tail, dict):
        raw = dict(mixed_tail)
    else:
        raise TypeError(
            "mixed_tail must be a MixedTailConfig, a dict, or None; "
            f"got {type(mixed_tail)!r}"
        )
    tail_bf16_native_blocks = int(raw.get("tail_bf16_native_blocks", 0))
    bf16_native_rung = resolve_bf16_native_rung(raw.get("bf16_native_rung")) or BF16_NATIVE_RUNG_B3
    return MixedTailConfig(
        tail_bf16_native_blocks=tail_bf16_native_blocks,
        bf16_native_rung=bf16_native_rung,
    )


@dataclass(frozen=True)
class ActivationTraceEvent:
    block_idx: int
    op_name: str
    mode_op_name: str
    mode: str
    tensor_format: str
    shape: tuple[int, ...]
    payload_shape: tuple[int, ...] | None
    scale_shape: tuple[int, ...] | None
    max_abs: float
    mean_abs: float
    nan_count: int
    inf_count: int
    scale_max_abs: float | None
    scale_mean_abs: float | None
    snapshot_index: int | None = None


@dataclass
class ActivationProbe:
    pair_precision_mode: str
    retain_tensors: bool = False
    events: list[ActivationTraceEvent] = field(default_factory=list)
    tensor_snapshots: list[torch.Tensor] = field(default_factory=list, repr=False)

    def record(
        self,
        block_idx: int,
        op_name: str,
        tensor: torch.Tensor | QuantizedPairTensor | Mxfp8PairTensor,
        *,
        mode_op_name: str | None = None,
    ) -> None:
        dequantized, payload_shape, scale_shape, tensor_format = _unwrap_trace_tensor(tensor)
        if dequantized.numel() == 0:
            max_abs = 0.0
            mean_abs = 0.0
            nan_count = 0
            inf_count = 0
        else:
            dequantized_f32 = dequantized.detach().to(torch.float32)
            abs_tensor = dequantized_f32.abs()
            max_abs = float(abs_tensor.amax().item())
            mean_abs = float(abs_tensor.mean().item())
            nan_count = int(torch.isnan(dequantized_f32).sum().item())
            inf_count = int(torch.isinf(dequantized_f32).sum().item())

        scale_max_abs = None
        scale_mean_abs = None
        if isinstance(tensor, (QuantizedPairTensor, Mxfp8PairTensor)):
            scale_f32 = tensor.scale.detach().to(torch.float32)
            scale_abs = scale_f32.abs()
            if scale_abs.numel() > 0:
                scale_max_abs = float(scale_abs.amax().item())
                scale_mean_abs = float(scale_abs.mean().item())

        snapshot_index = None
        if self.retain_tensors:
            snapshot_index = len(self.tensor_snapshots)
            self.tensor_snapshots.append(dequantized.detach().cpu())

        self.events.append(
            ActivationTraceEvent(
                block_idx=int(block_idx),
                op_name=op_name,
                mode_op_name=mode_op_name or op_name,
                mode=self.pair_precision_mode,
                tensor_format=tensor_format,
                shape=tuple(dequantized.shape),
                payload_shape=payload_shape,
                scale_shape=scale_shape,
                max_abs=max_abs,
                mean_abs=mean_abs,
                nan_count=nan_count,
                inf_count=inf_count,
                scale_max_abs=scale_max_abs,
                scale_mean_abs=scale_mean_abs,
                snapshot_index=snapshot_index,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_precision_mode": self.pair_precision_mode,
            "events": [asdict(event) for event in self.events],
        }


def _unwrap_trace_tensor(
    tensor: torch.Tensor | QuantizedPairTensor | Mxfp8PairTensor,
) -> tuple[torch.Tensor, tuple[int, ...] | None, tuple[int, ...] | None, str]:
    if isinstance(tensor, QuantizedPairTensor):
        return (
            tensor.dequantize(dtype=torch.float32),
            tuple(tensor.payload.shape),
            tuple(tensor.scale.shape),
            "quantized_fp8",
        )
    if isinstance(tensor, Mxfp8PairTensor):
        return (
            tensor.dequantize(dtype=torch.float32),
            tuple(tensor.payload.shape),
            tuple(tensor.scale.shape),
            "mxfp8_block32",
        )
    tensor_kind = str(tensor.dtype).replace("torch.", "")
    return tensor.detach(), None, None, tensor_kind


def _record_probe(
    probe: ActivationProbe | None,
    block_idx: int | None,
    op_name: str,
    tensor: torch.Tensor | QuantizedPairTensor | Mxfp8PairTensor,
    *,
    mode_op_name: str | None = None,
) -> None:
    if probe is not None and block_idx is not None:
        probe.record(block_idx, op_name, tensor, mode_op_name=mode_op_name)


def fp8_relu_quantized(tensor: QuantizedPairTensor) -> QuantizedPairTensor:
    payload, scale = fp8_unary_rows(tensor.payload, tensor.scale, op="relu")
    return QuantizedPairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def fp8_sigmoid_quantized(tensor: QuantizedPairTensor) -> QuantizedPairTensor:
    payload, scale = fp8_unary_rows(tensor.payload, tensor.scale, op="sigmoid")
    return QuantizedPairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def fp8_add_quantized(lhs: QuantizedPairTensor, rhs: QuantizedPairTensor) -> QuantizedPairTensor:
    payload, scale = fp8_binary_rows(lhs.payload, lhs.scale, rhs.payload, rhs.scale, op="add")
    return QuantizedPairTensor(payload=payload, scale=scale, logical_dtype=lhs.logical_dtype)


def fp8_mul_quantized(lhs: QuantizedPairTensor, rhs: QuantizedPairTensor) -> QuantizedPairTensor:
    payload, scale = fp8_binary_rows(lhs.payload, lhs.scale, rhs.payload, rhs.scale, op="mul")
    return QuantizedPairTensor(payload=payload, scale=scale, logical_dtype=lhs.logical_dtype)


def fp8_mask_mul_quantized(tensor: QuantizedPairTensor, mask: torch.Tensor) -> QuantizedPairTensor:
    payload, scale = fp8_mask_mul_rows(tensor.payload, tensor.scale, mask)
    return QuantizedPairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def fp8_layernorm_quantized(module: nn.LayerNorm, tensor: QuantizedPairTensor) -> QuantizedPairTensor:
    payload, scale = fp8_layernorm_rows(tensor.payload, tensor.scale, module.weight, module.bias, module.eps)
    return QuantizedPairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def fp8_unit_scale_quantized(payload: torch.Tensor, logical_dtype: torch.dtype, scale_dtype: torch.dtype) -> QuantizedPairTensor:
    scale = torch.ones((*payload.shape[:-1], 1), device=payload.device, dtype=scale_dtype)
    return QuantizedPairTensor(payload=payload, scale=scale, logical_dtype=logical_dtype)


def mxfp8_relu_quantized(tensor: Mxfp8PairTensor) -> Mxfp8PairTensor:
    payload, scale = fp8_unary_block32(tensor.payload, tensor.scale, op="relu")
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def mxfp8_sigmoid_quantized(tensor: Mxfp8PairTensor) -> Mxfp8PairTensor:
    payload, scale = fp8_unary_block32(tensor.payload, tensor.scale, op="sigmoid")
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def mxfp8_add_quantized(lhs: Mxfp8PairTensor, rhs: Mxfp8PairTensor) -> Mxfp8PairTensor:
    payload, scale = fp8_binary_block32(lhs.payload, lhs.scale, rhs.payload, rhs.scale, op="add")
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=lhs.logical_dtype)


def mxfp8_mul_quantized(lhs: Mxfp8PairTensor, rhs: Mxfp8PairTensor) -> Mxfp8PairTensor:
    payload, scale = fp8_binary_block32(lhs.payload, lhs.scale, rhs.payload, rhs.scale, op="mul")
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=lhs.logical_dtype)


def mxfp8_mask_mul_quantized(tensor: Mxfp8PairTensor, mask: torch.Tensor) -> Mxfp8PairTensor:
    payload, scale = fp8_mask_mul_block32(tensor.payload, tensor.scale, mask)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def mxfp8_layernorm_quantized(module: nn.LayerNorm, tensor: Mxfp8PairTensor) -> Mxfp8PairTensor:
    payload, scale = fp8_layernorm_block32(tensor.payload, tensor.scale, module.weight, module.bias, module.eps)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def native_mxfp8_relu_quantized(tensor: Mxfp8PairTensor) -> Mxfp8PairTensor:
    if minifold_native_raw is None:
        return mxfp8_relu_quantized(tensor)
    payload, scale = minifold_native_raw.relu_block32(tensor.payload.contiguous(), tensor.scale.contiguous())
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def native_mxfp8_add_quantized(lhs: Mxfp8PairTensor, rhs: Mxfp8PairTensor) -> Mxfp8PairTensor:
    if minifold_native_raw is None:
        return mxfp8_add_quantized(lhs, rhs)
    payload, scale = minifold_native_raw.add_block32(
        lhs.payload.contiguous(),
        lhs.scale.contiguous(),
        rhs.payload.contiguous(),
        rhs.scale.contiguous(),
    )
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=lhs.logical_dtype)


def native_mxfp8_layernorm_quantized(module: nn.LayerNorm, tensor: Mxfp8PairTensor) -> Mxfp8PairTensor:
    if minifold_native_raw is None:
        return mxfp8_layernorm_quantized(module, tensor)
    payload, scale = minifold_native_raw.layernorm_block32(
        tensor.payload.contiguous(),
        tensor.scale.contiguous(),
        module.weight.to(device=tensor.device, dtype=torch.bfloat16).contiguous(),
        module.bias.to(device=tensor.device, dtype=torch.bfloat16).contiguous(),
        float(module.eps),
    )
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=tensor.logical_dtype)


def native_transition_norm_fc1_quantized(
    norm_module: nn.LayerNorm,
    fc1_module: nn.Linear,
    x: Mxfp8PairTensor,
    stats: FP8ActivationStats | None = None,
) -> Mxfp8PairTensor:
    fallback = lambda: native_linear_forward_quantized(
        fc1_module,
        native_mxfp8_layernorm_quantized(norm_module, x),
        stats=stats,
        apply_relu=True,
        fuse_bias_epilogue=True,
    )
    if (
        minifold_native_raw is None
        or not hasattr(minifold_native_raw, "transition_norm_fc1_block32_fused")
        or not _has_mxfp8_linear_weight(fc1_module)
        or x.payload.shape[-1] != 128
        or fc1_module.weight.shape != (512, 128)
        or not getattr(fc1_module, "_native_transition_norm_fc1_enabled", False)
        or not getattr(fc1_module, "_native_transition_norm_fc1_supported", True)
        or not isinstance(getattr(fc1_module, "weight_mxfp8_cutlass_col", None), torch.Tensor)
    ):
        return fallback()
    original_shape = x.payload.shape[:-1]
    rows = x.payload.numel() // x.payload.shape[-1]
    try:
        payload, scale = minifold_native_raw.transition_norm_fc1_block32_fused(
            x.payload.contiguous(),
            x.scale.contiguous(),
            norm_module.weight.to(device=x.device, dtype=torch.bfloat16).contiguous(),
            norm_module.bias.to(device=x.device, dtype=torch.bfloat16).contiguous(),
            float(norm_module.eps),
            fc1_module.weight_mxfp8_cutlass_col.contiguous(),
            fc1_module.scale_w_mxfp8_swizzled.contiguous(),
            None if fc1_module.bias is None else fc1_module.bias.contiguous(),
        )
    except RuntimeError:
        fc1_module._native_transition_norm_fc1_supported = False
        return fallback()
    out_dim = fc1_module.weight.shape[0]
    payload = payload.reshape(*original_shape, out_dim).contiguous()
    scale = scale.reshape(*original_shape, out_dim // 32).contiguous()
    if stats is not None:
        stats.record_native_linear_fused(payload, scale)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def native_transition_norm_fc1_bf16(
    norm_module: nn.LayerNorm,
    fc1_module: nn.Linear,
    x: torch.Tensor,
) -> torch.Tensor:
    fallback = lambda: F.relu(fc1_module(norm_module(x)))
    if (
        minifold_native_raw is None
        or not hasattr(minifold_native_raw, "transition_norm_fc1_bf16_fused")
        or x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.shape[-1] != 128
        or fc1_module.weight.shape != (512, 128)
    ):
        return fallback()
    original_shape = x.shape[:-1]
    rows = x.numel() // x.shape[-1]
    out = minifold_native_raw.transition_norm_fc1_bf16_fused(
        x.reshape(1, rows, x.shape[-1]).contiguous(),
        norm_module.weight.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        norm_module.bias.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        float(norm_module.eps),
        fc1_module.weight.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        None if fc1_module.bias is None else fc1_module.bias.to(device=x.device, dtype=torch.bfloat16).contiguous(),
    )
    return out.reshape(*original_shape, fc1_module.weight.shape[0]).contiguous()


def native_transition_fc2_residual_bf16(
    fc2_module: nn.Linear,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    fallback = lambda: fc2_module(x) + residual
    if (
        minifold_native_raw is None
        or not hasattr(minifold_native_raw, "transition_fc2_residual_bf16_fused")
        or x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or x.shape[-1] != 512
        or residual.shape[-1] != 128
        or fc2_module.weight.shape != (128, 512)
    ):
        return fallback()
    original_shape = residual.shape[:-1]
    rows = residual.numel() // residual.shape[-1]
    out = minifold_native_raw.transition_fc2_residual_bf16_fused(
        x.reshape(1, rows, x.shape[-1]).contiguous(),
        fc2_module.weight.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        None if fc2_module.bias is None else fc2_module.bias.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        residual.reshape(1, rows, residual.shape[-1]).contiguous(),
    )
    return out.reshape(*original_shape, residual.shape[-1]).contiguous()


@dataclass
class FP8ActivationStats:
    pair_precision_mode: str
    quantized_bytes: int = 0
    scale_bytes: int = 0
    quantize_ops: int = 0
    dequantize_ops: int = 0
    tensorwise_repack_ops: int = 0
    tensorwise_repack_payload_bytes: int = 0
    tensorwise_repack_scale_bytes: int = 0
    linear_requant_ops: int = 0
    linear_requant_payload_bytes: int = 0
    linear_requant_scale_bytes: int = 0
    tri_pack_ops: int = 0
    tri_pack_payload_bytes: int = 0
    tri_pack_scale_bytes: int = 0
    tri_output_requant_ops: int = 0
    tri_output_requant_payload_bytes: int = 0
    tri_output_requant_scale_bytes: int = 0
    native_linear_fused_ops: int = 0
    native_linear_payload_bytes: int = 0
    native_linear_scale_bytes: int = 0
    native_gate_fused_ops: int = 0
    native_gate_payload_bytes: int = 0
    native_gate_scale_bytes: int = 0
    native_tri_fused_ops: int = 0
    native_tri_payload_bytes: int = 0
    native_tri_scale_bytes: int = 0
    boundary_counts: dict[str, int] = field(default_factory=dict)

    def record_quantize(self, tensor: QuantizedPairTensor | Mxfp8PairTensor) -> None:
        self.quantize_ops += 1
        self.quantized_bytes = max(self.quantized_bytes, tensor.quantized_bytes)
        self.scale_bytes = max(self.scale_bytes, tensor.scale_bytes)

    def record_dequantize(self, boundary: str) -> None:
        self.dequantize_ops += 1
        self.record_boundary(boundary)

    def record_tensorwise_repack(self, payload: torch.Tensor, scale: torch.Tensor) -> None:
        self.tensorwise_repack_ops += 1
        self.tensorwise_repack_payload_bytes = max(self.tensorwise_repack_payload_bytes, payload.numel() * payload.element_size())
        self.tensorwise_repack_scale_bytes = max(self.tensorwise_repack_scale_bytes, scale.numel() * scale.element_size())
        self.record_boundary("tensorwise_linear_pack")

    def record_linear_requant(self, payload: torch.Tensor, scale: torch.Tensor) -> None:
        self.linear_requant_ops += 1
        self.linear_requant_payload_bytes = max(self.linear_requant_payload_bytes, payload.numel() * payload.element_size())
        self.linear_requant_scale_bytes = max(self.linear_requant_scale_bytes, scale.numel() * scale.element_size())
        self.record_boundary("linear_output_requant")

    def record_tri_pack(self, payloads: tuple[torch.Tensor, ...], scales: tuple[torch.Tensor, ...]) -> None:
        self.tri_pack_ops += 1
        self.tri_pack_payload_bytes = max(
            self.tri_pack_payload_bytes,
            sum(tensor.numel() * tensor.element_size() for tensor in payloads),
        )
        self.tri_pack_scale_bytes = max(
            self.tri_pack_scale_bytes,
            sum(tensor.numel() * tensor.element_size() for tensor in scales),
        )
        self.record_boundary("tri_mxfp8_pack")

    def record_tri_output_requant(self, payload: torch.Tensor, scale: torch.Tensor) -> None:
        self.tri_output_requant_ops += 1
        self.tri_output_requant_payload_bytes = max(
            self.tri_output_requant_payload_bytes,
            payload.numel() * payload.element_size(),
        )
        self.tri_output_requant_scale_bytes = max(
            self.tri_output_requant_scale_bytes,
            scale.numel() * scale.element_size(),
        )
        self.record_boundary("tri_output_requant")

    def record_native_linear_fused(self, payload: torch.Tensor, scale: torch.Tensor) -> None:
        self.native_linear_fused_ops += 1
        self.native_linear_payload_bytes = max(
            self.native_linear_payload_bytes,
            payload.numel() * payload.element_size(),
        )
        self.native_linear_scale_bytes = max(
            self.native_linear_scale_bytes,
            scale.numel() * scale.element_size(),
        )
        self.record_boundary("native_linear_fused")

    def record_native_gate_fused(self, payload: torch.Tensor, scale: torch.Tensor) -> None:
        self.native_gate_fused_ops += 1
        self.native_gate_payload_bytes = max(
            self.native_gate_payload_bytes,
            payload.numel() * payload.element_size(),
        )
        self.native_gate_scale_bytes = max(
            self.native_gate_scale_bytes,
            scale.numel() * scale.element_size(),
        )
        self.record_boundary("native_gate_fused")

    def record_native_tri_fused(self, payload: torch.Tensor, scale: torch.Tensor) -> None:
        self.native_tri_fused_ops += 1
        self.native_tri_payload_bytes = max(
            self.native_tri_payload_bytes,
            payload.numel() * payload.element_size(),
        )
        self.native_tri_scale_bytes = max(
            self.native_tri_scale_bytes,
            scale.numel() * scale.element_size(),
        )
        self.record_boundary("native_tri_fused")

    def record_boundary(self, boundary: str) -> None:
        self.boundary_counts[boundary] = self.boundary_counts.get(boundary, 0) + 1


def _record_boundary(stats: FP8ActivationStats | None, boundary: str) -> None:
    if stats is not None:
        stats.record_boundary(boundary)


def _materialize_pair_tensor(
    tensor: torch.Tensor | QuantizedPairTensor | Mxfp8PairTensor,
    stats: FP8ActivationStats | None,
    boundary: str,
) -> torch.Tensor:
    if isinstance(tensor, (QuantizedPairTensor, Mxfp8PairTensor)):
        if stats is not None:
            stats.record_dequantize(boundary)
        return tensor.dequantize(dtype=torch.bfloat16)
    return tensor


def _gold_pack_block32_tensor(
    dense: torch.Tensor,
    logical_dtype: torch.dtype,
    stats: FP8ActivationStats | None,
    boundary: str,
) -> Mxfp8PairTensor:
    payload, scale = fp8_requantize_block32(dense.to(torch.float32), scale_dtype=torch.float32)
    if stats is not None:
        stats.record_linear_requant(payload, scale)
        stats.record_boundary(boundary)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=logical_dtype)


def _tri_mask_from_mask(mask: torch.Tensor | None, mode_name: str) -> torch.Tensor | None:
    if mask is None:
        return None
    mask_bool = mask.to(torch.bool)
    if mask_bool.dim() == 2:
        return (mask_bool[:, :, None] & mask_bool[:, None, :]).contiguous()
    if mask_bool.dim() == 3:
        return mask_bool.contiguous()
    raise ValueError(f"Unsupported mask rank for {mode_name}: {mask_bool.dim()}")


def quantize_linear_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_to_fp8(weight.detach().to(torch.bfloat16), scale_dtype=torch.float32)


def quantize_linear_weight_to_fp8_tensorwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f32 = weight.detach().to(torch.float32)
    scale = (weight_f32.abs().amax().clamp(min=1e-12) / _FP8_MAX).to(torch.float32)
    weight_fp8 = (weight_f32 / scale).to(torch.float8_e4m3fn)
    return weight_fp8.contiguous(), scale.reshape(())


def quantize_linear_weight_to_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_3d = weight.detach().to(torch.float32).unsqueeze(0).contiguous()
    if weight_3d.shape[-1] % 32 != 0:
        raise ValueError(f"MXFP8 weight quantization requires input dim divisible by 32, got {weight_3d.shape[-1]}")
    min_scale = 5.877471754111438e-39
    blocks = weight_3d.reshape(weight_3d.shape[0], weight_3d.shape[1], weight_3d.shape[2] // 32, 32)
    block_amax = blocks.abs().amax(dim=-1).clamp(min=min_scale)
    scale_f32 = torch.pow(2.0, torch.ceil(torch.log2(block_amax / _FP8_MAX))).clamp(min=min_scale).to(torch.float32)
    weight_mxfp8 = (blocks / scale_f32.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).reshape_as(weight_3d)
    scale = scale_f32.to(torch.float8_e8m0fnu).contiguous()
    scale_swizzled = _swizzle_mxfp8_scale_rowwise(scale.contiguous())
    return weight_mxfp8.contiguous(), scale.contiguous(), scale_swizzled.contiguous()


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor | None) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value)


def configure_fp8_linear_weight_(module: nn.Linear, enabled: bool) -> None:
    if enabled:
        weight_fp8, scale_w = quantize_linear_weight_to_fp8(module.weight)
        _set_buffer(module, "weight_fp8", weight_fp8.contiguous())
        _set_buffer(module, "scale_w", scale_w.contiguous())
        if module.weight.is_cuda and bmm_ext_raw is not None:
            weight_mxfp8, scale_w_mxfp8, scale_w_mxfp8_swizzled = quantize_linear_weight_to_mxfp8(module.weight)
            weight_mxfp8_cutlass_col = weight_mxfp8.transpose(1, 2).contiguous()
        else:
            weight_mxfp8, scale_w_mxfp8, scale_w_mxfp8_swizzled, weight_mxfp8_cutlass_col = None, None, None, None
        _set_buffer(module, "weight_mxfp8", weight_mxfp8)
        _set_buffer(module, "scale_w_mxfp8", scale_w_mxfp8)
        _set_buffer(module, "scale_w_mxfp8_swizzled", scale_w_mxfp8_swizzled)
        _set_buffer(module, "weight_mxfp8_cutlass_col", weight_mxfp8_cutlass_col)
        _set_buffer(module, "weight_fp8_tensorwise", None)
        _set_buffer(module, "scale_w_tensorwise", None)
    else:
        _set_buffer(module, "weight_fp8", None)
        _set_buffer(module, "scale_w", None)
        _set_buffer(module, "weight_mxfp8", None)
        _set_buffer(module, "scale_w_mxfp8", None)
        _set_buffer(module, "scale_w_mxfp8_swizzled", None)
        _set_buffer(module, "weight_mxfp8_cutlass_col", None)
        _set_buffer(module, "weight_fp8_tensorwise", None)
        _set_buffer(module, "scale_w_tensorwise", None)


def _has_fp8_linear_weight(module: nn.Linear) -> bool:
    return isinstance(getattr(module, "weight_fp8", None), torch.Tensor) and isinstance(getattr(module, "scale_w", None), torch.Tensor)


def _has_mxfp8_linear_weight(module: nn.Linear) -> bool:
    return (
        isinstance(getattr(module, "weight_mxfp8", None), torch.Tensor)
        and isinstance(getattr(module, "scale_w_mxfp8", None), torch.Tensor)
        and isinstance(getattr(module, "scale_w_mxfp8_swizzled", None), torch.Tensor)
    )


def _has_fp8_linear_weight_tensorwise(module: nn.Linear) -> bool:
    return isinstance(getattr(module, "weight_fp8_tensorwise", None), torch.Tensor) and isinstance(
        getattr(module, "scale_w_tensorwise", None), torch.Tensor
    )


def fp8_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    if not _has_fp8_linear_weight(module):
        raise RuntimeError("FP8 linear precision requested but quantized buffers are missing; call configure_linear_precision(model, 'fp8') after moving the model to its target device.")
    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)

    def run_scaled_mm(x_chunk: torch.Tensor) -> torch.Tensor:
        x_fp8, scale_x = quantize_to_fp8(x_chunk, scale_dtype=torch.float32)
        # Row-wise scaling on this torch build only supports bf16/fp16 output; resident FP8 storage is restored by the caller.
        return torch._scaled_mm(
            x_fp8,
            module.weight_fp8.t(),
            scale_x.contiguous(),
            module.scale_w.t().contiguous(),
            bias=module.bias,
            out_dtype=torch.bfloat16,
        )

    if x_2d.shape[0] <= _FP8_LINEAR_MAX_ROWS:
        y = run_scaled_mm(x_2d)
    else:
        # Large flattened sequence lengths can trigger CUTLASS internal failures; chunk the row dimension to preserve the FP8 path.
        y = torch.cat([run_scaled_mm(x_2d[start : start + _FP8_LINEAR_MAX_ROWS]) for start in range(0, x_2d.shape[0], _FP8_LINEAR_MAX_ROWS)], dim=0)
    return y.reshape(*original_shape, module.weight.shape[0])


def fp8_linear_forward_quantized(
    module: nn.Linear,
    x: QuantizedPairTensor,
    stats: FP8ActivationStats | None = None,
) -> QuantizedPairTensor:
    if not _has_fp8_linear_weight(module):
        raise RuntimeError(
            "FP8 linear precision requested but quantized buffers are missing; call configure_linear_precision(model, 'fp8', include_transition=True) after moving the model to its target device."
        )
    original_shape = x.payload.shape[:-1]
    x_payload_2d = x.payload.reshape(-1, x.payload.shape[-1]).contiguous()
    x_scale_2d = x.scale.reshape(-1, 1).contiguous()
    scale_dtype = x.scale.dtype

    def run_scaled_mm(x_chunk: torch.Tensor, scale_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch._scaled_mm(
            x_chunk,
            module.weight_fp8.t(),
            scale_chunk.to(torch.float32).contiguous(),
            module.scale_w.t().contiguous(),
            bias=module.bias,
            out_dtype=torch.bfloat16,
        )
        payload, scale = fp8_requantize_rows(y, scale_dtype=scale_dtype)
        if stats is not None:
            stats.record_linear_requant(payload, scale)
        return payload, scale

    if x_payload_2d.shape[0] <= _FP8_LINEAR_MAX_ROWS:
        payload_2d, scale_2d = run_scaled_mm(x_payload_2d, x_scale_2d)
    else:
        payload_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        for start in range(0, x_payload_2d.shape[0], _FP8_LINEAR_MAX_ROWS):
            payload_chunk, scale_chunk = run_scaled_mm(
                x_payload_2d[start : start + _FP8_LINEAR_MAX_ROWS],
                x_scale_2d[start : start + _FP8_LINEAR_MAX_ROWS],
            )
            payload_parts.append(payload_chunk)
            scale_parts.append(scale_chunk)
        payload_2d = torch.cat(payload_parts, dim=0)
        scale_2d = torch.cat(scale_parts, dim=0)
    payload = payload_2d.reshape(*original_shape, module.weight.shape[0]).contiguous()
    scale = scale_2d.reshape(*original_shape, 1).contiguous()
    return QuantizedPairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def mxfp8_linear_forward_quantized(
    module: nn.Linear,
    x: Mxfp8PairTensor,
    stats: FP8ActivationStats | None = None,
) -> Mxfp8PairTensor:
    if bmm_ext_raw is None or not _has_mxfp8_linear_weight(module):
        raise RuntimeError(
            "MXFP8 linear precision requested but raw MXFP8 buffers are missing; call configure_linear_precision(model, 'fp8', include_transition=True) after moving the model to its target device."
        )
    original_shape = x.payload.shape[:-1]
    in_dim = x.payload.shape[-1]
    out_dim = module.weight.shape[0]
    rows = x.payload.numel() // in_dim
    groups = in_dim // 32
    payload_2d = x.payload.reshape(rows, in_dim).contiguous()
    scale_2d = x.scale.reshape(rows, groups).contiguous()
    scale_dtype = x.scale.dtype

    def run_raw_linear(payload_chunk: torch.Tensor, scale_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale_e8 = scale_chunk.to(torch.float8_e8m0fnu).reshape(1, payload_chunk.shape[0], groups).contiguous()
        scale_swizzled = _swizzle_mxfp8_scale_rowwise(scale_e8)
        y = bmm_ext_raw.mxfp8_cublaslt_bmm(
            payload_chunk.reshape(1, payload_chunk.shape[0], in_dim).contiguous(),
            module.weight_mxfp8.contiguous(),
            scale_swizzled,
            module.scale_w_mxfp8_swizzled.contiguous(),
            "bfloat16",
        ).reshape(payload_chunk.shape[0], out_dim)
        payload, scale = fp8_requantize_block32(
            y,
            bias=module.bias,
            scale_dtype=scale_dtype,
        )
        if stats is not None:
            stats.record_linear_requant(payload, scale)
        return payload, scale

    if rows <= _FP8_LINEAR_MAX_ROWS:
        payload_2d_out, scale_2d_out = run_raw_linear(payload_2d, scale_2d)
    else:
        payload_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        for start in range(0, rows, _FP8_LINEAR_MAX_ROWS):
            payload_chunk, scale_chunk = run_raw_linear(
                payload_2d[start : start + _FP8_LINEAR_MAX_ROWS],
                scale_2d[start : start + _FP8_LINEAR_MAX_ROWS],
            )
            payload_parts.append(payload_chunk)
            scale_parts.append(scale_chunk)
        payload_2d_out = torch.cat(payload_parts, dim=0)
        scale_2d_out = torch.cat(scale_parts, dim=0)
    payload = payload_2d_out.reshape(*original_shape, out_dim).contiguous()
    scale = scale_2d_out.reshape(*original_shape, out_dim // 32).contiguous()
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def native_linear_forward_quantized(
    module: nn.Linear,
    x: Mxfp8PairTensor,
    stats: FP8ActivationStats | None = None,
    apply_relu: bool = False,
    direct_fp8_output: bool = False,
    fuse_bias_epilogue: bool = False,
    residual: Mxfp8PairTensor | None = None,
    use_gold_output_pack: bool = False,
) -> Mxfp8PairTensor:
    if minifold_native_raw is None or not _has_mxfp8_linear_weight(module):
        raise RuntimeError(
            "Native MXFP8 linear precision requested but native MXFP8 buffers are missing; build minifold_native_ext and call configure_linear_precision(model, 'fp8', include_transition=True) after moving the model to its target device."
        )
    original_shape = x.payload.shape[:-1]
    in_dim = x.payload.shape[-1]
    out_dim = module.weight.shape[0]
    rows = x.payload.numel() // in_dim
    groups = in_dim // 32

    payload_2d = x.payload.reshape(rows, in_dim).contiguous()
    scale_2d = x.scale.reshape(rows, groups).contiguous()
    residual_payload_2d = None
    residual_scale_2d = None
    if residual is not None:
        if residual.payload.shape[:-1] != original_shape or residual.payload.shape[-1] != out_dim:
            raise ValueError(
                f"Residual shape {tuple(residual.payload.shape)} does not match expected output shape {tuple((*original_shape, out_dim))}"
            )
        residual_payload_2d = residual.payload.reshape(rows, out_dim).contiguous()
        residual_scale_2d = residual.scale.reshape(rows, out_dim // 32).contiguous()

    def run_native_linear(
        payload_chunk: torch.Tensor,
        scale_chunk: torch.Tensor,
        residual_payload_chunk: torch.Tensor | None = None,
        residual_scale_chunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale_e8 = scale_chunk.to(torch.float8_e8m0fnu).reshape(1, payload_chunk.shape[0], groups).contiguous()
        scale_swizzled = _swizzle_mxfp8_scale_rowwise(scale_e8)
        use_fc1_direct = (
            apply_relu
            and residual_payload_chunk is None
            and residual_scale_chunk is None
            and not use_gold_output_pack
            and in_dim == 128
            and out_dim == 512
            and getattr(module, "_native_fc1_direct_enabled", False)
            and hasattr(minifold_native_raw, "linear_block32_fc1_direct")
            and isinstance(getattr(module, "weight_mxfp8_cutlass_col", None), torch.Tensor)
            and getattr(module, "_native_fc1_direct_supported", True)
        )
        use_direct_output = direct_fp8_output and getattr(module, "_native_direct_fp8_output_supported", True)
        support_attr = "_native_relu_bias_epilogue_supported" if apply_relu else "_native_bias_epilogue_supported"
        use_fused_bias_epilogue = fuse_bias_epilogue and getattr(module, support_attr, True)
        residual_payload_3d = None
        residual_scale_3d = None
        if residual_payload_chunk is not None and residual_scale_chunk is not None:
            residual_payload_3d = residual_payload_chunk.reshape(1, payload_chunk.shape[0], out_dim).contiguous()
            residual_scale_3d = residual_scale_chunk.reshape(1, payload_chunk.shape[0], out_dim // 32).contiguous()
        if use_fc1_direct:
            try:
                payload, scale = minifold_native_raw.linear_block32_fc1_direct(
                    payload_chunk.reshape(1, payload_chunk.shape[0], in_dim).contiguous(),
                    module.weight_mxfp8_cutlass_col.contiguous(),
                    scale_swizzled,
                    module.scale_w_mxfp8_swizzled.contiguous(),
                    None if module.bias is None else module.bias.contiguous(),
                )
            except RuntimeError:
                module._native_fc1_direct_supported = False
            else:
                payload = payload.reshape(payload_chunk.shape[0], out_dim)
                scale = scale.reshape(payload_chunk.shape[0], out_dim // 32)
                if stats is not None:
                    stats.record_native_linear_fused(payload, scale)
                return payload, scale
        current_direct_output = use_direct_output
        current_fused_bias_epilogue = use_fused_bias_epilogue
        while True:
            try:
                if use_gold_output_pack:
                    if not hasattr(minifold_native_raw, "linear_block32_raw_debug"):
                        raise RuntimeError("Native gold-pack linear debug binding is unavailable")
                    dense = minifold_native_raw.linear_block32_raw_debug(
                        payload_chunk.reshape(1, payload_chunk.shape[0], in_dim).contiguous(),
                        module.weight_mxfp8.contiguous(),
                        scale_swizzled,
                        module.scale_w_mxfp8_swizzled.contiguous(),
                        None if module.bias is None else module.bias.contiguous(),
                        "bfloat16",
                        apply_relu,
                        current_direct_output,
                        current_fused_bias_epilogue,
                        residual_payload_3d,
                        residual_scale_3d,
                    )
                else:
                    payload, scale = minifold_native_raw.linear_block32_fused(
                        payload_chunk.reshape(1, payload_chunk.shape[0], in_dim).contiguous(),
                        module.weight_mxfp8.contiguous(),
                        scale_swizzled,
                        module.scale_w_mxfp8_swizzled.contiguous(),
                        None if module.bias is None else module.bias.contiguous(),
                        "bfloat16",
                        apply_relu,
                        current_direct_output,
                        current_fused_bias_epilogue,
                        residual_payload_3d,
                        residual_scale_3d,
                    )
                break
            except RuntimeError as exc:
                exc_text = str(exc)
                if current_direct_output and (
                    "cublasLtMatmulAlgoGetHeuristic" in exc_text or "cublasLtMatmul(" in exc_text
                ):
                    module._native_direct_fp8_output_supported = False
                    current_direct_output = False
                    continue
                if current_fused_bias_epilogue and (
                    "cublasLtMatmulAlgoGetHeuristic" in exc_text or "cublasLtMatmul(" in exc_text
                ):
                    setattr(module, support_attr, False)
                    current_fused_bias_epilogue = False
                    continue
                raise
        if use_gold_output_pack:
            dense = dense.reshape(payload_chunk.shape[0], out_dim).contiguous()
            pair = _gold_pack_block32_tensor(
                dense,
                x.logical_dtype,
                stats,
                boundary="native_linear_gold_output_pack",
            )
            if stats is not None:
                stats.record_native_linear_fused(pair.payload, pair.scale)
            return pair.payload, pair.scale
        payload = payload.reshape(payload_chunk.shape[0], out_dim)
        scale = scale.reshape(payload_chunk.shape[0], out_dim // 32)
        if stats is not None:
            stats.record_native_linear_fused(payload, scale)
        return payload, scale

    row_limit = _FP8_LINEAR_MAX_ROWS
    if (
        apply_relu
        and residual_payload_2d is None
        and residual_scale_2d is None
        and in_dim == 128
        and out_dim == 512
        and getattr(module, "_native_fc1_direct_enabled", False)
    ):
        row_limit = min(row_limit, _NATIVE_FC1_DIRECT_MAX_ROWS)

    if rows <= row_limit:
        payload_2d_out, scale_2d_out = run_native_linear(payload_2d, scale_2d, residual_payload_2d, residual_scale_2d)
    else:
        payload_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        for start in range(0, rows, row_limit):
            payload_chunk, scale_chunk = run_native_linear(
                payload_2d[start : start + row_limit],
                scale_2d[start : start + row_limit],
                None if residual_payload_2d is None else residual_payload_2d[start : start + row_limit],
                None if residual_scale_2d is None else residual_scale_2d[start : start + row_limit],
            )
            payload_parts.append(payload_chunk)
            scale_parts.append(scale_chunk)
        payload_2d_out = torch.cat(payload_parts, dim=0)
        scale_2d_out = torch.cat(scale_parts, dim=0)
    payload = payload_2d_out.reshape(*original_shape, out_dim).contiguous()
    scale = scale_2d_out.reshape(*original_shape, out_dim // 32).contiguous()
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def native_gate_sigmoid_mul_quantized(
    lhs_module: nn.Linear,
    rhs_module: nn.Linear,
    x: Mxfp8PairTensor,
    stats: FP8ActivationStats | None = None,
    residual: Mxfp8PairTensor | None = None,
    use_gold_output_pack: bool = False,
) -> Mxfp8PairTensor:
    if minifold_native_raw is None or not _has_mxfp8_linear_weight(lhs_module) or not _has_mxfp8_linear_weight(rhs_module):
        raise RuntimeError(
            "Native MXFP8 fused gate requested but native MXFP8 buffers are missing; build minifold_native_ext and call configure_linear_precision(model, 'fp8', include_transition=True) after moving the model to its target device."
        )
    original_shape = x.payload.shape[:-1]
    in_dim = x.payload.shape[-1]
    out_dim = lhs_module.weight.shape[0]
    rows = x.payload.numel() // in_dim
    groups = in_dim // 32

    payload_2d = x.payload.reshape(rows, in_dim).contiguous()
    scale_2d = x.scale.reshape(rows, groups).contiguous()
    scale_e8 = scale_2d.to(torch.float8_e8m0fnu).reshape(1, rows, groups).contiguous()
    scale_swizzled = _swizzle_mxfp8_scale_rowwise(scale_e8)
    residual_payload_3d = None
    residual_scale_3d = None
    if residual is not None:
        if residual.payload.shape[:-1] != original_shape or residual.payload.shape[-1] != out_dim:
            raise ValueError(
                f"Residual shape {tuple(residual.payload.shape)} does not match expected output shape {tuple((*original_shape, out_dim))}"
            )
        residual_payload_3d = residual.payload.reshape(1, rows, out_dim).contiguous()
        residual_scale_3d = residual.scale.reshape(1, rows, out_dim // 32).contiguous()
    if use_gold_output_pack:
        if not hasattr(minifold_native_raw, "gate_sigmoid_mul_block32_raw_debug"):
            raise RuntimeError("Native gold-pack gate debug binding is unavailable")
        dense = minifold_native_raw.gate_sigmoid_mul_block32_raw_debug(
            payload_2d.reshape(1, rows, in_dim).contiguous(),
            scale_swizzled,
            lhs_module.weight_mxfp8.contiguous(),
            lhs_module.scale_w_mxfp8_swizzled.contiguous(),
            None if lhs_module.bias is None else lhs_module.bias.contiguous(),
            rhs_module.weight_mxfp8.contiguous(),
            rhs_module.scale_w_mxfp8_swizzled.contiguous(),
            None if rhs_module.bias is None else rhs_module.bias.contiguous(),
            "bfloat16",
            residual_payload_3d,
            residual_scale_3d,
        )
        pair = _gold_pack_block32_tensor(
            dense.reshape(*original_shape, out_dim).contiguous(),
            x.logical_dtype,
            stats,
            boundary="native_gate_gold_output_pack",
        )
        payload = pair.payload
        scale = pair.scale
    else:
        payload, scale = minifold_native_raw.gate_sigmoid_mul_block32_fused(
            payload_2d.reshape(1, rows, in_dim).contiguous(),
            scale_swizzled,
            lhs_module.weight_mxfp8.contiguous(),
            lhs_module.scale_w_mxfp8_swizzled.contiguous(),
            None if lhs_module.bias is None else lhs_module.bias.contiguous(),
            rhs_module.weight_mxfp8.contiguous(),
            rhs_module.scale_w_mxfp8_swizzled.contiguous(),
            None if rhs_module.bias is None else rhs_module.bias.contiguous(),
            "bfloat16",
            residual_payload_3d,
            residual_scale_3d,
        )
        payload = payload.reshape(*original_shape, out_dim).contiguous()
        scale = scale.reshape(*original_shape, out_dim // 32).contiguous()
    if stats is not None:
        stats.record_native_gate_fused(payload, scale)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def native_tri_mul_from_block32_quantized(
    x: Mxfp8PairTensor,
    mask: torch.Tensor | None = None,
    stats: FP8ActivationStats | None = None,
    *,
    use_gold_input_pack: bool = False,
    use_gold_output_pack: bool = False,
) -> Mxfp8PairTensor:
    if minifold_native_raw is None:
        raise RuntimeError("Native MXFP8 tri extension is not importable")
    tri_mask = _tri_mask_from_mask(mask, "native MXFP8 tri path")
    if use_gold_input_pack or use_gold_output_pack:
        if not hasattr(minifold_native_raw, "tri_mul_pair_from_packed_debug"):
            raise RuntimeError("Native gold-pack tri debug bindings are unavailable")
        batch = x.payload.shape[0]
        if use_gold_input_pack:
            a1, a1_scale = fp8_pack_block32_carrier_to_mxfp8_lhs(
                x.payload, x.scale, mask=tri_mask, channel_group=0, transpose=False
            )
            b1, b1_scale = fp8_pack_block32_carrier_to_mxfp8_lhs(
                x.payload, x.scale, mask=tri_mask, channel_group=1, transpose=False
            )
            a2_t, a2_t_scale = fp8_pack_block32_carrier_to_mxfp8_lhs(
                x.payload, x.scale, mask=tri_mask, channel_group=2, transpose=True
            )
            b2_rhs, b2_rhs_scale = fp8_pack_block32_carrier_to_mxfp8_rhs(
                x.payload, x.scale, mask=tri_mask, channel_group=3
            )
        else:
            a1, a1_scale, b1, b1_scale, a2_t, a2_t_scale, b2_rhs, b2_rhs_scale = (
                minifold_native_raw.pack_block32_to_mxfp8_fused_debug(
                    x.payload.contiguous(),
                    x.scale.contiguous(),
                    tri_mask,
                )
            )
        if stats is not None:
            stats.record_tri_pack((a1, b1, a2_t, b2_rhs), (a1_scale, b1_scale, a2_t_scale, b2_rhs_scale))
        x1, x2 = minifold_native_raw.tri_mul_pair_from_packed_debug(
            a1,
            b1,
            a2_t,
            b2_rhs,
            a1_scale,
            b1_scale,
            a2_t_scale,
            b2_rhs_scale,
            "float16",
        )
        if use_gold_output_pack:
            payload, scale = fp8_tri_outputs_to_block32_carrier(x1, x2, batch=batch, scale_dtype=x.scale.dtype)
            if stats is not None:
                stats.record_tri_output_requant(payload, scale)
                stats.record_boundary("native_tri_gold_output_pack")
        else:
            payload, scale = minifold_native_raw.tri_pair_to_block32_carrier_debug(x1, x2, batch)
        if stats is not None:
            stats.record_native_tri_fused(payload, scale)
        return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)
    payload, scale = minifold_native_raw.tri_mul_pair_from_block32_carrier(
        x.payload.contiguous(),
        x.scale.contiguous(),
        tri_mask,
        "float16",
    )
    if stats is not None:
        stats.record_native_tri_fused(payload, scale)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def native_tri_mul_from_gate_quantized(
    lhs_module: nn.Linear,
    rhs_module: nn.Linear,
    x: Mxfp8PairTensor,
    mask: torch.Tensor,
    stats: FP8ActivationStats | None = None,
) -> Mxfp8PairTensor:
    if (
        minifold_native_raw is None
        or not hasattr(minifold_native_raw, "tri_gate_block32_fused")
        or not _has_mxfp8_linear_weight(lhs_module)
        or not _has_mxfp8_linear_weight(rhs_module)
    ):
        gated = native_gate_sigmoid_mul_quantized(lhs_module, rhs_module, x, stats=stats)
        return native_tri_mul_from_block32_quantized(gated, mask=mask, stats=stats)
    original_shape = x.payload.shape[:-1]
    in_dim = x.payload.shape[-1]
    rows = x.payload.numel() // in_dim
    groups = in_dim // 32
    payload_2d = x.payload.reshape(rows, in_dim).contiguous()
    scale_2d = x.scale.reshape(rows, groups).contiguous()
    scale_e8 = scale_2d.to(torch.float8_e8m0fnu).reshape(1, rows, groups).contiguous()
    scale_swizzled = _swizzle_mxfp8_scale_rowwise(scale_e8)
    mask_bool = mask.to(torch.bool)
    if mask_bool.dim() == 2:
        tri_mask = (mask_bool[:, :, None] & mask_bool[:, None, :]).contiguous()
    elif mask_bool.dim() == 3:
        tri_mask = mask_bool.contiguous()
    else:
        raise ValueError(f"Unsupported mask rank for native fused tri path: {mask_bool.dim()}")
    payload, scale = minifold_native_raw.tri_gate_block32_fused(
        payload_2d.reshape(1, rows, in_dim).contiguous(),
        scale_swizzled,
        lhs_module.weight_mxfp8.contiguous(),
        lhs_module.scale_w_mxfp8_swizzled.contiguous(),
        None if lhs_module.bias is None else lhs_module.bias.contiguous(),
        rhs_module.weight_mxfp8.contiguous(),
        rhs_module.scale_w_mxfp8_swizzled.contiguous(),
        None if rhs_module.bias is None else rhs_module.bias.contiguous(),
        tri_mask,
        "float16",
    )
    payload = payload.reshape(*original_shape, payload.shape[-1]).contiguous()
    scale = scale.reshape(*original_shape, scale.shape[-1]).contiguous()
    if stats is not None:
        stats.record_native_tri_fused(payload, scale)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def native_tri_mul_from_input_quantized(
    input_norm_module: nn.LayerNorm,
    lhs_module: nn.Linear,
    rhs_module: nn.Linear,
    x: Mxfp8PairTensor,
    mask: torch.Tensor,
    stats: FP8ActivationStats | None = None,
) -> Mxfp8PairTensor:
    if (
        minifold_native_raw is None
        or not hasattr(minifold_native_raw, "tri_input_norm_gate_block32_fused")
        or not _has_mxfp8_linear_weight(lhs_module)
        or not _has_mxfp8_linear_weight(rhs_module)
    ):
        normalized = native_mxfp8_layernorm_quantized(input_norm_module, x)
        return native_tri_mul_from_gate_quantized(lhs_module, rhs_module, normalized, mask=mask, stats=stats)
    mask_bool = mask.to(torch.bool)
    if mask_bool.dim() == 2:
        tri_mask = (mask_bool[:, :, None] & mask_bool[:, None, :]).contiguous()
    elif mask_bool.dim() == 3:
        tri_mask = mask_bool.contiguous()
    else:
        raise ValueError(f"Unsupported mask rank for native fused tri path: {mask_bool.dim()}")
    payload, scale = minifold_native_raw.tri_input_norm_gate_block32_fused(
        x.payload.contiguous(),
        x.scale.contiguous(),
        input_norm_module.weight.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        input_norm_module.bias.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        float(input_norm_module.eps),
        lhs_module.weight_mxfp8.contiguous(),
        lhs_module.scale_w_mxfp8_swizzled.contiguous(),
        None if lhs_module.bias is None else lhs_module.bias.contiguous(),
        rhs_module.weight_mxfp8.contiguous(),
        rhs_module.scale_w_mxfp8_swizzled.contiguous(),
        None if rhs_module.bias is None else rhs_module.bias.contiguous(),
        tri_mask,
        "float16",
    )
    if stats is not None:
        stats.record_native_tri_fused(payload, scale)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def native_tri_gate_layernorm_quantized(
    lhs_module: nn.Linear,
    rhs_module: nn.Linear,
    output_norm_module: nn.LayerNorm,
    x: Mxfp8PairTensor,
    mask: torch.Tensor,
    stats: FP8ActivationStats | None = None,
) -> Mxfp8PairTensor:
    if (
        minifold_native_raw is None
        or not hasattr(minifold_native_raw, "tri_gate_layernorm_block32_fused")
        or not _has_mxfp8_linear_weight(lhs_module)
        or not _has_mxfp8_linear_weight(rhs_module)
    ):
        gated = native_gate_sigmoid_mul_quantized(lhs_module, rhs_module, x, stats=stats)
        tri_out_q = native_tri_mul_from_block32_quantized(gated, mask=mask, stats=stats)
        return native_mxfp8_layernorm_quantized(output_norm_module, tri_out_q)
    original_shape = x.payload.shape[:-1]
    in_dim = x.payload.shape[-1]
    rows = x.payload.numel() // in_dim
    groups = in_dim // 32
    payload_2d = x.payload.reshape(rows, in_dim).contiguous()
    scale_2d = x.scale.reshape(rows, groups).contiguous()
    scale_e8 = scale_2d.to(torch.float8_e8m0fnu).reshape(1, rows, groups).contiguous()
    scale_swizzled = _swizzle_mxfp8_scale_rowwise(scale_e8)
    mask_bool = mask.to(torch.bool)
    if mask_bool.dim() == 2:
        tri_mask = (mask_bool[:, :, None] & mask_bool[:, None, :]).contiguous()
    elif mask_bool.dim() == 3:
        tri_mask = mask_bool.contiguous()
    else:
        raise ValueError(f"Unsupported mask rank for native fused tri path: {mask_bool.dim()}")
    payload, scale = minifold_native_raw.tri_gate_layernorm_block32_fused(
        payload_2d.reshape(1, rows, in_dim).contiguous(),
        scale_swizzled,
        lhs_module.weight_mxfp8.contiguous(),
        lhs_module.scale_w_mxfp8_swizzled.contiguous(),
        None if lhs_module.bias is None else lhs_module.bias.contiguous(),
        rhs_module.weight_mxfp8.contiguous(),
        rhs_module.scale_w_mxfp8_swizzled.contiguous(),
        None if rhs_module.bias is None else rhs_module.bias.contiguous(),
        tri_mask,
        output_norm_module.weight.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        output_norm_module.bias.to(device=x.device, dtype=torch.bfloat16).contiguous(),
        float(output_norm_module.eps),
        "float16",
    )
    payload = payload.reshape(*original_shape, payload.shape[-1]).contiguous()
    scale = scale.reshape(*original_shape, scale.shape[-1]).contiguous()
    if stats is not None:
        stats.record_native_tri_fused(payload, scale)
    return Mxfp8PairTensor(payload=payload, scale=scale, logical_dtype=x.logical_dtype)


def collect_fp8_linear_storage_stats(module: nn.Module) -> dict[str, int]:
    stats = {
        "linear_count": 0,
        "weight_bytes": 0,
        "scale_bytes": 0,
        "tensorwise_weight_bytes": 0,
        "tensorwise_scale_bytes": 0,
        "mxfp8_weight_bytes": 0,
        "mxfp8_scale_bytes": 0,
    }
    for submodule in module.modules():
        if not isinstance(submodule, nn.Linear) or not _has_fp8_linear_weight(submodule):
            continue
        stats["linear_count"] += 1
        stats["weight_bytes"] += submodule.weight_fp8.numel() * submodule.weight_fp8.element_size()
        stats["scale_bytes"] += submodule.scale_w.numel() * submodule.scale_w.element_size()
        if _has_mxfp8_linear_weight(submodule):
            stats["mxfp8_weight_bytes"] += submodule.weight_mxfp8.numel() * submodule.weight_mxfp8.element_size()
            stats["mxfp8_scale_bytes"] += (
                submodule.scale_w_mxfp8.numel() * submodule.scale_w_mxfp8.element_size()
                + submodule.scale_w_mxfp8_swizzled.numel() * submodule.scale_w_mxfp8_swizzled.element_size()
            )
        if _has_fp8_linear_weight_tensorwise(submodule):
            stats["tensorwise_weight_bytes"] += submodule.weight_fp8_tensorwise.numel() * submodule.weight_fp8_tensorwise.element_size()
            stats["tensorwise_scale_bytes"] += submodule.scale_w_tensorwise.numel() * submodule.scale_w_tensorwise.element_size()
    return stats


def tri_mul_fp8_cublaslt_quantized(
    x: QuantizedPairTensor,
    stats: FP8ActivationStats | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> QuantizedPairTensor:
    if bmm_ext_raw is None:
        raise RuntimeError("fp8_cublaslt raw extension is not importable")
    batch, n, n2, dim = x.payload.shape
    if dim != 128:
        raise ValueError(f"tri_mul_fp8_cublaslt_quantized expects channel dim 128, got {dim}")
    if n != n2:
        raise ValueError("tri_mul_fp8_cublaslt_quantized expects square spatial dimensions")

    a1, a1_scale = fp8_pack_carrier_to_mxfp8_lhs(x.payload, x.scale, channel_offset=0, transpose=False)
    b1, b1_scale = fp8_pack_carrier_to_mxfp8_lhs(x.payload, x.scale, channel_offset=32, transpose=False)
    a2_t, a2_t_scale = fp8_pack_carrier_to_mxfp8_lhs(x.payload, x.scale, channel_offset=64, transpose=True)
    b2_rhs, b2_rhs_scale = fp8_pack_carrier_to_mxfp8_rhs(x.payload, x.scale, channel_offset=96)
    if stats is not None:
        stats.record_tri_pack((a1, b1, a2_t, b2_rhs), (a1_scale, b1_scale, a2_t_scale, b2_rhs_scale))

    out_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if out_dtype not in out_map:
        raise ValueError(f"Unsupported raw FP8 tri out_dtype {out_dtype}")
    x1, x2 = bmm_ext_raw.mxfp8_cublaslt_tri_mul_pair(
        a1,
        b1,
        a2_t,
        b2_rhs,
        a1_scale,
        b1_scale,
        a2_t_scale,
        b2_rhs_scale,
        out_map[out_dtype],
    )
    tri_payload, tri_scale = fp8_tri_outputs_to_carrier(x1, x2, batch=batch, scale_dtype=x.scale.dtype)
    if stats is not None:
        stats.record_tri_output_requant(tri_payload, tri_scale)
    return QuantizedPairTensor(payload=tri_payload, scale=tri_scale, logical_dtype=x.logical_dtype)


def mxfp8_tri_mul_fp8_cublaslt_quantized(
    x: Mxfp8PairTensor,
    stats: FP8ActivationStats | None = None,
    out_dtype: torch.dtype = torch.float16,
    mask: torch.Tensor | None = None,
) -> Mxfp8PairTensor:
    if bmm_ext_raw is None:
        raise RuntimeError("fp8_cublaslt raw extension is not importable")
    batch, n, n2, dim = x.payload.shape
    if dim != 128:
        raise ValueError(f"mxfp8_tri_mul_fp8_cublaslt_quantized expects channel dim 128, got {dim}")
    if n != n2:
        raise ValueError("mxfp8_tri_mul_fp8_cublaslt_quantized expects square spatial dimensions")

    if mask is None:
        tri_mask = None
    else:
        mask_bool = mask.to(torch.bool)
        if mask_bool.dim() == 2:
            tri_mask = (mask_bool[:, :, None] & mask_bool[:, None, :]).contiguous()
        elif mask_bool.dim() == 3:
            tri_mask = mask_bool.contiguous()
        else:
            raise ValueError(f"Unsupported mask rank for MXFP8 tri path: {mask_bool.dim()}")
    a1, a1_scale = fp8_pack_block32_carrier_to_mxfp8_lhs(
        x.payload, x.scale, mask=tri_mask, channel_group=0, transpose=False
    )
    b1, b1_scale = fp8_pack_block32_carrier_to_mxfp8_lhs(
        x.payload, x.scale, mask=tri_mask, channel_group=1, transpose=False
    )
    a2_t, a2_t_scale = fp8_pack_block32_carrier_to_mxfp8_lhs(
        x.payload, x.scale, mask=tri_mask, channel_group=2, transpose=True
    )
    b2_rhs, b2_rhs_scale = fp8_pack_block32_carrier_to_mxfp8_rhs(
        x.payload, x.scale, mask=tri_mask, channel_group=3
    )
    if stats is not None:
        stats.record_tri_pack((a1, b1, a2_t, b2_rhs), (a1_scale, b1_scale, a2_t_scale, b2_rhs_scale))

    out_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if out_dtype not in out_map:
        raise ValueError(f"Unsupported raw FP8 tri out_dtype {out_dtype}")
    x1, x2 = bmm_ext_raw.mxfp8_cublaslt_tri_mul_pair(
        a1,
        b1,
        a2_t,
        b2_rhs,
        a1_scale,
        b1_scale,
        a2_t_scale,
        b2_rhs_scale,
        out_map[out_dtype],
    )
    tri_payload, tri_scale = fp8_tri_outputs_to_block32_carrier(x1, x2, batch=batch, scale_dtype=x.scale.dtype)
    if stats is not None:
        stats.record_tri_output_requant(tri_payload, tri_scale)
    return Mxfp8PairTensor(payload=tri_payload, scale=tri_scale, logical_dtype=x.logical_dtype)


class RelativePosition(nn.Module):
    def __init__(self, bins: int, pairwise_state_dim: int):
        super().__init__()
        self.bins = bins
        self.embedding = nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1
        diff[mask == 0] = 0
        return self.embedding(diff)


class SequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim: int, inner_dim: int, pairwise_state_dim: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(sequence_state_dim, eps=1e-5)
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.o_proj.bias)

    def forward(self, sequence_state: torch.Tensor) -> torch.Tensor:
        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        return self.o_proj(torch.cat([prod, diff], dim=-1))


class TransitionUpdate(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        linear_precision: str | None = None,
        hybrid_precision: HybridPrecisionConfig | dict[str, bool] | None = None,
    ):
        super().__init__()
        self.linear_precision = resolve_linear_precision(linear_precision)
        self.hybrid_precision = resolve_hybrid_precision_config(hybrid_precision)
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def set_linear_precision(self, linear_precision: str | None, *, include_transition: bool = False) -> None:
        self.linear_precision = resolve_linear_precision(linear_precision)
        enabled = include_transition and self.linear_precision == LINEAR_PRECISION_FP8
        for linear in (self.fc1, self.fc2):
            configure_fp8_linear_weight_(linear, enabled=enabled)
        # The chunked CUTLASS transition-entry path remains in-tree but
        # non-default on this node/build. When disabled, forward_fp8_native()
        # falls back to the stable resident-LN + native fc1 flow.
        self.fc1._native_transition_norm_fc1_enabled = False
        self.fc1._native_transition_norm_fc1_supported = True
        self.fc1._native_fc1_direct_enabled = False
        self.fc2._native_fc1_direct_enabled = False

    def _apply_linear(self, module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self.linear_precision == LINEAR_PRECISION_FP8 and _has_fp8_linear_weight(module):
            return fp8_linear_forward(module, x)
        return module(x)

    def _hybrid_fuses_residual(self) -> bool:
        return self.hybrid_precision.use_resident_fp8_residual and self.hybrid_precision.use_native_linear

    def forward(
        self,
        x: torch.Tensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> torch.Tensor:
        _record_boundary(stats, "layernorm")
        x = self.norm(x)
        _record_probe(activation_probe, block_idx, "transition_norm", x)
        x = self._apply_linear(self.fc1, x)
        _record_boundary(stats, "relu")
        x = F.relu(x)
        _record_probe(activation_probe, block_idx, "transition_fc1_relu", x)
        x = self._apply_linear(self.fc2, x)
        _record_probe(activation_probe, block_idx, "transition_fc2", x)
        return x

    def forward_fp8_extreme(
        self,
        x: Mxfp8PairTensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> Mxfp8PairTensor:
        if self.linear_precision != LINEAR_PRECISION_FP8 or not _has_mxfp8_linear_weight(self.fc1) or not _has_mxfp8_linear_weight(self.fc2):
            raise RuntimeError("fp8_extreme requires TransitionUpdate.fc1/fc2 to be quantized; build with linear_precision=fp8.")
        x = mxfp8_layernorm_quantized(self.norm, x)
        _record_probe(activation_probe, block_idx, "transition_norm", x)
        x = mxfp8_linear_forward_quantized(self.fc1, x, stats=stats)
        x = mxfp8_relu_quantized(x)
        _record_probe(activation_probe, block_idx, "transition_fc1_relu", x)
        x = mxfp8_linear_forward_quantized(self.fc2, x, stats=stats)
        _record_probe(activation_probe, block_idx, "transition_fc2", x)
        return x

    def forward_fp8_native(
        self,
        x: Mxfp8PairTensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> Mxfp8PairTensor:
        if self.linear_precision != LINEAR_PRECISION_FP8 or not _has_mxfp8_linear_weight(self.fc1) or not _has_mxfp8_linear_weight(self.fc2):
            raise RuntimeError("fp8_native requires TransitionUpdate.fc1/fc2 to be quantized; build with linear_precision=fp8.")
        residual = x
        x = native_transition_norm_fc1_quantized(self.norm, self.fc1, x, stats=stats)
        _record_probe(
            activation_probe,
            block_idx,
            "transition_fc1_relu",
            x,
            mode_op_name="transition_native_norm_fc1_out",
        )
        if stats is not None:
            stats.record_boundary("fp8_residual_add")
        x = native_linear_forward_quantized(
            self.fc2,
            x,
            stats=stats,
            direct_fp8_output=True,
            fuse_bias_epilogue=True,
            residual=residual,
        )
        _record_probe(
            activation_probe,
            block_idx,
            "transition_fc2",
            x,
            mode_op_name="transition_native_fc2_out",
        )
        return x

    def forward_fp8_native_gold_packs(
        self,
        x: Mxfp8PairTensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> Mxfp8PairTensor:
        if self.linear_precision != LINEAR_PRECISION_FP8 or not _has_mxfp8_linear_weight(self.fc1) or not _has_mxfp8_linear_weight(self.fc2):
            raise RuntimeError("fp8_native_gold_packs requires TransitionUpdate.fc1/fc2 to be quantized; build with linear_precision=fp8.")
        residual = x
        x = native_mxfp8_layernorm_quantized(self.norm, x)
        _record_probe(activation_probe, block_idx, "transition_norm", x)
        x = native_linear_forward_quantized(
            self.fc1,
            x,
            stats=stats,
            apply_relu=True,
            direct_fp8_output=True,
            fuse_bias_epilogue=True,
            use_gold_output_pack=True,
        )
        _record_probe(
            activation_probe,
            block_idx,
            "transition_fc1_relu",
            x,
            mode_op_name="transition_native_gold_fc1_out",
        )
        if stats is not None:
            stats.record_boundary("fp8_residual_add")
        x = native_linear_forward_quantized(
            self.fc2,
            x,
            stats=stats,
            direct_fp8_output=True,
            fuse_bias_epilogue=True,
            residual=residual,
            use_gold_output_pack=True,
        )
        _record_probe(
            activation_probe,
            block_idx,
            "transition_fc2",
            x,
            mode_op_name="transition_native_gold_fc2_out",
        )
        return x

    def forward_bf16_native(
        self,
        x: torch.Tensor,
        bf16_native_rung: str,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> torch.Tensor:
        if bf16_native_rung == BF16_NATIVE_RUNG_B2:
            return self.forward(x, stats=None, block_idx=block_idx, activation_probe=activation_probe)
        residual = x
        if bf16_native_rung == BF16_NATIVE_RUNG_B3:
            x = native_transition_norm_fc1_bf16(self.norm, self.fc1, x)
            _record_probe(activation_probe, block_idx, "transition_fc1_relu", x)
            x = self._apply_linear(self.fc2, x)
            _record_probe(activation_probe, block_idx, "transition_fc2", x)
            return x
        x = native_transition_norm_fc1_bf16(self.norm, self.fc1, x)
        _record_probe(activation_probe, block_idx, "transition_fc1_relu", x)
        x = native_transition_fc2_residual_bf16(self.fc2, x, residual)
        _record_probe(activation_probe, block_idx, "transition_fc2", x)
        return x

    def forward_fp8_hybrid(
        self,
        x: Mxfp8PairTensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> Mxfp8PairTensor:
        if self.linear_precision != LINEAR_PRECISION_FP8 or not _has_mxfp8_linear_weight(self.fc1) or not _has_mxfp8_linear_weight(self.fc2):
            raise RuntimeError("fp8_hybrid requires TransitionUpdate.fc1/fc2 to be quantized; build with linear_precision=fp8.")
        residual = x
        if self.hybrid_precision.use_native_layernorm:
            x = native_mxfp8_layernorm_quantized(self.norm, x)
        else:
            x = mxfp8_layernorm_quantized(self.norm, x)
        _record_probe(activation_probe, block_idx, "transition_norm", x)

        if self.hybrid_precision.use_native_linear:
            x = native_linear_forward_quantized(
                self.fc1,
                x,
                stats=stats,
                apply_relu=True,
                direct_fp8_output=True,
                fuse_bias_epilogue=True,
            )
            _record_probe(
                activation_probe,
                block_idx,
                "transition_fc1_relu",
                x,
                mode_op_name="transition_native_fc1_relu_out",
            )
            x = native_linear_forward_quantized(
                self.fc2,
                x,
                stats=stats,
                direct_fp8_output=True,
                fuse_bias_epilogue=True,
                residual=residual if self._hybrid_fuses_residual() else None,
            )
            _record_probe(
                activation_probe,
                block_idx,
                "transition_fc2",
                x,
                mode_op_name="transition_native_fc2_out",
            )
            return x

        x = mxfp8_linear_forward_quantized(self.fc1, x, stats=stats)
        x = mxfp8_relu_quantized(x)
        _record_probe(activation_probe, block_idx, "transition_fc1_relu", x)
        x = mxfp8_linear_forward_quantized(self.fc2, x, stats=stats)
        _record_probe(activation_probe, block_idx, "transition_fc2", x)
        return x


class TriangularUpdate(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        tri_impl: str = "bmm",
        tri_einsum: str = "bf16",
        linear_precision: str | None = None,
        hybrid_precision: HybridPrecisionConfig | dict[str, bool] | None = None,
    ):
        super().__init__()
        self.tri_impl = tri_impl
        self.tri_einsum = tri_einsum
        self.linear_precision = resolve_linear_precision(linear_precision)
        self.hybrid_precision = resolve_hybrid_precision_config(hybrid_precision)
        self.input_norm = nn.LayerNorm(dim, eps=1e-5)
        self.pi = nn.Linear(dim, dim)
        self.gi = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim // 2, eps=1e-5)
        self.po = nn.Linear(dim // 2, dim)
        self.go = nn.Linear(dim // 2, dim)

    def set_linear_precision(self, linear_precision: str | None) -> None:
        self.linear_precision = resolve_linear_precision(linear_precision)
        enabled = self.linear_precision == LINEAR_PRECISION_FP8
        for linear in (self.pi, self.gi, self.po, self.go):
            configure_fp8_linear_weight_(linear, enabled=enabled)

    def _apply_linear(self, module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self.linear_precision == LINEAR_PRECISION_FP8:
            return fp8_linear_forward(module, x)
        return module(x)

    def _hybrid_fuses_residual(self) -> bool:
        return self.hybrid_precision.use_resident_fp8_residual and self.hybrid_precision.use_native_gate

    def _run_tri_backend(self, x_bdnn: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        if self.tri_impl == "cublas_xbdnn":
            return tri_mul_xbdnn(x_bdnn, out_dtype=out_dtype)
        if self.tri_impl == "bmm":
            a1, b1, a2, b2 = torch.chunk(x_bdnn, 4, dim=1)
            x1 = tri_mul_bmm_bdnn(a1, b1, k_dim=2)
            x2 = tri_mul_bmm_bdnn(a2, b2, k_dim=1)
            return torch.cat([x1, x2], dim=-1)
        if self.tri_impl == "fp8_cublaslt":
            return tri_mul_fp8_cublaslt(x_bdnn, out_dtype=out_dtype)
        if self.tri_impl == "fp8_grouped":
            return tri_mul_fp8_grouped(x_bdnn, out_dtype=out_dtype)
        raise ValueError(f"Unsupported tri_impl: {self.tri_impl}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> torch.Tensor:
        _record_boundary(stats, "layernorm")
        x = self.input_norm(x)
        _record_probe(activation_probe, block_idx, "tri_input_norm", x)
        pi = self._apply_linear(self.pi, x)
        _record_probe(activation_probe, block_idx, "tri_pi", pi)
        gi = self._apply_linear(self.gi, x)
        _record_probe(activation_probe, block_idx, "tri_gi", gi)
        _record_boundary(stats, "sigmoid")
        x = pi * torch.sigmoid(gi)
        _record_probe(activation_probe, block_idx, "tri_gated", x)
        _record_boundary(stats, "mask_mul")
        x = x * mask.unsqueeze(-1)
        if self.tri_einsum == "off":
            x = x.float()
        x_bdnn = x.permute(0, 3, 1, 2).contiguous()
        x = self._run_tri_backend(x_bdnn, out_dtype=x.dtype)
        x = x.to(self.output_norm.weight.dtype)
        _record_probe(activation_probe, block_idx, "tri_mul_out", x)
        _record_boundary(stats, "layernorm")
        x = self.output_norm(x)
        _record_probe(activation_probe, block_idx, "tri_output_norm", x)
        po = self._apply_linear(self.po, x)
        _record_probe(activation_probe, block_idx, "tri_po", po)
        go = self._apply_linear(self.go, x)
        _record_probe(activation_probe, block_idx, "tri_go", go)
        _record_boundary(stats, "sigmoid")
        x = po * torch.sigmoid(go)
        _record_probe(activation_probe, block_idx, "tri_out", x)
        return x

    def forward_fp8_extreme(
        self,
        x: Mxfp8PairTensor,
        mask: torch.Tensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> Mxfp8PairTensor:
        if self.linear_precision != LINEAR_PRECISION_FP8:
            raise RuntimeError("fp8_extreme requires linear_precision=fp8 for TriangularUpdate.")
        if self.tri_impl != "fp8_cublaslt":
            raise RuntimeError("fp8_extreme currently requires tri_impl=fp8_cublaslt.")
        x = mxfp8_layernorm_quantized(self.input_norm, x)
        _record_probe(activation_probe, block_idx, "tri_input_norm", x)
        pi = mxfp8_linear_forward_quantized(self.pi, x, stats=stats)
        _record_probe(activation_probe, block_idx, "tri_pi", pi)
        gi = mxfp8_linear_forward_quantized(self.gi, x, stats=stats)
        _record_probe(activation_probe, block_idx, "tri_gi", gi)
        gated = mxfp8_mul_quantized(pi, mxfp8_sigmoid_quantized(gi))
        _record_probe(activation_probe, block_idx, "tri_gated", gated)
        tri_out_q = mxfp8_tri_mul_fp8_cublaslt_quantized(gated, stats=stats, out_dtype=torch.float16, mask=mask)
        _record_probe(activation_probe, block_idx, "tri_mul_out", tri_out_q)
        tri_out_q = mxfp8_layernorm_quantized(self.output_norm, tri_out_q)
        _record_probe(activation_probe, block_idx, "tri_output_norm", tri_out_q)
        po = mxfp8_linear_forward_quantized(self.po, tri_out_q, stats=stats)
        _record_probe(activation_probe, block_idx, "tri_po", po)
        go = mxfp8_linear_forward_quantized(self.go, tri_out_q, stats=stats)
        _record_probe(activation_probe, block_idx, "tri_go", go)
        x = mxfp8_mul_quantized(po, mxfp8_sigmoid_quantized(go))
        _record_probe(activation_probe, block_idx, "tri_out", x)
        return x

    def forward_fp8_native(
        self,
        x: Mxfp8PairTensor,
        mask: torch.Tensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> Mxfp8PairTensor:
        if self.linear_precision != LINEAR_PRECISION_FP8:
            raise RuntimeError("fp8_native requires linear_precision=fp8 for TriangularUpdate.")
        if self.tri_impl != "fp8_cublaslt":
            raise RuntimeError("fp8_native currently requires tri_impl=fp8_cublaslt.")
        residual = x
        tri_out_q = native_tri_mul_from_input_quantized(self.input_norm, self.pi, self.gi, x, mask=mask, stats=stats)
        _record_probe(
            activation_probe,
            block_idx,
            "tri_mul_out",
            tri_out_q,
            mode_op_name="tri_native_input_gate_mul_out",
        )
        tri_out_q = native_mxfp8_layernorm_quantized(self.output_norm, tri_out_q)
        _record_probe(activation_probe, block_idx, "tri_output_norm", tri_out_q)
        if stats is not None:
            stats.record_boundary("fp8_residual_add")
        x = native_gate_sigmoid_mul_quantized(self.po, self.go, tri_out_q, stats=stats, residual=residual)
        _record_probe(
            activation_probe,
            block_idx,
            "tri_residual_out",
            x,
            mode_op_name="tri_native_output_gate_out",
        )
        return x

    def forward_fp8_native_gold_packs(
        self,
        x: Mxfp8PairTensor,
        mask: torch.Tensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> Mxfp8PairTensor:
        if self.linear_precision != LINEAR_PRECISION_FP8:
            raise RuntimeError("fp8_native_gold_packs requires linear_precision=fp8 for TriangularUpdate.")
        if self.tri_impl != "fp8_cublaslt":
            raise RuntimeError("fp8_native_gold_packs currently requires tri_impl=fp8_cublaslt.")
        residual = x
        x = native_mxfp8_layernorm_quantized(self.input_norm, x)
        _record_probe(activation_probe, block_idx, "tri_input_norm", x)
        gated = native_gate_sigmoid_mul_quantized(
            self.pi,
            self.gi,
            x,
            stats=stats,
            use_gold_output_pack=True,
        )
        _record_probe(
            activation_probe,
            block_idx,
            "tri_gated",
            gated,
            mode_op_name="tri_native_gold_input_gate_out",
        )
        tri_out_q = native_tri_mul_from_block32_quantized(
            gated,
            mask=mask,
            stats=stats,
            use_gold_input_pack=True,
            use_gold_output_pack=True,
        )
        _record_probe(
            activation_probe,
            block_idx,
            "tri_mul_out",
            tri_out_q,
            mode_op_name="tri_native_gold_mul_out",
        )
        tri_out_q = native_mxfp8_layernorm_quantized(self.output_norm, tri_out_q)
        _record_probe(activation_probe, block_idx, "tri_output_norm", tri_out_q)
        if stats is not None:
            stats.record_boundary("fp8_residual_add")
        x = native_gate_sigmoid_mul_quantized(
            self.po,
            self.go,
            tri_out_q,
            stats=stats,
            residual=residual,
            use_gold_output_pack=True,
        )
        _record_probe(
            activation_probe,
            block_idx,
            "tri_residual_out",
            x,
            mode_op_name="tri_native_gold_output_gate_out",
        )
        return x

    def forward_bf16_native(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        bf16_native_rung: str,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> torch.Tensor:
        x = self.input_norm(x)
        _record_probe(activation_probe, block_idx, "tri_input_norm", x)
        pi = self._apply_linear(self.pi, x)
        _record_probe(activation_probe, block_idx, "tri_pi", pi)
        gi = self._apply_linear(self.gi, x)
        _record_probe(activation_probe, block_idx, "tri_gi", gi)
        x = pi * torch.sigmoid(gi)
        _record_probe(activation_probe, block_idx, "tri_gated", x)
        x = x * mask.unsqueeze(-1)
        x_bdnn = x.permute(0, 3, 1, 2).contiguous()
        x = self._run_tri_backend(x_bdnn, out_dtype=x.dtype)
        x = x.to(self.output_norm.weight.dtype)
        _record_probe(activation_probe, block_idx, "tri_mul_out", x)
        x = self.output_norm(x)
        _record_probe(activation_probe, block_idx, "tri_output_norm", x)
        po = self._apply_linear(self.po, x)
        _record_probe(activation_probe, block_idx, "tri_po", po)
        go = self._apply_linear(self.go, x)
        _record_probe(activation_probe, block_idx, "tri_go", go)
        x = po * torch.sigmoid(go)
        _record_probe(activation_probe, block_idx, "tri_out", x)
        return x

    def forward_fp8_hybrid(
        self,
        x: Mxfp8PairTensor,
        mask: torch.Tensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> Mxfp8PairTensor:
        if self.linear_precision != LINEAR_PRECISION_FP8:
            raise RuntimeError("fp8_hybrid requires linear_precision=fp8 for TriangularUpdate.")
        if self.tri_impl != "fp8_cublaslt":
            raise RuntimeError("fp8_hybrid currently requires tri_impl=fp8_cublaslt.")

        residual = x
        if self.hybrid_precision.use_native_layernorm:
            x = native_mxfp8_layernorm_quantized(self.input_norm, x)
        else:
            x = mxfp8_layernorm_quantized(self.input_norm, x)
        _record_probe(activation_probe, block_idx, "tri_input_norm", x)

        if self.hybrid_precision.use_native_gate:
            gated = native_gate_sigmoid_mul_quantized(self.pi, self.gi, x, stats=stats)
            _record_probe(
                activation_probe,
                block_idx,
                "tri_gated",
                gated,
                mode_op_name="tri_native_input_gate_out",
            )
        else:
            pi = mxfp8_linear_forward_quantized(self.pi, x, stats=stats)
            _record_probe(activation_probe, block_idx, "tri_pi", pi)
            gi = mxfp8_linear_forward_quantized(self.gi, x, stats=stats)
            _record_probe(activation_probe, block_idx, "tri_gi", gi)
            gated = mxfp8_mul_quantized(pi, mxfp8_sigmoid_quantized(gi))
            _record_probe(activation_probe, block_idx, "tri_gated", gated)

        if self.hybrid_precision.use_native_tri:
            tri_out_q = native_tri_mul_from_block32_quantized(gated, mask=mask, stats=stats)
            _record_probe(
                activation_probe,
                block_idx,
                "tri_mul_out",
                tri_out_q,
                mode_op_name="tri_native_mul_out",
            )
        else:
            tri_out_q = mxfp8_tri_mul_fp8_cublaslt_quantized(gated, stats=stats, out_dtype=torch.float16, mask=mask)
            _record_probe(activation_probe, block_idx, "tri_mul_out", tri_out_q)

        if self.hybrid_precision.use_native_layernorm:
            tri_out_q = native_mxfp8_layernorm_quantized(self.output_norm, tri_out_q)
        else:
            tri_out_q = mxfp8_layernorm_quantized(self.output_norm, tri_out_q)
        _record_probe(activation_probe, block_idx, "tri_output_norm", tri_out_q)

        if self.hybrid_precision.use_native_gate:
            tri_out = native_gate_sigmoid_mul_quantized(
                self.po,
                self.go,
                tri_out_q,
                stats=stats,
                residual=residual if self._hybrid_fuses_residual() else None,
            )
            _record_probe(
                activation_probe,
                block_idx,
                "tri_residual_out" if self._hybrid_fuses_residual() else "tri_out",
                tri_out,
                mode_op_name="tri_native_output_gate_out",
            )
            return tri_out

        po = mxfp8_linear_forward_quantized(self.po, tri_out_q, stats=stats)
        _record_probe(activation_probe, block_idx, "tri_po", po)
        go = mxfp8_linear_forward_quantized(self.go, tri_out_q, stats=stats)
        _record_probe(activation_probe, block_idx, "tri_go", go)
        tri_out = mxfp8_mul_quantized(po, mxfp8_sigmoid_quantized(go))
        _record_probe(activation_probe, block_idx, "tri_out", tri_out)
        return tri_out


class Block(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        tri_impl: str = "bmm",
        tri_einsum: str = "bf16",
        linear_precision: str | None = None,
        hybrid_precision: HybridPrecisionConfig | dict[str, bool] | None = None,
        bf16_native_rung: str | None = None,
    ):
        super().__init__()
        self.hybrid_precision = resolve_hybrid_precision_config(hybrid_precision)
        self.bf16_native_rung = resolve_bf16_native_rung(bf16_native_rung) or BF16_NATIVE_RUNG_B4
        self.triangular = TriangularUpdate(
            dim=dim,
            tri_impl=tri_impl,
            tri_einsum=tri_einsum,
            linear_precision=linear_precision,
            hybrid_precision=self.hybrid_precision,
        )
        self.transition = TransitionUpdate(
            dim=dim,
            hidden=dim * 4,
            linear_precision=linear_precision,
            hybrid_precision=self.hybrid_precision,
        )

    def configure_bf16_native_tail(self, bf16_native_rung: str = BF16_NATIVE_RUNG_B3) -> None:
        self.bf16_native_rung = resolve_bf16_native_rung(bf16_native_rung) or BF16_NATIVE_RUNG_B3
        self.triangular.tri_impl = "cublas_xbdnn"
        self.triangular.set_linear_precision(LINEAR_PRECISION_BF16_NATIVE)
        self.transition.set_linear_precision(LINEAR_PRECISION_BF16_NATIVE, include_transition=False)

    def _hybrid_uses_tri_residual_fusion(self) -> bool:
        return self.hybrid_precision.use_resident_fp8_residual and self.hybrid_precision.use_native_gate

    def _hybrid_uses_transition_residual_fusion(self) -> bool:
        return self.hybrid_precision.use_resident_fp8_residual and self.hybrid_precision.use_native_linear

    def forward(
        self,
        x: torch.Tensor | QuantizedPairTensor | Mxfp8PairTensor,
        mask: torch.Tensor,
        stats: FP8ActivationStats | None = None,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> torch.Tensor:
        _record_probe(activation_probe, block_idx, "block_input", x)
        if isinstance(x, Mxfp8PairTensor) and stats is not None and stats.pair_precision_mode == PAIR_PRECISION_FP8_EXTREME:
            tri_out = self.triangular.forward_fp8_extreme(
                x,
                mask,
                stats=stats,
                block_idx=block_idx,
                activation_probe=activation_probe,
            )
            _record_boundary(stats, "fp8_residual_add")
            x = mxfp8_add_quantized(x, tri_out)
            _record_probe(activation_probe, block_idx, "tri_residual_out", x)
            trans_out = self.transition.forward_fp8_extreme(
                x,
                stats=stats,
                block_idx=block_idx,
                activation_probe=activation_probe,
            )
            _record_boundary(stats, "fp8_residual_add")
            x = mxfp8_add_quantized(x, trans_out)
            _record_probe(activation_probe, block_idx, "block_output", x)
            return x
        if isinstance(x, Mxfp8PairTensor) and stats is not None and stats.pair_precision_mode in (
            PAIR_PRECISION_FP8_NATIVE,
            PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL,
        ):
            x = self.triangular.forward_fp8_native(
                x,
                mask,
                stats=stats,
                block_idx=block_idx,
                activation_probe=activation_probe,
            )
            x = self.transition.forward_fp8_native(
                x,
                stats=stats,
                block_idx=block_idx,
                activation_probe=activation_probe,
            )
            _record_probe(
                activation_probe,
                block_idx,
                "block_output",
                x,
                mode_op_name="transition_native_fc2_out",
            )
            return x
        if isinstance(x, Mxfp8PairTensor) and stats is not None and stats.pair_precision_mode == PAIR_PRECISION_FP8_NATIVE_GOLD_PACKS:
            x = self.triangular.forward_fp8_native_gold_packs(
                x,
                mask,
                stats=stats,
                block_idx=block_idx,
                activation_probe=activation_probe,
            )
            x = self.transition.forward_fp8_native_gold_packs(
                x,
                stats=stats,
                block_idx=block_idx,
                activation_probe=activation_probe,
            )
            _record_probe(
                activation_probe,
                block_idx,
                "block_output",
                x,
                mode_op_name="transition_native_gold_fc2_out",
            )
            return x
        if isinstance(x, Mxfp8PairTensor) and stats is not None and stats.pair_precision_mode == PAIR_PRECISION_FP8_HYBRID:
            tri_out = self.triangular.forward_fp8_hybrid(
                x,
                mask,
                stats=stats,
                block_idx=block_idx,
                activation_probe=activation_probe,
            )
            if self._hybrid_uses_tri_residual_fusion():
                x = tri_out
            else:
                _record_boundary(stats, "fp8_residual_add")
                x = mxfp8_add_quantized(x, tri_out)
                _record_probe(activation_probe, block_idx, "tri_residual_out", x)
            trans_out = self.transition.forward_fp8_hybrid(
                x,
                stats=stats,
                block_idx=block_idx,
                activation_probe=activation_probe,
            )
            if self._hybrid_uses_transition_residual_fusion():
                x = trans_out
            else:
                _record_boundary(stats, "fp8_residual_add")
                x = mxfp8_add_quantized(x, trans_out)
            _record_probe(
                activation_probe,
                block_idx,
                "block_output",
                x,
                mode_op_name="transition_native_fc2_out" if self._hybrid_uses_transition_residual_fusion() else "block_output",
            )
            return x
        x = _materialize_pair_tensor(x, stats, "layernorm")
        tri_out = self.triangular(
            x,
            mask,
            stats=stats,
            block_idx=block_idx,
            activation_probe=activation_probe,
        )
        _record_boundary(stats, "residual_add")
        x = x + tri_out
        _record_probe(activation_probe, block_idx, "tri_residual_out", x)
        trans_out = self.transition(
            x,
            stats=stats,
            block_idx=block_idx,
            activation_probe=activation_probe,
        )
        _record_boundary(stats, "residual_add")
        x = x + trans_out
        _record_probe(activation_probe, block_idx, "block_output", x)
        return x

    def forward_bf16_native(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        block_idx: int | None = None,
        activation_probe: ActivationProbe | None = None,
    ) -> torch.Tensor:
        _record_probe(activation_probe, block_idx, "block_input", x)
        tri_out = self.triangular.forward_bf16_native(
            x,
            mask,
            self.bf16_native_rung,
            block_idx=block_idx,
            activation_probe=activation_probe,
        )
        x = x + tri_out
        _record_probe(activation_probe, block_idx, "tri_residual_out", x)
        trans_out = self.transition.forward_bf16_native(
            x,
            self.bf16_native_rung,
            block_idx=block_idx,
            activation_probe=activation_probe,
        )
        if self.bf16_native_rung in (BF16_NATIVE_RUNG_B4, BF16_NATIVE_RUNG_B5):
            x = trans_out
        else:
            x = x + trans_out
        _record_probe(activation_probe, block_idx, "block_output", x)
        return x


class MiniFormer(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        blocks: int = 48,
        tri_impl: str = "bmm",
        tri_einsum: str = "bf16",
        pair_precision: str | None = None,
        linear_precision: str | None = None,
        fp8_activations: bool | None = None,
        hybrid_precision: HybridPrecisionConfig | dict[str, bool] | None = None,
        bf16_native_rung: str | None = None,
        mixed_tail: MixedTailConfig | dict[str, object] | None = None,
    ):
        super().__init__()
        self.pair_precision = resolve_pair_precision(pair_precision=pair_precision, fp8_activations=fp8_activations)
        self.linear_precision = resolve_linear_precision(linear_precision)
        self.hybrid_precision = resolve_hybrid_precision_config(hybrid_precision)
        self.bf16_native_rung = resolve_bf16_native_rung(bf16_native_rung)
        self.mixed_tail = resolve_mixed_tail_config(mixed_tail)
        if self.pair_precision == PAIR_PRECISION_BF16_NATIVE and self.bf16_native_rung is None:
            self.bf16_native_rung = BF16_NATIVE_RUNG_B4
        self.fp8_activations = self.pair_precision not in (PAIR_PRECISION_BF16, PAIR_PRECISION_BF16_NATIVE)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    tri_impl=tri_impl,
                    tri_einsum=tri_einsum,
                    linear_precision=self.linear_precision,
                    hybrid_precision=self.hybrid_precision,
                    bf16_native_rung=self.bf16_native_rung,
                )
                for _ in range(blocks)
            ]
        )
        if self.mixed_tail.tail_bf16_native_blocks < 0 or self.mixed_tail.tail_bf16_native_blocks > len(self.blocks):
            raise ValueError(
                f"mixed_tail.tail_bf16_native_blocks must be between 0 and {len(self.blocks)}, "
                f"got {self.mixed_tail.tail_bf16_native_blocks}"
            )
        self._mixed_tail_start_idx = len(self.blocks) - self.mixed_tail.tail_bf16_native_blocks
        self.last_pair_precision_stats: FP8ActivationStats | None = None
        self.last_fp8_stats: FP8ActivationStats | None = None
        self.last_activation_probe: ActivationProbe | None = None

    def configure_mixed_tail_runtime(
        self,
        mixed_tail: MixedTailConfig | dict[str, object] | None = None,
    ) -> MixedTailConfig:
        self.mixed_tail = resolve_mixed_tail_config(mixed_tail)
        if self.mixed_tail.tail_bf16_native_blocks < 0 or self.mixed_tail.tail_bf16_native_blocks > len(self.blocks):
            raise ValueError(
                f"mixed_tail.tail_bf16_native_blocks must be between 0 and {len(self.blocks)}, "
                f"got {self.mixed_tail.tail_bf16_native_blocks}"
            )
        self._mixed_tail_start_idx = len(self.blocks) - self.mixed_tail.tail_bf16_native_blocks
        if self.pair_precision != PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL or self.mixed_tail.tail_bf16_native_blocks == 0:
            return self.mixed_tail
        for idx in range(self._mixed_tail_start_idx, len(self.blocks)):
            self.blocks[idx].configure_bf16_native_tail(self.mixed_tail.bf16_native_rung)
        return self.mixed_tail

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        activation_probe: ActivationProbe | None = None,
        dump_activation_stats: bool = False,
    ) -> torch.Tensor:
        stats = None if self.pair_precision in (PAIR_PRECISION_BF16, PAIR_PRECISION_BF16_NATIVE) else FP8ActivationStats(pair_precision_mode=self.pair_precision)
        if dump_activation_stats and activation_probe is None:
            activation_probe = ActivationProbe(pair_precision_mode=self.pair_precision)
        if self.pair_precision in FP8_BLOCK32_PAIR_PRECISIONS:
            x = Mxfp8PairTensor.from_tensor(x, scale_dtype=torch.float32)
            if stats is not None:
                stats.record_quantize(x)
        for idx, block in enumerate(self.blocks):
            if self.pair_precision == PAIR_PRECISION_BF16_NATIVE:
                x = block.forward_bf16_native(x, mask, block_idx=idx, activation_probe=activation_probe)
                continue
            if self.pair_precision == PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL and idx >= self._mixed_tail_start_idx:
                if isinstance(x, (QuantizedPairTensor, Mxfp8PairTensor)):
                    x = _materialize_pair_tensor(x, stats, "mixed_tail_boundary")
                    _record_probe(activation_probe, idx, "mixed_tail_boundary", x)
                x = block.forward_bf16_native(x, mask, block_idx=idx, activation_probe=activation_probe)
                continue
            x = block(x, mask, stats=stats, block_idx=idx, activation_probe=activation_probe)
            if self.pair_precision in FP8_BLOCK32_PAIR_PRECISIONS:
                if stats is not None and isinstance(x, Mxfp8PairTensor):
                    stats.record_quantize(x)
                continue
            is_last_block = idx == len(self.blocks) - 1
            if stats is None or is_last_block:
                continue
            quantized = QuantizedPairTensor.from_tensor(x)
            stats.record_quantize(quantized)
            if self.pair_precision == PAIR_PRECISION_FP8_STORAGE:
                stats.record_dequantize("storage_roundtrip")
                x = quantized.dequantize(dtype=x.dtype)
            elif self.pair_precision == PAIR_PRECISION_FP8_RESIDENT:
                x = quantized
            else:
                raise ValueError(f"Unsupported pair precision mode: {self.pair_precision}")
        if isinstance(x, (QuantizedPairTensor, Mxfp8PairTensor)):
            x = _materialize_pair_tensor(x, stats, "miniformer_exit")
        self.last_pair_precision_stats = stats
        self.last_fp8_stats = stats
        self.last_activation_probe = activation_probe
        return x


class FoldingTrunk(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        bins: int,
        disto_bins: int = 64,
        num_layers: int = 48,
        tri_impl: str = "bmm",
        tri_einsum: str = "bf16",
        pair_precision: str | None = None,
        linear_precision: str | None = None,
        fp8_activations: bool | None = None,
        hybrid_precision: HybridPrecisionConfig | dict[str, bool] | None = None,
        bf16_native_rung: str | None = None,
        mixed_tail: MixedTailConfig | dict[str, object] | None = None,
    ):
        super().__init__()
        self.disto_bins = disto_bins
        self.positional_embedding = RelativePosition(bins, c_z)
        self.seq_to_pair = SequenceToPair(c_s, c_z // 2, c_z)
        self.projection = nn.Linear(c_z * 3, c_z)
        self.recycle = nn.Linear(disto_bins, c_z)
        self.miniformer = MiniFormer(
            dim=c_z,
            blocks=num_layers,
            tri_impl=tri_impl,
            tri_einsum=tri_einsum,
            pair_precision=pair_precision,
            linear_precision=linear_precision,
            fp8_activations=fp8_activations,
            hybrid_precision=hybrid_precision,
            bf16_native_rung=bf16_native_rung,
            mixed_tail=mixed_tail,
        )
        self.fc_out_1 = nn.Linear(c_z, c_z)
        self.fc_out_2 = nn.Linear(c_z, disto_bins)
        nn.init.zeros_(self.seq_to_pair.o_proj.weight)
        nn.init.zeros_(self.seq_to_pair.o_proj.bias)

    def forward(
        self,
        s_s: torch.Tensor,
        s_z: torch.Tensor,
        mask: torch.Tensor,
        num_recycling: int = 0,
        *,
        activation_probe: ActivationProbe | None = None,
        dump_activation_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_mask = mask[:, None, :] * mask[:, :, None]
        residx = torch.arange(s_s.shape[1], device=s_s.device).unsqueeze(0).expand(s_s.shape[0], -1)
        s_z = torch.cat([s_z, self.seq_to_pair(s_s), self.positional_embedding(residx, mask=pair_mask)], dim=-1)
        s_z = self.projection(s_z)
        pair_mask = pair_mask.to(s_z)
        shape = tuple(s_z.shape[:3]) + (self.disto_bins,)
        dists = torch.zeros(shape, device=s_z.device, dtype=s_z.dtype)
        for _ in range(num_recycling + 1):
            s_z_c = s_z + self.recycle(dists)
            s_z_c = self.miniformer(
                s_z_c,
                pair_mask,
                activation_probe=activation_probe,
                dump_activation_stats=dump_activation_stats,
            )
            fc_out = self.fc_out_1(s_z_c + s_z_c.transpose(1, 2))
            fc_out = F.relu(fc_out)
            preds = self.fc_out_2(fc_out)
            dists = preds.detach().argmax(dim=-1)
            dists = F.one_hot(dists, self.disto_bins).to(s_z)
        return preds, s_z_c


class PlainESM2MiniFold(nn.Module):
    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t36_3B_UR50D",
        c_s: int = 1024,
        c_z: int = 128,
        num_blocks: int = 48,
        no_bins: int = 64,
        tri_impl: str = "bmm",
        tri_einsum: str = "bf16",
        pair_precision: str | None = None,
        linear_precision: str | None = None,
        fp8_activations: bool | None = None,
        hybrid_precision: HybridPrecisionConfig | dict[str, bool] | None = None,
        bf16_native_rung: str | None = None,
        mixed_tail: MixedTailConfig | dict[str, object] | None = None,
    ):
        super().__init__()
        from esm_backbone import ESM2Backbone

        self.backbone = ESM2Backbone(esm_model_name)
        embed_dim = self.backbone.embed_dim
        attn_dim = self.backbone.attn_dim
        self.fc_s_1 = nn.Linear(embed_dim, c_s)
        self.fc_s_2 = nn.Linear(c_s, c_s)
        self.fc_z_1 = nn.Linear(attn_dim, c_z)
        self.fc_z_2 = nn.Linear(c_z, c_z)
        self.fold = FoldingTrunk(
            c_s=c_s,
            c_z=c_z,
            bins=32,
            disto_bins=no_bins,
            num_layers=num_blocks,
            tri_impl=tri_impl,
            tri_einsum=tri_einsum,
            pair_precision=pair_precision,
            linear_precision=linear_precision,
            fp8_activations=fp8_activations,
            hybrid_precision=hybrid_precision,
            bf16_native_rung=bf16_native_rung,
            mixed_tail=mixed_tail,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        num_recycling: int = 0,
        *,
        activation_probe: ActivationProbe | None = None,
        dump_activation_stats: bool = False,
    ) -> dict[str, torch.Tensor]:
        esm_out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        s_s = self.fc_s_2(F.relu(self.fc_s_1(esm_out["representations"])))
        s_z = self.fc_z_2(F.relu(self.fc_z_1(esm_out["attentions"])))
        preds, pair = self.fold(
            s_s,
            s_z,
            mask=batch["mask"],
            num_recycling=num_recycling,
            activation_probe=activation_probe,
            dump_activation_stats=dump_activation_stats,
        )
        return {"preds": preds, "pair": pair}


def configure_linear_precision(
    module: nn.Module,
    linear_precision: str | None = None,
    *,
    include_transition: bool = False,
) -> str:
    resolved = resolve_linear_precision(linear_precision)
    for submodule in module.modules():
        if isinstance(submodule, TriangularUpdate):
            submodule.set_linear_precision(resolved)
        elif isinstance(submodule, TransitionUpdate):
            submodule.set_linear_precision(resolved, include_transition=include_transition)
    return resolved


def configure_mixed_tail_runtime(
    module: nn.Module,
    pair_precision: str,
    mixed_tail: MixedTailConfig | dict[str, object] | None = None,
) -> MixedTailConfig:
    resolved = resolve_mixed_tail_config(mixed_tail)
    if pair_precision != PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL:
        return resolved
    if isinstance(module, MiniFormer):
        return module.configure_mixed_tail_runtime(resolved)
    for submodule in module.modules():
        if isinstance(submodule, MiniFormer):
            return submodule.configure_mixed_tail_runtime(resolved)
    raise TypeError("configure_mixed_tail_runtime expected a module containing a MiniFormer")


def validate_bf16_native_configuration(
    pair_precision: str,
    linear_precision: str,
    tri_impl: str,
    bf16_native_rung: str | None = None,
) -> str | None:
    rung = resolve_bf16_native_rung(bf16_native_rung)
    if pair_precision != PAIR_PRECISION_BF16_NATIVE:
        return rung
    if rung is None:
        rung = BF16_NATIVE_RUNG_B4
    if tri_impl != "cublas_xbdnn":
        raise ValueError("pair_precision=bf16_native currently requires tri_impl=cublas_xbdnn.")
    if rung == BF16_NATIVE_RUNG_B2:
        if linear_precision != LINEAR_PRECISION_BF16:
            raise ValueError("bf16_native rung B2 currently requires linear_precision=bf16.")
    elif linear_precision != LINEAR_PRECISION_BF16_NATIVE:
        raise ValueError(f"bf16_native rung {rung} currently requires linear_precision=bf16_native.")
    return rung


def validate_fp8_extreme_configuration(pair_precision: str, linear_precision: str, tri_impl: str) -> None:
    if pair_precision not in FP8_BLOCK32_PAIR_PRECISIONS:
        return
    if linear_precision != LINEAR_PRECISION_FP8:
        raise ValueError(f"pair_precision={pair_precision} requires linear_precision=fp8.")
    if tri_impl != "fp8_cublaslt":
        raise ValueError(f"pair_precision={pair_precision} currently requires tri_impl=fp8_cublaslt.")


def validate_fp8_native_mixed_tail_configuration(
    pair_precision: str,
    linear_precision: str,
    tri_impl: str,
    num_blocks: int,
    mixed_tail: MixedTailConfig | dict[str, object] | None = None,
) -> MixedTailConfig:
    resolved = resolve_mixed_tail_config(mixed_tail)
    if pair_precision != PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL:
        return resolved
    validate_fp8_extreme_configuration(pair_precision, linear_precision, tri_impl)
    if resolved.tail_bf16_native_blocks < 0 or resolved.tail_bf16_native_blocks > num_blocks:
        raise ValueError(
            f"pair_precision={pair_precision} requires mixed_tail.tail_bf16_native_blocks between 0 and {num_blocks}, "
            f"got {resolved.tail_bf16_native_blocks}."
        )
    if resolved.bf16_native_rung != BF16_NATIVE_RUNG_B3:
        raise ValueError(
            f"pair_precision={pair_precision} currently requires mixed_tail.bf16_native_rung={BF16_NATIVE_RUNG_B3}."
        )
    return resolved
