from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions import split_quantize
from transformer_engine.pytorch.tensor import MXFP8Quantizer, MXFP8TensorStorage

from . import _C


_FP8_E4M3_MAX = 448.0
_FP8_E8M0_MIN = 2.0**-127
_MXFP8_INPUT_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
_MXFP8_SCALE_DTYPE = torch.float8_e8m0fnu
_NVFP4_SCALE_DTYPE = torch.float8_e4m3fn
_SUPPORTED_OUT_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


@dataclass(frozen=True)
class PackedNVFP4Tensor:
    data: torch.Tensor
    logical_shape: tuple[int, int, int]
    scale_inv: torch.Tensor | None = None
    amax: torch.Tensor | None = None
    rhs_transposed: bool = False

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def requires_grad(self) -> bool:
        return bool(self.data.requires_grad)


def _check_cuda_tensor(t: torch.Tensor, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def _normalize_shape(shape: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"{name} must describe a 3D tensor")
    return tuple(int(v) for v in shape)


def _logical_shape(arg, name: str) -> tuple[int, int, int]:
    if isinstance(arg, PackedNVFP4Tensor):
        return _normalize_shape(arg.logical_shape, f"{name}.logical_shape")
    if not isinstance(arg, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or PackedNVFP4Tensor")
    if arg.dim() != 3:
        raise ValueError(f"{name} must be 3D")
    return tuple(int(v) for v in arg.shape)


def _storage_tensor(arg, name: str) -> torch.Tensor:
    if isinstance(arg, PackedNVFP4Tensor):
        return arg.data
    if not isinstance(arg, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or PackedNVFP4Tensor")
    return arg


def _require_no_grad(*tensors: torch.Tensor) -> None:
    if any(bool(t.requires_grad) for t in tensors):
        raise ValueError("bmm_block_scaled is forward-only and rejects tensors requiring grad")


def quantize_mxfp8(
    x: torch.Tensor,
    *,
    role: str = "lhs",
    sf_vec_size: int = 32,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D CUDA tensor into explicit MXFP8 data and compact block scales."""
    _check_cuda_tensor(x, "x")
    if x.dim() != 3:
        raise ValueError("x must be 3D")
    if fp8_dtype not in _MXFP8_INPUT_DTYPES:
        raise TypeError(f"unsupported fp8_dtype {fp8_dtype}")
    if sf_vec_size <= 0:
        raise ValueError("sf_vec_size must be positive")
    role = role.lower()
    if role not in {"lhs", "rhs"}:
        raise ValueError("role must be 'lhs' or 'rhs'")

    x_f32 = x.float()
    if role == "lhs":
        if x.shape[-1] % sf_vec_size != 0:
            raise ValueError("lhs last dimension must be divisible by sf_vec_size")
        blocks = x_f32.reshape(*x.shape[:-1], x.shape[-1] // sf_vec_size, sf_vec_size)
        block_amax = blocks.abs().amax(dim=-1).clamp(min=_FP8_E8M0_MIN)
        scale_inv_f32 = torch.pow(2.0, torch.ceil(torch.log2(block_amax / _FP8_E4M3_MAX))).clamp(min=_FP8_E8M0_MIN)
        x_fp8 = (blocks / scale_inv_f32.unsqueeze(-1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(fp8_dtype)
    else:
        if x.shape[1] % sf_vec_size != 0:
            raise ValueError("rhs K dimension must be divisible by sf_vec_size")
        blocks = x_f32.reshape(x.shape[0], x.shape[1] // sf_vec_size, sf_vec_size, x.shape[2])
        block_amax = blocks.abs().amax(dim=2).clamp(min=_FP8_E8M0_MIN)
        scale_inv_f32 = torch.pow(2.0, torch.ceil(torch.log2(block_amax / _FP8_E4M3_MAX))).clamp(min=_FP8_E8M0_MIN)
        x_fp8 = (blocks / scale_inv_f32.unsqueeze(2)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(fp8_dtype)

    return x_fp8.reshape_as(x).contiguous(), scale_inv_f32.to(_MXFP8_SCALE_DTYPE).contiguous()


def _swizzle_mxfp8_scale_rowwise(scale: torch.Tensor) -> torch.Tensor:
    """Convert padded row-major MXFP8 scale tensor into the tiled layout cuBLASLt expects."""
    _check_cuda_tensor(scale, "scale")
    if scale.dim() != 3:
        raise ValueError("scale must be 3D")
    batch, rows, cols = scale.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    if padded_rows != rows or padded_cols != cols:
        padded = torch.full(
            (batch, padded_rows, padded_cols),
            _FP8_E8M0_MIN,
            device=scale.device,
            dtype=scale.dtype,
        )
        padded[:, :rows, :cols] = scale
        scale = padded
        rows = padded_rows
        cols = padded_cols
    return (
        scale.view(batch, rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view_as(scale)
    )


def _quantize_mxfp8_rowwise_swizzled(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data, scale = quantize_mxfp8(x, role="lhs")
    return data, _swizzle_mxfp8_scale_rowwise(scale)


def _make_te_mxfp8_quantizer() -> MXFP8Quantizer:
    quantizer = MXFP8Quantizer(TE_DType[torch.float8_e4m3fn], rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = False
    return quantizer


def _split_quantize_mxfp8(chunks: list[torch.Tensor]) -> list[MXFP8TensorStorage]:
    if not chunks:
        return []
    split_sections = [chunk.shape[0] for chunk in chunks]
    quantizers = [_make_te_mxfp8_quantizer() for _ in chunks]
    return list(split_quantize(torch.cat(chunks, dim=0), split_sections, quantizers))


def _as_fp8_data(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype in _MXFP8_INPUT_DTYPES else t.view(torch.float8_e4m3fn)


def _as_scale_dtype(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == _MXFP8_SCALE_DTYPE else t.view(_MXFP8_SCALE_DTYPE)


def _rowwise_mxfp8(
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    batch: int,
    m: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = _as_fp8_data(rowwise_data)
    scale = _as_scale_dtype(rowwise_scale_inv).reshape(batch, m, -1)[..., : k // 32]
    return data, _swizzle_mxfp8_scale_rowwise(scale)


def _transpose_rowwise_mxfp8(
    columnwise_data: torch.Tensor,
    columnwise_scale_inv: torch.Tensor,
    batch: int,
    m: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = _as_fp8_data(columnwise_data).transpose(1, 2).contiguous()
    scale = _as_scale_dtype(columnwise_scale_inv).reshape(batch, -1, m)[:, : k // 32, :].transpose(1, 2).contiguous()
    return data, _swizzle_mxfp8_scale_rowwise(scale)


def _rhs_mxfp8(
    columnwise_data: torch.Tensor,
    columnwise_scale_inv: torch.Tensor,
    batch: int,
    k: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = _as_fp8_data(columnwise_data)
    scale_rhs = _as_scale_dtype(columnwise_scale_inv).reshape(batch, -1, n)[:, : k // 32, :]
    return data, _swizzle_mxfp8_scale_rowwise(scale_rhs.transpose(1, 2).contiguous())


def _validate_mxfp8_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    sf_vec_size: int,
) -> None:
    if a.dtype not in _MXFP8_INPUT_DTYPES:
        raise TypeError("mxfp8 input a must use torch.float8_e4m3fn or torch.float8_e5m2")
    if b.dtype not in _MXFP8_INPUT_DTYPES:
        raise TypeError("mxfp8 input b must use torch.float8_e4m3fn or torch.float8_e5m2")
    if a_scale.dtype != _MXFP8_SCALE_DTYPE:
        raise TypeError("mxfp8 a_scale must use torch.float8_e8m0fnu")
    if b_scale.dtype != _MXFP8_SCALE_DTYPE:
        raise TypeError("mxfp8 b_scale must use torch.float8_e8m0fnu")
    if sf_vec_size <= 0:
        raise ValueError("sf_vec_size must be positive")
    if a.shape[2] % sf_vec_size != 0 or b.shape[1] % sf_vec_size != 0:
        raise ValueError("mxfp8 K must be divisible by sf_vec_size")


def _validate_nvfp4_inputs(
    a: PackedNVFP4Tensor,
    b: PackedNVFP4Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    sf_vec_size: int,
) -> None:
    if not isinstance(a, PackedNVFP4Tensor) or not isinstance(b, PackedNVFP4Tensor):
        raise TypeError("nvfp4 inputs must be PackedNVFP4Tensor values")
    if a.data.dtype != torch.uint8 or b.data.dtype != torch.uint8:
        raise TypeError("nvfp4 packed storage must use torch.uint8")
    if a_scale.dtype not in {torch.uint8, _NVFP4_SCALE_DTYPE}:
        raise TypeError("nvfp4 a_scale must use torch.float8_e4m3fn or torch.uint8")
    if b_scale.dtype not in {torch.uint8, _NVFP4_SCALE_DTYPE}:
        raise TypeError("nvfp4 b_scale must use torch.float8_e4m3fn or torch.uint8")
    if sf_vec_size != 16:
        raise ValueError("nvfp4 currently requires sf_vec_size=16")

    a_b, a_m, a_k = a.logical_shape
    b_b, b_k, b_n = b.logical_shape
    if a.data.shape != (a_b, a_m, a_k // 2):
        raise ValueError("nvfp4 packed a storage must have shape (B, M, K/2)")
    expected_b_storage = (b_b, b_n, b_k // 2) if b.rhs_transposed else (b_b, b_k, b_n // 2)
    if b.data.shape != expected_b_storage:
        raise ValueError(f"nvfp4 packed b storage must have shape {expected_b_storage}")
    if a_k % 2 != 0 or b_n % 2 != 0:
        raise ValueError("nvfp4 logical K for a and logical N for b must be even")
    if a_k % 32 != 0 or b_k % 32 != 0:
        raise ValueError("nvfp4 K must be divisible by 32")


def _validate_common(
    a_shape: tuple[int, int, int],
    b_shape: tuple[int, int, int],
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    sf_vec_size: int,
) -> None:
    if a_shape[0] != b_shape[0]:
        raise ValueError("batch dimensions must match exactly")
    if a_shape[2] != b_shape[1]:
        raise ValueError("inner dimensions must match exactly")
    if a_scale.dim() != 3 or b_scale.dim() != 3:
        raise ValueError("scale tensors must be 3D")

    expected_a_scale = (a_shape[0], a_shape[1], a_shape[2] // sf_vec_size)
    expected_b_scale = (b_shape[0], b_shape[1] // sf_vec_size, b_shape[2])
    padded_a_scale = (
        a_shape[0],
        ((a_shape[1] + 127) // 128) * 128,
        (((a_shape[2] // sf_vec_size) + 3) // 4) * 4,
    )
    padded_b_scale = (
        b_shape[0],
        ((b_shape[2] + 127) // 128) * 128,
        (((b_shape[1] // sf_vec_size) + 3) // 4) * 4,
    )
    if tuple(a_scale.shape) not in {expected_a_scale, padded_a_scale}:
        raise ValueError(
            f"a_scale must have shape {expected_a_scale} or {padded_a_scale}, got {tuple(a_scale.shape)}"
        )
    if tuple(b_scale.shape) not in {expected_b_scale, padded_b_scale}:
        raise ValueError(
            f"b_scale must have shape {expected_b_scale} or {padded_b_scale}, got {tuple(b_scale.shape)}"
        )


def bmm_block_scaled(
    a,
    b,
    *,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    format: str,
    out_dtype: torch.dtype = torch.float32,
    sf_vec_size: int = 32,
) -> torch.Tensor:
    format = format.lower()
    if out_dtype not in _SUPPORTED_OUT_DTYPES:
        raise TypeError(f"unsupported out_dtype {out_dtype}")
    _check_cuda_tensor(a_scale, "a_scale")
    _check_cuda_tensor(b_scale, "b_scale")

    a_tensor = _storage_tensor(a, "a")
    b_tensor = _storage_tensor(b, "b")
    _check_cuda_tensor(a_tensor, "a")
    _check_cuda_tensor(b_tensor, "b")
    _require_no_grad(a_tensor, b_tensor, a_scale, b_scale)

    if len({a_tensor.device, b_tensor.device, a_scale.device, b_scale.device}) != 1:
        raise ValueError("all tensors must be on the same CUDA device")

    a_shape = _logical_shape(a, "a")
    b_shape = _logical_shape(b, "b")

    if format == "mxfp8":
        if isinstance(a, PackedNVFP4Tensor) or isinstance(b, PackedNVFP4Tensor):
            raise TypeError("mxfp8 expects plain torch.Tensor inputs")
        _validate_mxfp8_inputs(a_tensor, b_tensor, a_scale, b_scale, sf_vec_size)
        _validate_common(a_shape, b_shape, a_scale, b_scale, sf_vec_size)
        a_override: list[int] = []
        b_override: list[int] = []
    elif format == "nvfp4":
        _validate_nvfp4_inputs(a, b, a_scale, b_scale, sf_vec_size)
        _validate_common(a_shape, b_shape, a_scale, b_scale, sf_vec_size)
        a_override = list(a_shape)
        b_override = list(b_shape)
    else:
        raise ValueError("format must be 'mxfp8' or 'nvfp4'")

    return _C.bmm_block_scaled(
        a_tensor,
        b_tensor,
        a_scale.contiguous(),
        b_scale.contiguous(),
        (
            a.amax.contiguous()
            if isinstance(a, PackedNVFP4Tensor) and a.amax is not None
            else torch.empty(0, device=a_tensor.device, dtype=torch.float32)
        ),
        (
            b.amax.contiguous()
            if isinstance(b, PackedNVFP4Tensor) and b.amax is not None
            else torch.empty(0, device=b_tensor.device, dtype=torch.float32)
        ),
        format,
        _SUPPORTED_OUT_DTYPES[out_dtype],
        int(sf_vec_size),
        a_override,
        b_override,
        bool(isinstance(a, PackedNVFP4Tensor) and a.rhs_transposed),
        bool(isinstance(b, PackedNVFP4Tensor) and b.rhs_transposed),
    )


class _MXFP8Bmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        if a.dim() != 3 or b.dim() != 3:
            raise ValueError("mxfp8_bmm expects 3D tensors")
        if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
            raise ValueError(f"incompatible bmm shapes: {tuple(a.shape)} and {tuple(b.shape)}")

        a_fp8, a_scale = quantize_mxfp8(a.detach(), role="lhs")
        b_fp8, b_scale = quantize_mxfp8(b.detach(), role="rhs")
        a_t_fp8, a_t_scale = quantize_mxfp8(a.detach().transpose(1, 2).contiguous(), role="lhs")
        b_t_fp8, b_t_scale = quantize_mxfp8(b.detach().transpose(1, 2).contiguous(), role="rhs")

        ctx.save_for_backward(a_fp8, a_scale, a_t_fp8, a_t_scale, b_fp8, b_scale, b_t_fp8, b_t_scale)
        ctx.a_dtype = a.dtype
        ctx.b_dtype = b.dtype

        return bmm_block_scaled(
            a_fp8,
            b_fp8,
            a_scale=a_scale,
            b_scale=b_scale,
            format="mxfp8",
            out_dtype=out_dtype,
            sf_vec_size=32,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a_fp8, a_scale, a_t_fp8, a_t_scale, b_fp8, b_scale, b_t_fp8, b_t_scale = ctx.saved_tensors
        g_lhs_fp8, g_lhs_scale = quantize_mxfp8(grad_output.detach(), role="lhs")
        g_rhs_fp8, g_rhs_scale = quantize_mxfp8(grad_output.detach(), role="rhs")

        grad_a = bmm_block_scaled(
            g_lhs_fp8,
            b_t_fp8,
            a_scale=g_lhs_scale,
            b_scale=b_t_scale,
            format="mxfp8",
            out_dtype=torch.float32,
            sf_vec_size=32,
        )
        grad_b = bmm_block_scaled(
            a_t_fp8,
            g_rhs_fp8,
            a_scale=a_t_scale,
            b_scale=g_rhs_scale,
            format="mxfp8",
            out_dtype=torch.float32,
            sf_vec_size=32,
        )
        return grad_a.to(ctx.a_dtype), grad_b.to(ctx.b_dtype), None


def mxfp8_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Autograd-backed MXFP8 batched GEMM with FP32 accumulation."""
    if out_dtype not in _SUPPORTED_OUT_DTYPES:
        raise TypeError(f"unsupported out_dtype {out_dtype}")
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("mxfp8_bmm expects 3D tensors")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError(f"incompatible bmm shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    if any(dim % 32 != 0 for dim in (a.shape[1], a.shape[2], b.shape[1], b.shape[2])):
        raise ValueError("mxfp8_bmm requires M, K, and N dimensions to be divisible by 32")
    return _MXFP8Bmm.apply(a, b, out_dtype)


def mxfp8_cublaslt_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Raw cuBLASLt MXFP8 strided batched GEMM benchmark path.

    This path quantizes `a` and `b` once in Python, materializes `b^T` as the
    column-major operand expected by cuBLASLt, swizzles scales into the tiled
    UE8M0 layout documented by cuBLASLt, and calls the native matmul directly.
    """
    if out_dtype not in _SUPPORTED_OUT_DTYPES:
        raise TypeError(f"unsupported out_dtype {out_dtype}")
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("mxfp8_cublaslt_bmm expects 3D tensors")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError(f"incompatible bmm shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    if any(dim % 32 != 0 for dim in (a.shape[1], a.shape[2], b.shape[1], b.shape[2])):
        raise ValueError("mxfp8_cublaslt_bmm requires M, K, and N dimensions to be divisible by 32")

    a_fp8, a_scale = quantize_mxfp8(a, role="lhs")
    b_t = b.transpose(1, 2).contiguous()
    b_t_fp8, b_t_scale = quantize_mxfp8(b_t, role="lhs")
    a_scale_swizzled = _swizzle_mxfp8_scale_rowwise(a_scale)
    b_scale_swizzled = _swizzle_mxfp8_scale_rowwise(b_t_scale)
    return _C.mxfp8_cublaslt_bmm(
        a_fp8,
        b_t_fp8,
        a_scale_swizzled,
        b_scale_swizzled,
        _SUPPORTED_OUT_DTYPES[out_dtype],
    )


def _reshape_chunk(x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    if x.dim() != 4:
        raise ValueError("expected chunk tensor with shape (B, D, N, N)")
    bsz, d_chunk, n1, n2 = x.shape
    return x.reshape(bsz * d_chunk, n1, n2), bsz, d_chunk, n1


class _MXFP8TriMulXBDNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bdnn: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        if x_bdnn.dim() != 4:
            raise ValueError("x_bdnn must have shape (B, D, N, N)")
        if x_bdnn.shape[1] % 4 != 0:
            raise ValueError("x_bdnn channel dimension must be divisible by 4")
        if x_bdnn.shape[2] != x_bdnn.shape[3]:
            raise ValueError("x_bdnn must have square sequence dimensions")
        if x_bdnn.shape[2] % 32 != 0 or (x_bdnn.shape[1] // 4) % 32 != 0:
            raise ValueError("packed fp8 tri-mul requires N and D_chunk divisible by 32")

        a1, b1, a2, b2 = torch.chunk(x_bdnn.detach(), 4, dim=1)
        a1_3d, bsz, d_chunk, n = _reshape_chunk(a1)
        b1_3d, _, _, _ = _reshape_chunk(b1)
        a2_3d, _, _, _ = _reshape_chunk(a2)
        b2_3d, _, _, _ = _reshape_chunk(b2)

        b1_t_3d = b1_3d.transpose(1, 2).contiguous()
        a2_t_3d = a2_3d.transpose(1, 2).contiguous()
        b2_t_3d = b2_3d.transpose(1, 2).contiguous()
        a1_t_3d = a1_3d.transpose(1, 2).contiguous()

        a1_fp8, a1_scale = quantize_mxfp8(a1_3d, role="lhs")
        a1_t_fp8, a1_t_scale = quantize_mxfp8(a1_t_3d, role="lhs")
        b1_fp8, b1_scale = quantize_mxfp8(b1_3d, role="rhs")
        b1_t_fp8, b1_t_scale = quantize_mxfp8(b1_t_3d, role="rhs")
        a2_fp8, a2_scale = quantize_mxfp8(a2_3d, role="lhs")
        a2_t_fp8, a2_t_scale = quantize_mxfp8(a2_t_3d, role="lhs")
        b2_fp8, b2_scale = quantize_mxfp8(b2_3d, role="rhs")
        b2_t_fp8, b2_t_scale = quantize_mxfp8(b2_t_3d, role="rhs")

        x1 = bmm_block_scaled(
            a1_fp8,
            b1_t_fp8,
            a_scale=a1_scale,
            b_scale=b1_t_scale,
            format="mxfp8",
            out_dtype=out_dtype,
            sf_vec_size=32,
        )
        x2 = bmm_block_scaled(
            a2_t_fp8,
            b2_fp8,
            a_scale=a2_t_scale,
            b_scale=b2_scale,
            format="mxfp8",
            out_dtype=out_dtype,
            sf_vec_size=32,
        )

        ctx.save_for_backward(
            a1_t_fp8,
            a1_t_scale,
            b1_fp8,
            b1_scale,
            a2_fp8,
            a2_scale,
            b2_t_fp8,
            b2_t_scale,
        )
        ctx.shape = (bsz, d_chunk, n)
        ctx.input_dtype = x_bdnn.dtype

        x1 = x1.reshape(bsz, d_chunk, n, n)
        x2 = x2.reshape(bsz, d_chunk, n, n)
        return torch.cat([x1, x2], dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a1_t_fp8, a1_t_scale, b1_fp8, b1_scale, a2_fp8, a2_scale, b2_t_fp8, b2_t_scale = ctx.saved_tensors
        bsz, d_chunk, n = ctx.shape
        g1, g2 = torch.chunk(grad_output.detach(), 2, dim=1)
        g1 = g1.reshape(bsz * d_chunk, n, n)
        g2 = g2.reshape(bsz * d_chunk, n, n)
        g1_t = g1.transpose(1, 2).contiguous()
        g2_t = g2.transpose(1, 2).contiguous()

        g1_lhs_fp8, g1_lhs_scale = quantize_mxfp8(g1, role="lhs")
        g1_rhs_fp8, g1_rhs_scale = quantize_mxfp8(g1, role="rhs")
        g2_lhs_fp8, g2_lhs_scale = quantize_mxfp8(g2, role="lhs")
        g2_rhs_fp8, g2_rhs_scale = quantize_mxfp8(g2, role="rhs")
        g1_t_lhs_fp8, g1_t_lhs_scale = quantize_mxfp8(g1_t, role="lhs")
        g2_t_lhs_fp8, g2_t_lhs_scale = quantize_mxfp8(g2_t, role="lhs")

        grad_a1 = bmm_block_scaled(
            g1_lhs_fp8,
            b1_fp8,
            a_scale=g1_lhs_scale,
            b_scale=b1_scale,
            format="mxfp8",
            out_dtype=torch.float32,
            sf_vec_size=32,
        )
        grad_b1_t = bmm_block_scaled(
            a1_t_fp8,
            g1_rhs_fp8,
            a_scale=a1_t_scale,
            b_scale=g1_rhs_scale,
            format="mxfp8",
            out_dtype=torch.float32,
            sf_vec_size=32,
        )
        grad_a2_t = bmm_block_scaled(
            g2_lhs_fp8,
            b2_t_fp8,
            a_scale=g2_lhs_scale,
            b_scale=b2_t_scale,
            format="mxfp8",
            out_dtype=torch.float32,
            sf_vec_size=32,
        )
        grad_b2 = bmm_block_scaled(
            a2_fp8,
            g2_rhs_fp8,
            a_scale=a2_scale,
            b_scale=g2_rhs_scale,
            format="mxfp8",
            out_dtype=torch.float32,
            sf_vec_size=32,
        )

        grad_b1 = grad_b1_t.transpose(1, 2).contiguous()
        grad_a2 = grad_a2_t.transpose(1, 2).contiguous()

        grad_x = torch.cat(
            [
                grad_a1.reshape(bsz, d_chunk, n, n),
                grad_b1.reshape(bsz, d_chunk, n, n),
                grad_a2.reshape(bsz, d_chunk, n, n),
                grad_b2.reshape(bsz, d_chunk, n, n),
            ],
            dim=1,
        )
        return grad_x.to(ctx.input_dtype), None


def mxfp8_tri_mul_xbdnn(
    x_bdnn: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Packed MXFP8 triangular multiplication on a `(B, 128, N, N)` tensor."""
    if out_dtype not in _SUPPORTED_OUT_DTYPES:
        raise TypeError(f"unsupported out_dtype {out_dtype}")
    return _MXFP8TriMulXBDNN.apply(x_bdnn, out_dtype)


class _MXFP8CuBLASLtTriMulXBDNN(torch.autograd.Function):
    @staticmethod
    def _forward_impl(
        x_bdnn: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[int, int, int], torch.dtype]:
        if x_bdnn.dim() != 4:
            raise ValueError("x_bdnn must have shape (B, D, N, N)")
        if x_bdnn.shape[1] % 4 != 0:
            raise ValueError("x_bdnn channel dimension must be divisible by 4")
        if x_bdnn.shape[2] != x_bdnn.shape[3]:
            raise ValueError("x_bdnn must have square sequence dimensions")
        if x_bdnn.shape[2] % 32 != 0 or (x_bdnn.shape[1] // 4) % 32 != 0:
            raise ValueError("packed fp8 tri-mul requires N and D_chunk divisible by 32")

        a1, b1, a2, b2 = torch.chunk(x_bdnn.detach(), 4, dim=1)
        a1_3d, bsz, d_chunk, n = _reshape_chunk(a1)
        b1_3d, _, _, _ = _reshape_chunk(b1)
        a2_3d, _, _, _ = _reshape_chunk(a2)
        b2_3d, _, _, _ = _reshape_chunk(b2)
        packed_a1, packed_b1, packed_a2, packed_b2 = _split_quantize_mxfp8([a1_3d, b1_3d, a2_3d, b2_3d])

        a1_fp8, a1_scale = _rowwise_mxfp8(packed_a1._rowwise_data, packed_a1._rowwise_scale_inv, bsz * d_chunk, n, n)
        b1_fp8, b1_scale = _rowwise_mxfp8(packed_b1._rowwise_data, packed_b1._rowwise_scale_inv, bsz * d_chunk, n, n)
        a1_t_fp8, a1_t_scale = _transpose_rowwise_mxfp8(
            packed_a1._columnwise_data, packed_a1._columnwise_scale_inv, bsz * d_chunk, n, n
        )
        b1_t_fp8, b1_t_scale = _transpose_rowwise_mxfp8(
            packed_b1._columnwise_data, packed_b1._columnwise_scale_inv, bsz * d_chunk, n, n
        )
        a2_fp8, a2_scale = _rowwise_mxfp8(packed_a2._rowwise_data, packed_a2._rowwise_scale_inv, bsz * d_chunk, n, n)
        a2_t_fp8, a2_t_scale = _transpose_rowwise_mxfp8(
            packed_a2._columnwise_data, packed_a2._columnwise_scale_inv, bsz * d_chunk, n, n
        )
        b2_fp8, b2_scale = _rowwise_mxfp8(packed_b2._rowwise_data, packed_b2._rowwise_scale_inv, bsz * d_chunk, n, n)
        b2_rhs_fp8, b2_rhs_scale = _rhs_mxfp8(
            packed_b2._columnwise_data, packed_b2._columnwise_scale_inv, bsz * d_chunk, n, n
        )

        x1, x2 = _C.mxfp8_cublaslt_tri_mul_pair(
            a1_fp8,
            b1_fp8,
            a2_t_fp8,
            b2_rhs_fp8,
            a1_scale,
            b1_scale,
            a2_t_scale,
            b2_rhs_scale,
            _SUPPORTED_OUT_DTYPES[out_dtype],
        )
        saved = (
            a1_t_fp8,
            a1_t_scale,
            b1_t_fp8,
            b1_t_scale,
            a2_fp8,
            a2_scale,
            b2_fp8,
            b2_scale,
        )
        shape = (bsz, d_chunk, n)
        input_dtype = x_bdnn.dtype
        x = torch.cat([x1.reshape(bsz, d_chunk, n, n), x2.reshape(bsz, d_chunk, n, n)], dim=1)
        return x, saved, shape, input_dtype

    @staticmethod
    def forward(ctx, x_bdnn: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        x, saved, shape, input_dtype = _MXFP8CuBLASLtTriMulXBDNN._forward_impl(x_bdnn, out_dtype)
        ctx.save_for_backward(*saved)
        ctx.shape = shape
        ctx.input_dtype = input_dtype
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a1_t_fp8, a1_t_scale, b1_t_fp8, b1_t_scale, a2_fp8, a2_scale, b2_fp8, b2_scale = ctx.saved_tensors
        bsz, d_chunk, n = ctx.shape
        g1, g2 = torch.chunk(grad_output.detach(), 2, dim=1)
        g1 = g1.reshape(bsz * d_chunk, n, n)
        g2 = g2.reshape(bsz * d_chunk, n, n)
        packed_g1, packed_g2 = _split_quantize_mxfp8([g1, g2])
        g1_fp8, g1_scale = _rowwise_mxfp8(packed_g1._rowwise_data, packed_g1._rowwise_scale_inv, bsz * d_chunk, n, n)
        g1_t_fp8, g1_t_scale = _transpose_rowwise_mxfp8(
            packed_g1._columnwise_data, packed_g1._columnwise_scale_inv, bsz * d_chunk, n, n
        )
        g2_fp8, g2_scale = _rowwise_mxfp8(packed_g2._rowwise_data, packed_g2._rowwise_scale_inv, bsz * d_chunk, n, n)
        g2_t_fp8, g2_t_scale = _transpose_rowwise_mxfp8(
            packed_g2._columnwise_data, packed_g2._columnwise_scale_inv, bsz * d_chunk, n, n
        )

        grad_a1, grad_b1, grad_a2, grad_b2 = _C.mxfp8_cublaslt_tri_mul_pair_backward(
            g1_fp8,
            g1_t_fp8,
            g2_fp8,
            g2_t_fp8,
            a1_t_fp8,
            b1_t_fp8,
            a2_fp8,
            b2_fp8,
            g1_scale,
            g1_t_scale,
            g2_scale,
            g2_t_scale,
            a1_t_scale,
            b1_t_scale,
            a2_scale,
            b2_scale,
        )

        grad_x = torch.cat(
            [
                grad_a1.reshape(bsz, d_chunk, n, n),
                grad_b1.reshape(bsz, d_chunk, n, n),
                grad_a2.reshape(bsz, d_chunk, n, n),
                grad_b2.reshape(bsz, d_chunk, n, n),
            ],
            dim=1,
        )
        return grad_x.to(ctx.input_dtype), None


def mxfp8_cublaslt_tri_mul_xbdnn(
    x_bdnn: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Packed MXFP8 triangular multiplication backed by raw cuBLASLt batched GEMM."""
    if out_dtype not in _SUPPORTED_OUT_DTYPES:
        raise TypeError(f"unsupported out_dtype {out_dtype}")
    return _MXFP8CuBLASLtTriMulXBDNN.apply(x_bdnn, out_dtype)


def mxfp8_cublaslt_tri_mul_xbdnn_inference(
    x_bdnn: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Forward-only packed MXFP8 triangular multiplication without autograd state."""
    if out_dtype not in _SUPPORTED_OUT_DTYPES:
        raise TypeError(f"unsupported out_dtype {out_dtype}")
    x, _, _, _ = _MXFP8CuBLASLtTriMulXBDNN._forward_impl(x_bdnn, out_dtype)
    return x
