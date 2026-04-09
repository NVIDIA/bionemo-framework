from __future__ import annotations

import torch

try:
    from . import _C
except ImportError:
    _C = None

try:
    from .triton_kernel import tri_mul_fused_triton
except ImportError:
    tri_mul_fused_triton = None


def extension_available() -> bool:
    return tri_mul_fused_triton is not None or _C is not None


def _tri_mul_reference(a: torch.Tensor, b: torch.Tensor, k_dim: int) -> torch.Tensor:
    if a.dim() != 4 or b.dim() != 4:
        raise ValueError("a and b must be rank-4 tensors with shape (B, N, N, D)")
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    if k_dim not in {1, 2}:
        raise ValueError(f"k_dim must be 1 or 2, got {k_dim}")

    B, N1, N2, D = a.shape
    a3 = a.permute(0, 3, 1, 2).contiguous().reshape(B * D, N1, N2)
    b3 = b.permute(0, 3, 1, 2).contiguous().reshape(B * D, N1, N2)
    if k_dim == 2:
        out3 = torch.bmm(a3, b3.transpose(1, 2))
    else:
        out3 = torch.bmm(a3.transpose(1, 2), b3)
    return out3.reshape(B, D, N1, N2).permute(0, 2, 3, 1)


def tri_mul_fused(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    k_dim: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused triangular multiplication API.

    The built extension is allowed to diverge internally from the current
    reshape-to-bmm implementation. The Python fallback keeps the same semantics
    so callers and benchmarks can be wired up before the real fused kernel lands.
    """
    out_dtype = a.dtype if out_dtype is None else out_dtype
    if tri_mul_fused_triton is not None and a.is_cuda and a.dtype == torch.bfloat16 and a.shape[-1] == 32:
        return tri_mul_fused_triton(a, b, k_dim=k_dim, out_dtype=out_dtype)
    if _C is None or a.shape[-1] != 32:
        return _tri_mul_reference(a, b, k_dim).to(out_dtype)
    return _C.tri_mul_fused(a, b, int(k_dim), str(out_dtype).replace("torch.", ""))


def tri_mul_bdnn_cublas(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    k_dim: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out_dtype = a.dtype if out_dtype is None else out_dtype
    if _C is None:
        raise RuntimeError("tri_mul_ext._C is not available")
    return _C.tri_mul_bdnn_cublas(a, b, int(k_dim), str(out_dtype).replace("torch.", ""))


def tri_mul_xbdnn_cublas(
    x_bdnn: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out_dtype = x_bdnn.dtype if out_dtype is None else out_dtype
    if _C is None:
        raise RuntimeError("tri_mul_ext._C is not available")
    return _C.tri_mul_xbdnn_cublas(x_bdnn, str(out_dtype).replace("torch.", ""))
