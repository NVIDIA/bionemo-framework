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


def _tri_mul_xbdnn_reference(x_bdnn: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    if x_bdnn.dim() != 4:
        raise ValueError("x_bdnn must have shape (B, D, N, N)")
    if x_bdnn.shape[1] != 128:
        raise ValueError(f"x_bdnn must have channel dimension 128, got {x_bdnn.shape[1]}")

    a1, b1, a2, b2 = torch.chunk(x_bdnn, 4, dim=1)
    batch, d_chunk, n, _ = a1.shape
    x1 = torch.bmm(a1.reshape(batch * d_chunk, n, n), b1.reshape(batch * d_chunk, n, n).transpose(1, 2))
    x2 = torch.bmm(a2.reshape(batch * d_chunk, n, n).transpose(1, 2), b2.reshape(batch * d_chunk, n, n))
    out = torch.cat(
        [x1.reshape(batch, d_chunk, n, n), x2.reshape(batch, d_chunk, n, n)],
        dim=1,
    )
    return out.permute(0, 2, 3, 1).to(out_dtype)


class _TriMulFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, k_dim: int, out_dtype: torch.dtype) -> torch.Tensor:
        ctx.k_dim = int(k_dim)
        ctx.save_for_backward(a, b)
        return _C.tri_mul_fused(a, b, int(k_dim), str(out_dtype).replace("torch.", ""))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_a, grad_b = _C.tri_mul_fused_backward(grad_output, a, b, int(ctx.k_dim))
        return grad_a, grad_b, None, None


class _TriMulXBDNNCuBLAS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bdnn: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        ctx.save_for_backward(x_bdnn)
        ctx.input_dtype = x_bdnn.dtype
        ctx.out_dtype = out_dtype
        return _C.tri_mul_xbdnn_cublas(x_bdnn, str(out_dtype).replace("torch.", ""))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x_bdnn,) = ctx.saved_tensors
        if grad_output.dtype != ctx.input_dtype:
            grad_output = grad_output.to(ctx.input_dtype)
        grad_output = grad_output.contiguous()

        a1, b1, a2, b2 = [
            t.permute(0, 2, 3, 1).contiguous() for t in torch.chunk(x_bdnn, 4, dim=1)
        ]
        g1, g2 = [t.contiguous() for t in torch.chunk(grad_output, 2, dim=-1)]
        grad_a1, grad_b1 = _C.tri_mul_fused_backward(g1, a1, b1, 2)
        grad_a2, grad_b2 = _C.tri_mul_fused_backward(g2, a2, b2, 1)
        grad_x = torch.cat(
            [
                grad_a1.permute(0, 3, 1, 2).contiguous(),
                grad_b1.permute(0, 3, 1, 2).contiguous(),
                grad_a2.permute(0, 3, 1, 2).contiguous(),
                grad_b2.permute(0, 3, 1, 2).contiguous(),
            ],
            dim=1,
        )
        return grad_x.to(ctx.input_dtype), None


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
    if (
        tri_mul_fused_triton is not None
        and a.is_cuda
        and a.dtype == torch.bfloat16
        and a.shape[-1] == 32
        and not torch.is_grad_enabled()
    ):
        return tri_mul_fused_triton(a, b, k_dim=k_dim, out_dtype=out_dtype)
    if _C is None or a.shape[-1] != 32:
        return _tri_mul_reference(a, b, k_dim).to(out_dtype)
    return _TriMulFused.apply(a, b, int(k_dim), out_dtype)


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
    if (
        not x_bdnn.requires_grad
        and not torch.is_grad_enabled()
    ):
        return _C.tri_mul_xbdnn_cublas(x_bdnn, str(out_dtype).replace("torch.", ""))
    if x_bdnn.shape[1] != 128:
        return _tri_mul_xbdnn_reference(x_bdnn, out_dtype)
    return _TriMulXBDNNCuBLAS.apply(x_bdnn, out_dtype)
