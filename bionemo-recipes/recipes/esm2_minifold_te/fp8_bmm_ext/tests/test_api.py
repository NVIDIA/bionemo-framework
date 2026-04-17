import pytest
import torch

import bmm_ext
from bmm_ext import (
    mxfp8_bmm,
    mxfp8_cublaslt_bmm,
    mxfp8_cublaslt_tri_mul_xbdnn,
    mxfp8_tri_mul_xbdnn,
    pack_nvfp4,
    quantize_mxfp8,
)
from bmm_ext.ops import PackedNVFP4Tensor, _swizzle_mxfp8_scale_rowwise, bmm_block_scaled


def test_rejects_broadcastable_shapes():
    a = torch.empty((1, 4, 8), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty((2, 8, 4), device="cuda", dtype=torch.float8_e4m3fn)
    a_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.float8_e8m0fnu)
    b_scale = torch.empty((2, 1, 4), device="cuda", dtype=torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="batch dimensions"):
        bmm_block_scaled(a, b, a_scale=a_scale, b_scale=b_scale, format="mxfp8", sf_vec_size=8)


def test_rejects_grad_tensors():
    a = torch.empty((1, 4, 8), device="cuda", dtype=torch.float8_e4m3fn, requires_grad=False)
    b = torch.empty((1, 8, 4), device="cuda", dtype=torch.float8_e4m3fn)
    a_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.float8_e8m0fnu, requires_grad=True)
    b_scale = torch.empty((1, 1, 4), device="cuda", dtype=torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="forward-only"):
        bmm_block_scaled(a, b, a_scale=a_scale, b_scale=b_scale, format="mxfp8", sf_vec_size=8)


def test_nvfp4_requires_packed_wrapper():
    a = torch.empty((1, 4, 8), device="cuda", dtype=torch.uint8)
    b = torch.empty((1, 8, 4), device="cuda", dtype=torch.uint8)
    a_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.float8_e4m3fn)
    b_scale = torch.empty((1, 1, 4), device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(TypeError, match="PackedNVFP4Tensor"):
        bmm_block_scaled(a, b, a_scale=a_scale, b_scale=b_scale, format="nvfp4", sf_vec_size=16)


def test_packed_wrapper_shape_validation():
    a = PackedNVFP4Tensor(torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8), (1, 4, 10))
    b = PackedNVFP4Tensor(torch.empty((1, 10, 2), device="cuda", dtype=torch.uint8), (1, 10, 4))
    a_scale = torch.empty((1, 4, 0), device="cuda", dtype=torch.float8_e4m3fn)
    b_scale = torch.empty((1, 0, 4), device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        bmm_block_scaled(a, b, a_scale=a_scale, b_scale=b_scale, format="nvfp4", sf_vec_size=16)


def test_mxfp8_smoke_matches_fp32_reference():
    a = torch.randn((2, 8, 32), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn((2, 32, 4), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    a_scale = torch.ones((2, 8, 1), device="cuda", dtype=torch.float8_e8m0fnu)
    b_scale = torch.ones((2, 1, 4), device="cuda", dtype=torch.float8_e8m0fnu)
    out = bmm_block_scaled(a, b, a_scale=a_scale, b_scale=b_scale, format="mxfp8", sf_vec_size=32)
    ref = torch.bmm(a.float(), b.float())
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref)


def test_quantize_mxfp8_returns_expected_layout():
    x = torch.randn((2, 8, 32), device="cuda", dtype=torch.bfloat16)
    data, scale = quantize_mxfp8(x)
    assert data.shape == x.shape
    assert data.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 8, 1)
    assert scale.dtype == torch.float8_e8m0fnu


def test_mxfp8_bmm_backward_matches_bf16_reference():
    a = torch.randn((2, 32, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn((2, 32, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = mxfp8_bmm(a, b, out_dtype=torch.float32)
    loss = out.square().mean()
    loss.backward()

    a_ref = a.detach().clone().float().requires_grad_(True)
    b_ref = b.detach().clone().float().requires_grad_(True)
    out_ref = torch.bmm(a_ref, b_ref)
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    assert out.dtype == torch.float32
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.allclose(out, out_ref, atol=1.0, rtol=0.2)
    assert torch.allclose(a.grad.float(), a_ref.grad, atol=0.5, rtol=0.3)
    assert torch.allclose(b.grad.float(), b_ref.grad, atol=0.5, rtol=0.3)


def test_mxfp8_cublaslt_matches_te_raw_backend():
    a = torch.randn((8, 256, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((8, 256, 256), device="cuda", dtype=torch.bfloat16)

    a_fp8, a_scale = quantize_mxfp8(a, role="lhs")
    b_fp8, b_scale = quantize_mxfp8(b, role="rhs")
    b_t_fp8, b_t_scale = quantize_mxfp8(b.transpose(1, 2).contiguous(), role="lhs")
    out_te = bmm_block_scaled(
        a_fp8,
        b_fp8,
        a_scale=a_scale,
        b_scale=b_scale,
        format="mxfp8",
        out_dtype=torch.float32,
        sf_vec_size=32,
    )
    out_lt = bmm_ext._C.mxfp8_cublaslt_bmm(
        a_fp8,
        b_t_fp8,
        _swizzle_mxfp8_scale_rowwise(a_scale),
        _swizzle_mxfp8_scale_rowwise(b_t_scale),
        "float32",
    )

    assert torch.equal(out_lt, out_te)


def test_mxfp8_cublaslt_wrapper_produces_finite_output():
    a = torch.randn((8, 256, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((8, 256, 256), device="cuda", dtype=torch.bfloat16)
    out = mxfp8_cublaslt_bmm(a, b, out_dtype=torch.float32)
    assert out.shape == (8, 256, 256)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_mxfp8_tri_mul_xbdnn_forward_backward():
    x = torch.randn((2, 128, 32, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = mxfp8_tri_mul_xbdnn(x, out_dtype=torch.float32)
    loss = out.square().mean()
    loss.backward()

    a1, b1, a2, b2 = torch.chunk(x.detach().float(), 4, dim=1)
    bsz = x.shape[0]
    d_chunk = a1.shape[1]
    x1 = torch.bmm(a1.reshape(-1, 32, 32), b1.reshape(-1, 32, 32).transpose(1, 2)).reshape(bsz, d_chunk, 32, 32)
    x2 = torch.bmm(a2.reshape(-1, 32, 32).transpose(1, 2), b2.reshape(-1, 32, 32)).reshape(bsz, d_chunk, 32, 32)
    ref = torch.cat([x1, x2], dim=1)

    assert out.shape == (2, 64, 32, 32)
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()
    delta = (out - ref).abs()
    assert delta.mean().item() < 0.25
    assert delta.flatten().quantile(0.99).item() < 0.8


def test_mxfp8_cublaslt_tri_mul_xbdnn_forward_backward():
    x = torch.randn((2, 128, 32, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = mxfp8_cublaslt_tri_mul_xbdnn(x, out_dtype=torch.float32)
    loss = out.square().mean()
    loss.backward()

    ref = mxfp8_tri_mul_xbdnn(x.detach(), out_dtype=torch.float32)
    assert out.shape == (2, 64, 32, 32)
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()
    assert torch.equal(out, ref)


def test_nvfp4_smoke_returns_expected_shape():
    a_src = torch.randn((1, 16, 32), device="cuda", dtype=torch.float32).clamp(-6, 6)
    b_src = torch.randn((1, 32, 16), device="cuda", dtype=torch.float32).clamp(-6, 6)
    a = pack_nvfp4(a_src)
    b = pack_nvfp4(b_src, role="rhs")
    a_scale = a.scale_inv
    b_scale = b.scale_inv
    out = bmm_block_scaled(a, b, a_scale=a_scale, b_scale=b_scale, format="nvfp4", sf_vec_size=16)
    ref = torch.bmm(a_src, b_src)
    assert tuple(out.shape) == (1, 16, 16)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.max(torch.abs(out - ref)) < 10.0
