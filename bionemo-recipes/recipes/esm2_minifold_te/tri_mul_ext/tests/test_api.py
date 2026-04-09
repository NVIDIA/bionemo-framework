from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tri_mul_ext"))

from tri_mul_ext import tri_mul_fused, tri_mul_xbdnn_cublas  # noqa: E402


def test_tri_mul_fused_matches_reference_kdim2():
    if not torch.cuda.is_available():
        return
    a = torch.randn((2, 8, 8, 4), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((2, 8, 8, 4), device="cuda", dtype=torch.bfloat16)
    ref = torch.einsum("bikd,bjkd->bijd", a.float(), b.float()).to(torch.bfloat16)
    out = tri_mul_fused(a, b, k_dim=2, out_dtype=torch.bfloat16)
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_tri_mul_fused_matches_reference_kdim1():
    if not torch.cuda.is_available():
        return
    a = torch.randn((2, 8, 8, 4), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((2, 8, 8, 4), device="cuda", dtype=torch.bfloat16)
    ref = torch.einsum("bkid,bkjd->bijd", a.float(), b.float()).to(torch.bfloat16)
    out = tri_mul_fused(a, b, k_dim=1, out_dtype=torch.bfloat16)
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_tri_mul_xbdnn_cublas_matches_reference():
    if not torch.cuda.is_available():
        return
    x = torch.randn((2, 8, 8, 128), device="cuda", dtype=torch.bfloat16)
    x_bdnn = x.permute(0, 3, 1, 2).contiguous()
    x1, x2, x3, x4 = x.chunk(4, dim=-1)
    ref1 = torch.einsum("bikd,bjkd->bijd", x1.float(), x2.float())
    ref2 = torch.einsum("bkid,bkjd->bijd", x3.float(), x4.float())
    ref = torch.cat([ref1, ref2], dim=-1).to(torch.bfloat16)
    out = tri_mul_xbdnn_cublas(x_bdnn, out_dtype=torch.bfloat16)
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
