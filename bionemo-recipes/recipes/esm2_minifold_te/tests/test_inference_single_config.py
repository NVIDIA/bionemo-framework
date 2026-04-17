import importlib.util
import sys
from pathlib import Path

import pytest
import torch


RECIPE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RECIPE_DIR))

PLAIN_SPEC = importlib.util.spec_from_file_location("plain_minifold_infer", RECIPE_DIR / "plain_minifold_infer.py")
PLAIN_MODULE = importlib.util.module_from_spec(PLAIN_SPEC)
assert PLAIN_SPEC is not None and PLAIN_SPEC.loader is not None
sys.modules[PLAIN_SPEC.name] = PLAIN_MODULE
PLAIN_SPEC.loader.exec_module(PLAIN_MODULE)

SCRIPT_SPEC = importlib.util.spec_from_file_location("inference_single_config", RECIPE_DIR / "inference_single_config.py")
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


def test_fp8_roundtrip_shapes():
    x = torch.randn(2, 4, 4, 8, dtype=torch.bfloat16)
    x_fp8, scale = PLAIN_MODULE.quantize_to_fp8(x)
    x_deq = PLAIN_MODULE.dequantize_from_fp8(x_fp8, scale)
    assert x_fp8.shape == x.shape
    assert scale.shape == (2, 4, 4, 1)
    assert x_deq.shape == x.shape


def test_default_config_loads():
    args = SCRIPT_MODULE.build_arg_parser().parse_args([])
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.seq_len == 256
    assert cfg.mbs == 16
    assert cfg.tri_impl == "bmm"
    assert cfg.pair_precision == "bf16"
    assert cfg.linear_precision == "bf16"


def test_compat_flag_maps_to_fp8_storage():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(["--fp8_activations"])
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.pair_precision == "fp8_storage"
    assert cfg.fp8_activations is True
    assert cfg.linear_precision == "bf16"


def test_linear_precision_flag_loads():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(["--linear_precision", "fp8"])
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.linear_precision == "fp8"


def test_fp8_extreme_config_loads():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(
        ["--pair_precision", "fp8_extreme", "--linear_precision", "fp8", "--tri_impl", "fp8_cublaslt"]
    )
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.pair_precision == "fp8_extreme"
    assert cfg.linear_precision == "fp8"
    assert cfg.tri_impl == "fp8_cublaslt"


def test_fp8_native_config_loads():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(
        ["--pair_precision", "fp8_native", "--linear_precision", "fp8", "--tri_impl", "fp8_cublaslt"]
    )
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.pair_precision == "fp8_native"
    assert cfg.linear_precision == "fp8"
    assert cfg.tri_impl == "fp8_cublaslt"


def test_fp8_extreme_validation_rejects_non_fp8_tri_backend():
    with pytest.raises(ValueError, match="pair_precision=fp8_extreme.*requires tri_impl"):
        PLAIN_MODULE.validate_fp8_extreme_configuration("fp8_extreme", "fp8", "bmm")


def test_fp8_native_validation_rejects_non_fp8_tri_backend():
    with pytest.raises(ValueError, match="pair_precision=fp8_native.*requires tri_impl"):
        PLAIN_MODULE.validate_fp8_extreme_configuration("fp8_native", "fp8", "bmm")


def test_render_markdown():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(
        ["--seq_len", "1024", "--mbs", "8", "--tri_impl", "cublas_xbdnn", "--fp8_activations", "--linear_precision", "fp8"]
    )
    cfg = SCRIPT_MODULE.load_config(args)
    table = SCRIPT_MODULE.render_markdown(
        cfg,
        {
            "median_ms": 12.5,
            "proteins_per_sec": 640.0,
            "unpadded_tokens_per_sec": 655360.0,
            "peak_memory_allocated_gib": 3.25,
        },
    )
    assert "| cublas_xbdnn | fp8_storage | fp8 | 1024 | 8 | 12.50 |" in table
    assert "fp8_storage" in table
    assert "linear=fp8" in table


def test_quantized_pair_tensor_storage_accounting():
    x = torch.randn(1, 4, 4, 8, dtype=torch.bfloat16)
    q = PLAIN_MODULE.QuantizedPairTensor.from_tensor(x)
    assert q.quantized_bytes == q.payload.numel() * q.payload.element_size()
    assert q.scale_bytes == q.scale.numel() * q.scale.element_size()


def test_configure_linear_precision_registers_fp8_buffers():
    block = PLAIN_MODULE.TriangularUpdate(dim=16, linear_precision="bf16").to(dtype=torch.bfloat16)
    block.set_linear_precision("fp8")
    assert block.linear_precision == "fp8"
    assert block.pi.weight_fp8.dtype == torch.float8_e4m3fn
    assert block.pi.scale_w.dtype == torch.float32


def test_configure_linear_precision_transition_buffers():
    block = PLAIN_MODULE.TransitionUpdate(dim=16, hidden=32, linear_precision="bf16").to(dtype=torch.bfloat16)
    block.set_linear_precision("fp8", include_transition=True)
    assert block.fc1.weight_fp8.dtype == torch.float8_e4m3fn
    assert block.fc2.scale_w.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused tri-output smoke")
def test_fp8_tri_outputs_to_carrier_shapes():
    x1 = torch.randn(32, 32, 32, device="cuda", dtype=torch.float16)
    x2 = torch.randn(32, 32, 32, device="cuda", dtype=torch.float16)
    payload, scale = PLAIN_MODULE.fp8_tri_outputs_to_carrier(x1, x2, batch=1, scale_dtype=torch.float32)
    assert payload.shape == (1, 32, 32, 64)
    assert scale.shape == (1, 32, 32, 1)
    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused linear-output requant smoke")
def test_fp8_requantize_rows_shapes():
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    payload, scale = PLAIN_MODULE.fp8_requantize_rows(x, scale_dtype=torch.float32)
    assert payload.shape == x.shape
    assert scale.shape == (16, 1)
    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for block32 bias-requant smoke")
def test_fp8_requantize_block32_with_bias_shapes():
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    payload, scale = PLAIN_MODULE.fp8_requantize_block32(x, bias=bias, scale_dtype=torch.float32)
    assert payload.shape == x.shape
    assert scale.shape == (8, 4)
    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native fused linear smoke",
)
def test_native_linear_forward_quantized_shapes():
    linear = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(linear, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    y = PLAIN_MODULE.native_linear_forward_quantized(linear, x)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 8, 8, 128)
    assert y.scale.shape == (1, 8, 8, 4)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native fused linear smoke",
)
def test_native_linear_forward_quantized_direct_output_shapes():
    linear = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(linear, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    y = PLAIN_MODULE.native_linear_forward_quantized(linear, x, direct_fp8_output=True)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 8, 8, 128)
    assert y.scale.shape == (1, 8, 8, 4)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native gate smoke",
)
def test_native_gate_sigmoid_mul_quantized_shapes():
    pi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    gi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(pi, enabled=True)
    PLAIN_MODULE.configure_fp8_linear_weight_(gi, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    y = PLAIN_MODULE.native_gate_sigmoid_mul_quantized(pi, gi, x)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 8, 8, 128)
    assert y.scale.shape == (1, 8, 8, 4)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native block32 pointwise smoke",
)
def test_native_block32_pointwise_shapes():
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    norm = torch.nn.LayerNorm(128, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    y = PLAIN_MODULE.native_mxfp8_layernorm_quantized(norm, x)
    z = PLAIN_MODULE.native_mxfp8_relu_quantized(y)
    added = PLAIN_MODULE.native_mxfp8_add_quantized(z, z)
    assert added.payload.shape == (1, 8, 8, 128)
    assert added.scale.shape == (1, 8, 8, 4)
    assert added.payload.dtype == torch.float8_e4m3fn
    assert added.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native block32 pointwise smoke",
)
def test_native_block32_layernorm_64_shapes():
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 64, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    norm = torch.nn.LayerNorm(64, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    y = PLAIN_MODULE.native_mxfp8_layernorm_quantized(norm, x)
    assert y.payload.shape == (1, 8, 8, 64)
    assert y.scale.shape == (1, 8, 8, 2)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native tri smoke",
)
def test_native_tri_mul_from_block32_quantized_shapes():
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 32, 32, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    mask = torch.ones(1, 32, device="cuda", dtype=torch.bool)
    y = PLAIN_MODULE.native_tri_mul_from_block32_quantized(x, mask=mask)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 32, 32, 64)
    assert y.scale.shape == (1, 32, 32, 2)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32
