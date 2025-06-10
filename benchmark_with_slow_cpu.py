"""
[te_linear           ] | batch_first=False    | swap_bl=False    | Time: 3.3304 s
[pt_linear           ] | batch_first=False    | swap_bl=False    | Time: 2.8248 s
[rmsnormlinear       ] | batch_first=False    | swap_bl=False    | Time: 5.5260 s
[te_layernormlinear  ] | batch_first=False    | swap_bl=False    | Time: 6.7206 s
"""

import time
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from einops import rearrange

fp8_enabled = False

def set_format_recipe():
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    return fp8_format, fp8_recipe

class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super(RMSNorm, self).__init__()
        self.eps, self.hidden_size = config.eps, config.hidden_size
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size, dtype=config.params_dtype))
        self.register_parameter("scale", self.scale)
        self.use_flash_rmsnorm = config.get("use_flash_rmsnorm", False)

        if self.use_flash_rmsnorm:
            from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func

            self.rmsnorm_func = rmsnorm_func

    def forward(self, x):
        if self.use_flash_rmsnorm:
            return self.rmsnorm_func(x, self.scale, self.eps)
        else:
            y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
            return self.scale * y


class RMSNormLinear(torch.nn.Module):
    def __init__(self, config, output_size):
        super(RMSNormLinear, self).__init__()
        self.rmsnorm = RMSNorm(config)
        self.linear = nn.Linear(config.hidden_size, output_size)

    def forward(self, x):
        x = self.rmsnorm(x)
        x = self.linear(x)
        return x

class RMSNormTELinear(torch.nn.Module):
    def __init__(self, config, output_size):
        super(RMSNormTELinear, self).__init__()
        self.rmsnorm = RMSNorm(config)
        self.linear = te.Linear(config.hidden_size, output_size)
        self.fp8_format, self.fp8_recipe = set_format_recipe()

    def forward(self, x):
        x = self.rmsnorm(x)
        with te.fp8_autocast(enabled=fp8_enabled, fp8_recipe=self.fp8_recipe):
            x = self.linear(x)
        return x


class TELayerNormLinearWrapper(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(TELayerNormLinearWrapper, self).__init__()
        self.layer = te.LayerNormLinear(hidden_size, output_size, eps=1e-5, normalization="RMSNorm")
        self.fp8_format, self.fp8_recipe = set_format_recipe()

    def forward(self, x):
        with te.fp8_autocast(enabled=fp8_enabled, fp8_recipe=self.fp8_recipe):
            return self.layer(x)


def benchmark_layer(layer, inputs):
    # Warm-up
    _ = layer(inputs)
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(50000):
        _ = layer(inputs)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def get_layer(layer_type: str, hidden_size: int):
    output_size = 12288
    if layer_type == "te_layernorm":
        return te.LayerNorm(hidden_size, eps=1e-5).cuda().bfloat16()
    elif layer_type == "pt_layernorm":
        return nn.LayerNorm(hidden_size, eps=1e-5).cuda().bfloat16()
    elif layer_type == "te_linear":
        return te.Linear(hidden_size, output_size).cuda().bfloat16()
    elif layer_type == "pt_linear":
        return nn.Linear(hidden_size, output_size).cuda().bfloat16()
    elif layer_type == "te_layernormlinear":
        # te.pytorch.LayerNormLinear combines LayerNorm + Linear into one
        return TELayerNormLinearWrapper(hidden_size, output_size).cuda().bfloat16()
    elif layer_type == "rmsnorm":
        class Config:
            def __init__(self, hidden_size, eps=1e-5, params_dtype=torch.bfloat16):
                self.hidden_size = hidden_size
                self.eps = eps
                self.params_dtype = params_dtype

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config(hidden_size)
        return RMSNorm(config).cuda()
    elif layer_type == "rmsnormlinear":
        class Config:
            def __init__(self, hidden_size, eps=1e-5, params_dtype=torch.bfloat16):
                self.hidden_size = hidden_size
                self.eps = eps
                self.params_dtype = params_dtype

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config(hidden_size)
        return RMSNormLinear(config, output_size).cuda().bfloat16()
    elif layer_type == "rmsnormtelinear":
        class Config:
            def __init__(self, hidden_size, eps=1e-5, params_dtype=torch.bfloat16):
                self.hidden_size = hidden_size
                self.eps = eps
                self.params_dtype = params_dtype

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config(hidden_size)
        return RMSNormTELinear(config, output_size).cuda().bfloat16()
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

# got bitten so many times by measuring the wrong thing, so make sure we assert
# preconditions.
def assert_cpu_speed():
    from pathlib import Path
    cpu = Path("/sys/devices/system/cpu")
    for cpufreq in cpu.glob("cpu*/cpufreq/"):
        gov = cpufreq/"scaling_governor"
        assert gov.read_text().strip() == "userspace", f"Please set {gov} to userspace"
        minfreq = cpufreq/"cpuinfo_min_freq"
        setfreq = cpufreq/"scaling_setspeed"
        assert setfreq.read_text() == minfreq.read_text(), f"Please set {setfreq} to {minfreq}"
assert_cpu_speed()

def benchmark_layers(layer_type: str, batch_first, swap_bl):
    hidden_size = 4096
    input_size = hidden_size
    batch_size = 1
    seq_len = 16

    shape = (batch_size, seq_len, input_size) if batch_first else (seq_len, batch_size, input_size)
    inputs = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    if swap_bl:
        inputs = rearrange(inputs, "a b d -> b a d")

    layer = get_layer(layer_type, hidden_size)

    time_taken = benchmark_layer(layer, inputs)
    print(f"[{layer_type:<20}] | {batch_first=}    | {swap_bl=}    | Time: {time_taken:.4f} s")

if __name__ == "__main__":

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    with torch.inference_mode():
        [benchmark_layers(layer_type, batch_first, swap_bl)
         for batch_first in [False]
         for swap_bl in [False]
         for layer_type in [
            "te_linear",
            "pt_linear",
            "rmsnormlinear",
            "te_layernormlinear",
            #"rmsnorm",
            #"te_layernorm",
            #"pt_layernorm",
            #"rmsnormtelinear",
            ]
        ]
