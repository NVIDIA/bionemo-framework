# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# TODO: Remove this file once Apex version in container contains at least SHA 58acf96915eecd7e13adff61d2c389fba3efede2
# © 2023 NVIDIA CORPORATION & AFFILIATES

from distutils.version import LooseVersion

import triton
import triton.language as tl
from triton import Config


if LooseVersion("2.0.0") < LooseVersion(triton.__version__):
    rsqrt = tl.math.rsqrt
else:
    rsqrt = tl.libdevice.rsqrt


# %% Forward kernel for contiguous inputs.
@triton.autotune(
    configs=[
        Config({"M_BLOCK": 1}, num_warps=1),
        Config({"M_BLOCK": 2}, num_warps=1),
        Config({"M_BLOCK": 4}, num_warps=2),
        Config({"M_BLOCK": 8}, num_warps=4),
        Config({"M_BLOCK": 16}, num_warps=8),
        Config({"M_BLOCK": 32}, num_warps=8),
        Config({"M_BLOCK": 64}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={
        "N_BLOCK": lambda kwargs: triton.next_power_of_2(kwargs["N"]),
    },
)
@triton.jit
def _layer_norm_forward(
    x_ptr,
    w_ptr,
    b_ptr,
    eps,
    x_invstd_ptr,
    x_mean_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
):
    m_idx = tl.program_id(0) * M_BLOCK + tl.arange(0, M_BLOCK)
    m_mask = m_idx < M
    n_idx = tl.arange(0, N_BLOCK)
    n_mask = n_idx < N
    mask = m_mask[:, None] & n_mask[None, :]
    x = tl.load(x_ptr + N * m_idx[:, None] + n_idx[None, :], mask, other=0).to(tl.float32)
    x_mean = tl.sum(x, 1) / N
    tl.store(x_mean_ptr + m_idx, x_mean, m_mask)
    x_bar = x - x_mean[:, None]
    x_var = tl.sum(x_bar * x_bar, 1) / N
    x_invstd = rsqrt(x_var + eps)
    tl.store(x_invstd_ptr + m_idx, x_invstd, m_mask)
    x_hat = x_bar * x_invstd[:, None]
    w = tl.load(w_ptr + n_idx, n_mask, other=0).to(tl.float32)[None, :]
    b = tl.load(b_ptr + n_idx, n_mask, other=0).to(tl.float32)[None, :]
    y = w * x_hat + b
    tl.store(y_ptr + N * m_idx[:, None] + n_idx[None, :], y, mask)


# %% Forward kernel for noncontiguous inputs. Using strided access to avoid extra memory overhead.
@triton.autotune(
    configs=[
        Config({"M_BLOCK": 1}, num_warps=1),
        Config({"M_BLOCK": 2}, num_warps=1),
        Config({"M_BLOCK": 4}, num_warps=2),
        Config({"M_BLOCK": 8}, num_warps=4),
        Config({"M_BLOCK": 16}, num_warps=8),
        Config({"M_BLOCK": 32}, num_warps=8),
        Config({"M_BLOCK": 64}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={
        "N_BLOCK": lambda kwargs: triton.next_power_of_2(kwargs["N"]),
    },
)
@triton.jit
def _layer_norm_forward_strided(
    x_ptr,
    w_ptr,
    b_ptr,
    eps,
    x_invstd_ptr,
    x_mean_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    m_logic_idx = tl.program_id(0) * M_BLOCK + tl.arange(0, M_BLOCK)
    m_mask = m_logic_idx < M
    m_logic_idx_0 = m_logic_idx // (D1 * D2) % D0
    m_logic_idx_1 = m_logic_idx // D2 % D1
    m_logic_idx_2 = m_logic_idx % D2
    m_idx = m_logic_idx_0 * S0 + m_logic_idx_1 * S1 + m_logic_idx_2 * S2
    n_logic_idx = tl.arange(0, N_BLOCK)
    n_mask = n_logic_idx < N
    n_idx = n_logic_idx * S3
    mask = m_mask[:, None] & n_mask[None, :]
    x_idx = m_idx[:, None] + n_idx[None, :]
    x = tl.load(x_ptr + x_idx, mask, other=0).to(tl.float32)
    x_mean = tl.sum(x, 1) / N
    tl.store(x_mean_ptr + m_logic_idx, x_mean, m_mask)
    x_bar = x - x_mean[:, None]
    x_var = tl.sum(x_bar * x_bar, 1) / N
    x_invstd = rsqrt(x_var + eps)
    tl.store(x_invstd_ptr + m_logic_idx, x_invstd, m_mask)
    x_hat = x_bar * x_invstd[:, None]
    w = tl.load(w_ptr + n_logic_idx, n_mask, other=0).to(tl.float32)[None, :]
    b = tl.load(b_ptr + n_logic_idx, n_mask, other=0).to(tl.float32)[None, :]
    y = w * x_hat + b
    tl.store(y_ptr + N * m_logic_idx[:, None] + n_logic_idx[None, :], y, mask)
