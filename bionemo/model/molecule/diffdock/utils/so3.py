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


import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation


package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 1000
X_N = 2000

omegas = np.linspace(0, np.pi, X_N + 1)[1:]

# TODO generating these arrays is super slow, we should vectorize this


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(
        Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()
    ).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * np.sin(omega * (l + 1 / 2)) / np.sin(omega / 2)
    return p


def _expansion_vectorized(omega, eps, L=2000):
    l = np.arange(L).reshape((-1, 1))
    omega = omega.reshape((1, -1))
    eps = eps.reshape((1, -1))

    p1 = (2 * l + 1) * np.exp(-l * (l + 1) * eps**2)
    p2 = np.sin(omega * (l + 1 / 2)) / np.sin(omega / 2)
    p = np.matmul(p2.T, p1).T
    return p


def _density(expansion, omega, marginal=True):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return expansion / 8 / np.pi**2  # the constant factor doesn't affect any actual calculations though


def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * (lo * dhi - hi * dlo) / lo**2
    return dSigma / exp


def _score_vectorized(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    l = np.arange(L).reshape((-1, 1))
    omega = omega.reshape((1, -1))
    eps = eps.reshape((1, -1))

    hi = np.sin(omega * (l + 1 / 2))
    dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
    lo = np.sin(omega / 2)
    dlo = 1 / 2 * np.cos(omega / 2)
    dSigma1 = (2 * l + 1) * np.exp(-l * (l + 1) * eps**2)
    dSigma2 = (lo * dhi - hi * dlo) / lo**2
    dSigma = np.matmul(dSigma2.T, dSigma1).T
    return dSigma / exp


def _score_small_eps(omega, eps):
    # formula for f(omega, eps) in eq (5) https://openreview.net/pdf?id=jHA-yCyBGb
    # score = d(log(f(omega, eps^2)) / d omega
    # for our range of omegas, this approximation works well for eps up to ~0.7
    # note that for numerical stability it is important to combine
    # exp(pi*omega/eps) * exp(-pi**2/eps) into exp(pi*(omega-pi)/eps)

    x = omega.reshape((1, -1))
    a = eps.reshape((-1, 1)) ** 2

    return (
        -0.5 * x / a
        + (
            1
            + -np.exp(np.pi * (x - np.pi) / a)
            + -np.exp(-np.pi * (x + np.pi) / a)
            + -(np.pi * (x - 2 * np.pi) / a) * np.exp(np.pi * (x - np.pi) / a)
            + np.pi * (x + 2 * np.pi) / a * np.exp(-np.pi * (x + np.pi) / a)
        )
        / (x + -(x - 2 * np.pi) * np.exp(np.pi * (x - np.pi) / a) + (x + 2 * np.pi) * np.exp(-np.pi * (x + np.pi) / a))
        - 0.5 * np.cos(x / 2) / np.sin(x / 2)
    )


if os.path.exists(os.path.join(package_path, ".so3.npz")):
    so3 = np.load(os.path.join(package_path, ".so3.npz"))
    _omegas_array = so3["_omegas_array"]
    _cdf_vals = so3["_cdf_vals"]
    _score_norms = so3["_score_norms"]
    _exp_score_norms = so3["_exp_score_norms"]
else:
    _eps_array = (10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)).astype(np.float128)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:].astype(np.float128)

    _exp_vals = _expansion_vectorized(_omegas_array, _eps_array)
    _pdf_vals = _density(_exp_vals, _omegas_array, marginal=True)
    _cdf_vals = _pdf_vals.cumsum(1) / X_N * np.pi
    _score_norms = np.zeros((N_EPS, X_N))
    _small_eps_idx = _eps_array < 0.5
    _score_norms[_small_eps_idx] = _score_small_eps(_omegas_array, _eps_array[_small_eps_idx])
    _score_norms[~_small_eps_idx] = _score_vectorized(
        _exp_vals[~_small_eps_idx], _omegas_array, _eps_array[~_small_eps_idx]
    )

    _exp_score_norms = np.sqrt(np.sum(_score_norms**2 * _pdf_vals, axis=1) / np.sum(_pdf_vals, axis=1) / np.pi)

    _omegas_array = _omegas_array.astype(np.float64)
    _cdf_vals = _cdf_vals.astype(np.float64)
    _score_norms = _score_norms.astype(np.float64)
    _exp_score_norms = _exp_score_norms.astype(np.float64)

    np.savez(
        os.path.join(package_path, ".so3.npz"),
        _omegas_array=_omegas_array,
        _cdf_vals=_cdf_vals,
        _score_norms=_score_norms,
        _exp_score_norms=_exp_score_norms,
    )


def sample(eps):
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def sample_vec(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps, vec):
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps):
    device = eps.device
    eps = eps.cpu().numpy()
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).to(device=device, dtype=torch.float)
