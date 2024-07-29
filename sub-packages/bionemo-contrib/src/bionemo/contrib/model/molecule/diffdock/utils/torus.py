# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import numpy as np
import tqdm


"""
    Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def p(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-((x + 2 * np.pi * i) ** 2) / 2 / sigma**2)
    return p_


def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma**2 * np.exp(-((x + 2 * np.pi * i) ** 2) / 2 / sigma**2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

if os.path.exists(os.path.join(package_path, ".torus.npz")):
    torus = np.load(os.path.join(package_path, ".torus.npz"))
    p_ = torus["p_"]
    score_ = torus["score_"]
else:
    p_ = p(x, sigma[:, None], N=100)
    score_ = grad(x, sigma[:, None], N=100) / p_

    np.savez(os.path.join(package_path, ".torus.npz"), p_=p_, score_=score_)


def score(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma, seed=None):
    if seed is None:
        out = sigma * np.random.randn(*sigma.shape)
    else:
        rng = np.random.default_rng(seed)
        out = sigma * rng.normal(size=sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


class TorusScoreNorm:
    _score_norm = None

    def __init__(self, seed=None):
        if TorusScoreNorm._score_norm is None:
            _score_norm = score(
                sample(sigma[None].repeat(10000, 0).flatten(), seed=seed), sigma[None].repeat(10000, 0).flatten()
            ).reshape(10000, -1)
            TorusScoreNorm._score_norm = (_score_norm**2).mean(0)

    def __call__(self, sigma):
        sigma = np.log(sigma / np.pi)
        sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
        sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
        return TorusScoreNorm._score_norm[sigma]
