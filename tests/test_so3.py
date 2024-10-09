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


import numpy as np

from bionemo.model.molecule.diffdock.utils import so3


def test_density():
    # two orders of magnitude smaller values - otherwise non-vectorized takes ages
    N_EPS = 10
    X_N = 20

    _eps_array = (10 ** np.linspace(np.log10(so3.MIN_EPS), np.log10(so3.MAX_EPS), N_EPS)).astype(np.float128)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:].astype(np.float128)

    exp_vals = so3._expansion_vectorized(_omegas_array, _eps_array)
    exp_vals_ref = np.asarray([so3._expansion(_omegas_array, eps) for eps in _eps_array])
    assert np.allclose(exp_vals, exp_vals_ref)

    pdf_vals = so3._density(exp_vals, _omegas_array, marginal=True)
    pdf_vals_ref = np.asarray([so3._density(_exp, _omegas_array, marginal=True) for _exp in exp_vals_ref])
    assert np.allclose(pdf_vals, pdf_vals_ref)

    cdf_vals = pdf_vals.cumsum(1) / X_N * np.pi
    cdf_vals_ref = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in pdf_vals_ref])
    assert np.allclose(cdf_vals, cdf_vals_ref)


def test_score():
    # compare results between two formulas on a range where both are numerically stable
    _eps_array = (10 ** np.linspace(np.log10(0.26), np.log10(0.7), so3.N_EPS)).astype(np.float128)
    _omegas_array = np.linspace(0, np.pi, so3.X_N + 1)[1:].astype(np.float128)

    exp_vals = so3._expansion_vectorized(_omegas_array, _eps_array)

    score_norms1 = so3._score_vectorized(exp_vals, _omegas_array, _eps_array)
    score_norms2 = so3._score_small_eps(_omegas_array, _eps_array)
    assert np.allclose(score_norms1, score_norms2, atol=1)
