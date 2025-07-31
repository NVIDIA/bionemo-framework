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


from unittest import mock

import anndata
import pytest

from bionemo.scspeedtest.common import (
    BenchmarkResult,
    get_batch_size,
    measure_peak_memory_full,
)


def test_benchmark_result_derives_metrics_correctly():
    epoch_results = [
        {"samples": 100, "elapsed": 2.0, "peak_memory": 300.0, "avg_memory": 250.0},
        {"samples": 200, "elapsed": 3.0, "peak_memory": 320.0, "avg_memory": 260.0},
    ]
    result = BenchmarkResult(name="test", epoch_results=epoch_results, memory_before_instantiation_mb=100.0)
    assert result.samples_per_second == 300 / 5.0
    assert result.peak_memory_mb == 220.0  # 320 - 100
    assert result.avg_memory_mb == ((250 + 260) / 2) - 100


def test_get_batch_size_anndata():
    import numpy as np

    X = np.zeros((32, 10))
    adata = anndata.AnnData(X)
    assert get_batch_size(adata) == 32


def test_get_batch_size_tensorlike():
    class DummyTensor:
        def __init__(self, shape):
            self.shape = shape

    dummy = DummyTensor((64, 128))
    assert get_batch_size(dummy) == 64


def test_get_batch_size_list():
    assert get_batch_size([1, 2, 3]) == 3


@pytest.mark.filterwarnings("ignore:This process.*is multi-threaded.*:DeprecationWarning")
def test_measure_peak_memory_full_with_mocked_rss():
    class DummyMemInfo:
        def __init__(self, rss_bytes):
            self.rss = rss_bytes

    # Use return_value instead of side_effect for consistent behavior across OS
    mock_memory = DummyMemInfo(100 * 1024 * 1024)  # 100MB baseline
    with mock.patch("psutil.Process.memory_info", return_value=mock_memory):

        def dummy_func():
            return "ok"

        result, baseline, peak, avg, delta, final, duration = measure_peak_memory_full(
            dummy_func,
            sample_interval=0.01,
            multi_worker=False,
        )

        assert result == "ok"
        assert round(baseline, 1) == 100.0
        assert round(peak, 1) == 100.0  # All measurements return same mock value
        assert round(final, 1) == 100.0
        assert round(delta, 1) == 0.0  # No delta since all values are the same
        assert 0 < duration < 1.0
