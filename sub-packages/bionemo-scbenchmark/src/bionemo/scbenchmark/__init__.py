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


"""BioNeMo Single Cell Benchmarking Framework.

A simple, flexible framework for benchmarking any dataloader.
"""

from .benchmark import (
    BenchmarkConfig,
    benchmark_dataloader,
    run_benchmark,
)
from .common import (
    BenchmarkResult,
    export_benchmark_results,
)


__version__ = "0.1.0"
__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "benchmark_dataloader",
    "export_benchmark_results",
    "run_benchmark",
]
