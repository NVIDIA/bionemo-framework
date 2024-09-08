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

import gc
import sys
from typing import Any, Generator, List, Optional, Tuple, TypeVar

import torch


Feature = TypeVar("Feature")


def collect_cuda_peak_alloc(
    workflow: Generator[Optional[Feature], bool, Any],
    device: torch.device,
) -> Tuple[List[Feature], List[int]]:
    """
    Collects CUDA peak memory allocation statistics for a given workflow.
    This function iterates through the workflow, runs each iteration to completion,
    and records the peak CUDA memory allocation during this process.

    Note: the first iteration of the workflow might result in smaller memory
    allocation due to uninitialized data, .e.g,  those internal to Pytorch, so the
    user might want to skip the first few resulting data points

    Args:
        workflow: A generator that performs the work whose CUDA memory
            allocation is to be monitored and collected. It should yield features
            that can be used to fit a memory consumption model. The features yield
            from it should not consume significant amount of GPU memory to avoid
            bias in peak allocation measurement.
        device: The target Torch CUDA device

    Returns:
        Tuple[List[Feature], List[int]]: A tuple containing the collected features and memory usage statistics.

    Raises:
        ValueError: If the provided device is not a CUDA device.
    """
    if device.type != "cuda":
        raise ValueError("This function is intended for CUDA devices only.")

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    features = []
    alloc_peaks = []

    # prime the generator with None
    do_cleanup = None
    try:
        while True:
            try:
                data = workflow.send(do_cleanup)
                if do_cleanup is None:
                    do_cleanup = cleanup
                alloc_peak = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
                alloc_peaks.append(alloc_peak)
                features.append(data)
            except torch.cuda.OutOfMemoryError:
                print("Encounter CUDA out-of-memory error. Skipping sample", file=sys.stderr, flush=True)
                continue
            except Exception as e:
                raise e
    except StopIteration:
        pass
    except Exception as e:
        raise RuntimeError("An error occurred during CUDA memory collection.") from e
    return features, alloc_peaks


class PolynomialRegression(torch.nn.Module):
    """
    A class for performing polynomial regression using PyTorch.

    This class allows users to create a model that fits data points
    with a polynomial of a specified degree. It also provides methods
    to evaluate the fitted polynomial and fit it to new data.
    """

    def __init__(self, degree: int):
        """
        Initializes a PolynomialRegression object.

        Args:
            degree (int): The degree of the polynomial regression model.
                Must be a non-negative integer.

        Raises:
            ValueError: If degree is not a non-negative integer.
        """
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("degree must be a non-negative integer")
        self.degree = degree
        self.coeffs = torch.zeros(degree + 1, dtype=torch.float32)
        super().__init__()

    def _polynomial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the polynomial at point(s) x.

        Args:
            x (torch.Tensor): A 1D tensor containing the points to evaluate
                the polynomial at.

        Returns:
            torch.Tensor: A 2D tensor where each row corresponds to a data
                point and each column corresponds to a term in the polynomial.
        """
        if x.ndim != 1:
            raise TypeError("x must be a 1D tensor.")
        ans = x.unsqueeze(-1).repeat(1, self.degree + 1)
        ans[:, 0] = 1
        ans = torch.cumprod(ans, dim=-1)
        return ans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the polynomial at point(s) x.

        Args:
            x (torch.Tensor): A 1D tensor containing the points to evaluate
                the polynomial at.

        Returns:
            torch.Tensor: The value of the polynomial at each data point.
        """
        if x.ndim != 1:
            raise TypeError("x must be a 1D tensor.")
        polynomial = self._polynomial(x)
        return polynomial @ self.coeffs

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fits the polynomial regression model to data points (x, y).

        Args:
            x (torch.Tensor): A 1D tensor containing the input data points.
            y (torch.Tensor): A 1D tensor containing the output data points.

        Raises:
            TypeError: If x or y is not a 1D tensor.
            ValueError: If the number of samples in x and y are not equal.
            TypeError: If x or y is not a floating point tensor.
        """
        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating point tensor.")
        if not torch.is_floating_point(y):
            raise TypeError("y must be a floating point tensor.")
        if x.ndim != 1:
            raise TypeError("x must be a 1D tensor.")
        if y.ndim != 1:
            raise TypeError("y must be a 1D tensor.")
        n_samples = x.shape[0]
        if y.shape[0] != n_samples:
            raise ValueError("The number of samples in x and y must be equal.")
        polynomial = self._polynomial(x)
        self.coeffs = torch.linalg.lstsq(polynomial, y).solution
