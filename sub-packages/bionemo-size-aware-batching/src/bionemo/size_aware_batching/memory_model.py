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
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

import torch


Data = TypeVar("Data")
Feature = TypeVar("Feature")


def collect_cuda_peak_alloc(
    model: torch.nn.Module,
    dataset: Iterable[Data],
    device: torch.device,
    feature_fn: Optional[Callable[[Data], Feature]] = None,
) -> Tuple[List[Feature], List[int]]:
    """
    Collects CUDA peak memory allocation statistics for a given model and dataset.
    This function iterates through the dataset, runs the model's forward and backward
    on each data sample, and records the peak CUDA memory allocation during this process.

    Args:
        model (torch.nn.Module): The PyTorch model to collect memory usage for.
        dataset (Iterable[Data]): The dataset to iterate over.
        device (torch.device): The device to run the model on (e.g. GPU or CPU).
        feature_fn (Optional[Callable[[Data], Any]], optional):
            A function that takes a data point and returns a feature representation
            to be used in modelling the memory allocation for that data point.
            Its return must not consume significant amount of GPU memory to avoid
            contaminating memory samples.
            If None, the original data points will be used. Defaults to None.

    Returns:
        Tuple[List[Any], List[int]]: A tuple containing the collected memory features and memory usage statistics.
    """
    model = model.to(device)
    # warm up a few rounds before collecting stats
    n_warmup = 2
    for i_warmup, data in enumerate(dataset):
        try:
            y = model(data.to(device))
            y.backward()
        except torch.cuda.OutOfMemoryError:
            print("Encounter CUDA out-of-memory error. Skipping sample", file=sys.stderr, flush=True)
            continue
        except Exception as e:
            raise e
        finally:
            del data
            if "y" in locals():
                del y
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            if i_warmup == n_warmup - 1:
                break
    torch.cuda.reset_peak_memory_stats(device)
    features = []
    alloc_peaks = []
    for data in dataset:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            y = model(data.to(device))
            y.backward()
            alloc_peak = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
            alloc_peaks.append(alloc_peak)
            if feature_fn is None:
                features.append(data.to(torch.device("cpu")))
            else:
                features.append(feature_fn(data))
        except torch.cuda.OutOfMemoryError:
            print("Encounter CUDA out-of-memory error. Skipping sample", file=sys.stderr, flush=True)
            continue
        except Exception as e:
            raise e
        finally:
            del data
            if "y" in locals():
                del y
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    return features, alloc_peaks


def polynomial_regression(degree: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Performs polynomial regression on the given data points.

    This function takes in a 1D tensor of input values `x`, a 1D tensor of output values `y`,
    and an integer degree for the polynomial. It returns a tensor of coefficients
    that represent the best-fit polynomial of the specified degree.

    The function uses least squares regression to fit the polynomial.

    Args:
        degree (int): The degree of the polynomial.
        x (torch.Tensor): A 1D tensor of features, where the dimension is along the sample.
        y (torch.Tensor): A 1D tensor of labels, where the dimension is along the sample.

    Returns:
        torch.Tensor: A tensor of coefficients representing the best-fit polynomial.

    Raises:
        TypeError: If `x` or `y` is not a 1D tensor or if `x` and `y` do not have the same length.
        ValueError: If `x` and `y` are not floating point tensors.
    """
    if degree < 0:
        raise ValueError("The degree of the polynomial must be non-negative.")
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
    polynomials = x.unsqueeze(-1).repeat(1, degree + 1)
    polynomials[:, 0] = 1
    polynomials = torch.cumprod(polynomials, dim=-1)
    coeffs = torch.linalg.lstsq(polynomials, y).solution
    return coeffs
