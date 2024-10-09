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
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

import torch


__all__: Sequence[str] = ("collect_cuda_peak_alloc",)

Data = TypeVar("Data")
Feature = TypeVar("Feature")


def collect_cuda_peak_alloc(
    dataset: Iterable[Data],
    work: Callable[[Data], Feature],
    device: torch.device,
    cleanup: Optional[Callable[[], None]] = None,
) -> Tuple[List[Feature], List[int]]:
    """Collects CUDA peak memory allocation statistics for a given workflow.

    This function iterates through the provided dataset, applies the given feature function to each data point,
    and records the peak CUDA memory allocation during this process. The features extracted from the data points
    are collected along with their corresponding memory usage statistics.

    Note that the first few iterations of the workflow might result in smaller memory allocations due to uninitialized
    data (e.g., internal PyTorch buffers). Therefore, users may want to skip these initial data points when analyzing the results.

    Args:
        dataset: An iterable containing the input data.
        work: A function that takes a data point and returns its corresponding feature. This is where
            the main computation happens and memory allocations are tracked.
        device: The target Torch CUDA device.
        cleanup: A function that is called after each iteration to perform any necessary cleanup.

    Returns:
        A tuple containing the collected features and their corresponding memory usage statistics.

    Raises:
        ValueError: If the provided device is not a CUDA device.

    -------

    Examples:
    ```python
    >>> import torch
    >>> from bionemo.size_aware_batching.utils import collect_cuda_peak_alloc


    >>> # prepare dataset, model and other components of a workflow
    >>> # for which the user want to collect CUDA peak memory allocation statistics
    >>> dataset, model, optimizer = ...
    >>> # Set the target Torch CUDA device.
    >>> device = torch.device("cuda:0")
    >>> model = model.to(device)

    >>> # Define a function that takes an element of the dataset as input and
    >>> # do a training step
    >>> def work(data):
    ...     # example body of a training loop
    ...     optimizer.zero_grad()
    ...     output = model(data.to(device))
    ...     loss = compute_loss(output)
    ...     loss.backward()
    ...     optimizer.step()
    ...     # extract the feature for later to be modeled or analyzed
    ...     return featurize(data)

    >>> # can optionally use a cleanup function to release the references
    >>> # hold during the work(). This cleanup function will be called
    >>> # at the end of each step before garbage collection and memory allocations measurement
    >>> def cleanup():
    ...     model.zero_grad(set_to_none=True)

    >>> # Collect features (i.e., model outputs) and memory usage statistics for the workflow.
    >>> features, alloc_peaks = collect_cuda_peak_alloc(
    ...     dataset=batches,
    ...     work=work,
    ...     device=device,
    ...     cleanup=cleanup,
    ... )


    >>> # use features and alloc_peaks as needed, e.g., fit a model
    >>> # that can use these statistics to predict memory usage
    >>> memory_model = ...
    >>> memory_model.fit(features, alloc_peaks)
    ```


    """
    if device.type != "cuda":
        raise ValueError("This function is intended for CUDA devices only.")

    features = []
    alloc_peaks = []

    for data in dataset:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            feature = work(data)
            alloc_peak = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
            alloc_peaks.append(alloc_peak)
            features.append(feature)
        except torch.cuda.OutOfMemoryError:
            print("Encounter CUDA out-of-memory error. Skipping sample", file=sys.stderr, flush=True)
            continue
        finally:
            # ensures cleanup is done next round even in case of exception
            del data
            if "feature" in locals():
                del feature
            if cleanup is not None:
                cleanup()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
    return features, alloc_peaks
