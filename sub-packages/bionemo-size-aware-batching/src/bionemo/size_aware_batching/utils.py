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
from typing import Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TypeVar

import torch


__all__: Sequence[str] = ("collect_cuda_peak_alloc", "create_buckets", "Buckets")

Data = TypeVar("Data")
Feature = TypeVar("Feature")
TorchIntegerDataTypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


class Buckets(NamedTuple):
    """A container for storing bucket boundaries and sizes.

    Attributes:
        bucket_boundaries (torch.Tensor): A 1D tensor with the boundaries of all the bucket.
        bucket_sizes (torch.Tensor): The number of elements in each bucket.
    """

    bucket_boundaries: torch.Tensor
    bucket_sizes: torch.Tensor


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


def create_buckets(sizes: torch.Tensor, max_width: int, min_bucket_count: int) -> Buckets:
    """Create buckets for a list of integers with pre-defined maximal width of interval and minimal bucket count.

    It will return a named tuple containing the bucket boundaries and the actual bucket sizes.
    e.g. torch.tensor([0, 5, 7]), torch.tensor([3,2]): specifies 2 buckets: one with range 0<= sizes < 5, width=5 and 3 elements
    and the other one with range 5 <= sizes < 7, width=2 and 2 elements.


    Args:
        sizes: An 1D tensor of integers.
        max_width: The maximum width of a bucket, should be a positive integer.
        min_bucket_count: The minimum count of a bucket, should be a positive integer.
            Bucket size may be smaller than min_bucket_count if its width reaches max_width.

    Raises:
        ValueError: If the provided sizes is empty, or not integers.
        ValueError: If max_width is not a positive integer or min_bucket_count is not a positive integer.

    Returns:
        A namedtuple containing bucket boundaries in ascending order and the number of elements in each bucket.

    ---------

    Examples:
    ```python
    >>> import torch
    >>> from bionemo.size_aware_batching.utils import create_buckets

    >>> sizes = torch.tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 22, 22, 22, 22])
    >>> buckets = create_buckets(sizes, max_width=5, min_bucket_count=10)
    >>> # 5 buckets: 1 <= sizes < 6, 6 <= sizes < 11, 11 <= sizes < 16, 16 <= sizes < 21, 21 <= sizes < 23
    >>> print(buckets.bucket_boundaries)
    tensor([ 1,  6, 11, 16, 21, 23])

    >>> # each with 12, 0, 0, 0, 4 elements respectively.
    >>> print(buckets.bucket_sizes)
    tensor([12,  0,  0,  0,  4])

    >>> sizes = torch.arange(20)
    >>> # min_bucket_count is used to control bucket size
    >>> buckets = create_buckets(sizes, max_width=10, min_bucket_count=5)
    >>> print(buckets.bucket_boundaries)
    tensor([ 0,  5, 10, 15, 20])

    >>> print(buckets.bucket_sizes)
    tensor([5, 5, 5, 5])
    ```

    """
    if not torch.is_tensor(sizes):
        raise TypeError(f"sizes should be a torch tensor, but got sizes={sizes}")

    if sizes.ndim != 1:
        raise ValueError(f"sizes should be a 1D tensor, but got sizes with shape {sizes.shape}")

    if sizes.dtype not in TorchIntegerDataTypes:
        raise ValueError(f"sizes should contain only integers, but got sizes.dtype={sizes.dtype}")

    if len(sizes) == 0:
        raise ValueError("sizes should not be empty")

    if not isinstance(max_width, int) or max_width <= 0:
        raise ValueError(f"max_width should be a positive integer but got max_width={max_width}")

    if not isinstance(min_bucket_count, int) or min_bucket_count <= 0:
        raise ValueError(f"min_bucket_count should be a positive integer but got min_bucket_count={min_bucket_count}")

    unique_values, counts = torch.unique(sizes, return_counts=True, sorted=True)

    bucket_boundaries = [unique_values[0]]
    bucket_sizes = []
    start = 0
    end = 0
    upper_bound = unique_values[0] + 1
    bucket_count = 0

    while start < len(unique_values):
        while (
            end < len(unique_values)
            and bucket_count < min_bucket_count
            and unique_values[end] - bucket_boundaries[-1] < max_width
        ):
            bucket_count += counts[end]
            end += 1

        bucket_sizes.append(sum(counts[start:end]))
        if end == len(unique_values):
            upper_bound = unique_values[-1] + 1
        else:
            upper_bound = unique_values[end]

        # Adjust the end of the range to ensure that no width exceeds 'max_width'
        n_empty_buckets = (upper_bound - bucket_boundaries[-1]) // max_width
        if n_empty_buckets > 0:
            bucket_boundaries.extend(
                list(
                    range(
                        bucket_boundaries[-1] + max_width,
                        bucket_boundaries[-1] + max_width * (n_empty_buckets + 1),
                        max_width,
                    )
                )
            )
            bucket_sizes.extend([0] * (n_empty_buckets - 1))
        else:
            bucket_boundaries.append(upper_bound)

        start = end
        end = start + 1
        # index start may be out of bounds
        bucket_count = counts[start:end].sum()

    bucket_boundaries = torch.tensor(bucket_boundaries)
    bucket_sizes = torch.tensor(bucket_sizes)

    return Buckets(bucket_boundaries, bucket_sizes)
