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


import datetime
import hashlib
import multiprocessing
import os
import random
from typing import Callable, Iterator, List, Optional, Tuple

import torch
from nemo.utils import logging
from tqdm import tqdm


_GPU_ARCH_TYPE = None


def datetime_from_string(
    datetime_string: str,
    datetime_format: str = "%Y-%m-%d %H:%M:%S",
) -> datetime.datetime:
    """Converts string to datetime object."""
    return datetime.datetime.strptime(datetime_string, datetime_format)


def get_seed_from_string(s: str) -> int:
    """Hashes input string and returns uint64-like integer seed value."""
    rng = random.Random(s)
    seed = rng.getrandbits(64)
    return seed


def get_seed_randomly() -> int:
    """Returns truly pseduorandom uint64-like integer seed value."""
    rng = random.Random(None)
    seed = rng.getrandbits(64)
    return seed


def hash_string_into_number(s: str) -> int:
    """Hashes string into uint64-like integer number."""
    b = s.encode("utf-8")
    d = hashlib.sha256(b).digest()
    i = int.from_bytes(d[:8], byteorder="little", signed=False)
    return i


def all_equal(values: list) -> bool:
    """Checks if all values in list are equal to each other."""
    if not values:
        return True
    first_val = values[0]
    for val in values:
        if val != first_val:
            return False
    return True


def list_zip(*arglists) -> list:
    """Transforms given columns into list of rows."""
    if len(arglists) == 0:
        return []
    lengths = [len(arglist) for arglist in arglists]
    if not all_equal(lengths):
        raise ValueError(f"unequal list lengths: {lengths}")
    return list(zip(*arglists))


def split_list_into_n_chunks(arglist: list, n: int) -> Iterator[list]:
    """Splits list into given number of chunks."""
    assert len(arglist) >= 0
    assert n > 0
    min_chunk_size, remainder = divmod(len(arglist), n)
    left = 0
    for i in range(n):
        right = left + min_chunk_size
        if i < remainder:
            right += 1
        yield arglist[left:right]
        left = right


def flatten_list(arglist: list) -> list:
    return [element for sublist in arglist for element in sublist]


def map_dict_values(fn: Callable, d: dict) -> dict:
    """Maps dictionary values using given function."""
    return {k: fn(v) for k, v in d.items()}


def map_tree_leaves(fn: Callable, tree: dict, leaf_type: Optional[type] = None) -> dict:
    """Maps tree leaf nodes using given function."""
    output = {}
    assert isinstance(tree, dict)
    for k, v in tree.items():
        if isinstance(v, dict):
            # non-leaf node encountered -> recursive call
            output[k] = map_tree_leaves(fn, v)
        elif leaf_type is None:
            # leaf type not specified -> apply function
            output[k] = fn(v)
        elif isinstance(v, leaf_type):
            # leaf type specified and matches -> apply function
            output[k] = fn(v)
        else:
            # leaf type specified and doesn't match -> identity
            output[k] = v
    return output


def slice_generator(start: int, end: int, size: int) -> Iterator[Tuple[int, int]]:
    """Returns slice indices iterator from start to end."""
    for i in range(start, end, size):
        left = i
        right = min(i + size, end)
        yield left, right


def apply_func_parallel(
    func: Callable,
    args_list: List[tuple],
    num_parallel_processes: int,
) -> list:
    if not isinstance(args_list, list):
        raise TypeError(f"args_list is of type {type(args_list)}, but it should be of type {list}.")
    for args in args_list:
        if not isinstance(args, tuple):
            raise TypeError(f"args is of type {type(args)}, but it should be of type {tuple}.")

    if num_parallel_processes > 0:
        async_results = []
        pool = multiprocessing.Pool(num_parallel_processes)
        for args in args_list:
            ar = pool.apply_async(func, args)
            async_results.append(ar)
        results = [ar.get() for ar in tqdm(async_results)]
        pool.close()
        pool.join()

    else:
        results = []
        for args in tqdm(args_list):
            r = func(*args)
            results.append(r)

    return results


def collate(samples: List[dict]) -> dict:
    """Converts list of samples into a batch dict."""
    assert isinstance(samples, list)
    assert len(samples) > 0
    sample0 = samples[0]
    assert isinstance(sample0, dict)
    batch = {}
    for key in list(sample0.keys()):
        batch[key] = [sample[key] for sample in samples]
        if isinstance(sample0[key], torch.Tensor):
            batch[key] = torch.stack(batch[key], dim=0)
    return batch


def _set_gpu_arch_type() -> None:
    global _GPU_ARCH_TYPE
    dgxsystem = os.environ.get("DGXSYSTEM", "")
    if dgxsystem.startswith("DGXH100"):
        _GPU_ARCH_TYPE = "hopper"
    elif dgxsystem.startswith("DGXA100"):
        _GPU_ARCH_TYPE = "ampere"
    else:
        # assume hopper if reached here:
        # TODO: this should probably be changed to ampere; mlperf assumes hopper
        _GPU_ARCH_TYPE = "hopper"
        logging.warning("could not recognize GPU architecture in the current environment" " - assuming hopper")


def is_hopper_arch() -> bool:
    global _GPU_ARCH_TYPE
    if _GPU_ARCH_TYPE is None:
        _set_gpu_arch_type()
    return _GPU_ARCH_TYPE == "hopper"


def is_ampere_arch() -> bool:
    global _GPU_ARCH_TYPE
    if _GPU_ARCH_TYPE is None:
        _set_gpu_arch_type()
    return _GPU_ARCH_TYPE == "ampere"
