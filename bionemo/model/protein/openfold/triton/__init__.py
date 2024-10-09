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


# TODO: Remove this file once Apex version in container contains at least SHA 58acf96915eecd7e13adff61d2c389fba3efede2
# © 2023 NVIDIA CORPORATION & AFFILIATES

import pickle
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
from typing import BinaryIO, Union

import torch
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.jit import JITFunction

from bionemo.data.protein.openfold.helpers import is_ampere_arch, is_hopper_arch
from bionemo.model.protein.openfold.triton._layer_norm_backward_kernels import (
    _layer_norm_backward_dw_db_partial,
    _layer_norm_backward_dw_db_partial_strided,
    _layer_norm_backward_dx,
    _layer_norm_backward_dx_strided,
)
from bionemo.model.protein.openfold.triton._layer_norm_config_ampere import _auto_tuned_config_ampere
from bionemo.model.protein.openfold.triton._layer_norm_config_hopper import _auto_tuned_config_hopper
from bionemo.model.protein.openfold.triton._layer_norm_forward_kernels import (
    _layer_norm_forward,
    _layer_norm_forward_strided,
)
from bionemo.model.protein.openfold.triton.layer_norm import LayerNormSmallShapeOptImpl
from bionemo.model.protein.openfold.triton.mha import (
    AttnBiasJIT,
    AttnNoBiasJIT,
    AttnTri,
    CanSchTriMHA,
)


__all__ = (
    "LayerNormSmallShapeOptImpl",
    "sync_triton_auto_tune_cache_across_gpus",
    "CanSchTriMHA",
    "AttnTri",
    "AttnBiasJIT",
    "AttnNoBiasJIT",
)


def _get_tuneable_triton_func_name(f: Union[Autotuner, Heuristics, JITFunction]) -> str:
    if isinstance(f, JITFunction):
        return f.__name__
    else:
        return _get_tuneable_triton_func_name(f.fn)


_tuneable_triton_kernels = OrderedDict(
    (_get_tuneable_triton_func_name(func), func)
    for func in (
        _layer_norm_backward_dw_db_partial,
        _layer_norm_backward_dw_db_partial_strided,
        _layer_norm_backward_dx,
        _layer_norm_backward_dx_strided,
        _layer_norm_forward,
        _layer_norm_forward_strided,
    )
)


def _save_triton_auto_tune_cache(f: BinaryIO, verbose: bool = False) -> None:
    caches = OrderedDict()
    for func_name, func in _tuneable_triton_kernels.items():
        if len(func.cache) < 1:
            raise ValueError(f"Triton JIT kernel {func.__name__} didn't have tuning cache")
        caches[func_name] = deepcopy(func.cache)
    pickle.dump(caches, f)
    if verbose:
        print(f"Triton kernel auto-tuning caches written to {f}")


def _load_triton_auto_tune_cache(f: BinaryIO, strict: bool = True, verbose: bool = False) -> None:
    caches = pickle.load(f)
    if strict:
        loaded_func_name = set(caches.keys())
        tuneable_func_name = set(_tuneable_triton_kernels.keys())
        if loaded_func_name != tuneable_func_name:
            raise ValueError(
                f"Tuneable Triton kernels don't match with provided auto-tuning cache file {f}\n"
                f"Missing kernel caches: {tuneable_func_name - loaded_func_name}\n"
                f"Unexpected kernel caches: {loaded_func_name - tuneable_func_name}"
            )
    for func_name, cache in caches.items():
        if func_name not in _tuneable_triton_kernels:
            raise ValueError(f"{func_name} from {f} doesn't match any tuneable Triton kernels")
        _tuneable_triton_kernels[func_name].cache = cache
    if verbose:
        print(f"Triton kernel auto-tuning caches loaded from {f}")


def load_triton_auto_tuned_cache_params(dap_size: int, arch_type: str = None, verbose: bool = True) -> None:
    if arch_type is None:
        if is_hopper_arch():
            arch_type = "hopper"
        elif is_ampere_arch():
            arch_type = "ampere"
        else:
            arch_type = "hopper"
    if arch_type not in ("hopper", "ampere"):
        raise ValueError(f"Unknown arch type {repr(arch_type)}")
    if verbose:
        print(f"Loading auto-tuned Triton configs for DAP={dap_size} and arch_type={arch_type}...")
    auto_tuned_config = {
        "hopper": _auto_tuned_config_hopper,
        "ampere": _auto_tuned_config_ampere,
    }[arch_type]
    config_for_current_dap = auto_tuned_config[dap_size]
    for func_name, cache in config_for_current_dap.items():
        _tuneable_triton_kernels[func_name].cache = cache
    if verbose:
        print("Loaded auto-tuned Triton configs successfully!")


def sync_triton_auto_tune_cache_across_gpus() -> None:
    if not torch.distributed.is_initialized():
        return
    if torch.distributed.get_rank() == 0:
        print("Broadcasting Triton auto-tuning cache from rank 0 to other ranks...")
        cache = BytesIO()
        _save_triton_auto_tune_cache(cache)
        cache.seek(0)
        cache_list = [
            cache,
        ]
    else:
        print(f"Rank {torch.distributed.get_rank()} is waiting for Triton auto-tuning cache from rank 0...")
        cache_list = [
            None,
        ]
    torch.distributed.broadcast_object_list(cache_list)
    cache = cache_list[0]
    _load_triton_auto_tune_cache(cache)
    print("Succeed!")
