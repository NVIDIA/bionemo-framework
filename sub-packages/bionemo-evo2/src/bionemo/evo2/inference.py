# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

from functools import lru_cache
import torch
import logging

log = logging.getLogger(__name__)

def ckpt_dir_to_name(ckpt_dir):
    from json import loads
    j = loads((ckpt_dir/"context"/"io.json").read_text())
    return next(
        x for x in j["objects"].keys()
        if x.startswith("hyena") and x.endswith("_config_1")
    )

@lru_cache
def getenv(var, default=None, *args, **kwargs):
    from os import getenv as _getenv
    val = _getenv(var, *args, **kwargs)
    if val is None:
        val = default
    else:
        log.info(f"Using env variable {var}={val}")
    if type(default) is bool:
        return str(val).lower() in ["y", "yes", "1", "t", "true"]
    if type(default) is not type(None):
        return type(default)(val)
    return val

def prune_caches(): # Helps to cleanup memory before various runs
    if getenv("EVO2_PRUNE_CACHES", True):
        return
    import gc
    gc.collect()
    torch.cuda.empty_cache()

@lru_cache
def detect_max_seq_len(ckpt_name):
    ret = getenv("EVO2_MAX_SEQ_LEN")
    if ret is not None:
        return int(ret)

    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
    gpus = torch.cuda.device_count()

    # For both 40B and 7B, values are somewhat conservative and made to
    # work for most cases; feel free to increase them via EVO2_MAX_SEQ_LEN
    # as needed.
    if "40b" in ckpt_name:
        if mem_gb > 120 and gpus >= 2: # e.g. h200-x2
            ret = 100_000
        elif mem_gb > 120: # e.g. h200-x1
            ret = 20_000
        elif mem_gb > 60 and gpus >= 2: # e.g. h100-x2
            ret = 100_000
        else:
            ret = 10_000
    else:
        if mem_gb > 40: # e.g. l40
            ret = 100_000
        else:
            ret = 20_000
    log.info(f"Guessed EVO2_MAX_SEQ_LEN={ret} {locals()}")
    return ret

@lru_cache
def detect_pipeline_parallel(ckpt_name):
    env_pp = getenv("EVO2_PIPELINE_PARALLEL")
    if env_pp is not None:
        return int(env_pp)

    gpus = torch.cuda.device_count()
    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3

    from math import ceil
    min_gpus = ceil(90/mem_gb) if "40b" in ckpt_name else 1
    if gpus < min_gpus:
        log.error(f"Not enough memory. Set EVO2_PIPELINE_PARALLEL to bypass. {locals()}")
        exit(1)

    is_testing = getenv("PYTEST_CURRENT_TEST") is not None
    if is_testing:
        ret = min_gpus # to speed up test start-up, avoid pp, if possible
    else:
        ret = gpus # use all GPUs in production: allows for longer sequences
    log.info(f"Guessed EVO2_PIPELINE_PARALLEL={ret}, {locals()}")
    return ret

def get_trainer(ckpt_name):
    import nemo.lightning as nl

    fp8 = True
    full_fp8 = False
    return nl.Trainer(
        accelerator="gpu",
        devices=detect_pipeline_parallel(ckpt_name),
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=detect_pipeline_parallel(ckpt_name),
            context_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )

def detect_pst(ckpt_name):
    ret = getenv("EVO2_PST")
    if ret is not None:
        return int(ret)

    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
    gpus = torch.cuda.device_count()

    ret = None
    if "40b" in ckpt_name:
        if gpus >= 2 and mem_gb > 120: # e.g. h200-x2
            ret = 8192
        elif mem_gb > 120: # e.g. h200-x1
            ret = 4096
        elif gpus >= 3 and mem_gb > 60: # e.g. h100-x3
            ret = 512
        elif gpus >= 2 and mem_gb > 60: # e.g. h100-x2
            ret = 256
        elif gpus >= 3: # e.g. l40-x3
            ret = 128
        else: # e.g. l40-x2
            ret = 64
    elif "7b" in ckpt_name:
        if mem_gb < 60:
            ret = 2048
    log.info(f"Guessed EVO2_PST={ret} {locals()}")
    return ret

def get_default_config(ckpt_name):
    # Below are parameters that need to be tuned based on GPU memory size.
    return {
        "params_dtype": torch.bfloat16,
        "vortex_style_fp8": getenv("EVO2_VORTEX_STYLE_FP8", True),
        "enable_flash_decode": getenv("EVO2_FLASH_DECODE", True),
        "fp32_residual_connection": getenv("EVO2_FP32_RESIDUAL_CONNECTION", False),
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "recompute_method": None,

        # Affects KV cache size, can cause OOM easily, set to 1 to save memory (i.e. running 40b model)
        "inference_max_requests": getenv("EVO2_INFERENCE_MAX_REQUESTS", 1),

        # This mostly determines size of KV cache.
        "inference_max_seq_length": detect_max_seq_len(ckpt_name),

        # This is used to split batch into mini-batches.
        # If use batch size = 1, set same as inference_max_seq_length to avoid
        # more complex code path.
        "inference_batch_times_seqlen_threshold": detect_max_seq_len(ckpt_name),

        # Affects max memory usage during parallel hyena filters pass.
        "prompt_segmentation_threshold": detect_pst(ckpt_name),
    }

@torch.inference_mode
def get_model_and_tokenizer(*, ckpt_name = None, ckpt_dir = None, **kwargs):
    prune_caches()

    if ckpt_dir is not None and ckpt_name is None:
        ckpt_name = ckpt_dir_to_name(ckpt_dir)

    from bionemo.core.data.load import load
    try:
        ckpt_dir: Path = load(ckpt_name) if ckpt_dir is None else ckpt_dir
    except ValueError as e:
        if "40b" in str(ckpt_name): # NeMo 40b is not yet published
            raise ValueError(f"Place 40b checkpoint to ~/.cache/bionemo/overrides/{ckpt_name}") from e
        raise

    from nemo.collections.llm import inference

    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=ckpt_dir,
        trainer=get_trainer(ckpt_name),
        **(get_default_config(ckpt_name) | kwargs),
    )
    return inference_wrapped_model, mcore_tokenizer
