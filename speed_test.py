# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse
from typing import Literal, Optional
import time
import nemo.lightning as nl
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm import generate
from nemo.utils import logging

def get_inference_params():
    # Below are parameters that need to be tuned based on GPU memory size.
    return {
        # Affects KV cache size, can cause OOM easily, set to 1 to save memory (i.e. running 40b model)
        "inference_max_requests": 1,
    }

def get_trainer(pipeline_parallel=1):
    import nemo.lightning as nl

    fp8 = True
    full_fp8 = False
    return nl.Trainer(
        accelerator="gpu",
        devices=pipeline_parallel,
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pipeline_parallel,
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

def get_model_and_tokenizer(ckpt_dir="ckpt/7b", vortex_style_fp8=True):
    trainer = get_trainer()
    from nemo.collections.llm import inference

    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=ckpt_dir,
        trainer=trainer,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=8192,  # TODO
        inference_max_seq_length=8192,  # TODO
        vortex_style_fp8=vortex_style_fp8,
        fp32_residual_connection=False,
        # use_te_rng_tracker=True,
        # te_rng_tracker=True,
        # inference_rng_tracker=True,
        # enable_cuda_graph=True,
        # cudagraph_rng_tracker=True,
        enable_flash_decode=True,
        cuda_graph=True,
        recompute_granularity=None,
        recompute_num_layers=None,
        recompute_method=None,
        **get_inference_params(),
    )
    return inference_wrapped_model, mcore_tokenizer

def do_generate(n=300):
    inference_wrapped_model, mcore_tokenizer = get_model_and_tokenizer()

    from megatron.core.inference.common_inference_params import CommonInferenceParams
    from nemo.collections.llm.inference import generate
    _ = generate(
        model=inference_wrapped_model,
        max_batch_size=1,  # vortex only supports batch size 1
        tokenizer=mcore_tokenizer,
        prompts=["AA"],
        random_seed=42,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=5,
        ),
    )

    for _ in range(3):
        t0 = time.perf_counter_ns()
        results = generate(
            model=inference_wrapped_model,
            max_batch_size=1,  # vortex only supports batch size 1
            tokenizer=mcore_tokenizer,
            prompts=["AAA"],
            random_seed=42,
            inference_params=CommonInferenceParams(
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                return_log_probs=False,
                num_tokens_to_generate=n,
            ),
        )
        dt = (time.perf_counter_ns() - t0) / 1e9  # seconds
        tokens_per_sec = (len(results[0].generated_text)+1) / dt  # +1 for the prompt
        print(results[0].generated_text)
        print(tokens_per_sec)

# got bitten so many times by measuring the wrong thing, so make sure we assert
# preconditions.
def assert_cpu_speed():
    from pathlib import Path
    cpu = Path("/sys/devices/system/cpu")
    for cpufreq in cpu.glob("cpu*/cpufreq/"):
        gov = cpufreq/"scaling_governor"
        assert gov.read_text().strip() == "userspace", f"Please set {gov} to userspace"
        minfreq = cpufreq/"cpuinfo_min_freq"
        setfreq = cpufreq/"scaling_setspeed"
        assert setfreq.read_text() == minfreq.read_text(), f"Please set {setfreq} to {minfreq}"
#assert_cpu_speed()

if __name__ == "__main__":
    with torch.inference_mode():
        do_generate()
