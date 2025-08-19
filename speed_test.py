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

def do_generate(n=300):
    from bionemo.evo2.inference import get_model_and_tokenizer
    inference_wrapped_model, mcore_tokenizer = get_model_and_tokenizer(ckpt_name="evo2/7b-1m:1.0")

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
