# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pickle

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support, check_mxfp8_support


def requires_fp8(func):
    """Decorator to skip tests that require FP8 support."""
    fp8_available, reason = check_fp8_support()
    return pytest.mark.skipif(not fp8_available, reason=f"FP8 is not supported on this GPU: {reason}")(func)


def requires_mxfp8(func):
    """Decorator to skip tests that require MXFP8 support."""
    mxfp8_available, reason = check_mxfp8_support()
    if torch.cuda.get_device_capability() == (12, 0):
        mxfp8_available = False
        reason = "MXFP8 is not supported on sm120"
    return pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8 is not supported on this GPU: {reason}")(func)


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


if __name__ == "__main__":
    import transformer_engine.pytorch
    from transformer_engine.pytorch.fp8 import DelayedScaling, Format

    from esm.modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank())

    config = NVEsmConfig.from_pretrained("nvidia/esm2_t6_8M_UR50D", dtype=torch.bfloat16)
    model = NVEsmForMaskedLM(config)
    model = model.to(torch.cuda.current_device())
    model = torch.nn.parallel.DistributedDataParallel(model)

    model.train()

    generator = torch.Generator()
    generator.manual_seed(torch.distributed.get_rank())

    fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")

    for _ in range(3):
        input_data = {
            "input_ids": torch.randint(0, config.vocab_size, (1, 32), generator=generator),
            "labels": torch.randint(0, config.vocab_size, (1, 32), generator=generator),
            "attention_mask": torch.ones(1, 32),
        }
        input_data = {k: v.to(torch.cuda.current_device()) for k, v in input_data.items()}

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                outputs = model(**input_data)

        outputs.loss.backward()

    fp8_extra_states = {key: val for key, val in model.state_dict().items() if key.endswith("_extra_state")}
    outputs_list = [None] * torch.distributed.get_world_size() if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(fp8_extra_states, outputs_list, dst=0)
    if torch.distributed.get_rank() == 0:
        assert outputs_list is not None

        for key in outputs_list[0]:
            state_1 = outputs_list[0][key]
            state_2 = outputs_list[1][key]
            dict_1 = pickle.loads(state_1.detach().numpy(force=True).tobytes())
            dict_2 = pickle.loads(state_2.detach().numpy(force=True).tobytes())
            recipe_1 = dict_1.pop("recipe", None)
            recipe_2 = dict_2.pop("recipe", None)
            torch.testing.assert_close(dict_1, dict_2)
            assert recipe_1 == recipe_2

    torch.distributed.destroy_process_group()
