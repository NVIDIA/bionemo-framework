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

import peft
import torch

from esm.modeling_esm_te import NVEsmForMaskedLM


def test_create_peft_model(te_model_checkpoint):
    model = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        target_parameters=["layernorm_qkv.weight"],
        bias="none",
    )

    peft_model = peft.get_peft_model(model, peft_config)
    status = peft_model.get_model_status()
    assert status.trainable_params == 145_024
    assert status.total_params == 7_645_184
