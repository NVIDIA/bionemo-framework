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

import pytest
import torch
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends

from esm.collator import MLMDataCollatorWithFlattening
from esm.modeling_esm_te import NVEsmForMaskedLM


@pytest.fixture
def input_data_thd(tokenizer, tokenized_proteins):
    data_collator = MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        seed=42,
        bshd_equivalent=True,
        bshd_pad_to_multiple_of=256,
    )
    return data_collator(tokenized_proteins)


def test_thd_from_collator_output(te_model_checkpoint, input_data_thd):
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_thd.to("cuda")
    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model_thd(**input_data_thd, output_hidden_states=True)

    assert outputs.loss < 3.0


def test_thd_values_match(te_model_checkpoint, input_data, input_data_thd):
    torch.testing.assert_close(
        input_data["input_ids"][input_data["attention_mask"].to(bool)],
        input_data_thd["input_ids"].flatten(0),
    )

    torch.testing.assert_close(
        input_data["labels"][input_data["attention_mask"].to(bool)],
        input_data_thd["labels"].flatten(0),
    )

    model_bshd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_bshd.to("cuda")
    model_thd.to("cuda")

    input_data_bshd = {k: v.to("cuda") for k, v in input_data.items()}
    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}

    bshd_outputs = model_bshd(**input_data_bshd)
    thd_outputs = model_thd(**input_data_thd)

    torch.testing.assert_close(bshd_outputs.loss, thd_outputs.loss)

    # bshd_logits = bshd_outputs.logits[input_data_bshd["attention_mask"].to(bool)]
    # TODO(BIONEMO-2801): Investigate why these are not close on sm89 but pass on sm120.
    # torch.testing.assert_close(bshd_logits, thd_outputs.logits)


def test_thd_backwards(te_model_checkpoint, input_data_thd, monkeypatch):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
        _attention_backends["backend_selection_requires_update"] = True

    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_thd.to("cuda")
    input_data = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    outputs = model_thd(**input_data)
    outputs.loss.backward()
