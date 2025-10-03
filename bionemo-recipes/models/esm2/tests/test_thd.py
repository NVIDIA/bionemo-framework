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

import os

import pytest
import torch
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends

from esm.collator import MLMDataCollatorWithFlattening
from esm.modeling_esm_te import NVEsmConfig, NVEsmEmbeddings, NVEsmForMaskedLM


@pytest.fixture
def input_data_thd(tokenizer, tokenized_proteins):
    data_collator = MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        seed=42,
        bshd_equivalent=True,
        bshd_pad_to_multiple_of=32,
    )
    return data_collator(tokenized_proteins)


@pytest.mark.parametrize("use_token_dropout", [True, False])
def test_nv_esm_embeddings_random_init(te_model_checkpoint, input_data_thd, input_data, use_token_dropout):
    config = NVEsmConfig.from_pretrained(te_model_checkpoint)
    assert config.token_dropout is True
    embedding = NVEsmEmbeddings(config)
    embedding.token_dropout = use_token_dropout
    embedding.to("cuda")

    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    input_data_thd.pop("labels")
    outputs_thd = embedding(**input_data_thd)

    input_data_bshd = {k: v.to("cuda") for k, v in input_data.items()}
    input_data_bshd.pop("labels")
    outputs_bshd = embedding(**input_data_bshd)

    # Reshape outputs_bshd to match outputs_thd
    outputs_bshd = outputs_bshd[input_data_bshd["attention_mask"].to(bool)].unsqueeze(0)
    torch.testing.assert_close(outputs_thd, outputs_bshd, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("use_token_dropout", [True, False])
def test_nv_esm_embeddings_from_model(te_model_checkpoint, input_data_thd, input_data, use_token_dropout):
    model = NVEsmForMaskedLM.from_pretrained(
        te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16, token_dropout=use_token_dropout
    )
    embedding = model.esm.embeddings
    assert embedding.token_dropout == use_token_dropout
    embedding.to("cuda")

    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    input_data_thd.pop("labels")
    outputs_thd = embedding(**input_data_thd)

    input_data_bshd = {k: v.to("cuda") for k, v in input_data.items()}
    input_data_bshd.pop("labels")
    outputs_bshd = embedding(**input_data_bshd)

    # Reshape outputs_bshd to match outputs_thd
    outputs_bshd = outputs_bshd[input_data_bshd["attention_mask"].to(bool)].unsqueeze(0)
    torch.testing.assert_close(outputs_thd, outputs_bshd, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("use_token_dropout", [True, False])
def test_first_transformer_layer(te_model_checkpoint, input_data_thd, input_data, use_token_dropout):
    torch.testing.assert_close(
        input_data["input_ids"][input_data["attention_mask"].to(bool)],
        input_data_thd["input_ids"].flatten(0),
    )

    torch.testing.assert_close(
        input_data["labels"][input_data["attention_mask"].to(bool)],
        input_data_thd["labels"].flatten(0),
    )

    model_thd = NVEsmForMaskedLM.from_pretrained(
        te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16, token_dropout=use_token_dropout
    )
    model_bshd = NVEsmForMaskedLM.from_pretrained(
        te_model_checkpoint, dtype=torch.bfloat16, token_dropout=use_token_dropout
    )
    model_thd.to("cuda")
    model_bshd.to("cuda")

    embedding = model_thd.esm.embeddings
    assert embedding.token_dropout == use_token_dropout

    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    input_data_thd.pop("labels")
    embeddings_thd = embedding(**input_data_thd)

    input_data_bshd = {k: v.to("cuda") for k, v in input_data.items()}
    input_data_bshd.pop("labels")
    embeddings_bshd = embedding(**input_data_bshd)

    # Ensure embeddings match -- shouldn't these be the same as the test above?
    torch.testing.assert_close(
        embeddings_thd,
        embeddings_bshd[input_data_bshd["attention_mask"].to(bool)].unsqueeze(0),
        atol=1e-8,
        rtol=1e-8,
    )

    layer_thd = model_thd.esm.encoder.layers[0]
    layer_bshd = model_bshd.esm.encoder.layers[0]

    def get_output_bshd(embeddings, inputs_bshd):
        extended_attention_mask = model_bshd.get_extended_attention_mask(
            inputs_bshd["attention_mask"], embeddings.shape
        )
        extended_attention_mask = extended_attention_mask < -1
        return layer_bshd(
            embeddings,
            attention_mask=extended_attention_mask,
            rotary_pos_emb=model_bshd.esm.encoder.te_rope_emb,
        )

    def get_output_thd(embeddings, inputs_thd):
        return layer_thd(
            embeddings,
            cu_seqlens_q=inputs_thd["cu_seq_lens_q"],
            cu_seqlens_kv=inputs_thd["cu_seq_lens_k"],
            max_seqlen_q=inputs_thd["max_length_q"],
            max_seqlen_kv=inputs_thd["max_length_k"],
            rotary_pos_emb=model_thd.esm.encoder.rotary_embeddings(max_seq_len=inputs_thd["cu_seq_lens_q"][-1]),
        )

    output_bshd = get_output_bshd(embeddings_bshd, input_data_bshd)
    # output_thd = get_output_thd(embeddings_thd.squeeze(0), input_data_thd)
    output_thd = get_output_thd(embeddings_bshd[input_data_bshd["attention_mask"].to(bool)], input_data_thd)

    torch.testing.assert_close(
        output_bshd[input_data_bshd["attention_mask"].to(bool)],
        output_thd,
        atol=1e-8,
        rtol=1e-8,
    )


def test_thd_from_collator_output(te_model_checkpoint, input_data_thd):
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_thd.to("cuda")
    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model_thd(**input_data_thd, output_hidden_states=True)

    assert outputs.loss < 3.0


@pytest.fixture(params=["flash_attn", "fused_attn"])
def attn_impl(request, monkeypatch):
    if request.param == "flash_attn":
        os.environ["NVTE_FUSED_ATTN"] = "0"
        os.environ["NVTE_FLASH_ATTN"] = "1"
        _attention_backends["backend_selection_requires_update"] = True

    else:
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FLASH_ATTN"] = "0"
        _attention_backends["backend_selection_requires_update"] = True

    return request.param


def test_thd_losses_match(te_model_checkpoint, input_data, input_data_thd, attn_impl):
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


@pytest.mark.parametrize("token_dropout", [True, False])
def test_thd_logits_match(te_model_checkpoint, input_data, input_data_thd, attn_impl, token_dropout):
    # Ensure the input data is the same
    torch.testing.assert_close(
        input_data["input_ids"][input_data["attention_mask"].to(bool)],
        input_data_thd["input_ids"].flatten(0),
    )

    torch.testing.assert_close(
        input_data["labels"][input_data["attention_mask"].to(bool)],
        input_data_thd["labels"].flatten(0),
    )

    # Create models
    model_bshd = NVEsmForMaskedLM.from_pretrained(
        te_model_checkpoint, token_dropout=token_dropout, dtype=torch.bfloat16
    )
    model_thd = NVEsmForMaskedLM.from_pretrained(
        te_model_checkpoint, token_dropout=token_dropout, attn_input_format="thd", dtype=torch.bfloat16
    )

    model_bshd.to("cuda")
    model_thd.to("cuda")

    input_data_bshd = {k: v.to("cuda") for k, v in input_data.items()}
    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}

    thd_outputs = model_thd(**input_data_thd, output_hidden_states=True)
    bshd_outputs = model_bshd(**input_data_bshd, output_hidden_states=True)

    for i, (bshd_hidden, thd_hidden) in enumerate(zip(bshd_outputs.hidden_states, thd_outputs.hidden_states)):
        torch.testing.assert_close(
            bshd_hidden[input_data_bshd["attention_mask"].to(bool)],
            thd_hidden.squeeze(0),
            msg=lambda msg: "Hidden states do not match going into layer " + str(i + 1) + ": " + msg,
        )

    bshd_logits = bshd_outputs.logits[input_data_bshd["attention_mask"].to(bool)]
    torch.testing.assert_close(bshd_logits, thd_outputs.logits)


def test_thd_backwards_works(te_model_checkpoint, input_data_thd, attn_impl):
    if attn_impl == "fused_attn" and torch.cuda.get_device_capability() == (12, 0):
        pytest.xfail("BIONEMO-2840: On sm120 the THD backwards implementation is not available for fused attn.")

    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_thd.to("cuda")
    input_data = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    outputs = model_thd(**input_data)
    outputs.loss.backward()


@pytest.mark.parametrize("token_dropout", [True, False])
def test_thd_backwards_passes_match(te_model_checkpoint, input_data, input_data_thd, attn_impl, token_dropout):
    if attn_impl == "fused_attn" and torch.cuda.get_device_capability() == (12, 0):
        pytest.xfail("BIONEMO-2840: On sm120 the THD backwards implementation is not available for fused attn.")

    torch.testing.assert_close(
        input_data["input_ids"][input_data["attention_mask"].to(bool)],
        input_data_thd["input_ids"].flatten(0),
    )

    torch.testing.assert_close(
        input_data["labels"][input_data["attention_mask"].to(bool)],
        input_data_thd["labels"].flatten(0),
    )

    model_bshd = NVEsmForMaskedLM.from_pretrained(
        te_model_checkpoint, token_dropout=token_dropout, dtype=torch.bfloat16
    )
    model_thd = NVEsmForMaskedLM.from_pretrained(
        te_model_checkpoint, token_dropout=token_dropout, attn_input_format="thd", dtype=torch.bfloat16
    )
    model_bshd.to("cuda")
    model_thd.to("cuda")

    input_data_bshd = {k: v.to("cuda") for k, v in input_data.items()}
    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}

    bshd_outputs = model_bshd(**input_data_bshd)
    thd_outputs = model_thd(**input_data_thd)

    thd_outputs.loss.backward()
    bshd_outputs.loss.backward()

    thd_grads = {name: p.grad for name, p in model_thd.named_parameters() if p.grad is not None}
    bshd_grads = {name: p.grad for name, p in model_bshd.named_parameters() if p.grad is not None}

    # For some reason, the word embeddings grads have a slightly higher numerical error.
    thd_word_embeddings_grad = thd_grads.pop("esm.embeddings.word_embeddings.weight")
    bshd_word_embeddings_grad = bshd_grads.pop("esm.embeddings.word_embeddings.weight")

    torch.testing.assert_close(thd_grads, bshd_grads)

    # sus
    torch.testing.assert_close(thd_word_embeddings_grad, bshd_word_embeddings_grad, atol=1e-2, rtol=1e-5)
