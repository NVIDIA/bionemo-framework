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
import copy
import pytest
import torch
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp

from esm.collator import MLMDataCollatorWithFlattening
from esm.modeling_esm_te import NVEsmConfig, NVEsmEmbeddings, NVEsmForMaskedLM


compute_capability = torch.cuda.get_device_capability()



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

@pytest.fixture
def input_data_thd_padded_from_input_data_thd(input_data_thd):
    input_data_thd_padded = copy.deepcopy(input_data_thd)
    input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
        input_data_thd_padded["input_ids"],
        input_data_thd_padded["labels"],
        input_data_thd_padded["cu_seq_lens_q"],
        16,
        padding_token_id=1,
        padding_label_id=-100,
    )
    
    input_data_thd_padded["input_ids"] = input_ids_padded.unsqueeze(0)
    input_data_thd_padded["labels"] = labels_padded.unsqueeze(0)
    input_data_thd_padded["cu_seq_lens_q_padded"] = cu_seqlens_padded.to(torch.int32)
    input_data_thd_padded["cu_seq_lens_k_padded"] = cu_seqlens_padded.to(torch.int32)
    input_data_thd_padded["pad_between_seqs"] = True
    return input_data_thd_padded


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


def test_thd_from_collator_output(te_model_checkpoint, input_data_thd):
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_thd.to("cuda")
    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    with torch.no_grad():
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
    if attn_impl == "fused_attn" and torch.cuda.get_device_capability()[0] == 8:
        pytest.xfail("On Ada and Ampere, no THD implementation is available for fused attn.")

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


def test_thd_logits_match_with_bf16_autocast(te_model_checkpoint, input_data, input_data_thd, attn_impl):
    if attn_impl == "fused_attn" and torch.cuda.get_device_capability()[0] == 8:
        pytest.xfail("On Ada and Ampere, no THD implementation is available for fused attn.")
    elif attn_impl == "flash_attn" and torch.cuda.get_device_capability()[0] == 8:
        pytest.xfail("BIONEMO-2801: On Ada and Ampere, the flash attention logits don't seem to match.")

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
    model_bshd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)

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
            atol=1e-1 if compute_capability[0] == 8 else 1e-5,
            rtol=1.6e-2,
        )

        if compute_capability[0] == 8:
            break  # On Ada and Ampere, we see much larger numerical errors so we stop after the first layer

    bshd_logits = bshd_outputs.logits[input_data_bshd["attention_mask"].to(bool)]
    torch.testing.assert_close(bshd_logits, thd_outputs.logits, atol=1e-8, rtol=1e-8)


def test_thd_backwards_works(te_model_checkpoint, input_data_thd, attn_impl):
    if attn_impl == "fused_attn" and torch.cuda.get_device_capability() == (12, 0):
        pytest.xfail("BIONEMO-2840: On sm120 the THD backwards implementation is not available for fused attn.")
    elif attn_impl == "fused_attn" and torch.cuda.get_device_capability()[0] == 8:
        pytest.xfail("On Ada and Ampere, no THD implementation is available for fused attn.")

    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_thd.to("cuda")
    input_data = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    outputs = model_thd(**input_data)
    outputs.loss.backward()


def test_thd_backwards_passes_match(te_model_checkpoint, input_data, input_data_thd, attn_impl):
    if attn_impl == "fused_attn" and torch.cuda.get_device_capability() == (12, 0):
        pytest.xfail("BIONEMO-2840: On sm120 the THD backwards implementation is not available for fused attn.")
    elif attn_impl == "fused_attn" and torch.cuda.get_device_capability()[0] == 8:
        pytest.xfail("On Ada and Ampere, no THD implementation is available for fused attn.")

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

    thd_outputs.loss.backward()
    bshd_outputs.loss.backward()

    thd_grads = {name: p.grad for name, p in model_thd.named_parameters() if p.grad is not None}
    bshd_grads = {name: p.grad for name, p in model_bshd.named_parameters() if p.grad is not None}

    # max_diff_by_layer = {key: (thd_grads[key] - bshd_grads[key]).abs().max().item() for key in thd_grads.keys()}

    # For some reason, the word embeddings grads have a slightly higher numerical error.
    thd_word_embeddings_grad = thd_grads.pop("esm.embeddings.word_embeddings.weight")
    bshd_word_embeddings_grad = bshd_grads.pop("esm.embeddings.word_embeddings.weight")
    torch.testing.assert_close(
        thd_grads,
        bshd_grads,
        atol=1e-2 if compute_capability[0] == 8 else 1e-5,
        rtol=1.6e-2,
    )

    torch.testing.assert_close(thd_word_embeddings_grad, bshd_word_embeddings_grad, atol=1e-2, rtol=1e-5)


def test_thd_vs_padded_thd_equivalence(te_model_checkpoint, input_data_thd, input_data_thd_padded_from_input_data_thd, attn_impl):
    if attn_impl == "flash_attn":
        pytest.xfail("Flash attention is not supported for padded sequences.")
        
    input_data_thd_padded = input_data_thd_padded_from_input_data_thd
    seqlens_q = input_data_thd_padded["cu_seq_lens_q_padded"][1:] - input_data_thd_padded["cu_seq_lens_q_padded"][:-1]
    max_length_q = int((seqlens_q.max().item() + 63) // 64 * 64) # TODO(@jomitchell): Not sure if I need this anymore.
    max_length_k = max_length_q
    input_data_thd_padded["max_length_q"] = max_length_q
    input_data_thd_padded["max_length_k"] = max_length_k

    input_data_thd_gpu = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    input_data_thd_padded_gpu = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd_padded.items()}

    # Run the input data thd through.
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", token_dropout=False, dtype=torch.bfloat16)
    model_thd.to("cuda")
    outputs_thd = model_thd(**input_data_thd_gpu)
    outputs_thd_padded = model_thd(**input_data_thd_padded_gpu)

    padding_token_id = 1
    vocab = outputs_thd.logits.size(-1)

    cu_seq_lens_q = input_data_thd['cu_seq_lens_q']
    cu_seq_lens_q_padded = input_data_thd_padded['cu_seq_lens_q_padded']
    cu_num_pads = cu_seq_lens_q_padded - cu_seq_lens_q
    seq_lengths_real = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]

    num_real_tokens = outputs_thd.logits.shape[0] # should be cu_seq_lens_q[-1]

    # How much we need to shift each sequence by.
    offsets = torch.repeat_interleave(cu_num_pads[:-1], seq_lengths_real, dim=0)

    # The indices of the real tokens as appears in the padded logits.
    real_idx = torch.arange(0, num_real_tokens) + offsets

    assert ((input_data_thd['input_ids'].squeeze() - input_data_thd_padded['input_ids'].squeeze().index_select(0, real_idx)).abs().max().item() == 0)

    # Now index select the padded logits to get the real logits.
    logits_unpadded = outputs_thd_padded.logits.index_select(0, real_idx.cuda())

    torch.testing.assert_close(outputs_thd.logits, logits_unpadded, atol=1e-8, rtol=1e-5)
