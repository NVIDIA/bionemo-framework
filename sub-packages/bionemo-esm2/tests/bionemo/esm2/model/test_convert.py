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


import torch
from nemo.lightning import io
from transformers import AutoModelForMaskedLM

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.model import ESM2Config
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.testing import megatron_parallel_state_utils


def test_convert_esm2_hf_to_nemo(tmp_path):
    from bionemo.esm2.model.convert import HFESM2Importer  # noqa: F401

    model_tag = "facebook/esm2_t6_8M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "output_ckpt")
    tokenizer = get_tokenizer()

    test_proteins = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA",
        "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG",
    ]

    tokens = tokenizer(test_proteins, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # HF 650M model
    hf_model = AutoModelForMaskedLM.from_pretrained(model_tag, torch_dtype=get_autocast_dtype(32)).cuda()

    with torch.inference_mode(), megatron_parallel_state_utils.distributed_model_parallel_state():
        nemo_model = (
            ESM2Config(initial_ckpt_path=tmp_path / "output_ckpt", include_embeddings=True, include_hiddens=True)
            .configure_model(tokenizer)
            .to("cuda")
            .eval()
        )

        for i in range(len(hf_model.esm.encoder.layer)):
            torch.testing.assert_close(
                hf_model.esm.encoder.layer[i].attention.self.rotary_embeddings.inv_freq,
                nemo_model.rotary_pos_emb.inv_freq,
                atol=1e-4,
                rtol=1e-6,
            )

        hf_output_all = hf_model(input_ids, attention_mask, output_hidden_states=True)

        nemo_output = nemo_model(input_ids, attention_mask)
        nemo_logits = nemo_output["token_logits"].transpose(0, 1).contiguous()[..., : tokenizer.vocab_size]

        nemo_hidden_state = nemo_output["hidden_states"]
        hf_hidden_state = hf_output_all.hidden_states[-1]

        # Rather than directly comparing the logit or hidden state tensors, we compare their cosine similarity. These
        # should be essentially 1 if the outputs are equivalent, but is less sensitive to small numerical differences.
        # We don't care about the padding tokens, so we only compare the non-padding tokens.
        logit_similarity = torch.nn.functional.cosine_similarity(nemo_logits, hf_output_all.logits, dim=2)
        logit_similarity = logit_similarity[attention_mask == 1]

        hidden_state_similarity = torch.nn.functional.cosine_similarity(nemo_hidden_state, hf_hidden_state, dim=2)
        hidden_state_similarity = hidden_state_similarity[attention_mask == 1]

        torch.testing.assert_close(logit_similarity, torch.ones_like(logit_similarity))
        torch.testing.assert_close(hidden_state_similarity, torch.ones_like(hidden_state_similarity))
