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


import gc
from pathlib import Path

import torch
from megatron.core.transformer.module import Float16Module
from transformers import AutoModelForMaskedLM

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.model import ESM2Config


def assert_model_equivalence(
    ckpt_path: Path | str,
    model_tag: str,
    precision: PrecisionTypes = "fp32",
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Testing utility to compare the outputs of a NeMo2 checkpoint to the original HuggingFace model weights.

    Compares the cosine similarity of the logit and hidden state outputs of a NeMo2 model checkpoint to the outputs of
    the corresponding HuggingFace model.

    Args:
        ckpt_path: A path to a NeMo2 checkpoint for an ESM-2 model.
        model_tag: The HuggingFace model tag for the model to compare against.
        precision: The precision type to use for the comparison. Defaults to "fp32".
        rtol: The relative tolerance to use for the comparison. Defaults to None, which chooses the tolerance based on
            the precision.
        atol: The absolute tolerance to use for the comparison. Defaults to None, which chooses the tolerance based on
            the precision.
    """
    tokenizer = get_tokenizer()

    test_proteins = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA",
        "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG",
    ]
    tokens = tokenizer(test_proteins, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    dtype = get_autocast_dtype(precision)
    nemo_config = ESM2Config(
        initial_ckpt_path=str(ckpt_path),
        include_embeddings=True,
        include_hiddens=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        autocast_dtype=dtype,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
    )

    nemo_model = nemo_config.configure_model(tokenizer).to("cuda").eval()

    if dtype is torch.float16 or dtype is torch.bfloat16:
        nemo_model = Float16Module(nemo_config, nemo_model)

    nemo_output = nemo_model(input_ids, attention_mask)
    nemo_logits = nemo_output["token_logits"].transpose(0, 1).contiguous()[..., : tokenizer.vocab_size]
    nemo_hidden_state = nemo_output["hidden_states"]

    del nemo_model
    gc.collect()
    torch.cuda.empty_cache()

    hf_model = AutoModelForMaskedLM.from_pretrained(model_tag, torch_dtype=get_autocast_dtype(precision)).cuda().eval()
    hf_output_all = hf_model(input_ids, attention_mask, output_hidden_states=True)
    hf_hidden_state = hf_output_all.hidden_states[-1]

    # Rather than directly comparing the logit or hidden state tensors, we compare their cosine similarity. These
    # should be essentially 1 if the outputs are equivalent, but is less sensitive to small numerical differences.
    # We don't care about the padding tokens, so we only compare the non-padding tokens.
    logit_similarity = torch.nn.functional.cosine_similarity(nemo_logits, hf_output_all.logits, dim=2)
    logit_similarity = logit_similarity[attention_mask == 1]

    hidden_state_similarity = torch.nn.functional.cosine_similarity(nemo_hidden_state, hf_hidden_state, dim=2)
    hidden_state_similarity = hidden_state_similarity[attention_mask == 1]

    torch.testing.assert_close(logit_similarity, torch.ones_like(logit_similarity), rtol=rtol, atol=atol)
    torch.testing.assert_close(hidden_state_similarity, torch.ones_like(hidden_state_similarity), rtol=rtol, atol=atol)
