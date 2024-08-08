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
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import pytest
import torch
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from transformers import EsmForMaskedLM

from bionemo import esm2
from bionemo.esm2.api import ESM2Config, ESM2Model
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.testing import megatron_parallel_state_utils


bionemo2_root: Path = (
    # esm2 module's path is the most dependable --> don't expect this to change!
    Path(esm2.__file__)
    # This gets us from 'sub-packages/bionemo-esm2/src/bionemo/esm2/__init__.py' to 'sub-packages/bionemo-esm2'
    .parent.parent.parent.parent
    # From here, we want to get to the root of the repository: _before_ sub-packages/
    .parent.parent
).absolute()
assert bionemo2_root != Path("/")
nemo1_checkpoint_path: Path = bionemo2_root / "models/protein/esm2nv/esm2nv_650M_converted.nemo"


def register_hooks(model, hook_fn):
    for name, module in model.named_modules():
        # if 'encoder.layers.1.' in name:
        module.register_forward_hook(partial(hook_fn, name))


@pytest.fixture(scope="module")
def esm2_config() -> ESM2Config:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        esm2_config = ESM2Config(
            gradient_accumulation_fusion=False,
            apply_residual_connection_post_layernorm=False,
            biobert_spec_option=BiobertSpecOption.esm2_bert_layer_local_spec.value,
        )
        yield esm2_config


@pytest.fixture(scope="module")
def esm2_model(esm2_config) -> ESM2Model:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        tokenizer = AutoTokenizer(pretrained_model_name="facebook/esm2_t33_650M_UR50D")
        model = esm2_config.configure_model(tokenizer)
        yield model


@pytest.fixture(scope="module")
def esm2_model_from_nemo1(esm2_config) -> ESM2Model:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        tokenizer = AutoTokenizer(pretrained_model_name="facebook/esm2_t33_650M_UR50D")
        esm2_config.nemo1_ckpt_path = nemo1_checkpoint_path
        # esm2_config.return_only_hidden_states = True
        model = esm2_config.configure_model(tokenizer)
        yield model


def load_sample_protein_sequence_data(max_length: int = 1022) -> List[Tuple[str, str]]:
    """
    Generates sample protein sequences for sanity checks, including mask tokens.

    Parameters:
    ----------
    max_length: The maximum length of the protein sequences to be considered.
            Defaults to 1022.

    Returns:
    -------
    List[Tuple[str, str]]: list of sample proteins with a name and a sequence.
    """

    sample_data = [
        (
            "protein1",
            "MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVLGGFTSTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLAGWSRYIPEGLQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIIIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWVPYASVAFYIFTHQGSNFGPIFMTIPAFFAKSAAIYNPVIYIMMNKQFRNCMLTTICCGKNPLGDDEASATVSKTETSQVAPA",
        ),
        ("protein2", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"),
        (
            "protein3",
            "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG",
        ),
        (
            "protein4",
            "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLA",
        ),
    ]

    # add another sample protein that uses the maximum length to test this edge case
    sample_data.append(("protein5", (sample_data[0][1] * 3)[:max_length]))
    return sample_data


def dtype_from_precision(precision: Union[str, int]) -> torch.dtype:
    """
    Determines the appropriate PyTorch data type (dtype) based on the given precision.

    The function handles the following cases:
        - 32: Maps to torch.float32.
        - 16 or "16" or "16-mixed": Maps to torch.float16.
        - "bf16" or "bf16-mixed": Maps to torch.bfloat16.
        - For any other values, the function defaults to torch.float32 as a fallback.

    Parameters:
    ----------
    precision: The desired float precision for setting model's weights. This can be an integer (e.g., 32, 16)
            or a string (e.g., "16", "16-mixed", "bf16", "bf16-mixed").

    Returns:
    --------
        torch.dtype: The corresponding PyTorch data type for the specified precision level.
    """
    if precision == 32:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  # fallback
    return dtype


def test_esm2_logits(esm2_model_from_nemo1):
    device = "cuda"
    sample_data = load_sample_protein_sequence_data()
    tokenizer = AutoTokenizer(pretrained_model_name="facebook/esm2_t33_650M_UR50D")
    tokens = tokenizer.tokenizer([row[1] for row in sample_data], return_tensors="pt", padding=True)
    tokens["input_ids"] = tokens["input_ids"].to(device)
    tokens["attention_mask"] = tokens["attention_mask"].to(device)

    hf_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")  # HF 650M model
    hf_model = hf_model.to(device=device, dtype=dtype_from_precision(32))

    with torch.no_grad():
        esm2_model_from_nemo1.eval()

        hf_output_all = hf_model(**tokens, output_hidden_states=False)
        hf_logits = hf_output_all.logits

        # free GPU RAM for the NeMo model
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        # Get hidden embeddings from the converted BioNeMo model
        esm2_model_from_nemo1 = esm2_model_from_nemo1.to(device)
        result = esm2_model_from_nemo1(tokens["input_ids"], tokens["attention_mask"])

        logits = result["token_logits"][:, :, : tokenizer.vocab_size]
        diff = (logits - hf_logits) * tokens["attention_mask"].unsqueeze(-1)
        max_abs_diff = diff.abs().max().item()
    assert max_abs_diff < 8e-2
