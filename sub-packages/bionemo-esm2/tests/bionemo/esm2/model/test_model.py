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
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from torch import Tensor
from transformers import EsmForMaskedLM

from bionemo import esm2
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.api import ESM2Config, ESM2Model
from bionemo.esm2.model.embedding import ESM2Embedding
from bionemo.llm.model.biobert.model import MegatronBioBertModel
from bionemo.llm.utils.weight_utils import nemo1_to_nemo2_biobert_key_mapping
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


def reduce_hiddens(hiddens: Tensor, attention_mask: Tensor) -> Tensor:
    """reduce last layer's hidden values to embeddings

    Args:
        hiddens: [b, s, h] tensor of hidden values
        attention_mask: [b, s] attention mask tensor

    Returns:
        reduced embedding tensor [b, h]
    """
    masks = torch.sum(attention_mask, dim=1)
    embeddings = torch.zeros(
        size=(hiddens.shape[0], hiddens.shape[2]),
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )
    for i, (hidden, mask) in enumerate(zip(hiddens, masks)):
        embeddings[i, :] = torch.mean(hidden[1 : mask - 1], dim=0)
    return embeddings


@pytest.fixture(scope="module")
def esm2_config() -> ESM2Config:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        yield ESM2Config()


@pytest.fixture(scope="module")
def esm2_650M_config_w_ckpt() -> ESM2Config:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        yield ESM2Config(nemo1_ckpt_path=nemo1_checkpoint_path)


@pytest.fixture(scope="module")
def esm2_model(esm2_config) -> ESM2Model:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        tokenizer = AutoTokenizer(pretrained_model_name="facebook/esm2_t33_650M_UR50D")
        model = esm2_config.configure_model(tokenizer)
        yield model


@pytest.fixture(scope="module")
def sample_data() -> List[Tuple[str, str]]:
    """Generates sample protein sequences for sanity checks, including mask tokens."""
    max_length = 1022  # The maximum length of the protein sequences to be considered.
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
    yield sample_data


def test_esm2_model_initialized(esm2_model):
    assert isinstance(esm2_model, MegatronBioBertModel)
    assert isinstance(esm2_model, ESM2Model)
    assert isinstance(esm2_model.embedding, ESM2Embedding)


def test_esm2_650m_checkpoint(esm2_model):
    with tarfile.open(nemo1_checkpoint_path, "r") as ckpt, torch.no_grad():
        ckpt_file = ckpt.extractfile("./model_weights.ckpt")

        old_state_dict = torch.load(ckpt_file)
        # megatron is not registering inv_freq params anymore.
        # TODO: update Bionemo checkpoints
        old_state_dict.pop("model.language_model.rotary_pos_emb.inv_freq")

        new_state_dict = esm2_model.state_dict_for_save_checkpoint()

        # Set the new_model_prefix to "" since we are looking at the base megatron model and not the lightning module which stores a copy of
        #  this model into self.module
        old_keys = {nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="") for k in old_state_dict}
        assert len(old_keys) == len(old_state_dict), "Mapping unexpectedly discarded some keys."

        new_keys = set(new_state_dict)
        for k, v in old_state_dict.items():
            # Make sure the shapes of the weights match.
            assert new_state_dict[nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="")].shape == v.shape

        extra_keys = new_keys.difference(old_keys)
        extra_non_null_keys = {k for k in extra_keys if new_state_dict[k] is not None}
        assert not extra_non_null_keys, "There are new keys that have state that is missing from the old checkpoint."

        missing_old_keys = old_keys.difference(new_keys)
        assert not missing_old_keys, "There are keys in the old checkpoint that are missing from the new model."


def test_esm2_golden_values(esm2_650M_config_w_ckpt, sample_data):
    device = "cuda"

    tokenizer = AutoTokenizer(pretrained_model_name="facebook/esm2_t33_650M_UR50D")
    tokens = tokenizer.tokenizer([row[1] for row in sample_data], return_tensors="pt", padding=True).to(device)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # HF 650M model
    hf_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", torch_dtype=get_autocast_dtype(32)).to(
        device
    )

    with torch.no_grad():
        hf_output_all = hf_model(input_ids, attention_mask, output_hidden_states=True)
        hf_logits = hf_output_all.logits * attention_mask.unsqueeze(-1)
        hf_embeddings = reduce_hiddens(hf_output_all.hidden_states[-1], attention_mask)

        # free GPU RAM
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        # configure the model to return logits
        model = esm2_650M_config_w_ckpt.configure_model(tokenizer).to(device)
        model.eval()
        result = model(input_ids, attention_mask)
        logits = result["token_logits"][..., : tokenizer.vocab_size]
        logits = logits * attention_mask.unsqueeze(-1)  # incorporate masking logic

        # free GPU RAM
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # configure the model to return hiddens
        esm2_650M_config_hiddens = deepcopy(esm2_650M_config_w_ckpt)
        esm2_650M_config_hiddens.return_only_hidden_states = True
        model = esm2_650M_config_hiddens.configure_model(tokenizer).to(device)
        model.eval()
        hiddens = model(input_ids, attention_mask)
        embeddings = reduce_hiddens(torch.transpose(hiddens, 0, 1).float(), attention_mask)

        torch.testing.assert_close(logits, hf_logits, atol=9e-2, rtol=0.0)
        torch.testing.assert_close(embeddings, hf_embeddings, atol=5e-3, rtol=0.0)
