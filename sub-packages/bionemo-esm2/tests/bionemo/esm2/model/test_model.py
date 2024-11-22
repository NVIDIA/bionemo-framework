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
import io
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest
import torch
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from torch import Tensor
from transformers import EsmForMaskedLM

from bionemo.core.data.load import load
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.core.utils.random_utils import random_numpy_context
from bionemo.esm2.api import ESM2Config, ESM2Model
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.embedding import ESM2Embedding
from bionemo.llm.model.biobert.model import MegatronBioBertModel
from bionemo.llm.utils.weight_utils import nemo1_to_nemo2_biobert_key_mapping
from bionemo.testing import megatron_parallel_state_utils


nemo1_checkpoint_path: Path = load("esm2/nv_650m:1.0")


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
        tokenizer = get_tokenizer()
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


def _compute_loss(model, dataloader, vocab_size=None):
    loss = 0
    n = 0
    limit_batches = 10
    for i, batch in enumerate(dataloader):
        assert isinstance(batch, dict)
        result = model(input_ids=batch["text"].cuda(), attention_mask=batch["attention_mask"].cuda())

        # bionemo ESM2 vocab_size
        if vocab_size is not None:
            # token_logits is s,b and for simplicity here let's transpose to b,s. In general this reduces performance.
            logits = result["token_logits"].transpose(0, 1).contiguous()[..., :vocab_size]
        else:
            logits = result.logits

        loss_mask = batch["loss_mask"].cuda()
        target = batch["labels"].cuda()

        loss += torch.nn.functional.cross_entropy(logits[loss_mask].float(), target[loss_mask], reduction="sum")
        n += loss_mask.sum()

        if limit_batches is not None and i + 1 >= limit_batches:
            break
    mean_loss: Tensor = loss / n
    return mean_loss


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
        old_keys = {
            nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=True) for k in old_state_dict
        }
        assert len(old_keys) == len(old_state_dict), "Mapping unexpectedly discarded some keys."

        new_keys = set(new_state_dict)
        for k, v in old_state_dict.items():
            # Make sure the shapes of the weights match.
            assert (
                new_state_dict[nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=True)].shape
                == v.shape
            )

        extra_keys = new_keys.difference(old_keys)
        extra_non_null_keys = {
            k for k in extra_keys if new_state_dict[k] is not None and not isinstance(new_state_dict[k], io.BytesIO)
        }
        assert not extra_non_null_keys, "There are new keys that have state that is missing from the old checkpoint."

        missing_old_keys = old_keys.difference(new_keys)
        assert not missing_old_keys, "There are keys in the old checkpoint that are missing from the new model."


def test_esm2_golden_values(esm2_650M_config_w_ckpt, sample_data):
    tokenizer = AutoTokenizer(pretrained_model_name="facebook/esm2_t33_650M_UR50D")
    tokens = tokenizer.tokenizer([row[1] for row in sample_data], return_tensors="pt", padding=True).to("cuda")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # HF 650M model
    hf_model = EsmForMaskedLM.from_pretrained(
        "facebook/esm2_t33_650M_UR50D", torch_dtype=get_autocast_dtype(32)
    ).cuda()

    with torch.no_grad():
        hf_output_all = hf_model(input_ids, attention_mask, output_hidden_states=True)
        hf_logits = hf_output_all.logits * attention_mask.unsqueeze(-1)
        hf_embeddings = reduce_hiddens(hf_output_all.hidden_states[-1], attention_mask)

        # free GPU RAM
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        # configure the model to return logits
        model = esm2_650M_config_w_ckpt.configure_model(get_tokenizer()).cuda()
        model.eval()
        result = model(input_ids, attention_mask)
        # token_logits is s,b and for simplicity here let's transpose to b,s. In general this reduces performance.
        logits = result["token_logits"].transpose(0, 1).contiguous()[..., : tokenizer.vocab_size]
        logits = logits * attention_mask.unsqueeze(-1)  # incorporate masking logic

        # free GPU RAM
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # configure the model to return hiddens
        esm2_650M_config_hiddens = deepcopy(esm2_650M_config_w_ckpt)
        esm2_650M_config_hiddens.set_hparam("return_only_hidden_states", True)
        model = esm2_650M_config_hiddens.configure_model(get_tokenizer()).cuda()
        model.eval()
        hiddens = model(input_ids, attention_mask)
        embeddings = reduce_hiddens(torch.transpose(hiddens, 0, 1).float(), attention_mask)

        torch.testing.assert_close(logits, hf_logits, atol=0.2, rtol=0.0)
        torch.testing.assert_close(embeddings, hf_embeddings, atol=5e-3, rtol=0.0)


def test_esm2_loss(esm2_650M_config_w_ckpt, dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    compute_hf_reference: bool = True
    seed: int = 42

    with (
        torch.inference_mode(),
        megatron_parallel_state_utils.distributed_model_parallel_state(seed),
        random_numpy_context(seed),
    ):
        tokenizer = get_tokenizer()

        # ESM2 model initialized with 650M params
        model = esm2_650M_config_w_ckpt.configure_model(tokenizer).cuda()

        # Initialize the data module.
        data_module = ESMDataModule(
            train_cluster_path=train_cluster_path,
            train_database_path=dummy_protein_dataset,
            valid_cluster_path=valid_cluster_path,
            valid_database_path=dummy_protein_dataset,
            global_batch_size=4,
            micro_batch_size=2,
            min_seq_length=None,
            max_seq_length=1024,
            seed=seed,
            num_workers=1,
        )
        assert data_module is not None
        data_module.trainer = mock.Mock()
        data_module.trainer.max_epochs = 1
        data_module.trainer.max_steps = 10
        data_module.trainer.val_check_interval = 2
        data_module.trainer.limit_val_batches = 1

        data_module.setup()

        train_dataloader = data_module.train_dataloader()
        assert isinstance(train_dataloader, torch.utils.data.DataLoader)

        val_dataloader = data_module.val_dataloader()
        assert isinstance(val_dataloader, torch.utils.data.DataLoader)

        mean_loss = _compute_loss(model, train_dataloader, vocab_size=tokenizer.vocab_size)

        if compute_hf_reference:
            # HF model initialized with 650M params
            hf_model = EsmForMaskedLM.from_pretrained(
                "facebook/esm2_t33_650M_UR50D", torch_dtype=get_autocast_dtype(32)
            ).cuda()
            hf_mean_loss = _compute_loss(hf_model, train_dataloader)
            print(f"hf_mean_loss: {hf_mean_loss}")
        else:
            hf_mean_loss = torch.tensor(2.9279041290283203).cuda()

        torch.testing.assert_close(mean_loss, hf_mean_loss, atol=1e-3, rtol=0.0)
