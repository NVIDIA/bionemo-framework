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

import tarfile
from pathlib import Path

import pytest
import torch
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

from bionemo import esm2
from bionemo.esm2.api import ESM2Config, ESM2Model
from bionemo.esm2.model.embedding import ESM2Embedding
from bionemo.llm.model.biobert.model import BiobertSpecOption, MegatronBioBertModel
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


@pytest.fixture(scope="module")
def esm2_model() -> ESM2Model:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        tokenizer = AutoTokenizer(pretrained_model_name="facebook/esm2_t33_650M_UR50D")
        esm2_config = ESM2Config(
            gradient_accumulation_fusion=False,
            apply_residual_connection_post_layernorm=False,
            biobert_spec_option=BiobertSpecOption.esm2_bert_layer_local_spec.value,
        )
        model = esm2_config.configure_model(tokenizer)
        yield model


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
