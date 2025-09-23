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

import importlib
import os

import pytest
import torch
import transformer_engine.pytorch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from esm.collator import MLMDataCollatorWithFlattening
from esm.convert import convert_esm_hf_to_te


# Fix Triton UTF-8 decoding issue by setting CUDA library path
# This bypasses the problematic ldconfig -p call that contains non-UTF-8 characters
if not os.environ.get("TRITON_LIBCUDA_PATH"):
    # Set the path to CUDA libraries in the NVIDIA PyTorch container
    os.environ["TRITON_LIBCUDA_PATH"] = "/usr/local/cuda/lib64"


@pytest.fixture(autouse=True)
def use_te_debug(monkeypatch):
    monkeypatch.setenv("NVTE_DEBUG", "1")
    monkeypatch.setenv("NVTE_DEBUG_LEVEL", "2")
    importlib.reload(transformer_engine.pytorch)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


@pytest.fixture
def bshd_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=1024,
        seed=42,
    )


@pytest.fixture
def thd_data_collator(tokenizer):
    return MLMDataCollatorWithFlattening(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=8,
        seed=42,
    )


def get_test_proteins():
    return [
        "MLSATEKLSDYISSLFASVSIINSISTEDLFFLKLTCQTFSKDSEEYKAAYRILRGVQRGKVQIIEEALVS",
        "MFVFFAGTLVNQDTLNFRDQLNINVVGTVRGIAQDASKYLEYAIDSV",
        "MAATGSLILSDEEQAELIALAVRIVLACAGGSQNKELAAQLGVIETTVGEWRRRFAQNRVEGLRDEARPGAPSDDQ",
        "MSAVLSAVASDDWTAFAKLVHPYVHWTADGITTRGRTRVMARLSGHDGVKPASSYELRDGQVYRWTS",
        "MSDPAAEPPADTSGIAWRKSSYSGPNGNCVELAQISGDHVGIRNSRDLHGSVLTCTRAEFAALLCDIKAGRFDSLIL",
        "MRRPKLRRSGVLMSHPARGQPIKDASTEAAAERRPHVTSSERQDVSDQDTR",
        "MQTITVAGGNLFQIAAQYLGDATQWIRIAQLNGLADPVLSGVVTLTIPQPNPLAGGGVVGQ",
        "MVFSLEQFVRGQGWQSITSNSDNEVPKPRQVYEVKAVCHPGAWRVKARVFGTSQGIPFDYSQASMERRVAQDECDRRPQ",
        "AGDGTGCNPTLSKAAGVELDNSDSGEVFVIYLHIIIAIIVLISINLIGFLYF",
        "MKVGVDPSVCEAHGACMSILPEVFDLDDDEVLQIRDGELAPSEEESAERAVASCPMGALRLSR",
        "MWISERPPSRMALGSQSQMSLPGIPARCLHS",
        "MIDNSIRLFDADDSELFSLAEVPLDNKPIQRDTDSLSQWGDTWLREIQHS",
        "MVKNLFFNKIKNATLKVANISRCYLPFPPPPCPPPEPLEPPEPPAPLEPAPDPPPLPPFPVPDILPAI",
        "MSYINDITQSNSSILNVNVKINDHNSDEMYRNETKWYGEQFRYQSNPRFSRSSTSKNEKGFVQKKT",
        "MQILILPIPDQLQNPNKISQHLICITFVSEQTLPI",
    ]


@pytest.fixture
def test_proteins():
    return get_test_proteins()


def get_input_data(tokenizer, data_collator):
    torch.manual_seed(42)

    dataset = Dataset.from_list([{"sequence": p} for p in get_test_proteins()])

    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=1024,
        )

    tokenized_proteins = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sequence"],
    )

    dataloader = DataLoader(
        tokenized_proteins,
        batch_size=len(tokenized_proteins),
        collate_fn=data_collator,
    )

    batch = next(iter(dataloader))
    return batch


@pytest.fixture
def input_data(tokenizer, bshd_data_collator):
    return get_input_data(tokenizer, bshd_data_collator)


@pytest.fixture
def input_data_thd(tokenizer, thd_data_collator):
    return get_input_data(tokenizer, thd_data_collator)


@pytest.fixture
def te_model_checkpoint(tmp_path):
    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.save_pretrained(tmp_path / "te_model_checkpoint")
    return tmp_path / "te_model_checkpoint"
