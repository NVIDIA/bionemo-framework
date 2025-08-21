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
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


# Fix Triton UTF-8 decoding issue by setting CUDA library path
# This bypasses the problematic ldconfig -p call that contains non-UTF-8 characters
if not os.environ.get("TRITON_LIBCUDA_PATH"):
    # Set the path to CUDA libraries in the NVIDIA PyTorch container
    os.environ["TRITON_LIBCUDA_PATH"] = "/usr/local/cuda/lib64"


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


@pytest.fixture
def input_data(tokenizer):
    torch.manual_seed(42)

    test_proteins = [
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

    dataset = Dataset.from_list([{"sequence": p} for p in test_proteins])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=1024,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt",
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
