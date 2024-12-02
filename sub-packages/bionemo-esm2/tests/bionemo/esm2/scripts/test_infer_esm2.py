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

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from bionemo.core.data.load import load
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule, InMemoryCSVDataset
from bionemo.esm2.scripts.infer_esm2 import infer_model


esm2_650m_checkpoint_path = load("esm2/650m:2.0")


@pytest.fixture
def dummy_protein_sequences():
    """Create a list of artificial protein sequences"""
    artificial_sequence_data = [
        "TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
        "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN",
        "DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
        "LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
        "LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
        "ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
        "SGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKT",
    ]
    return artificial_sequence_data


@pytest.fixture
def dummy_protein_csv(tmp_path, dummy_protein_sequences):
    """Create a mock protein dataset."""
    csv_file = tmp_path / "protein_dataset.csv"
    # Create a DataFrame
    df = pd.DataFrame(dummy_protein_sequences, columns=["sequences"])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def dataset(dummy_protein_csv):
    return InMemoryCSVDataset(dummy_protein_csv)


@pytest.fixture
def data_module(dataset):
    return ESM2FineTuneDataModule(predict_dataset=dataset)


def test_in_memory_csv_dataset(dataset):
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "text" in sample
    assert "labels" in sample


def test_in_memory_csv_dataset_load_data(dataset, dummy_protein_csv):
    sequences, labels = dataset.load_data(dummy_protein_csv)
    assert isinstance(sequences, list)
    assert isinstance(labels, list)


def test_esm2_fine_tune_data_module_init(data_module):
    assert data_module.train_dataset is None
    assert data_module.valid_dataset is None
    assert data_module.predict_dataset is not None


def test_esm2_fine_tune_data_module_predict_dataloader(data_module):
    predict_dataloader = data_module.predict_dataloader()
    assert isinstance(predict_dataloader, DataLoader)
    batch = next(iter(predict_dataloader))
    assert isinstance(batch, dict)
    assert "text" in batch


def test_esm2_fine_tune_data_module_setup(data_module):
    with pytest.raises(RuntimeError):
        data_module.setup("fit")


def test_esm2_fine_tune_data_module_train_dataloader(data_module):
    with pytest.raises(AttributeError):
        data_module.train_dataloader()


def test_esm2_fine_tune_data_module_val_dataloader(data_module):
    with pytest.raises(AttributeError):
        data_module.val_dataloader()


def test_in_memory_csv_dataset_tokenizer():
    tokenizer = get_tokenizer()
    sequence = "sequence"
    tokenized_sequence = tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
    assert isinstance(tokenized_sequence, torch.Tensor)


@pytest.mark.parametrize("precision", ["fp32", "bf16-mixed"])
def test_infer_runs(tmpdir, dummy_protein_csv, dummy_protein_sequences, precision):
    data_path = dummy_protein_csv
    result_dir = Path(tmpdir.mkdir("results"))
    results_path = result_dir / "esm2_infer_results.pt"

    max_dataset_seq_len = max(len(seq) for seq in dummy_protein_sequences)

    infer_model(
        data_path=data_path,
        checkpoint_path=esm2_650m_checkpoint_path,
        results_path=results_path,
        min_seq_length=max_dataset_seq_len,
        include_hiddens=True,
        precision=precision,
        include_embeddings=True,
        include_input_ids=True,
        include_logits=True,
        micro_batch_size=3,  # dataset length (10) is not multiple of 3; this validates partial batch inference
        # config_class=SUPPORTED_CONFIGS[config_class_name],
        config_class=ESM2Config,
    )
    assert results_path.exists(), "Could not find test results pt file."

    results = torch.load(results_path)
    assert isinstance(results, dict)
    keys_included = ["token_logits", "hidden_states", "embeddings", "binary_logits", "input_ids"]
    assert all(key in results for key in keys_included)
    assert results["binary_logits"] is None
    assert results["embeddings"].shape[0] == len(dummy_protein_sequences)
    assert results["embeddings"].dtype == get_autocast_dtype(precision)
    # hidden_states are [batch, sequence, hidden_dim]
    assert results["hidden_states"].shape[:-1] == (len(dummy_protein_sequences), max_dataset_seq_len)
    # input_ids are [batch, sequence]
    assert results["input_ids"].shape == (len(dummy_protein_sequences), max_dataset_seq_len)
    # token_logits are [sequence, batch, num_tokens]
    assert results["token_logits"].shape[:-1] == (max_dataset_seq_len, len(dummy_protein_sequences))
