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


import pandas as pd
import pytest
import torch
from torch import Tensor

from bionemo.esm2.model.finetune.dataset import (
    InMemoryPerTokenValueDataset,
    InMemoryProteinDataset,
    InMemorySingleValueDataset,
)
from bionemo.llm.data.collate import MLM_LOSS_IGNORE_INDEX
from bionemo.llm.data.label2id_tokenizer import Label2IDTokenizer


def data_to_csv(data, tmp_path, with_label=True):
    """Create a mock protein dataset."""
    csv_file = tmp_path / "protein_dataset.csv"
    # Create a DataFrame
    df = pd.DataFrame(data, columns=["sequences", "labels"] if with_label else ["sequences"])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def dataset_no_labels(dummy_protein_sequences, tmp_path):
    csv_path = data_to_csv(dummy_protein_sequences, tmp_path, with_label=False)
    return InMemoryProteinDataset.from_csv(csv_path)


@pytest.fixture
def dataset_regression_labels(dummy_data_single_value_regression_ft, tmp_path):
    csv_path = data_to_csv(dummy_data_single_value_regression_ft, tmp_path, with_label=True)
    return InMemorySingleValueDataset.from_csv(csv_path)


@pytest.fixture
def dataset_per_token_classification_labels(dummy_data_per_token_classification_ft, tmp_path):
    csv_path = data_to_csv(dummy_data_per_token_classification_ft, tmp_path, with_label=True)
    return InMemoryPerTokenValueDataset.from_csv(csv_path)


def test_in_memory_protein_dataset_length_no_labels(dataset_no_labels, dummy_protein_sequences):
    assert len(dataset_no_labels) == len(dummy_protein_sequences)


def test_in_memory_protein_dataset_length_with_regression_labels(
    dataset_regression_labels, dummy_data_single_value_regression_ft
):
    assert len(dataset_regression_labels) == len(dummy_data_single_value_regression_ft)


def test_in_memory_protein_dataset_length_with_class_labels(
    dataset_per_token_classification_labels, dummy_data_per_token_classification_ft
):
    assert len(dataset_per_token_classification_labels) == len(dummy_data_per_token_classification_ft)


def test_in_memory_protein_dataset_getitem_no_labels(dataset_no_labels):
    sample = dataset_no_labels[0]
    assert isinstance(sample, dict)
    assert "text" in sample
    assert "labels" in sample
    assert isinstance(sample["text"], Tensor)
    assert isinstance(sample["labels"], Tensor)


def test_in_memory_protein_dataset_getitem_with_regression_labels(dataset_regression_labels):
    assert isinstance(dataset_regression_labels, InMemoryProteinDataset)
    sample = dataset_regression_labels[0]
    assert isinstance(sample, dict)
    assert "text" in sample
    assert "labels" in sample
    assert isinstance(sample["text"], Tensor)
    assert isinstance(sample["labels"], Tensor)
    assert sample["labels"].dtype == torch.float


def test_in_memory_protein_dataset_getitem_with_class_labels(dataset_per_token_classification_labels):
    assert isinstance(dataset_per_token_classification_labels, InMemoryProteinDataset)
    assert isinstance(dataset_per_token_classification_labels.label_tokenizer, Label2IDTokenizer)
    assert dataset_per_token_classification_labels.label_cls_eos_id == MLM_LOSS_IGNORE_INDEX

    sample = dataset_per_token_classification_labels[0]
    assert isinstance(sample, dict)
    assert "text" in sample
    assert "labels" in sample
    assert isinstance(sample["text"], Tensor)
    assert isinstance(sample["labels"], Tensor)
    assert sample["labels"].dtype == torch.int64


def test_in_memory_protein_dataset_tokenization(dataset_no_labels):
    sample = dataset_no_labels[0]
    tokenized_sequence = sample["text"]
    assert isinstance(tokenized_sequence, Tensor)
    assert tokenized_sequence.ndim == 1  # Ensure it's flattened.


def test_transofrm_classification_label(
    dataset_per_token_classification_labels, dummy_data_per_token_classification_ft
):
    pre_transfrom = dummy_data_per_token_classification_ft[0][1]
    label_ids = torch.tensor(dataset_per_token_classification_labels.label_tokenizer.text_to_ids(pre_transfrom))
    cls_eos = torch.tensor([dataset_per_token_classification_labels.label_cls_eos_id])
    post_transform = torch.cat((cls_eos, label_ids, cls_eos))

    assert torch.equal(dataset_per_token_classification_labels.transform_label(pre_transfrom), post_transform)


def test_transofrm_regression_label(dataset_regression_labels):
    """Ensure labels are transformed correctly."""
    transformed_label = dataset_regression_labels.transform_label(1.0)
    assert isinstance(transformed_label, Tensor)
    assert transformed_label.dtype == torch.float


def test_in_memory_protein_dataset_no_labels_fallback(dataset_no_labels):
    """Ensure the dataset works even when labels are missing."""
    sample = dataset_no_labels[0]
    assert isinstance(sample, dict)
    assert "labels" in sample
    assert isinstance(sample["labels"], Tensor)


def test_in_memory_protein_dataset_invalid_index(dataset_no_labels):
    """Test if out-of-range index raises an error."""
    with pytest.raises(KeyError):
        _ = dataset_no_labels[100]


def test_in_memory_protein_dataset_missing_sequences_column(tmp_path):
    """Test behavior when the CSV file is empty."""
    csv_file = tmp_path / "invalid.csv"
    pd.DataFrame({"wrong_column": ["MKTFFS"]}).to_csv(csv_file, index=False)
    with pytest.raises(KeyError):
        _ = InMemoryProteinDataset.from_csv(csv_file)


def test_in_memory_protein_dataset_special_tokens_masking(dataset_no_labels):
    """Ensure loss mask correctly handles special tokens."""
    sample = dataset_no_labels[0]
    assert "loss_mask" in sample
    assert isinstance(sample["loss_mask"], Tensor)
    assert sample["loss_mask"].dtype == torch.bool


def test_in_memory_protein_dataset_non_existent_file():
    """Ensure proper error handling for missing files."""
    with pytest.raises(FileNotFoundError):
        InMemoryProteinDataset.from_csv("non_existent_file.csv")
