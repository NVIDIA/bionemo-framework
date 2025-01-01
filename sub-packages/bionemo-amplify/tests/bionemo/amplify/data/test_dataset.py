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

from bionemo.amplify.data.dataset import (
    AMPLIFYMaskedResidueDataset,
    create_train_dataset,
    create_valid_dataset,
)


def test_protein_sqlite_dataset(dummy_protein_dataset):
    """Test the ProteinSQLiteDataset class."""

    dataset = ProteinSQLiteDataset(dummy_protein_dataset)

    assert len(dataset) == 5

    assert dataset["UniRef90_A"] == "ACDEFGHIKLMNPQRSTVWY"
    assert dataset["UniRef90_B"] == "DEFGHIKLMNPQRSTVWYAC"
    assert dataset["UniRef90_C"] == "MGHIKLMNPQRSTVWYACDE"
    assert dataset["UniRef50_A"] == "MKTVRQERLKSIVRI"
    assert dataset["UniRef50_B"] == "MRILERSKEPVSGAQLA"


def test_AMPLIFYPreTrainingDataset_getitem_has_expected_structure(dummy_protein_dataset, tokenizer):
    """Test that the AMPLIFYPreTrainingDataset's __getitem__ method is deterministic."""

    protein_dataset = ProteinSQLiteDataset(dummy_protein_dataset)
    clusters = [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]]
    amplify_dataset = AMPLIFYMaskedResidueDataset(
        protein_dataset=protein_dataset, clusters=clusters, total_samples=10, seed=123
    )

    sample = amplify_dataset[0]
    assert len(sample["text"]) == len(protein_dataset["UniRef90_A"]) + 2

    # Make sure all masked tokens are standard amino acids.
    for token in sample["labels"][sample["loss_mask"]].tolist():
        assert token in range(4, 24)

    # Make sure non-masked tokens are -1.
    assert torch.all(sample["labels"][~sample["loss_mask"]] == -1)

    assert sample["text"][0] == tokenizer.cls_token_id
    assert sample["text"][-1] == tokenizer.eos_token_id


def test_AMPLIFYPreTrainingDataset_getitem_match_for_identical_seeds(dummy_protein_dataset):
    """Test that the AMPLIFYPreTrainingDataset's __getitem__ method is deterministic."""

    dataset = ProteinSQLiteDataset(dummy_protein_dataset)
    clusters = [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]]

    dataset1 = AMPLIFYMaskedResidueDataset(protein_dataset=dataset, clusters=clusters, total_samples=10, seed=123)
    dataset2 = AMPLIFYMaskedResidueDataset(protein_dataset=dataset, clusters=clusters, total_samples=10, seed=123)

    # Check that the datasets are equal.
    for i in range(len(dataset1)):
        sample1 = dataset1[i]
        sample2 = dataset2[i]

        for key in sample1:
            torch.testing.assert_close(sample1[key], sample2[key])


def test_AMPLIFYPreTrainingDataset_getitem_is_deterministic(dummy_protein_dataset):
    """Test that the AMPLIFYPreTrainingDataset's __getitem__ method is deterministic."""

    dataset = ProteinSQLiteDataset(dummy_protein_dataset)
    clusters = [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]]

    dataset = AMPLIFYMaskedResidueDataset(protein_dataset=dataset, clusters=clusters, total_samples=10, seed=123)

    sample1 = dataset[8]

    for _ in range(10):
        sample2 = dataset[8]
        for key in sample1:
            torch.testing.assert_close(sample1[key], sample2[key])


def test_AMPLIFYPreTrainingDataset_getitem_differs_with_different_seeds(dummy_protein_dataset):
    """Test that the AMPLIFYPreTrainingDataset's __getitem__ method is deterministic."""

    dataset = ProteinSQLiteDataset(dummy_protein_dataset)
    clusters = [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]]

    dataset1 = AMPLIFYMaskedResidueDataset(protein_dataset=dataset, clusters=clusters, total_samples=10, seed=123)
    dataset2 = AMPLIFYMaskedResidueDataset(protein_dataset=dataset, clusters=clusters, total_samples=10, seed=321)

    for i in range(len(dataset)):
        sample1 = dataset1[i]
        sample2 = dataset2[i]
        assert not torch.equal(sample1["text"], sample2["text"])


def test_AMPLIFYPreTrainingDataset_getitem_changes_each_epoch(dummy_protein_dataset):
    """Test that the AMPLIFYPreTrainingDataset's __getitem__ method is deterministic."""

    dataset = ProteinSQLiteDataset(dummy_protein_dataset)
    clusters = [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]]

    dataset = AMPLIFYMaskedResidueDataset(protein_dataset=dataset, clusters=clusters, total_samples=10, seed=123)

    sample1 = dataset[0]
    sample2 = dataset[2]
    assert len(sample1["text"]) == len(sample2["text"])  # These should both be UniRef90_A
    assert not torch.equal(sample1["text"], sample2["text"])

    sample1 = dataset[0]
    sample2 = dataset[4]
    assert len(sample1["text"]) == len(sample2["text"])
    assert not torch.equal(sample1["text"], sample2["text"])


def test_AMPLIFYPreTrainingDataset_fails_with_empty_cluster(dummy_protein_dataset):
    """Test that the AMPLIFYPreTrainingDataset's __getitem__ method is deterministic."""

    dataset = ProteinSQLiteDataset(dummy_protein_dataset)
    clusters = [["UniRef90_A"], [], ["UniRef90_B", "UniRef90_C"]]

    dataset = AMPLIFYMaskedResidueDataset(protein_dataset=dataset, clusters=clusters, total_samples=10, seed=123)

    with pytest.raises(ValueError, match="Cluster 1 is empty."):
        dataset[1]


def test_AMPLIFYPreTrainingDataset_crops_out_start_and_end(dummy_protein_dataset, tokenizer):
    prot_dataset = ProteinSQLiteDataset(dummy_protein_dataset)
    clusters = [["UniRef90_A"]]

    dataset = AMPLIFYMaskedResidueDataset(
        protein_dataset=prot_dataset, clusters=clusters, seed=123, total_samples=10, max_seq_length=1024
    )

    assert len(dataset[0]["text"]) == len(prot_dataset["UniRef90_A"]) + 2
    assert dataset[0]["text"][0] == tokenizer.cls_token_id
    assert dataset[0]["text"][-1] == tokenizer.eos_token_id

    dataset = AMPLIFYMaskedResidueDataset(
        protein_dataset=prot_dataset, clusters=clusters, seed=123, total_samples=10, max_seq_length=3
    )

    assert len(dataset[0]["text"]) == 3

    # With a max length of 3, both the start and end tokens cant be present.
    assert not ((dataset[0]["text"][0] == tokenizer.cls_token_id) & (dataset[0]["text"][-1] == tokenizer.eos_token_id))


def test_AMPLIFYPreTrainingDataset_raises_index_error_outside_bounds(dummy_protein_dataset):
    """Test that the AMPLIFYPreTrainingDataset's __getitem__ method is deterministic."""

    dataset = ProteinSQLiteDataset(dummy_protein_dataset)
    clusters = [["UniRef90_A"], [], ["UniRef90_B", "UniRef90_C"]]

    dataset = AMPLIFYMaskedResidueDataset(protein_dataset=dataset, clusters=clusters, total_samples=10, seed=123)

    with pytest.raises(IndexError, match="Index 10 out of range \\[0, 10\\)."):
        dataset[10]

    with pytest.raises(IndexError, match="Index -1 out of range \\[0, 10\\)."):
        dataset[-1]


def test_create_train_dataset(dummy_protein_dataset, tmp_path):
    cluster_file = pd.DataFrame(
        {
            "ur90_id": [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]],
        }
    )

    cluster_file.to_parquet(tmp_path / "train_clusters.parquet")

    dataset = create_train_dataset(
        cluster_file=tmp_path / "train_clusters.parquet", db_path=dummy_protein_dataset, total_samples=10, seed=123
    )
    assert len(dataset) == 10
    dataset[6]  # Make sure it doesn't crash.


def test_create_valid_dataset(dummy_protein_dataset, tmp_path):
    cluster_file = pd.DataFrame(
        {
            "ur50_id": ["UniRef90_A", "UniRef90_B", "UniRef90_C"],
        }
    )

    cluster_file.to_parquet(tmp_path / "valid_clusters.parquet")

    dataset = create_valid_dataset(
        clusters=tmp_path / "valid_clusters.parquet", db_path=dummy_protein_dataset, total_samples=10, seed=123
    )
    assert len(dataset) == 10
    dataset[6]  # Make sure it doesn't crash.
