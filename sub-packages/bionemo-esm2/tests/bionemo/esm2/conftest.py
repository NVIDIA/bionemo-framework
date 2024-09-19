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


import sqlite3

import pandas as pd
import pytest

from bionemo.esm2.data.tokenizer import get_tokenizer


@pytest.fixture
def tokenizer():
    """Return the ESM2 tokenizer."""
    return get_tokenizer()


@pytest.fixture
def dummy_protein_dataset(tmp_path):
    """Create a mock protein dataset."""
    db_file = tmp_path / "protein_dataset.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE protein (
            id TEXT PRIMARY KEY,
            sequence TEXT
        )
    """
    )

    proteins = [
        ("UniRef90_A", "ACDEFGHIKLMNPQRSTVWY"),
        ("UniRef90_B", "DEFGHIKLMNPQRSTVWYAC"),
        ("UniRef90_C", "MGHIKLMNPQRSTVWYACDE"),
        ("UniRef50_A", "MKTVRQERLKSIVRI"),
        ("UniRef50_B", "MRILERSKEPVSGAQLA"),
    ]
    cursor.executemany("INSERT INTO protein VALUES (?, ?)", proteins)

    conn.commit()
    conn.close()

    return db_file


@pytest.fixture
def dummy_parquet_train_val_inputs(tmp_path):
    """Create a mock protein train and val cluster parquet."""
    train_cluster_path = tmp_path / "train_clusters.parquet"
    train_clusters = pd.DataFrame(
        {
            "ur90_id": [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]],
        }
    )
    train_clusters.to_parquet(train_cluster_path)

    valid_cluster_path = tmp_path / "valid_clusters.parquet"
    valid_clusters = pd.DataFrame(
        {
            "ur50_id": ["UniRef50_A", "UniRef50_B"],
        }
    )
    valid_clusters.to_parquet(valid_cluster_path)
    return train_cluster_path, valid_cluster_path


@pytest.fixture
def dummy_data_per_token_classification_ft():
    """Fixture providing dummy data for per-token classification fine-tuning.

    Returns:
        list: A list of dummy data for per-token classification fine-tuning.
    """
    data = [
        (
            "seq_0",
            "TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
            "EEEECCCCCHHHHHHHHHHHHHHHCCCEEEEEECCCHHHHHHHHHCCCCCCCCCEEE",
        ),
        ("seq_1", "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF", "CCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC"),
        ("seq_2", "GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN", "HHHHHCCCCCHHHHHHHHHHHHHHCCCHHHHHHHHHH"),
        (
            "seq_3",
            "DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
            "HHHHHHHHHHCCCHHHHHCCCCCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC",
        ),
        (
            "seq_4",
            "KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
            "CHHHHHHHHHHHHHHHCCCEEEEEECCCHHHHHHHHHCCCCCCCCCEEE",
        ),
        (
            "seq_5",
            "LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
            "HHHHHHHHHHHHHCHHHHHHHHHHHHCCCEECCCEEEECCEEEEECC",
        ),
        (
            "seq_6",
            "LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
            "HHHHHCCCHHHHHCCCCCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC",
        ),
        ("seq_7", "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF", "CCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC"),
        ("seq_8", "ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP", "HHHHHCHHHHHHHHHHHHCCCEECCCEEEECCEEEEECC"),
        (
            "seq_9",
            "SGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKT",
            "CCCCCCCCCCCCCCCCCCCCCCCCCCEEECCCCEEECHHHHHHHHHCCCCCCCCEEECCCCCC",
        ),
    ]
    return data


@pytest.fixture
def dummy_data_single_value_regression_ft(dummy_data_per_token_classification_ft):
    """Fixture providing dummy data for per-token classification fine-tuning.

    Returns:
        list: A list of dummy data for per-token classification fine-tuning.
    """
    data = [(seq_id, seq, len(seq) / 100.0) for seq_id, seq, _ in dummy_data_per_token_classification_ft]
    return data
