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


import pytest

from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.testing.data.esm2 import create_mock_parquet_train_val_inputs, create_mock_protein_dataset


@pytest.fixture
def tokenizer():
    """Return the ESM2 tokenizer."""
    return get_tokenizer()


@pytest.fixture
def dummy_protein_dataset(tmp_path):
    """Create a mock protein dataset."""
    return create_mock_protein_dataset(tmp_path)


@pytest.fixture
def dummy_parquet_train_val_inputs(tmp_path):
    """Create a mock protein train and val cluster parquet."""
    return create_mock_parquet_train_val_inputs(tmp_path)


@pytest.fixture
def dummy_data_per_token_classification_ft():
    """Fixture providing dummy data for per-token classification fine-tuning.

    Returns:
        list: A list of dummy data for per-token classification fine-tuning.
    """
    data = [
        (
            "TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
            "EEEECCCCCHHHHHHHHHHHHHHHCCCEEEEEECCCHHHHHHHHHCCCCCCCCCEEE",
        ),
        ("LYSGDHSTQGARFLRDLAENTGRAEYELLSLF", "CCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC"),
        ("GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN", "HHHHHCCCCCHHHHHHHHHHHHHHCCCHHHHHHHHHH"),
        (
            "DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
            "HHHHHHHHHHCCCHHHHHCCCCCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC",
        ),
        (
            "KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
            "CHHHHHHHHHHHHHHHCCCEEEEEECCCHHHHHHHHHCCCCCCCCCEEE",
        ),
        (
            "LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
            "HHHHHHHHHHHHHCHHHHHHHHHHHHCCCEECCCEEEECCEEEEECC",
        ),
        (
            "LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
            "HHHHHCCCHHHHHCCCCCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC",
        ),
        ("LYSGDHSTQGARFLRDLAENTGRAEYELLSLF", "CCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC"),
        ("ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP", "HHHHHCHHHHHHHHHHHHCCCEECCCEEEECCEEEEECC"),
        (
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
    data = [(seq, len(seq) / 100.0) for seq, _ in dummy_data_per_token_classification_ft]
    return data
