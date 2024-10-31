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
from esm2_infer import infer_model

from bionemo.esm2.api import ESM2Config
from bionemo.testing.data.load import load


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


# TODO: @pytest.mark.parametrize("config_class_name", list(SUPPORTED_CONFIGS))
def test_infer_runs(tmpdir, dummy_protein_csv, dummy_protein_sequences):
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
        include_embeddings=True,
        include_logits=True,
        micro_batch_size=2,
        # config_class=SUPPORTED_CONFIGS[config_class_name],
        config_class=ESM2Config,
    )
    assert results_path.exists(), "Could not find test results pt file."

    results = torch.load(results_path)
    assert isinstance(results, dict)
    keys_included = ["token_logits", "hidden_states", "embeddings", "binary_logits"]
    assert all(key in results for key in keys_included)
    assert results["binary_logits"] is None
    assert results["embeddings"].shape[0] == len(dummy_protein_sequences)
    assert results["hidden_states"].shape[:-1] == (len(dummy_protein_sequences), max_dataset_seq_len)
    assert results["token_logits"].shape[:-1] == (len(dummy_protein_sequences), max_dataset_seq_len)