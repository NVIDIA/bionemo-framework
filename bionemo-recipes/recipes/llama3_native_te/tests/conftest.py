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

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())


@pytest.fixture
def recipe_path() -> Path:
    """Return the root directory of the recipe."""
    return Path(__file__).parent.parent


@pytest.fixture
def tokenizer_path(recipe_path):
    """Get the path to the nucleotide tokenizer."""
    return str(recipe_path / "tokenizers" / "nucleotide_fast_tokenizer")


@pytest.fixture(scope="session")
def mock_genomic_parquet(tmp_path_factory) -> Path:
    """Create a mock genomic sequences parquet file for testing.

    This fixture creates a small parquet file with synthetic genomic sequences
    that can be used for training tests without relying on external data files.

    Returns:
        Path to the generated parquet file
    """
    tmp_dir = tmp_path_factory.mktemp("data")
    parquet_path = tmp_dir / "test_genomic_sequences.parquet"

    # Create mock genomic sequences with simple repeating patterns
    # These are easy for the model to overfit to, which is perfect for sanity tests
    sequences = [
        "ATCG" * 300,  # 1200 bp - simple ATCG repeat
        "AAAA" * 250 + "TTTT" * 250,  # 2000 bp - alternating A and T blocks
        "GCGC" * 200,  # 800 bp - GC repeat
        "ACGT" * 400,  # 1600 bp - all 4 nucleotides
        "TGCA" * 350,  # 1400 bp - reverse pattern
    ]

    # Create parquet table with 'sequence' column
    table = pa.table(
        {
            "sequence": sequences,
        }
    )

    pq.write_table(table, parquet_path)
    return parquet_path
