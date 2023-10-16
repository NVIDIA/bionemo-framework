# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

import gzip
import tempfile

# TODO change all references to new fasta memmap dataset instead of janky import
import pyfastx
import pytest

from bionemo.data.fasta_dataset import (
    BACKENDS,
    ConcatFastaDataset,
    DiscretizeFastaDataset,
    InternallyIndexedFastaMemMapDataset,
    PyfastxFastaDataset,
)


def test_fastx_dataset_check_len():
    fasta_file = tempfile.NamedTemporaryFile()
    with open(fasta_file.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGACCC\n>seq2\nTACAT\n')

    fa = pyfastx.Fasta(fasta_file.name)
    dataset = PyfastxFastaDataset(fa, 4)

    assert len(dataset) == 18


test_examples = [
    (-1, 'T'),
    (-2, 'AT'),
    (-5, 'TACA'),
    (-6, 'C'),
    (-10, 'GACC'),
    (0, 'ACAG'),
    (1, 'CAGA'),
    (2, 'AGAT'),
    (3, 'GATT'),
    (4, 'ATTC'),
    (10, 'CCC'),
    (11, 'CC'),
    (12, 'C'),
    (13, 'TACA'),
    (-18, 'ACAG'),
]


@pytest.mark.parametrize('backend', BACKENDS.keys())
@pytest.mark.parametrize('idx,sequence', test_examples)
def test_fasta_dataset(idx, sequence, backend):
    fasta_file = tempfile.NamedTemporaryFile()
    with open(fasta_file.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGACCC\n>seq2\nTACAT\n')

    fa = pyfastx.Fasta(fasta_file.name)
    dataset = PyfastxFastaDataset(fa, 4, backend=backend)
    assert dataset[idx]['seq'] == sequence


@pytest.fixture(scope="module")
def memmap_fasta():
    fasta_file = tempfile.NamedTemporaryFile()
    with open(fasta_file.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGACCC\n>seq2\nTACAT\n')
    dataset = InternallyIndexedFastaMemMapDataset(fasta_file.name, 4)
    return dataset


@pytest.mark.parametrize('idx,sequence', test_examples)
def test_fasta_memmap_dataset(idx, sequence, memmap_fasta):
    dataset = memmap_fasta

    assert dataset[idx]['seq'] == sequence


def test_fasta_memmap_dataset_len(memmap_fasta):
    dataset = memmap_fasta
    assert len(dataset) == 18


test_discrete_examples = [
    (0, 'ACAG'),
    (1, 'ATTC'),
    (2, 'GAC'),
    (3, 'TAT'),
    (4, 'TACC'),
    (5, 'AT'),
]


@pytest.mark.parametrize('backend', BACKENDS.keys())
@pytest.mark.parametrize('idx,sequence', test_discrete_examples)
def test_discrete_fasta_dataset(idx, sequence, backend):
    fasta_file = tempfile.NamedTemporaryFile()
    with open(fasta_file.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGAC\n>seq2\nTAT\n>seq3\nTACCAT\n')

    fa = pyfastx.Fasta(fasta_file.name)
    dataset = DiscretizeFastaDataset(PyfastxFastaDataset(fa, 4, backend=backend))
    assert dataset[idx]['seq'] == sequence


@pytest.fixture(scope='module')
def discretized_fasta_memmap():
    fasta_file = tempfile.NamedTemporaryFile()
    with open(fasta_file.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGAC\n>seq2\nTAT\n>seq3\nTACCAT\n')

    dataset = DiscretizeFastaDataset(InternallyIndexedFastaMemMapDataset([fasta_file.name, fasta_file.name], 4))

    return dataset


test_multiple_discrete_examples = [
    (0, 'ACAG'),
    (1, 'ATTC'),
    (2, 'GAC'),
    (3, 'TAT'),
    (4, 'TACC'),
    (5, 'AT'),
    (6, 'ACAG'),
    (7, 'ATTC'),
    (8, 'GAC'),
    (9, 'TAT'),
    (10, 'TACC'),
    (11, 'AT'),
]


@pytest.mark.parametrize('idx,sequence', test_multiple_discrete_examples)
def test_discrete_fasta_memmap_dataset(idx, sequence, discretized_fasta_memmap):
    assert discretized_fasta_memmap[idx]['seq'] == sequence


test_examples_multiple_files = [
    (0, 'ACAG', 'seq1', 0),
    (3, 'GATT', 'seq1', 3),
    (8, 'GACC', 'seq1', 8),
    (13, 'TACA', 'seq2', 0),
    (17, 'T', 'seq2', 4),
    (18, 'CCCA', 'seq1:2', 0),
    (30, 'ACAT', 'seq2:2', 1),
    (37, 'TATT', 'seq2:2', 8),
]


@pytest.mark.parametrize('idx,sequence,contig,start', test_examples_multiple_files)
def test_fasta_memmap_dataset_multiple_files(
    idx,
    sequence,
    contig,
    start,
):
    fasta_file_1 = tempfile.NamedTemporaryFile()

    with open(fasta_file_1.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGACCC\n>seq2\nTACAT\n')

    fasta_file_2 = tempfile.NamedTemporaryFile()
    with open(fasta_file_2.name, 'w') as fh:
        fh.write('>seq1:2\nCCCATTNA\nNAT\n>seq2:2\nTACATAC\nATATTC\n')

    dataset = InternallyIndexedFastaMemMapDataset(
        [fasta_file_1.name, fasta_file_2.name],
        max_length=4,
        sort_dataset_paths=False,
    )
    assert len(dataset) == 42
    assert dataset[idx]['start'] == start
    assert dataset[idx]['contig'] == contig
    assert dataset[idx]['seq'] == sequence


def test_file_nointernal_newline():
    fasta_file_1 = tempfile.NamedTemporaryFile()

    with open(fasta_file_1.name, 'w') as fh:
        fh.write('>seq1\nACAGAT')

    dataset = InternallyIndexedFastaMemMapDataset(
        [fasta_file_1.name],
        max_length=4,
        sort_dataset_paths=False,
    )

    idx, start, contig, sequence = 4, 4, 'seq1', 'AT'
    assert len(dataset) == 6
    assert dataset[idx]['start'] == start
    assert dataset[idx]['contig'] == contig
    assert dataset[idx]['seq'] == sequence


def test_file_nointernal_newline_multiple_lines():
    fasta_file_1 = tempfile.NamedTemporaryFile()

    with open(fasta_file_1.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\n>seq2\nACA')

    dataset = InternallyIndexedFastaMemMapDataset(
        [fasta_file_1.name],
        max_length=4,
        sort_dataset_paths=False,
    )

    idx, start, contig, sequence = 7, 1, 'seq2', 'CA'
    assert len(dataset) == 9
    assert dataset[idx]['start'] == start
    assert dataset[idx]['contig'] == contig
    assert dataset[idx]['seq'] == sequence


def test_file_nointernal_newline_multiple_lines_multiple_files():
    fasta_file_1 = tempfile.NamedTemporaryFile()

    with open(fasta_file_1.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\n>seq2\nACA')

    dataset = InternallyIndexedFastaMemMapDataset(
        [fasta_file_1.name, fasta_file_1.name],
        max_length=4,
        sort_dataset_paths=False,
    )

    idx, start, contig, sequence = 10, 1, 'seq1', 'CAGA'
    assert len(dataset) == 18
    assert dataset[idx]['start'] == start
    assert dataset[idx]['contig'] == contig
    assert dataset[idx]['seq'] == sequence

    idx, start, contig, sequence = 17, 2, 'seq2', 'A'
    assert len(dataset) == 18
    assert dataset[idx]['start'] == start
    assert dataset[idx]['contig'] == contig
    assert dataset[idx]['seq'] == sequence


test_examples_multiple_files_discrete = [
    (0, 'ACAG'),
    (1, 'ATTC'),
    (2, 'GACC'),
    (3, 'C'),
    (4, 'TACA'),
    (5, 'T'),
    (6, 'CCCA'),
    (7, 'TTNA'),
    (8, 'NAT'),
    (9, 'TACA'),
    (10, 'TACA'),
    (11, 'TATT'),
    (12, 'C'),
]


@pytest.mark.parametrize('backend', BACKENDS.keys())
@pytest.mark.parametrize('idx,sequence', test_examples_multiple_files_discrete)
def test_fasta_dataset_multiple_files_discrete(idx, sequence, backend):
    fasta_file_1 = tempfile.NamedTemporaryFile()
    with open(fasta_file_1.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGACCC\n>seq2\nTACAT\n')

    fasta_file_2 = tempfile.NamedTemporaryFile()
    with gzip.open(fasta_file_2.name, 'wb') as fh:
        fh.write(b'>seq1:2\nCCCATTNA\nNAT\n>seq2:2\nTACATAC\nATATTC\n')

    dataset = ConcatFastaDataset(
        [fasta_file_1.name, fasta_file_2.name],
        max_length=4,
        backend=backend,
        transforms=[
            DiscretizeFastaDataset,
        ],
    )
    assert dataset[idx]['seq'] == sequence
