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


import os
import pathlib
import random
import tempfile

import pyfaidx
import pytest
import torch

from bionemo.noodles import PyIndexedMmapFastaReader
from bionemo.noodles.nvfaidx import NvFaidx


@pytest.fixture
def sample_fasta():
    return str(pathlib.Path(__file__).parent.parent.parent / "bionemo/noodles/data/sample.fasta")


@pytest.fixture
def dupes_fasta():
    return str(pathlib.Path(__file__).parent.parent.parent / "bionemo/noodles/data/dupes.fasta")


def test_create_faidx_rustbind():
    filename = create_test_fasta(num_seqs=2, seq_length=200)
    faidx_filename = PyIndexedMmapFastaReader.create_faidx(filename, force=False)
    assert os.path.exists(faidx_filename)
    assert faidx_filename == filename + ".fai"

    index = PyIndexedMmapFastaReader.from_fasta_and_faidx(filename, faidx_filename)
    # By default does not build the index from an existing file, but the result should be equivalent.
    index2 = PyIndexedMmapFastaReader(filename)
    assert index.read_sequence_mmap("contig1:1-1") == index2.read_sequence_mmap("contig1:1-1")
    assert index.read_sequence_mmap("contig1:1-10") == index2.read_sequence_mmap("contig1:1-10")
    assert index.read_sequence_mmap("contig1:10-200") == index2.read_sequence_mmap("contig1:10-200")


def test_from_fasta_and_faidx_no_such_faidx():
    filename = create_test_fasta(num_seqs=2, seq_length=200)
    faidx_filename = PyIndexedMmapFastaReader.create_faidx(filename, force=False)
    os.remove(faidx_filename)
    # And this should fail.
    with pytest.raises(FileNotFoundError):
        _ = PyIndexedMmapFastaReader.from_fasta_and_faidx(filename, faidx_filename)


def test_from_fasta_and_faidx():
    # Smoke test, this should all work
    filename = create_test_fasta(num_seqs=2, seq_length=200)
    faidx_filename = PyIndexedMmapFastaReader.create_faidx(filename, force=False)
    index = PyIndexedMmapFastaReader.from_fasta_and_faidx(filename, faidx_filename)
    index2 = PyIndexedMmapFastaReader(filename, ignore_existing_fai=True)
    # Test against constructor for equivalence.
    assert index.read_sequence_mmap("contig1:1-1") == index2.read_sequence_mmap("contig1:1-1")
    assert index.read_sequence_mmap("contig1:1-10") == index2.read_sequence_mmap("contig1:1-10")
    assert index.read_sequence_mmap("contig1:10-200") == index2.read_sequence_mmap("contig1:10-200")

    # now we are going to rename the file, and then try again, we expect the same outcome.
    new_faidx_name = os.path.dirname(faidx_filename) + "/asdfasdfasdf"
    os.rename(faidx_filename, new_faidx_name)
    # Sanity checks for our test.
    assert not os.path.exists(faidx_filename)

    # Now we expect equivalent output even though its using the asdfasdfasdf fai file, and we know the implicit one is missing.
    index = PyIndexedMmapFastaReader.from_fasta_and_faidx(filename, new_faidx_name)
    # Test against constructor for equivalence.
    assert index.read_sequence_mmap("contig1:1-1") == index2.read_sequence_mmap("contig1:1-1")
    assert index.read_sequence_mmap("contig1:1-10") == index2.read_sequence_mmap("contig1:1-10")
    assert index.read_sequence_mmap("contig1:10-200") == index2.read_sequence_mmap("contig1:10-200")


def test_memmap_index(sample_fasta):
    # There exists an equivalent test in Rust.
    fasta_path = sample_fasta
    index = PyIndexedMmapFastaReader(fasta_path)
    assert index.read_sequence_mmap("chr1:1-1") == "A"
    assert index.read_sequence_mmap("chr1:1-2") == "AC"
    assert index.read_sequence_mmap("chr1:1-100000") == "ACTGACTGACTG"
    assert index.read_sequence_mmap("chr2:1-2") == "GG"
    assert index.read_sequence_mmap("chr2:1-1000000") == "GGTCAAGGTCAA"
    # Recall to get python based indexing we add 1 to both start and end, so 1-13 is a 12 character string(full sequence)
    assert index.read_sequence_mmap("chr2:1-11") == "GGTCAAGGTCA"
    assert index.read_sequence_mmap("chr2:1-12") == "GGTCAAGGTCAA"
    assert index.read_sequence_mmap("chr2:1-13") == "GGTCAAGGTCAA"

    assert index.read_sequence_mmap("chr3:1-2") == "AG"
    assert index.read_sequence_mmap("chr3:1-13") == "AGTCAAGGTCCAC"
    assert index.read_sequence_mmap("chr3:1-14") == "AGTCAAGGTCCACG"  # adds first character from next line
    assert (
        index.read_sequence_mmap("chr3:1-83")
        == "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCA"
    )
    assert (
        index.read_sequence_mmap("chr3:1-84")
        == "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG"
    )
    assert (
        index.read_sequence_mmap("chr3:1-10000")
        == "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG"
    )
    assert index.read_sequence_mmap("chr3:84-84") == "G"

    # Handles End of Index
    # Full sequence
    assert index.read_sequence_mmap("chr5:1-1000000") == "A"
    # Only one char, should succeed
    assert index.read_sequence_mmap("chr5:1-2") == "A"

    # Handles end of multi line but non-full sequence entry
    # Full sequence
    assert index.read_sequence_mmap("chr4:1-16") == "CCCCCCCCCCCCACGT"
    assert index.read_sequence_mmap("chr4:1-17") == "CCCCCCCCCCCCACGT"
    assert index.read_sequence_mmap("chr4:1-1000000") == "CCCCCCCCCCCCACGT"

    assert index.read_sequence_mmap("chr4:1-17") == "CCCCCCCCCCCCACGT"

    assert index.read_sequence_mmap("chr4:3-16") == "CCCCCCCCCCACGT"
    assert index.read_sequence_mmap("chr4:17-17") == ""


def test_len(sample_fasta):
    index = NvFaidx(sample_fasta)
    assert len(index) == 5


def test_contains(sample_fasta):
    index = NvFaidx(sample_fasta)
    for i in range(1, 6):
        assert f"chr{i}" in index


def test_create_faidx_nvfaidx(sample_fasta):
    test_fasta_fn = create_test_fasta()

    faidx_fn = NvFaidx(test_fasta_fn, None, ignore_existing_fai=False)

    faidx_fn = NvFaidx.create_faidx(test_fasta_fn, force=False)
    _ = NvFaidx(sample_fasta, faidx_path=faidx_fn, ignore_existing_fai=False)
    assert os.path.exists(faidx_fn)

    faidx_fn = NvFaidx.create_faidx(test_fasta_fn, force=True)
    _ = NvFaidx(sample_fasta, faidx_path=faidx_fn, ignore_existing_fai=False)
    assert os.path.exists(faidx_fn)

    with pytest.raises(FileExistsError):
        faidx_fn = NvFaidx.create_faidx(test_fasta_fn, force=False)

    _ = NvFaidx(sample_fasta, faidx_path=faidx_fn, ignore_existing_fai=False)


def test_iter_all_id_seqs(sample_fasta):
    expected = {
        "chr1": "ACTGACTGACTG",
        "chr2": "GGTCAAGGTCAA",
        "chr3": "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG",
        "chr4": "CCCCCCCCCCCCACGT",
        "chr5": "A",
    }
    fasta_path = sample_fasta
    index = NvFaidx(fasta_path)
    for seq_id in index:
        full_seq = index[seq_id][:]
        assert full_seq == expected[seq_id], seq_id

    for seq_id in index.keys():
        full_seq = index[seq_id][:]
        assert full_seq == expected[seq_id], seq_id

    # Same test different syntax
    for seq_id in index.keys():
        assert index[seq_id].sequence() == expected[seq_id], seq_id

    for_next_test = []
    for seq_id, full_seq in index.items():
        assert full_seq == expected[seq_id], seq_id
        for_next_test.append(full_seq)

    for full_seq, seq_via_items in zip(index.values(), for_next_test):
        assert full_seq == seq_via_items


def test_getitem_bounds(sample_fasta):
    # NOTE make this the correct path, check this file in since we are checking exactness of queries.
    index = NvFaidx(sample_fasta)
    # first element
    assert index["chr1"][0] == "A"
    # normal, in range, query
    assert index["chr1"][1:4] == "CTG"
    # Going beyond the max bound in a slice should truncate at the end of the sequence
    assert index["chr1"][1:10000] == "CTGACTGACTG"
    # Slice up to the last element
    assert index["chr1"][0:-1] == "ACTGACTGACT"
    # Get the full sequence
    assert index["chr1"][:] == "ACTGACTGACTG"
    # equivalent to above
    assert index["chr1"][:-1] == "ACTGACTGACT"
    # -1 should get the last element
    assert index["chr1"][-1:] == "G"
    # non slices return empty string
    assert index["chr1"][100:1] == ""
    # Negative integer indexing is allowed.
    assert index["chr1"][-1] == "G"
    assert index["chr1"][-1 * len(index["chr1"])] == "A"

    with pytest.raises(IndexError):
        # Negative indexing is not allowed to wrap
        index["chr1"][-1000000]

    # Invalid contig should throw an exception
    with pytest.raises(KeyError):
        index["asdfasdfasdfsadf"][-1:]


def _test_faidx_generic(faidx_obj):
    # This is a generic test that should work for both the pyfaidx and nvfaidx implementations.
    index = faidx_obj
    assert index["chr1"][0:1] == "A"
    assert index["chr1"][0:2] == "AC"
    assert index["chr1"][0:100000] == "ACTGACTGACTG"
    assert index["chr2"][0:2] == "GG"
    assert index["chr2"][0:100000] == "GGTCAAGGTCAA"

    assert index["chr3"][0:2] == "AG"
    assert index["chr3"][0:13] == "AGTCAAGGTCCAC"
    # in progress
    assert index["chr3"][0:14] == "AGTCAAGGTCCACG"  # adds first character from next line
    assert index["chr3"][0:83] == "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCA"
    assert (
        index["chr3"][0:84] == "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG"
    )
    assert (
        index["chr3"][0:10000]
        == "AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG"
    )
    assert index["chr3"][83:84] == "G"

    # Handles End of Index
    # Full sequence
    assert index["chr5"][0:1000000] == "A"
    # chr5 has one char, even though this spans 2, it returns len(1)
    assert index["chr5"][0:2] == "A"

    # Handles end of multi line but non-full sequence entry
    # Full sequence
    assert index["chr4"][0:16] == "CCCCCCCCCCCCACGT"
    assert index["chr4"][0:17] == "CCCCCCCCCCCCACGT"
    assert index["chr4"][0:1000000] == "CCCCCCCCCCCCACGT"

    # This one failing is bad, it means we are not calculating the newlines correctly in some conditions.
    assert index["chr4"][0:17] == "CCCCCCCCCCCCACGT"

    # Should see this is out of bounds and return empty or throw an error
    assert index["chr4"][17:17] == ""

    assert index["chr4"][17:] == ""


def test_nvfaidx_python_interface(sample_fasta):
    nvfaidx_index = NvFaidx(sample_fasta)
    pyfaidx_index = pyfaidx.Fasta(sample_fasta)
    _test_faidx_generic(nvfaidx_index)
    _test_faidx_generic(pyfaidx_index)


def test_pyfaidx_nvfaidx_equivalence():
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)
    pyfaidx_fasta = pyfaidx.Fasta(fasta)
    nvfaidx_fasta = NvFaidx(fasta)

    for i in range(100):
        # Deterministically generate regions to grab
        seqid = f"contig{i % 2 + 1}"
        start = i * 1000
        end = start + 1000

        if not pyfaidx_fasta[seqid][start:end] == nvfaidx_fasta[seqid][start:end]:
            raise Exception(f"Pyfaidx and NvFaidx do not match. correct={i}")


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_path, fasta_cls):
        self.fasta = fasta_cls(fasta_path)
        self.keys = list(self.fasta.keys())

    def __len__(self):
        # Gigantic, we dont care.
        return 99999999999

    def __getitem__(self, idx):
        # Always return the same thing to keep it easy, we assume the fasta_created is doing the right thing.
        return str(self.fasta["contig1"][150000:160000])


@pytest.mark.skip
@pytest.mark.xfail(reason="This is a known failure mode for pyfaidx that we are trying to prevent with nvfaidx.")
def test_parallel_index_creation_pyfaidx():
    """
    PyFaidx is a python replacement for faidx that provides a dictionary-like interface to reference genomes. Pyfaidx
    is not process safe, and therefore does not play nice with pytorch dataloaders.

    Ref: https://github.com/mdshw5/pyfaidx/issues/211

    Naively, this problem can be fixed by keeping index objects private to each process. However, instantiating this object can be quite slow.
        In the case of hg38, this can take between 15-30 seconds.

    For a good solution we need three things:
        1) Safe index creation, in multi-process or multi-node scenarios, this should be restricted to a single node where all workers block until it is complete (not implemented above)
        2) Index object instantion must be fast.
        3) Read-only use of the index object must be both thread safe and process safe with python.
    """
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)
    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls=pyfaidx.Fasta), batch_size=16, num_workers=16)
    max_i = 1000
    for i, batch in enumerate(dl):
        # assert length of all elements in batch is 10000
        if i > max_i:
            break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
        assert all(lens_equal), (set(lens), sum(lens_equal))


def test_parallel_index_creation_nvfaidx():
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)

    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls=NvFaidx), batch_size=32, num_workers=16)
    max_i = 1000
    # NOTE this shouldnt be failing uh oh
    for i, batch in enumerate(dl):
        if i > max_i:
            break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
        assert all(lens_equal), (set(lens), sum(lens_equal))


def test_duplicate_seqids(dupes_fasta):
    # Fails since we will get back 1 entry in our dict with 5 in our records list.
    with pytest.raises(ValueError):
        index = NvFaidx(dupes_fasta, allow_duplicate_seqids=False)

    index = NvFaidx(dupes_fasta, allow_duplicate_seqids=True)
    assert list(index.records.keys()) == list(range(5))
    assert len(index) == 5


def test_file_errors():
    # test missing fasta file
    # test failure to parse fasta file
    # test incomplete fai file
    with pytest.raises(FileNotFoundError):
        _ = PyIndexedMmapFastaReader("asdflasdfaslkdfasdf.fasta")

    temp_dir = tempfile.mkdtemp()
    fasta_path = os.path.join(temp_dir, "not_a_fasta.fasta")
    with open(fasta_path, "w") as fasta_file:
        fasta_file.write("this is not a fasta file.\n")

    # Should fail due to invalid fasta file when it tries to create the faidx
    with pytest.raises(RuntimeError):
        _ = PyIndexedMmapFastaReader(fasta_path, ignore_existing_fai=True)

    test_fa = create_test_fasta(num_seqs=2, seq_length=20)
    # now we are going to corrupt the .fai file
    with open(test_fa + ".fai", "w") as f:
        f.write("this is not a valid fai file")
    with pytest.raises(RuntimeError):
        _ = PyIndexedMmapFastaReader(test_fa, ignore_existing_fai=False)

    # But if we create an index in memory, should work!
    _ = PyIndexedMmapFastaReader(test_fa, ignore_existing_fai=True)

    # Should work because 'ignore' implies it only occurs with the fai exists.
    new_test_fasta = create_test_fasta(num_seqs=1, seq_length=200)
    _ = PyIndexedMmapFastaReader(new_test_fasta, ignore_existing_fai=False)


## Benchmarks
def measure_index_creation_time():
    """Observed performance.

    8x speedup for NvFaidx when using
    """
    import time

    # Too slow gen a big genome
    fasta = create_test_fasta(num_seqs=10, seq_length=200_000)
    if os.path.exists(fasta + ".fai"):
        os.remove(fasta + ".fai")
    start = time.time()
    _ = pyfaidx.Fasta(fasta)
    end = time.time()
    elapsed_pyfaidx = end - start

    # Remove the .fai file to prevent cheating.
    if os.path.exists(fasta + ".fai"):
        os.remove(fasta + ".fai")
    start = time.time()
    _ = NvFaidx(fasta, ignore_existing_fai=True)
    end = time.time()
    elapsed_nvfaidx = end - start

    # Now time the creation of the index file
    start = time.time()
    _ = PyIndexedMmapFastaReader.create_faidx(fasta)
    end = time.time()
    elapsed_creation = end - start

    start = time.time()
    NvFaidx(fasta, ignore_existing_fai=False)
    end = time.time()
    elapsed_existing = end - start

    print(f"pyfaidx instantiation: {elapsed_pyfaidx=}")
    print(f"nvfaidx instantiation: {elapsed_nvfaidx=}")
    print(f"nvfaidx instantiation faster by: {elapsed_pyfaidx/elapsed_nvfaidx=}")

    print(f"NvFaidx Index creation time to disk: {elapsed_creation=}")
    print(f"NvFaidx instantiation with existing: {elapsed_existing=}")


def measure_query_time():
    """Observed perf:

    2.3x faster nvfaidx when doing queries through our SequenceAccessor implementation in python land.
    """
    import time

    import numpy as np

    num_iters = 1000
    fasta = create_test_fasta(num_seqs=10, seq_length=200000)

    start_points = np.random.randint(0, 200000, size=num_iters)
    end_points = start_points + np.random.randint(1, 1000, size=num_iters)  # Adjust range size

    # So we are a little slower
    fasta_idx = NvFaidx(fasta)
    start = time.time()
    for i in range(num_iters):
        _ = fasta_idx["contig1"][start_points[i] : end_points[i]]
    end = time.time()
    elapsed_nvfaidx = end - start

    fasta_idx = pyfaidx.Fasta(fasta)
    start = time.time()
    for _ in range(num_iters):
        _ = fasta_idx["contig1"][150000:160000]
    end = time.time()
    elapsed_pyfaidx = end - start

    print(f"pyfaidx query/s: {elapsed_pyfaidx/num_iters=}")
    print(f"nvfaidx query/s: {elapsed_nvfaidx/num_iters=}")
    print(f"nvfaidx query faster by: {elapsed_pyfaidx/elapsed_nvfaidx=}")


# Utility function
def create_test_fasta(num_seqs=2, seq_length=1000):
    """
    Creates a FASTA file with random sequences.

    Args:
        num_seqs (int): Number of sequences to include in the FASTA file.
        seq_length (int): Length of each sequence.

    Returns:
        str: File path to the generated FASTA file.
    """
    temp_dir = tempfile.mkdtemp()
    fasta_path = os.path.join(temp_dir, "test.fasta")

    with open(fasta_path, "w") as fasta_file:
        for i in range(1, num_seqs + 1):
            # Write the header
            fasta_file.write(f">contig{i}\n")

            # Generate a random sequence of the specified length
            sequence = "".join(random.choices("ACGT", k=seq_length))

            # Split the sequence into lines of 60 characters for FASTA formatting
            for j in range(0, len(sequence), 80):
                fasta_file.write(sequence[j : j + 80] + "\n")

    return fasta_path
