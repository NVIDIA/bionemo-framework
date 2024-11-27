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


from noodles_fasta_wrapper import IndexedFastaReader


class SequenceAccessor:
    # NOTE: we could totally handle this stuff in Rust if we want.
    def __init__(self, reader, seqid, length):
        self.reader = reader
        self.seqid = seqid
        self.length = length

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slice for range queries
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.length

            # Bounds checking for slice
            if start < 0 or stop > self.length:
                raise IndexError(
                    f"Range [{start}:{stop}] is out of bounds for '{self.seqid}' with length {self.length}."
                )

            region_str = f"{self.seqid}:{start + 1}-{stop}"  # +1 for 1-based indexing
            return self.reader.query_region(region_str)

        elif isinstance(key, int):
            # Handle single integer for single nucleotide queries
            if key < 0 or key >= self.length:
                raise IndexError(f"Position {key} is out of bounds for '{self.seqid}' with length {self.length}.")

            # Query single nucleotide by creating a 1-length region
            region_str = f"{self.seqid}:{key + 1}-{key + 1}"  # +1 for 1-based indexing
            return self.reader.query_region(region_str)

        else:
            raise KeyError("query must be a slice or integer")


class NvFaidx:
    def __init__(self, fasta_path):
        self.reader = IndexedFastaReader(fasta_path)
        self.records = {record.name: record for record in self.reader.records()}

    def __getitem__(self, seqid):
        if seqid not in self.records:
            raise KeyError(f"Sequence '{seqid}' not found in index.")

        # Return a SequenceAccessor for slicing access
        record_length = self.records[seqid].length
        return SequenceAccessor(self.reader, seqid, record_length)

    def __contains__(self, seqid):
        return seqid in self.records

    def __len__(self):
        return len(self.records)

    def keys(self):
        return self.records.keys()


def tests():
    index = NvFaidx("sample.fasta")
    print(index["chr1"][1:10])
    try:
        print(index["chr1"][1:10000])
    except Exception:
        pass

    try:
        print(index["chr1"][1])
    except Exception:
        pass


def test_process_parallel_bug():
    """PyFaidx is a python replacement for faidx that provides a dictionary-like interface to reference genomes. Pyfaidx
    is not process safe, and therefore does not play nice with pytorch dataloaders.

    Ref: https://github.com/mdshw5/pyfaidx/issues/211

    Naively, this problem can be fixed by keeping index objects private to each process. However, instantiating this object can be quite slow.
        In the case of hg38, this can take between 20-30 seconds.

    For a good solution we need three things:
        1) Safe index creation, in multi-process or multi-node scenarios, this should be restricted to a single node where all workers block until it is complete (not implemented above)
        2) Index object instantion must be fast.
        3) Read-only use of the index object must be both thread safe and process safe with python.
    """
    from multiprocessing import Pool

    # NOTE: taken from github issue above, this reproduces the issue with inconsistent reads in parallel.
    fasta = NvFaidx("hg38.fa")

    def read_region(region):
        res = fasta[region["chr"]][region["start"] : region["end"]]
        # This fails when using Pyfaidx ephimerally.
        assert len(res) == 10000

    # This test is quite slow btw
    region_list = [{"chr": "NC_000019.10", "start": 150000, "end": 160000} for i in range(10000000)]

    with Pool(processes=16) as pool:
        results = pool.map(read_region, region_list)


def test_construction_time_hg38():
    # Should safely produce a .fai file in a single process.
    ...


def test_parallel_index_creation():
    # Should safely produce a .fai file in a single process.
    ...


def test_pyfaidx_equivalence():
    # Should produce the same results as pyfaidx.
    ...
