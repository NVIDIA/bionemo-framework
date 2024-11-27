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
from typing import Dict, Optional, Sequence

from bionemo.noodles import PyFaidxRecord, PyIndexedMmapFastaReader


__all__: Sequence[str] = (
    "SequenceAccessor",
    "NvFaidx",
)


class SequenceAccessor:
    """SequenceAccessor provides a dictionary-like interface to a single sequence in an indexed FASTA file.

    This allows for random access to the sequence, either by index, or by slice.
    """

    def __init__(self, reader: PyIndexedMmapFastaReader, seqid: str, length: int) -> None:
        """Construct a SequenceAccessor object.

        Args:
            reader (PyIndexedMmapFastaReader): The indexed reader object that provides access to the underlying FASTA file.
            seqid (str): The sequence identifier.
            length (int): The length of the sequence.
        """
        self.reader = reader
        self.seqid = seqid
        self.length = length

    def __getitem__(self, key: int | slice) -> str:  # noqa: D105
        if isinstance(key, slice):
            # Provide defaults for missing arguments in the slice.
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.length

            # Handle negative cases, remember, you can be arbitrarily negative in a slice.
            if start < 0:
                start += self.length
            if stop < 0:
                stop += self.length

            # Clamp normalized indices to valid range
            start = max(0, min(self.length, start))
            stop = max(0, min(self.length, stop))

            # Bounds checking after normalization
            if start > stop:
                return ""  # Return empty string for an empty slice

            # Construct region string
            region_str = f"{self.seqid}:{start + 1}-{stop}"  # +1 for 1-based indexing
            return self.reader.read_sequence_mmap(region_str)

        elif isinstance(key, int):
            # Normalize single integer for negative indexing
            if key < 0:
                key += self.length

            # Bounds checking
            if key < 0 or key >= self.length:
                raise IndexError(f"Position {key} is out of bounds for '{self.seqid}' with length {self.length}.")

            # Query single nucleotide by creating a 1-length region
            region_str = f"{self.seqid}:{key + 1}-{key + 1}"  # +1 for 1-based indexing
            return self.reader.read_sequence_mmap(region_str)

        else:
            raise TypeError("Index must be an integer or a slice.")


class NvFaidx:
    """NvFaidx is a rest + pyo3 replacement for PyFaidx that provides a dictionary-like interface to reference genomes.

    NvFaidx is built using Noodles as a backend for Fai objects, and memory maps for backing the underlying fasta.
    Using a backend of Memmaps provide the following benefits:
        - The kernel implements this mechanism by using page faults
        - Each read in a mmap'd file results in a page fault: there's nothing in memory to read!
        - The kernel handles this page fault by going to the disk, reading the file in the specified offset + index,
            then returning to the user process with what it just read, preventing penalties from context switching.

    *Context*: PyFaidx or _any_ buffered read based index is not process safe, and therefore does not play nice with pytorch dataloaders.
    Due to the order of operations, the underlying file handle is shared between processes, when `seek()` is called to perform random lookups,
    this can cause unexpected behavior in the forked processes.
    Ref: https://github.com/mdshw5/pyfaidx/issues/211

    For a good solution we need three things:
        1) Safe index creation, in multi-process or multi-node scenarios, this should be restricted to a single node
            where all workers block until it is complete (not implemented above)
        2) Index object instantion must be fast.
        3) Read-only use of the index object must be both thread safe and process safe with python.
    """

    def __init__(self, fasta_path: str | Path, faidx_path: Optional[str | Path] = None, ignore_existing_fai=True):
        """Construct a dict-like object representing a memmapped, indexed FASTA file.

        Args:
            fasta_path (str): Path to the FASTA file.
            faidx_path (str): Path to the FAI index file. If None, one will be created.
            ignore_existing_fai (bool): If True, ignore any existing FAI file and create an in-memory index. Note that
                this will also ignore `faidx_path`.
        """
        if isinstance(fasta_path, Path):
            fasta_path = str(fasta_path)
        elif not isinstance(fasta_path, str):
            raise TypeError(f"fasta_path must be a `str` or `pathlib.Path`, got: {type(fasta_path)}")

        if isinstance(faidx_path, Path):
            faidx_path = str(faidx_path)
        elif not isinstance(faidx_path, str) and faidx_path is not None:
            raise TypeError(f"faidx_path must be a `str`, `pathlib.Path`, or None. got: {type(faidx_path)}")

        if ignore_existing_fai:
            self.reader = PyIndexedMmapFastaReader(fasta_path, ignore_existing_fai=ignore_existing_fai)
        elif faidx_path is not None:
            self.reader = PyIndexedMmapFastaReader.from_fasta_and_faidx(fasta_path, faidx_path)
        else:
            # Builds a FAIDX object in memory by default.
            self.reader = PyIndexedMmapFastaReader(fasta_path)

        self.records: Dict[str, PyFaidxRecord] = {record.name: record for record in self.reader.records()}

    def __getitem__(self, seqid: str) -> SequenceAccessor:  # noqa: D105
        if seqid not in self.records:
            raise KeyError(f"Sequence '{seqid}' not found in index.")

        # Return a SequenceAccessor for slicing access
        record_length = self.records[seqid].length
        return SequenceAccessor(self.reader, seqid, record_length)

    def __contains__(self, seqid: str) -> bool:  # noqa: D105
        return seqid in self.records

    def __len__(self) -> int:  # noqa: D105
        return len(self.records)

    def keys(self) -> set[str]:  # noqa: D102
        return set(self.records.keys())

    @staticmethod
    def create_faidx(fasta_filename: str | Path) -> str:
        """Create a FAI index for a FASTA file, the result is saved in the same location as `fasta_filename`, with a .fai extension.

        Args:
            fasta_filename (str): Path to the FASTA file to be indexed.
        """
        if isinstance(fasta_filename, Path):
            fasta_filename = str(fasta_filename)
        return PyIndexedMmapFastaReader.create_faidx(fasta_filename)
