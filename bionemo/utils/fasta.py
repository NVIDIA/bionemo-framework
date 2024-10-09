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


import gzip
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, TypeVar


class FastaUtil:
    """Packaged this into a class so we can control how the things we do are generated.

    1) Meaning, read from a filename, yield back a dictionary of seq_ids + sequences.

    2) We can provide functions that operate on the dictionary of seq_ids: sequences

    Thus, we are controlling how this structure is created (from a filename), and methods that work
        on the constructed object.

    Alternatively this could be handled with types, but because we are not really using mypy,
        it would be best to capture all this behavior in a class.
    """

    def __init__(self, seq_lookup):
        self.seq_lookup = seq_lookup

    @classmethod
    def from_filename(cls, filename):
        """
        Parse a FASTA file as a FastaUtil object.

        Arguments:
            filename (str): Path to a FASTA-formatted file

        Returns:
            FastaUtil: a `FastaUtil` object.
        """
        seq_lookup = cls.read_fasta(filename)
        return cls(seq_lookup)

    @staticmethod
    def read_fasta(filename):
        """
        Parse a FASTA file as a `Dict` object.

        Arguments:
            filename (str): Path to a FASTA-formatted file

        Returns:
            Dict[str, List[str]] : A map of sequence ID's to a list of sequences.
        """
        seq_lookup = defaultdict(list)
        # Read a FQ, hack for supporting gzipped FQs
        open_fn = gzip.open if filename.endswith(".gz") else open
        for line in open_fn(filename, "rb"):
            line = line.decode("utf-8")
            if line[0] == ">":
                seq_id = line.strip()
            else:
                seq = line.strip()
                seq_lookup[seq_id].append(seq)

        seq_lookup = seq_lookup
        return seq_lookup

    def split_on_ns(self) -> "FastaUtil":
        """Splits a fasta file into new contigs wherever a 'N' is observed.
        Assumes that fasta_entries contains sequences of nucleotides under each entry,
        representing the individual lines in the fasta. Sequences are then concatenated and
        split on 'N's, before being written to a new file. Streams the result to a file.

        Arguments:
            dest_filename (str): where the resulting fasta file is written.
        Returns:
            str: where the resulting fasta file was written.
        """
        new_seqs = {}
        for id, sequences in self.seq_lookup.items():
            new_entries = self._split_ns(sequences)
            for counter, entry in enumerate(new_entries):
                new_id = id + f"_chunk{counter}"
                new_seqs[new_id] = entry
        self.seq_lookup = new_seqs
        return self

    def write(self, dest_filename):
        with open(dest_filename, "w") as fd:
            for id, sequence in self.seq_lookup.items():
                fd.write(f"{id}\n")
                fd.write(sequence + "\n")

        return dest_filename

    @staticmethod
    def _split_ns(sequences):
        """Splits sequences on N's

        Arguments:
            sequences (List[str]): Sequences to split
        Returns:
            List[str]: Sequences without N's and splits where N's were.
        """
        return [seq for seq in "".join(sequences).split("N") if len(seq) > 0]


# NOTE these serve no purpose other than showing the expected inputs.
T = TypeVar("T")
S = TypeVar("S")


class FastaTransform(ABC):
    """Generic preprocessing method. Given an applicable (function) and a sequence of objects,
    we can apply the passed function to every object in the sequence with the `map` method.

    This method captures one interface for representing these constraints. Both `get_elements` and `apply` are
    entirely up to the implementor, if there is a set of files and a set of operations that make sense for a
    specific preprocessing protocol, implement it!

    ResourcePreparer().prepare() always should return a list where every element has the same type.

    Examples:
    ```python
    Resource
    elements: T = ResourcePreparer().prepare() -> List[T]
    filename: S = 'a_filename_we_guarantee_exists.fa'

    elements: List[T] = ResourcePreparer().prepare()
    Preprocessor: Callable[[T], S] # A function that accepts an object of type T, and returns type S.

    def my_preprocsesor(obj: T) -> S:
        # Do work
        yield S

    preprocess = map(my_preprocessor, elements)
    ```
    """

    @abstractmethod
    def apply(self, thing: T) -> S:
        """User implemented function that is applied to all objects in `get_elements`."""
        raise NotImplementedError

    @abstractmethod
    def get_elements(self) -> List[T]:
        """User implemented function that defines 'things'"""
        raise NotImplementedError

    def map(self):
        yield from map(self.apply, self.get_elements())


@dataclass
class FastaSplitNs(FastaTransform):
    """Preprocessor that removes Ns, and creates new contigs between Ns. Runs of Ns are split once."""

    fasta_filenames: List[str]

    def apply(self, fasta_filename):
        new_filename = (
            FastaUtil.from_filename(fasta_filename).split_on_ns().write(self.get_chunked_name(fasta_filename))
        )
        return new_filename

    def get_chunked_name(self, unchunked_name) -> str:
        return unchunked_name + ".chunked.fa"

    def get_elements(self):
        return self.fasta_filenames
