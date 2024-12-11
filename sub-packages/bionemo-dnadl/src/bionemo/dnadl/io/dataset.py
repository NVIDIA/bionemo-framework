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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from abc import ABC
from pathlib import Path
from typing import Any, Callable, List, Optional

import polars as pl
import pysam
import torch
from torch.utils.data import Dataset

from bionemo.dnadl.tools.genome_editing import create_personal_sequence
from bionemo.dnadl.tools.genome_interval import GenomeInterval
from bionemo.dnadl.tools.vcf import SampleId, read_variants_in_interval
from bionemo.noodles.nvfaidx import NvFaidx


DNATokenizer = Callable[[str], torch.Tensor]


class Genome:
    """Class that creates a genome object from a fasta file. It can be used to extract sequences from the genome."""

    def __init__(self, fasta_file: str | os.PathLike):
        """Instantiate the class."""
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "path to fasta file must exist"
        self.seqs = NvFaidx(str(fasta_file))

    def adjust_interval(self, interval: GenomeInterval, context_length: int) -> GenomeInterval:
        """Used to extend an interval by a fixed amount on both sides."""
        start, end = interval.start, interval.end
        interval_length = interval.end - interval.start
        chromosome_length = self.seqs[interval.chromosome].length
        if interval_length >= context_length:
            return interval

        extra_seq = context_length - interval_length
        extra_left_seq = extra_seq // 2
        extra_right_seq = extra_seq - extra_left_seq

        start -= extra_left_seq
        end += extra_right_seq

        if start <= 0:
            start = 1

        if end > chromosome_length:
            end = chromosome_length

        return GenomeInterval(interval.chromosome, start, end)

    def extract_sequence(self, genome_interval: GenomeInterval) -> str:
        """Extract a sequence from the genome."""
        return str(self.seqs[genome_interval.chromosome][genome_interval.start : genome_interval.end])


class GenomeDataset(Dataset, ABC):
    """Abstract class for datasets that return a DNA sequence."""

    def __init__(self, tokenizer: DNATokenizer):
        """Instantiate the class."""
        self.tokenizer = tokenizer

    def __getitem__(self, ind: int) -> torch.tensor:
        """Get an item from the dataset."""
        sequence = self.get_dna_sequence(ind)
        return self.tokenizer(sequence)

    def get_dna_sequence(self, ind: int) -> str:
        """Get the DNA sequence."""
        ...


class GenomeIntervalDataset(GenomeDataset):
    """Datasets that extract sequences from a genome based on a bed file."""

    def __init__(
        self,
        genome: Genome,
        bed_file_df: pl.DataFrame,
        context_length: int | None = None,
        chr_bed_to_fasta_map: dict = {},
        **kwargs: Any,
    ):
        """Instantiate the class.

        Args:
            genome: Genome object
            bed_file_df: A bed file in the form of a polars DataFrame with chrom, start, and end columns at least
            context_length: Size of the window
            chr_bed_to_fasta_map: Mapping between chromosome names in the bed file and the fasta file
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        assert {"chrom", "start", "end"}.issubset(
            bed_file_df.columns
        ), "bed file df must contain columns ['chrom', 'start', 'end']"
        self.bed_file_df = bed_file_df

        # if the chromosome name in the bed file is different than the keyname in the fasta
        # can remap on the fly
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map
        self.context_length = context_length
        self.genome = genome

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.bed_file_df)

    def get_interval(self, ind: int) -> GenomeInterval:
        """Get the interval from the bed file."""
        chrom = self.chr_bed_to_fasta_map.get(self.bed_file_df["chrom"][ind], self.bed_file_df["chrom"][ind])
        genome_interval = GenomeInterval(chrom, self.bed_file_df["start"][ind], self.bed_file_df["end"][ind])
        if self.context_length is not None:
            return self.genome.adjust_interval(genome_interval, self.context_length)
        return genome_interval

    def get_dna_sequence(self, ind: int) -> str:
        """Get the DNA sequence from the genome."""
        genome_interval = self.get_interval(ind)
        return self.genome.extract_sequence(genome_interval)


class VCFDataset(GenomeDataset):
    """Dataset that creates sequences based on a VCF. It needs a genome and a bed file to specify the windows."""

    def __init__(
        self,
        vcf_file: str | Path,
        genome_interval_dataset: GenomeIntervalDataset,
        sample_ids: Optional[List[SampleId]] = None,
        **kwargs: Any,
    ):
        """Initialize the class."""
        super().__init__(**kwargs)
        self.genome_interval_dataset = genome_interval_dataset
        self.vcf_file = Path(vcf_file)
        assert self.vcf_file.exists(), "path to VCF file must exist"

        if sample_ids is None:
            self._populate_sample_ids_from_vcf()
        else:
            self.sample_ids = sample_ids

    def _populate_sample_ids_from_vcf(self) -> None:
        with pysam.VariantFile(str(self.vcf_file)) as vcf_file:
            self.sample_ids = list(vcf_file.header.samples)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.genome_interval_dataset) * len(self.sample_ids)

    def get_dna_sequence(self, ind: int) -> str:
        """Get an item from the dataset."""
        interval_index = ind // len(self.sample_ids)
        sample_index = ind % len(self.sample_ids)

        ref_sequence = self.genome_interval_dataset.get_dna_sequence(interval_index)
        interval = self.genome_interval_dataset.get_interval(interval_index)
        sample_variants = read_variants_in_interval(interval, self.vcf_file, self.sample_ids)[
            self.sample_ids[sample_index]
        ]
        return create_personal_sequence(ref_sequence, interval, sample_variants)
