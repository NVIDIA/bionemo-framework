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


import polars as pl
import pysam
import pytest

from bionemo.dnadl.io.dataset import Genome, GenomeIntervalDataset, VCFDataset
from bionemo.dnadl.tools.genome_interval import GenomeInterval


@pytest.fixture(scope="module")
def temp_fasta_file(tmp_path_factory):
    """Fixture that creates a temporary FASTA file with dummy content and yields the path to the file."""
    temp_fasta_path = tmp_path_factory.mktemp("data") / "temp_genome.fa"
    fasta_content = """>chr1
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
>chr2
TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
"""

    # Write the dummy FASTA content to the temporary file
    with open(temp_fasta_path, "w") as fasta_file:
        fasta_file.write(fasta_content)
    return temp_fasta_path


@pytest.fixture(scope="module")
def temp_bed_file(tmp_path_factory):
    """Fixture that creates a temporary FASTA file with dummy content and yields the path to the file."""
    temp_bed_path = tmp_path_factory.mktemp("data") / "temp_genome.bed"
    bed_content = """chr1 10 15 train 0
chr2 1 10 train 1
"""

    # Write the dummy BED file content to the temporary file
    with open(temp_bed_path, "w") as bed_file:
        bed_file.write(bed_content)

    return temp_bed_path


@pytest.fixture(scope="module")
def temp_vcf_file(tmp_path_factory):
    """Fixture that creates a temporary VCF file with dummy content and yields the path to the file."""
    temp_vcf_path = tmp_path_factory.mktemp("data") / "temp_variants.vcf"

    # Dummy VCF content for chromosome 1 and 2
    vcf_content = """##fileformat=VCFv4.2
##source=TestVCF
##contig=<ID=chr1,length=1000000>
##contig=<ID=chr2,length=2000000>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_1\tsample_2
chr1\t5\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/1
chr1\t15\t.\tA\tT\t.\tPASS\t.\tGT\t0/0\t0/1
chr2\t10\t.\tT\tA\t.\tPASS\t.\tGT\t1/1\t0/0
chr2\t18\t.\tT\tC\t.\tPASS\t.\tGT\t0/1\t0/0
"""

    # Write the dummy VCF content to the temporary file
    with open(temp_vcf_path, "w") as vcf_file:
        vcf_file.write(vcf_content)

    # Compress the VCF file using bgzip
    compressed_vcf_path = str(temp_vcf_path) + ".gz"
    pysam.tabix_compress(str(temp_vcf_path), compressed_vcf_path, force=True)

    # Index the compressed VCF file using tabix
    pysam.tabix_index(compressed_vcf_path, preset="vcf")

    return compressed_vcf_path


def test_genome(temp_fasta_file):
    genome = Genome(temp_fasta_file)

    seq_1 = genome.extract_sequence(GenomeInterval("chr1", 10, 15))
    assert len(seq_1) == 5
    assert set(seq_1) == {"A"}

    seq_2 = genome.extract_sequence(GenomeInterval("chr2", 1, 15))
    assert len(seq_2) == 14
    assert set(seq_2) == {"T"}

    with pytest.raises(KeyError):
        genome.extract_sequence(GenomeInterval("chr3", 10, 15))


def test_genome_interval(temp_fasta_file, temp_bed_file):
    genome = Genome(temp_fasta_file)
    bed_file_df = pl.read_csv(
        str(temp_bed_file), separator=" ", has_header=False, new_columns=["chrom", "start", "end"]
    )
    genome_interval = GenomeIntervalDataset(genome, bed_file_df, context_length=20, tokenizer=None)

    assert len(genome_interval) == 2
    assert genome_interval.get_dna_sequence(0) == "AAAAAAAAAAAAAAAAAAAA"
    assert genome_interval.get_dna_sequence(1) == "TTTTTTTTTTTTTTT"


def test_vcf_file(temp_fasta_file, temp_bed_file, temp_vcf_file):
    genome = Genome(temp_fasta_file)
    bed_file_df = pl.read_csv(
        str(temp_bed_file), separator=" ", has_header=False, new_columns=["chrom", "start", "end"]
    )
    genome_interval_dataset = GenomeIntervalDataset(
        genome=genome, tokenizer=None, bed_file_df=bed_file_df, context_length=20
    )
    vcf_dataset = VCFDataset(
        vcf_file=temp_vcf_file,
        genome_interval_dataset=genome_interval_dataset,
        sample_ids=["sample_1", "sample_2"],
        tokenizer=None,
    )

    assert len(vcf_dataset) == 4

    assert vcf_dataset.get_dna_sequence(0) == "AGAAAAAAAAAAAAAAAAAA"
    assert vcf_dataset.get_dna_sequence(2) == "TTTTTTTTATTTTTT"
