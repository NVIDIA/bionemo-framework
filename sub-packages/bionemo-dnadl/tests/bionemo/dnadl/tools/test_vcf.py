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


import pandas as pd
import pysam
import pytest

from bionemo.dnadl.tools.genome_interval import GenomeInterval
from bionemo.dnadl.tools.vcf import read_variants_in_interval


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


@pytest.fixture(scope="module")
def interval():
    return GenomeInterval("chr1", 1, 10)


def test_read_variants_in_interval(temp_vcf_file, interval):
    sample_to_alleles = read_variants_in_interval(interval, temp_vcf_file, ["sample_1"])

    assert sample_to_alleles.keys() == {"sample_1"}
    assert sample_to_alleles["sample_1"].equals(pd.DataFrame([{"POS": 5, "REF": "A", "ALT": "G"}]))
