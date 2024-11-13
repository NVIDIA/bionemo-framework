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
import pysam

from bionemo.dnadl.tools.genome_interval import GenomeInterval


SampleId = str | int


def _get_max_allele_index(allele_indices: tuple[int | None, int | None]) -> int:
    """Get the maximum allele index from a tuple of allele indices.

    `sample.allele_indices` is a tuple of two indices that associate the sample with alleles
    listed in the list 'variant.alleles', where the zeroth entry is the REF allele. A sample
    could have one or both entries being 0, indicating that one or both strands are REF at
    the specified position. By taking max(sample.allele_indices) you ensure to always extract
    an alternative allele if present or REF otherwise.

    Sometimes, None is contained in sample.allele_indices. This is treated as if it refers
    to the REF allele (ie 0)
    """
    adjusted_allele_indices = [0 if idx is None else idx for idx in allele_indices]
    max_allele_ind = max(adjusted_allele_indices)
    return max_allele_ind


def _match_contig_name(chrm: str, available_contigs: set[int | str]) -> str:
    """Let's try either the standardized version that i expect to be "chr22" or only the number string "22"."""
    assert chrm.startswith("chr")
    if chrm in available_contigs:
        return chrm
    elif chrm[3:] in available_contigs:
        return chrm[3:]
    else:
        raise ValueError(f"`{chrm}` is not in the available contigs {available_contigs}")


def _append_alleles(variant: pysam.libcbcf.VariantRecord, sample_ids: list[SampleId], sample_to_alleles):
    """Extract the alleles depending on the sample_id and haplotype defined at the initialization."""
    for sample_id in sample_ids:
        sample = variant.samples[sample_id]
        allele_ind = _get_max_allele_index(sample.allele_indices)
        if allele_ind > 0:
            sample_to_alleles[sample_id].append(
                {"POS": variant.pos, "REF": variant.alleles[0], "ALT": variant.alleles[allele_ind]}
            )

    return sample_to_alleles


def read_variants_in_interval(
    interval: GenomeInterval,
    vcf_path: Path,
    sample_ids: list[SampleId],
) -> dict[SampleId, pd.DataFrame]:
    """Read variants in an interval from a vcf file."""
    with pysam.VariantFile(str(vcf_path)) as fid:
        contig = _match_contig_name(interval.chromosome, set(fid.header.contigs))
        variants_iter = fid.fetch(contig=contig, start=interval.start - 1, end=interval.end, reopen=True)
        sample_to_alleles_for_interval = {sample_id: [] for sample_id in sample_ids}
        for variant in variants_iter:
            if variant.pos <= interval.start:
                # TODO: sometimes the variant overlaps the boundary of a window. The fetch function is zero base
                # so the interval works fine the problem is that sometimes a variant can indeed
                # overlap the region even if its start is outside.
                # I saw a 45818bp deletion on chr11:3290735, that overlaps with an interval chr11:3313611:3444682
                continue

            _append_alleles(variant, sample_ids, sample_to_alleles_for_interval)

        for sample_id in sample_ids:
            sample_to_alleles_for_interval[sample_id] = pd.DataFrame(sample_to_alleles_for_interval[sample_id])

        return sample_to_alleles_for_interval
