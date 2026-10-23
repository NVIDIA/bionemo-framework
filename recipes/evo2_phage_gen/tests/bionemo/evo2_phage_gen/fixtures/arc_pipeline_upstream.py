# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# fmt: off
"""Upstream ARC functions used as independent regression oracles.

Source: https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/phage_gen/pipelines/genome_design_filtering_pipeline.py
Whole-source SHA-256: ab1a5d388dcd2085ad79909ea15f3666740d76486f8e4b43d25fdf6fb5553091

Only trailing whitespace is stripped from function bodies; code tokens, comments
and other formatting are preserved. Both raw and normalized per-function hashes
are recorded. The import set is reduced to these excerpts' dependencies; neither
architecture imports nor the upstream CLI run on fixture import. Do not format
or change these oracle bodies to agree with the recipe under test. Additions
must be sourced from the same pinned revision, with their original line spans
and hashes recorded below. Upstream license: Apache-2.0 (repository LICENSE).

These functions intentionally include historical behaviors. Tests must state
where parity is required and where a documented correction deliberately differs.
"""

import itertools
import os
import re
import shutil
import subprocess
import time

import pandas as pd
from Bio import SeqIO, SeqFeature
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation
from Bio.SeqRecord import SeqRecord


UPSTREAM_COMMIT = "53f195997257c56c00e5ef8d33a54f5baad143a6"
UPSTREAM_SOURCE_SHA256 = "ab1a5d388dcd2085ad79909ea15f3666740d76486f8e4b43d25fdf6fb5553091"

UPSTREAM_FUNCTIONS = {
    "valid_nt_chars": {
        "lines": [
            140,
            143
        ],
        "source_sha256": "5c03c7d865a7b1a0584f0cf153db2294c0f53aa31215cb568438cc61fa1187f5",
        "sha256": "5c03c7d865a7b1a0584f0cf153db2294c0f53aa31215cb568438cc61fa1187f5"
    },
    "valid_genome_len": {
        "lines": [
            146,
            157
        ],
        "source_sha256": "567e65a109c2989e95f318f53117d1e68c8d8e5bd3b27cd4afc3117df12ce03e",
        "sha256": "708521db0393e125b2df35164478ef900075b31b75e79f49c8de5090aa856abe"
    },
    "calculate_gc_content": {
        "lines": [
            160,
            164
        ],
        "source_sha256": "0fac5e4af6bc35b05eddbd5f760d624a2b16e2124a5b1a262cd5cc7fac5fd21b",
        "sha256": "0fac5e4af6bc35b05eddbd5f760d624a2b16e2124a5b1a262cd5cc7fac5fd21b"
    },
    "valid_gc_content": {
        "lines": [
            167,
            178
        ],
        "source_sha256": "472628223c93e454a64144bc7d1de0a606a4336104b77b96f6f4a235f376be3d",
        "sha256": "dff6f51e6272334a6cf635a7af4309b1f5eda9b12098519e6dd9997d6793a304"
    },
    "calculate_nt_homopolymer_len": {
        "lines": [
            181,
            198
        ],
        "source_sha256": "8fc5b63d9126a18735ef15d8cb4d1f4b63ced11dc499303b86b0a664aba97b6c",
        "sha256": "b76b8f3e75ab946d1ecf6168d46794aebccc948dfdd5ba9396a7606a087f0986"
    },
    "valid_nt_homopolymer_len": {
        "lines": [
            201,
            212
        ],
        "source_sha256": "9fd4ed4090629585f209071672e63302ba86aed2b415b960cb7adf5e510ae26d",
        "sha256": "a247709efd840488d45120346d03e8ed7e9f9e5a1e7f810b1846632703d809aa"
    },
    "append_upstream_of_last_frame_stop": {
        "lines": [
            548,
            596
        ],
        "source_sha256": "e77157682c0a45360aebf988044e7066813bfd6cb41bd7f1281a2767e17c7190",
        "sha256": "821988dc034ef493f313f16647402cf929ae9d18db5d947d1a6984dd0e43c305"
    },
    "valid_mmseqs_pident": {
        "lines": [
            721,
            773
        ],
        "source_sha256": "f9849e206bb68b91e35a8449997da4311c94477af9deb9350888cd5fcff2fbcf",
        "sha256": "f9849e206bb68b91e35a8449997da4311c94477af9deb9350888cd5fcff2fbcf"
    },
    "mmseqs_search_proteins": {
        "lines": [
            915,
            938
        ],
        "source_sha256": "ed5d675ded3f828b7231d96c44db21291d6093ed0f6180efc26e580721d856d9",
        "sha256": "0a509d79f33b5c6e6c10cbd0c1f51d5336f38a281f7299e21a730aa968113009"
    },
    "parse_mmseqs_results": {
        "lines": [
            941,
            951
        ],
        "source_sha256": "a7f524cf07d0e37cff3cb9b22cc8dbecfb805779d89d0ab8a8c96e323c28c1fe",
        "sha256": "707eef3ee75c71a6154e340f9716e65b28b886069b86138dd1c752d7db8da979"
    },
    "mmseqs_results_to_df": {
        "lines": [
            954,
            970
        ],
        "source_sha256": "7283af48e35f1d07e63b1171c769913f4c6658b8b3432479259e25766e4efee7",
        "sha256": "8478fb373713c8e014af337f03e5962d42849b7367d03a07398ae11a66ee2ba3"
    },
    "run_mmseqs_search_proteins": {
        "lines": [
            973,
            1004
        ],
        "source_sha256": "54a4c33f572b22635fcce9ea58109fdd5a4611eb661c8485c88daa1faff48036",
        "sha256": "3feef18b8b05b8519053ccca3de722844f34c0ccbd55d651b32d198a1cc7a2bd"
    },
    "valid_protein_database_hit_count": {
        "lines": [
            1007,
            1039
        ],
        "source_sha256": "b106f4c93bbd0aefd5d7ff9e9543f289497739a95944b0ae7a9b4634f55eb430",
        "sha256": "aafd137f4ec08e7e994cb69afeec21f2b8edcb40e3e0559ee3d352ea7bbff21e"
    },
    "save_df_as_fasta": {
        "lines": [
            1126,
            1141
        ],
        "source_sha256": "439c6ee7e8b711f25e9f1a45fa8ae85c65ba9f47eac443ea63a6f8c3e3205f95",
        "sha256": "25267c08cb9d04a26375a8a6d15dbba6c3c3f634dabc19504d8e3f9544a58b0c"
    },
    "assign_numeric_genome_ids": {
        "lines": [
            1152,
            1161
        ],
        "source_sha256": "68e9d338e699346a413f67fbbaa2aeb25a7b559e7950b126ecc1dd06b62ee17d",
        "sha256": "68e9d338e699346a413f67fbbaa2aeb25a7b559e7950b126ecc1dd06b62ee17d"
    },
    "extract_orf_positions_from_protein_database_hits": {
        "lines": [
            1164,
            1241
        ],
        "source_sha256": "c575fc3abd0542729b116c4cb9e1dc281ff31a1c95eb1a0c9736723ec353210d",
        "sha256": "0c30ff9624c454138601bedf1856f3ad0fa7c07b93d93f473707853dc015d548"
    },
    "create_gff_file": {
        "lines": [
            1244,
            1266
        ],
        "source_sha256": "8535f1556ae300e32d16b08f5c5f627e65bf78c275e757303beff1096343e31f",
        "sha256": "8535f1556ae300e32d16b08f5c5f627e65bf78c275e757303beff1096343e31f"
    },
    "batch_create_gff_files": {
        "lines": [
            1322,
            1384
        ],
        "source_sha256": "e1c7d368df0c62903b78f10a3dd477fd6d3cb3cf40eb71c7af1deb27cd6d3331",
        "sha256": "e1c7d368df0c62903b78f10a3dd477fd6d3cb3cf40eb71c7af1deb27cd6d3331"
    },
    "parse_gff_attributes": {
        "lines": [
            1416,
            1424
        ],
        "source_sha256": "77336772e7863d78c3e740205c71b1dbbbcf9d040db414d78c9ddb864ec076ef",
        "sha256": "77336772e7863d78c3e740205c71b1dbbbcf9d040db414d78c9ddb864ec076ef"
    },
    "convert_gff_to_gbk": {
        "lines": [
            1427,
            1497
        ],
        "source_sha256": "ea34310377f82eb97942b161954cd5110f31e87459359aef8f213235a1e964f8",
        "sha256": "5f03017e144272c84e93fd0c299af6af5e0b1a90e26f5cd4e3bd1be213a82e12"
    },
    "batch_convert_gff_to_gbk": {
        "lines": [
            1500,
            1514
        ],
        "source_sha256": "e1a6d193dc0f9bc913f6c077ff1dfd50d67bb204b823169ca1f0e7b5fd1c5757",
        "sha256": "e1a6d193dc0f9bc913f6c077ff1dfd50d67bb204b823169ca1f0e7b5fd1c5757"
    },
    "calculate_average_protein_percent_identity": {
        "lines": [
            1648,
            1711
        ],
        "source_sha256": "5c71ef3bfe8cc20116799641d82db2511f0dbcea6daec86a06c602c9718dacdf",
        "sha256": "5c71ef3bfe8cc20116799641d82db2511f0dbcea6daec86a06c602c9718dacdf"
    },
    "count_total_num_genes": {
        "lines": [
            1809,
            1861
        ],
        "source_sha256": "99414991c209d8e579be695c2ef6eeb07a8bfedf765eb7440bbbf299ef792f42",
        "sha256": "040a43effcf28eab0865d521e98f89c8ca61405e71a1f1b18376090f8f1cd60e"
    },
    "count_syntenic_genes_all": {
        "lines": [
            1864,
            1987
        ],
        "source_sha256": "045dfcd5c99da8dc2aa98063a3e4fd1066c41468540265ded854ed40af10772f",
        "sha256": "045dfcd5c99da8dc2aa98063a3e4fd1066c41468540265ded854ed40af10772f"
    },
    "valid_syntenic_gene_count": {
        "lines": [
            1990,
            2019
        ],
        "source_sha256": "7a3dd2c83798ca1eaa436d00bc4d1f2c64498c4e582967677bcfe6284cc08b99",
        "sha256": "7a3dd2c83798ca1eaa436d00bc4d1f2c64498c4e582967677bcfe6284cc08b99"
    }
}


def valid_nt_chars(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sequences containing only A, C, G, T characters."""
    filtered_df = sequences_df[sequences_df['sequence'].apply(lambda seq: not re.search(r'[^ACGTacgt]', seq))]
    return filtered_df


def valid_genome_len(sequences_df: pd.DataFrame, length_range: range) -> pd.DataFrame:
    """Filter for sequences within a specified range of lengths."""
    min_len = min(length_range)
    max_len = max(length_range)
    sequences_df = sequences_df.copy()

    # Calculate genome lengths, filter, and record
    sequences_df['genome_length'] = sequences_df['sequence'].apply(len)
    filtered_df = sequences_df[
        (sequences_df['genome_length'] >= min_len) &
        (sequences_df['genome_length'] <= max_len)]
    return filtered_df


def calculate_gc_content(sequence) -> float:
    """Calculate % GC content of a sequence."""
    gc_count = sequence.count('G') + sequence.count('C') + sequence.count('g') + sequence.count('c')
    gc_content = (gc_count / len(sequence)) * 100
    return gc_content


def valid_gc_content(sequences_df: pd.DataFrame, gc_range: range) -> pd.DataFrame:
    """Filter for sequences within a specified range of % GC content."""
    min_gc = min(gc_range)
    max_gc = max(gc_range)
    sequences_df = sequences_df.copy()

    # Calculate GC contents, filter, and record
    sequences_df['gc_content'] = sequences_df['sequence'].apply(calculate_gc_content)
    filtered_df = sequences_df[
        (sequences_df['gc_content'] >= min_gc) &
        (sequences_df['gc_content'] <= max_gc)]
    return filtered_df


def calculate_nt_homopolymer_len(sequence) -> int:
    """Calculate longest homopolymer in a given sequence."""
    sequence = sequence.upper()

    # Create regex patterns for each nucleotide
    patterns = [r'(A+)', r'(C+)', r'(G+)', r'(T+)']

    longest_homopolymer = 0
    for pattern in patterns:
        matches = re.findall(pattern, sequence)
        if matches:
            # Find the longest match for the current nucleotide
            max_len = max(len(match) for match in matches)
            # Update longest_homopolymer if a longer stretch is found
            if max_len > longest_homopolymer:
                longest_homopolymer = max_len

    return longest_homopolymer


def valid_nt_homopolymer_len(sequences_df: pd.DataFrame, homopolymer_range: range) -> pd.DataFrame:
    """Filter for sequences that contain a longest homopolymer within a specified range."""
    min_len = min(homopolymer_range)
    max_len = max(homopolymer_range)
    sequences_df = sequences_df.copy()

    # Calculate homopolymer lengths, filter, and record
    sequences_df['max_nt_homopolymer_length'] = sequences_df['sequence'].apply(calculate_nt_homopolymer_len)
    filtered_df = sequences_df[
        (sequences_df['max_nt_homopolymer_length'] >= min_len) &
        (sequences_df['max_nt_homopolymer_length'] <= max_len)]
    return filtered_df


def append_upstream_of_last_frame_stop(input_fasta: str, output_fasta: str) -> None:
    """
    Finds the first stop codon but the last relative to the three reading frames in a sequence, and appends
    the upstream portion before it to the end of the sequence. This process can be called 'pseudo-circularization'.
    """
    def find_last_frame_stop(seq: Seq) -> int:
        """Find the first stop codon in each reading frame and return the furthest one (including the stop codon itself)."""
        stop_codons = ['TAA', 'TAG', 'TGA']
        first_stops = []

        # Check each of the three reading frames
        for frame in range(3):
            sub_seq = seq[frame:]  # Start from the current reading frame
            for i in range(0, len(sub_seq) - 3, 3):  # Iterate through the sequence in codon steps
                codon = sub_seq[i:i + 3]
                if codon in stop_codons:
                    first_stops.append(i + frame + 3)  # Include the stop codon itself
                    break  # Stop at the first stop codon in this frame

        # Return the furthest stop codon position (including the stop codon)
        return max(first_stops) if first_stops else len(seq)  # If no stop codon found, return full length

    records = []

    # Read input FASTA
    with open(input_fasta, "r") as input_handle:
        counter = 0
        for record in SeqIO.parse(input_handle, "fasta"):
            seq = record.seq

            # Find the last relative stop codon
            last_stop = find_last_frame_stop(seq)

            # Append the sequence upstream of the last stop codon to the end of the full sequence
            new_seq = seq + seq[:last_stop]

            # Create a new record with the updated sequence
            new_record = record
            new_record.seq = new_seq
            records.append(new_record)

            # Increment counter and print every 1000 iterations
            counter += 1
            if counter % 1000 == 0:
                print(f"Pseudo-circularized {counter} sequences")

    # Save to output FASTA
    with open(output_fasta, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")


def valid_mmseqs_pident(mmseqs_df: pd.DataFrame, descriptive_prefix: str, pident_range: tuple, sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sequences that have a % sequence identity to an MMseqs hit within a specified range."""

    min_pident = min(pident_range)
    max_pident = max(pident_range)

    # Drop the sequences column in mmseqs_df (if any) since we don't want to redundantly add it to sequences_df
    if 'sequence' in mmseqs_df.columns:
        mmseqs_df = mmseqs_df.drop(columns=['sequence'])

    # Drop the '_ORF.#' from orfipy prediction results if applicable
    if 'id_prompt' in mmseqs_df.columns and 'ORF' in mmseqs_df['id_prompt'].iloc[0]:
        mmseqs_df['id_prompt'] = mmseqs_df['id_prompt'].str.split('_').str[:-1].str.join('_')
        mmseqs_df = mmseqs_df.loc[mmseqs_df.groupby('id_prompt')[f'{descriptive_prefix}_mmseqs_percent_identity'].idxmax()].reset_index(drop=True)

    # Fix id_prompt to match those in sequences_df
    for index, row in mmseqs_df.iterrows():
        matching_prompt = sequences_df['id_prompt'][sequences_df['id_prompt'].str.contains(row['id_prompt'], regex=False)]
        if not matching_prompt.empty:
            mmseqs_df.at[index, 'id_prompt'] = matching_prompt.values[0]

    # Handle filtering based on pident range
    #results_df = mmseqs_df[
    #    (mmseqs_df[f'{descriptive_prefix}_mmseqs_percent_identity'] >= min_pident) &
    #    (mmseqs_df[f'{descriptive_prefix}_mmseqs_percent_identity'] <= max_pident)
    #]

    #if min_pident > 0:
    #    merged_df = pd.merge(sequences_df, results_df, on='id_prompt', how='left')
    #    filtered_df = merged_df.dropna(subset=[f'{descriptive_prefix}_mmseqs_percent_identity'])
    #elif min_pident == 0:
    #    # Identify hits that exceed max_pident
    #    over_max_df = mmseqs_df[mmseqs_df[f'{descriptive_prefix}_mmseqs_percent_identity'] > max_pident]
    #    over_max_ids = set(over_max_df['id_prompt'])
    #    merged_df = pd.merge(sequences_df, results_df, on='id_prompt', how='left')
    #    merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'] = (merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'].fillna(0))
    #    filtered_df = merged_df[~merged_df['id_prompt'].isin(over_max_ids)]

    # Merge full MMseqs results to preserve no-hit sequences as NaN
    merged_df = pd.merge(sequences_df, mmseqs_df, on='id_prompt', how='left')

    # Fill missing (no-hit) percent identities with 0
    merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'] = (
        merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'].fillna(0)
    )

    # Apply the min/max filter (now can include 0% identity sequences)
    filtered_df = merged_df[
        (merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'] >= min_pident) &
        (merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'] <= max_pident)
    ]

    return filtered_df


def mmseqs_search_proteins(query_fasta: str, mmseqs_db: str, results_dir: str, threads: int=8, split: int=0, sensitivity: float=4.0) -> None:
    """Run MMseqs2 search on the input protein fasta file against the given MMseqs database."""

    os.makedirs(results_dir, exist_ok=True)
    mmseqs_out = os.path.join(results_dir, "mmseqs_result.m8")
    log_file = os.path.join(results_dir, "mmseqs_search.log")

    # Run MMseqs search
    cmd = f"mmseqs easy-search {query_fasta} {mmseqs_db} {mmseqs_out} {results_dir} --threads {threads} --split {split} -s {sensitivity} --remove-tmp-files 1 --format-output 'query,target,evalue,pident'"
    #cmd = f"mmseqs easy-search {query_fasta} {mmseqs_db} {mmseqs_out} {results_dir} --threads {threads} --split {split} -s {sensitivity} -e 1e-15 --remove-tmp-files 1 --format-output 'query,target,evalue,pident'"
    print(f"Running command: {cmd}")
    start_time = time.time()
    try:
        with open(log_file, "w") as log:
            result = subprocess.run(cmd, shell=True, check=True, stdout=log, stderr=log, text=True)
    except subprocess.CalledProcessError as e:
        print(f"MMseqs2 search failed with error: {e.stderr}")
        raise e
    end_time = time.time()
    print(f"MMseqs2 search completed in {end_time - start_time:.2f} seconds.")

    if not os.path.isfile(mmseqs_out):
        print(f"Output file not found: {mmseqs_out}")
    return mmseqs_out


def parse_mmseqs_results(mmseqs_out):
    """Parse MMseqs search results including percent identity."""
    if not os.path.isfile(mmseqs_out):
        raise FileNotFoundError(f"Output file not found: {mmseqs_out}")

    hits = []
    with open(mmseqs_out, "r") as f:
        for line in f:
            query, target, evalue, pident = line.strip().split('\t')
            hits.append((query, target, float(evalue), float(pident)))
    return hits


def mmseqs_results_to_df(hits, query_fasta: str, output_csv: str, descriptive_prefix: str, only_top_hits: bool=True) -> pd.DataFrame:
    """Write the hits and corresponding sequences to a CSV file."""
    sequences = {record.id: record.seq for record in SeqIO.parse(query_fasta, "fasta")}

    data = []
    for query, target, evalue, pident in hits:
        if query in sequences:
            data.append([query, sequences[query], target, evalue, pident])

    df = pd.DataFrame(data, columns=["id_prompt", "sequence", f"{descriptive_prefix}_mmseqs_target", f"{descriptive_prefix}_mmseqs_e_value", f"{descriptive_prefix}_mmseqs_percent_identity"])
    if only_top_hits==True:
        df_top_hits = df.loc[df.groupby("id_prompt")[f"{descriptive_prefix}_mmseqs_e_value"].idxmin()]  # Keep only the top hit per query based on e-value
        df_top_hits.to_csv(output_csv, index=False)
        return df_top_hits
    else:
        df.to_csv(output_csv, index=False)
        return df


def run_mmseqs_search_proteins(query_fasta: str,
                               mmseqs_db: str,
                               results_dir: str,
                               output_csv: str,
                               descriptive_prefix: str,
                               threads: int=8,
                               split: int=0,
                               sensitivity: float=4.0,
                               only_top_hits: bool=True) -> pd.DataFrame:
    """
    Run MMseqs search with the specified directories and files.

    Arguments:
    - query_fasta: Path to the query proteins FASTA file.
    - mmseqs_db: Path to the target proteins MMseqs database.
    - results_dir: Directory to store the MMseqs results.
    - output_csv: Desired filename for MMseqs results.
    - threads: Number of threads for parallel processing (default: 8).
    - split: Desired chunk size for splitting queries in MMseqs search (default: 0).
    - sensitivity: Sensitivity for the MMseqs search (default: 4.0).
    - only_top_hits: Whether or not to keep only the top MMseqs hit for each protein.
    """
    if not os.path.isfile(query_fasta):
        raise FileNotFoundError(f"FASTA file not found: {query_fasta}")
    if not os.path.isdir(mmseqs_db) and not os.path.isfile(mmseqs_db):
        raise FileNotFoundError(f"MMseqs database not found: {mmseqs_db}")

    mmseqs_out = mmseqs_search_proteins(query_fasta, mmseqs_db, results_dir, threads, split, sensitivity)
    hits = parse_mmseqs_results(mmseqs_out)
    df = mmseqs_results_to_df(hits, query_fasta, output_csv, descriptive_prefix, only_top_hits)
    df.to_csv(output_csv, index=False)
    return df


def valid_protein_database_hit_count(protein_database_hits_df: pd.DataFrame, sequences_df: pd.DataFrame, id_column: str='query', min_hits: int=7) -> pd.DataFrame:
    """
    Filters the sequences dataframe to keep only entries with genome IDs that have at least a specified number of MMseqs hits to a given protein database.

    Args:
    - protein_database_hits_df (pd.DataFrame): DataFrame of MMseqs protein database hits.
    - id_column (str): Column name in `protein_database_hits_df` containing query IDs (default: 'query').
    - min_hits (int): Minimum number of hits required to keep a genome (default: 7).
    - sequences_df (pd.DataFrame): DataFrame containing genome IDs to filter.

    Returns:
    - pd.DataFrame: Filtered sequences_df with a column recording the number of hits per genome.
    """

    # Extract genome ID part before the last underscore
    protein_database_hits_df['genome_id'] = protein_database_hits_df[id_column].str.split('_').str[:-1].str.join('_')

    # Count hits for each genome ID
    genome_counts = protein_database_hits_df['genome_id'].value_counts()

    # Filter to keep only genome IDs with at least min_hits
    genomes_to_keep = genome_counts[genome_counts >= min_hits].index

    # Filter protein_database_hits_df to include only relevant genome IDs
    filtered_protein_database_hits_df = protein_database_hits_df[protein_database_hits_df['genome_id'].isin(genomes_to_keep)].copy()

    # Filter the sequences DataFrame
    filtered_df = sequences_df[sequences_df['id_prompt'].isin(genomes_to_keep)].copy()

    # Add a column to filtered_df with the hit counts
    filtered_df['protein_database_hit_count'] = filtered_df['id_prompt'].map(genome_counts)

    return filtered_df


def save_df_as_fasta(sequences_df: pd.DataFrame, output_fasta: str) -> None:
    """Save a pandas DataFrame as a fasta file using the 'id_prompt' as headers and 'sequence' as sequences."""
    records = []

    # Automatically select all columns except 'id_prompt' and 'sequence' as description columns
    description_cols = [col for col in sequences_df.columns if col not in ['id_prompt', 'sequence']]

    for _, row in sequences_df.iterrows():
        # Create a description string by joining all columns except 'id_prompt' and 'sequence'
        description = " ".join([f"{col}:{row[col]}" for col in description_cols])

        # Create SeqRecord with id, sequence, and description
        record = SeqRecord(Seq(row['sequence']), id=row['id_prompt'], description=description)
        records.append(record)

    SeqIO.write(records, output_fasta, "fasta")


def assign_numeric_genome_ids(fasta_file: str, query_genomes: set) -> dict:
    """Assigns numeric IDs to each unique genome in the FASTA file that matches the genomes in query_genomes."""
    genome_id_map = {}
    next_id = 0
    for record in SeqIO.parse(fasta_file, "fasta"):
        genome_name = record.id
        if genome_name in query_genomes and genome_name not in genome_id_map:
            genome_id_map[genome_name] = f"genome_{next_id}"
            next_id += 1
    return genome_id_map


def extract_orf_positions_from_protein_database_hits(
    genomes_fasta_file: str,
    orfs_fasta_file: str,
    protein_database_hits_df: pd.DataFrame,
    genome_id_map: dict
) -> dict:
    """
    Parses the ORFs FASTA file, extracts ORF positions and matching headers of genomes using a protein database hits dataframe,
    and maps them to their corresponding genome IDs using the genome ID map. Adds a GFF entry for the original whole genome length
    using the genomes fasta file (should be the non-pseudo-circularized genomes fasta).
    """
    orf_data = {}
    protein_database_hits_dict = protein_database_hits_df.set_index('id_prompt')[['sequence', 'protein_database_mmseqs_target', 'protein_database_mmseqs_percent_identity', 'annot', 'category']].to_dict('index')  # Fast lookup

    # Process genome lengths from the genomes_fasta_file
    genome_lengths = {}
    for record in SeqIO.parse(genomes_fasta_file, "fasta"):
        genome_name = record.id
        genome_length = len(record.seq)
        genome_lengths[genome_name] = genome_length

    # Process ORF data
    for record in SeqIO.parse(orfs_fasta_file, "fasta"):
        header = record.description
        genome_name = record.id.split('_ORF')[0]

        # Only process ORFs for genomes in the ID map
        if genome_name in genome_id_map and record.id in protein_database_hits_dict:
            genome_id = genome_id_map[genome_name]  # Map to numeric ID
            # Retrieve function and product information
            sequence_value = protein_database_hits_dict[record.id]['sequence']
            percent_identity_value = protein_database_hits_dict[record.id]['protein_database_mmseqs_percent_identity']
            function_value = protein_database_hits_dict[record.id]['category']
            product_value = protein_database_hits_dict[record.id]['annot']
            # Extract the 'ORF.#' part from the record ID
            match = re.search(r'ORF\.\d+', record.id)
            if match:
                orf_id = match.group(0)  # Extract the entire matched 'ORF.#' string
            else:
                orf_id = None  # Handle cases where no match is found
            # Extract start and end positions
            match = re.search(r'\[(\d+)-(\d+)\]', header)
            if match:
                start, end = match.groups()
                # Extract strand (this is new)
                strand_match = re.search(r'\[\d+-\d+\]\((\+|\-)\)', header)
                strand = strand_match.group(1) if strand_match else '+'  # Default to '+' if no strand is found
                orf_entry = {
                    'seq_id': genome_id,
                    'feature_type': 'CDS',
                    'start': start,
                    'end': end,
                    'score': '.',  # Placeholder for score
                    'strand': strand,
                    'phase': '0',
                    'attributes': f"ID={orf_id};function={function_value};product={product_value};seq={sequence_value};percent_identity={percent_identity_value}"
                }
                # Append ORF entry to the corresponding genome ID
                orf_data.setdefault(genome_id, []).append(orf_entry)

    # Add genome length entries to ORF data
    for genome_name, genome_length in genome_lengths.items():
        if genome_name in genome_id_map:
            genome_id = genome_id_map[genome_name]
            genome_entry = {
                'seq_id': genome_id,
                'feature_type': 'region',
                'start': 1,
                'end': genome_length,
                'score': '.',  # Placeholder for score
                'strand': '+',
                'phase': '.',  # Not applicable for regions
                'attributes': f"ID={genome_id};length={genome_length}"
            }
            # Append genome length entry to the corresponding genome ID
            orf_data.setdefault(genome_id, []).append(genome_entry)

    return orf_data


def create_gff_file(orf_data: dict, genome_name: str, genome_id: str, genome_seq: str, output_dir) -> None:
    """Write a .gff file for each genome using ORF data, its given name, ID, and sequence, and saves to an output directory."""
    output_gff = os.path.join(output_dir, f"{genome_id}.gff")

    with open(output_gff, 'w') as gff:
        gff.write("##gff-version 3\n")
        gff.write(f"##sequence-region {genome_id} 1 {len(genome_seq)}\n")
        gff.write(f"##description {genome_name}\n")

        # Write ORF features if they exist for this genome
        if genome_id in orf_data:
            for orf in orf_data[genome_id]:
                gff_line = (
                    f"{orf['seq_id']}\tPredicted genome annotation\t{orf['feature_type']}\t"
                    f"{orf['start']}\t{orf['end']}\t{orf['score']}\t{orf['strand']}\t"
                    f"{orf['phase']}\t{orf['attributes']}\n"
                )
                gff.write(gff_line)

        # Append genome sequence in FASTA format
        gff.write("##FASTA\n")
        gff.write(f">{genome_id}\n")
        gff.write(f"{genome_seq}\n")


def batch_create_gff_files(genomes_fasta_file: str,
                           circular_genomes_fasta_file: str,
                           genomes_csv_file: pd.DataFrame,
                           circular_orfs_fasta_file: str,
                           protein_database_hits_df: pd.DataFrame,
                           output_dir: str,
                           genome_id_map_output_dir: str) -> None:
    """
    Write a .gff file for each genome in a fasta file (should be the pseudo-circularized genomes fasta)
    using information from the original genomes fasta file, and associated ORF data, protein database hits, and tropism protein hits,
    and saves to an output directory.
    Note: This function is currently written to be compatible with PHROGs database protein hits and orfipy-predicted ORFs.
    """

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(genome_id_map_output_dir, exist_ok=True)

    matching_queries = set(genomes_csv_file['id_prompt'].to_list())  # Retrieve genome descriptors

    # Filter genomes_fasta_file to include only matching queries
    filtered_genomes_fasta = f"{output_dir}/filtered_genomes.fasta"
    with open(filtered_genomes_fasta, "w") as out_handle:
        for record in SeqIO.parse(genomes_fasta_file, "fasta"):
            if record.id in matching_queries:
                SeqIO.write(record, out_handle, "fasta")

    # Filter circular_genomes_fasta_file to include only matching queries
    filtered_circular_genomes_fasta = f"{output_dir}/filtered_circular_genomes.fasta"
    with open(filtered_circular_genomes_fasta, "w") as out_handle:
        for record in SeqIO.parse(circular_genomes_fasta_file, "fasta"):
            if record.id in matching_queries:
                SeqIO.write(record, out_handle, "fasta")

    # Assign numeric IDs only to genomes in matching_queries
    genome_id_map = assign_numeric_genome_ids(filtered_genomes_fasta, matching_queries)

    # Load protein database hits as dataframe
    #protein_database_hits_df = pd.read_csv(protein_database_hits_csv_file)

    # Extract ORF positions only for genomes in genome_id_map
    orf_data = extract_orf_positions_from_protein_database_hits(filtered_genomes_fasta, circular_orfs_fasta_file, protein_database_hits_df, genome_id_map)

    # Process each genome in the filtered circular genomes FASTA and create GFF files for matches only
    for genome_record in SeqIO.parse(filtered_circular_genomes_fasta, "fasta"):
        genome_name = genome_record.id
        if genome_name in genome_id_map:  # Only write GFF files for matching genomes
            genome_id = genome_id_map[genome_name]
            genome_seq = str(genome_record.seq)
            create_gff_file(orf_data, genome_name, genome_id, genome_seq, output_dir)
            print(f"GFF file created for {genome_name} as {genome_id}.gff")

    # Save genome ID mapping
    genome_id_map_df = pd.DataFrame(list(genome_id_map.items()), columns=["genome_name", "genome_id"])
    genome_id_map_df.to_csv(f'{genome_id_map_output_dir}/genome_id_map.csv', index=False)

    # Clean up temporary filtered FASTA files
    try:
        os.remove(filtered_genomes_fasta)
        os.remove(filtered_circular_genomes_fasta)
        print("Temporary filtered FASTA files removed.")
    except OSError as e:
        print(f"Error deleting temporary files: {e}")


def parse_gff_attributes(attributes_str: str) -> dict:
    """Parses the attributes field in a GFF line to handle values with spaces."""
    attributes = {}
    for attribute in attributes_str.split(';'):
        key_value = attribute.strip().split('=', 1)  # Split on the first '=' only
        if len(key_value) == 2:
            key, value = key_value
            attributes[key] = value
    return attributes


def convert_gff_to_gbk(gff_file: str, output_gbk: str) -> None:
    """
    Converts a GFF file (with embedded FASTA) to GenBank format.

    Parameters:
        gff_file (str): Path to the GFF file.
        output_gbk (str): Path to save the output GenBank file.
    """
    features = []
    fasta_sequence = None
    seq_id = None

    with open(gff_file) as gff_handle:
        for line in gff_handle:
            if line.startswith("##sequence-region"):
                seq_id = line.split()[1]  # Extract sequence ID
            elif line.startswith("##FASTA"):
                break  # End of GFF section, start of FASTA
            elif not line.startswith("#") and line.strip():
                columns = line.strip().split("\t")

                # Extract feature information
                feature_type = columns[2]
                start = int(columns[3])
                end = int(columns[4])
                strand = 1 if columns[6] == '+' else -1
                attributes_str = columns[8]

                # Parse attributes properly
                attributes = parse_gff_attributes(attributes_str)

                # Extract specific fields
                feature_id = attributes.get("ID", "")
                function = attributes.get("function", "")
                product = attributes.get("product", "")
                percent_identity = attributes.get("percent_identity", "")
                sequence = attributes.get("seq", "")

                # Create the SeqFeature
                feature = SeqFeature.SeqFeature(
                    location=FeatureLocation(start, end, strand=strand),
                    type=feature_type,
                    qualifiers={
                        "ID": feature_id,
                        "function": function,
                        "product": product,
                        "percent_identity": percent_identity,
                        "seq": sequence
                    }
                )
                # Add translation if provided
                if sequence:
                    feature.qualifiers["translation"] = sequence
                features.append(feature)

        # Parse FASTA sequence after ##FASTA section
        fasta_sequence = next(SeqIO.parse(gff_handle, "fasta"))

    # Construct SeqRecord for GenBank format
    seq_record = SeqRecord(
        fasta_sequence.seq,
        id=seq_id,
        name=seq_id,
        description=fasta_sequence.description,
        annotations={"molecule_type": "DNA"}  # Required annotation for GenBank
    )
    seq_record.features.extend(features)

    # Write to GenBank file
    SeqIO.write(seq_record, output_gbk, "genbank")
    print(f"Converted {gff_file} to {output_gbk}")


def batch_convert_gff_to_gbk(input_dir: str, output_dir: str) -> None:
    """
    Converts all GFF files in an input directory to GenBank format in an output directory.

    Parameters:
    - input_dir (str): Path to the directory containing GFF files.
    - output_dir (str): Path to the directory to save the GenBank files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for gff_file in sorted(os.listdir(input_dir)):
        if gff_file.endswith(".gff"):
            input_gff_path = os.path.join(input_dir, gff_file)
            output_gbk_path = os.path.join(output_dir, f"{os.path.splitext(gff_file)[0]}.gbk")
            convert_gff_to_gbk(input_gff_path, output_gbk_path)


def calculate_average_protein_percent_identity(gff_directory: str, results_csv: str, output_csv: str) -> None:
    """
    Calculates the average percent_identity for each genome in a directory of GFF files.
    Combines the results with the 'id_prompt' column in the results_csv and writes a final CSV file.

    Parameters:
    - gff_directory (str): Path to the directory containing GFF files.
    - results_csv (str): Path to the CSV file containing 'id_prompt'.
    - output_csv (str): Path to the output CSV file.
    """
    results = []

    # Loop through all files in the directory
    for gff_file in os.listdir(gff_directory):
        if gff_file.endswith(".gff"):
            file_path = os.path.join(gff_directory, gff_file)

            # Initialize variables for percent_identity calculations
            total_percent_identity = 0
            num_entries = 0
            id_prompt = None

            # Parse the GFF file
            with open(file_path, "r") as f:
                for line in f:
                    # Extract id_prompt from the ##description line
                    if line.startswith("##description"):
                        id_prompt = line.strip().replace("##description", "").strip()

                    # Skip other header or comment lines
                    if line.startswith("#") or line.strip() == "":
                        continue

                    # Split the line into columns
                    columns = line.strip().split("\t")

                    # Check if it's a CDS feature
                    if len(columns) > 8 and columns[2] == "CDS":
                        attributes = columns[8]
                        # Extract percent_identity from attributes
                        attributes_dict = {key: value for key, value in [attr.split("=") for attr in attributes.split(";")]}
                        if "percent_identity" in attributes_dict:
                            percent_identity = float(attributes_dict["percent_identity"])
                            total_percent_identity += percent_identity
                            num_entries += 1

            # Calculate average percent_identity for the genome
            if id_prompt is None:
                id_prompt = gff_file.replace(".gff", "")  # Fallback to filename if description is missing
            if num_entries > 0:
                average_percent_identity = total_percent_identity / num_entries
            else:
                average_percent_identity = 0  # No CDS entries

            # Append the result
            results.append({
                "id_prompt": id_prompt,
                "average_protein_percent_identity": average_percent_identity
            })

    results_df = pd.DataFrame(results)
    latest_results_df = pd.read_csv(results_csv)
    merged_df = pd.merge(latest_results_df, results_df, on="id_prompt", how="left")
    merged_df.to_csv(output_csv, index=False)


def count_total_num_genes(gff_directory: str, results_csv: str) -> None:
    """
    Counts the number of CDS features in each GFF file and adds this count as a new column 'total_num_genes'.
    Merges the results with the 'id_prompt' column in the results_csv and writes the final CSV file.

    Parameters:
    - gff_directory (str): Path to the directory containing GFF files.
    - results_csv (str): Path to the CSV file containing 'id_prompt' which will be overwritten with the CDS counts
    """
    results = []

    # Loop through all files in the directory
    for gff_file in os.listdir(gff_directory):
        if gff_file.endswith(".gff"):
            file_path = os.path.join(gff_directory, gff_file)

            # Initialize variables
            num_genes = 0
            id_prompt = None

            # Parse the GFF file
            with open(file_path, "r") as f:
                for line in f:
                    # Extract id_prompt from the ##description line
                    if line.startswith("##description"):
                        id_prompt = line.strip().replace("##description", "").strip()

                    # Skip header or comment lines
                    if line.startswith("#") or line.strip() == "":
                        continue

                    # Split the line into columns
                    columns = line.strip().split("\t")

                    # Check if it's a CDS feature
                    if len(columns) > 8 and columns[2] == "CDS":
                        num_genes += 1  # Count CDS features

            # Use the filename as fallback if description is missing
            if id_prompt is None:
                id_prompt = gff_file.replace(".gff", "")

            # Append the result
            results.append({
                "id_prompt": id_prompt,
                "total_num_genes": num_genes
            })

    # Update and overwrite the csv
    results_df = pd.DataFrame(results)
    latest_results_df = pd.read_csv(results_csv)
    merged_df = pd.merge(latest_results_df, results_df, on="id_prompt", how="left")
    merged_df.to_csv(results_csv, index=False)


def count_syntenic_genes_all(root_dir: str, gff_dir: str, input_csv: str, output_csv: str) -> None:
    """
    Counts the number of syntenic genes in MMseqs results from all lovis4u folders and merges the results with an input CSV by mapping to `id_prompt`.
    Also records non-syntenic genes and their annotations as columns.
    """

    def extract_prefix(identifier: str) -> str:
        """ Extracts the portion of the string before 'ORF'. """
        match = re.match(r"^(.*?)ORF", identifier)
        return match.group(1) if match else identifier  # Return full string if no match

    def extract_gene_name(identifier: str) -> str:
        """ Extracts the gene name from the given identifier in the format 'genome_#-ORF.#'. """
        match = re.search(r"ORF\.\d+", identifier)
        return match.group(0) if match else identifier

    def count_syntenic_genes(file_path: str, gff_path: str) -> tuple:
        """
        Counts the number of syntenic genes between two genomes and identifies non-syntenic genes.
        Also retrieves gene annotations from the GFF file.

        Parameters:
            file_path (str): Path to the TSV file.
            gff_path (str): Path to the GFF file.

        Returns:
            tuple: The count of valid non-identical gene pairs, a comma-separated list of non-syntenic genes,
                and a comma-separated list of their annotations.
        """
        df = pd.read_csv(file_path, sep="\t", header=None, names=["col1", "col2"])

        # Find mismatches
        mismatched = df[df["col1"] != df["col2"]].copy()

        # Extract prefixes before "ORF"
        mismatched["prefix1"] = mismatched["col1"].apply(extract_prefix)
        mismatched["prefix2"] = mismatched["col2"].apply(extract_prefix)

        # Ensure prefixes are also not identical
        valid_syntenic_pairs = mismatched[mismatched["prefix1"] != mismatched["prefix2"]]
        #print(f'Valid syntenic pairs: {valid_syntenic_pairs}')

        # Count valid syntenic pairs
        syntenic_count = len(valid_syntenic_pairs)

        # Get syntenic genes from valid syntenic pairs where the genome prefix matches
        syntenic_genes_col1 = set(valid_syntenic_pairs[valid_syntenic_pairs["col1"].str.startswith("genome_")]["col1"].apply(extract_gene_name))
        #print(f'Syntenic genes col1: {syntenic_genes_col1}')
        syntenic_genes_col2 = set(valid_syntenic_pairs[valid_syntenic_pairs["col2"].str.startswith("genome_")]["col2"].apply(extract_gene_name))
        #print(f'Syntenic genes col2: {syntenic_genes_col2}')
        syntenic_genes = syntenic_genes_col1.union(syntenic_genes_col2)
        #print(f'Final syntenic genes: {syntenic_genes}')

        # Extract all genes and their annotations from the GFF file
        all_genes = {}
        with open(gff_path, "r") as gff_file:
            for line in gff_file:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue
                attributes = fields[8]

                # Extract gene ID
                gene_id_match = re.search(r"ID=([^;]+)", attributes)
                gene_id = gene_id_match.group(1) if gene_id_match else None

                # Extract gene annotation (product)
                product_match = re.search(r"product=([^;]+)", attributes)
                product = product_match.group(1) if product_match else "Unknown gene"

                # Handle cases where the product is empty or 'nan'
                if not product or product.lower() == "nan":
                    product = "Unknown gene"

                # Record the gene and its annotation if it has a valid ID
                if gene_id and gene_id.startswith("ORF"):
                    all_genes[gene_id] = product

        # Identify non-syntenic genes and their annotations (starting with ORF only)
        non_syntenic_genes = {gene for gene in all_genes.keys() if gene.startswith("ORF")} - syntenic_genes
        print(f'Non-syntenic genes: {non_syntenic_genes}')
        non_syntenic_annotations = [all_genes[gene] for gene in sorted(non_syntenic_genes)]

        return syntenic_count, ",".join(sorted(non_syntenic_genes)), ",".join(non_syntenic_annotations)

    # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    syntenic_counts = {}
    non_syntenic_genes_dict = {}
    non_syntenic_annotations_dict = {}

    # Iterate through lovis4u subfolders
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)

        if os.path.isdir(subfolder_path) and subfolder.startswith("genome_"):
            mmseqs_path = os.path.join(subfolder_path, "mmseqs", "mmseqs_clustering.tsv")
            genome_name = subfolder
            gff_path = os.path.join(gff_dir, f"{genome_name}.gff")

            if os.path.exists(mmseqs_path) and os.path.exists(gff_path):
                syntenic_count, non_syntenic_genes, non_syntenic_annotations = count_syntenic_genes(mmseqs_path, gff_path)
                syntenic_counts[genome_name] = syntenic_count
                non_syntenic_genes_dict[genome_name] = non_syntenic_genes
                non_syntenic_annotations_dict[genome_name] = non_syntenic_annotations
                print(f"Processed {genome_name}: {syntenic_count} syntenic genes, non-syntenic: {non_syntenic_genes}")
            else:
                print(f"Warning: Required file not found for {genome_name}")

    # Load the input CSV
    input_df = pd.read_csv(input_csv)

    # Map syntenic counts, non-syntenic genes, and annotations to the DataFrame
    input_df["num_syntenic_genes"] = input_df["genome_id"].map(syntenic_counts).fillna(0).astype(int)
    input_df["non_syntenic_genes"] = input_df["genome_id"].map(non_syntenic_genes_dict).fillna("")
    input_df["non_syntenic_annotations"] = input_df["genome_id"].map(non_syntenic_annotations_dict).fillna("")

    # Save the updated DataFrame
    input_df.to_csv(output_csv, index=False)


def valid_syntenic_gene_count(input_csv: str, output_csv: str,
                              syntenic_gene_count_range: list, total_gene_count_range: list, syntenic_total_gene_count_remove: set,
                              gff_dir: str, gbk_dir: str, pdf_dir: str, metadata_dir: str) -> None:
    """
    Filters sequences based on combinations of `num_syntenic_genes` and `total_num_genes` ranges.
    Also deletes corresponding GFF, GBK, PDF files and metadata folders for filtered-out genomes.
    """
    df = pd.read_csv(input_csv)

    syntenic_values = range(syntenic_gene_count_range[0], syntenic_gene_count_range[1] + 1)
    total_genes_values = range(total_gene_count_range[0], total_gene_count_range[1] + 1)
    valid_combinations = set(itertools.product(syntenic_values, total_genes_values))

    syntenic_total_gene_count_remove = {tuple(pair) for pair in syntenic_total_gene_count_remove}
    valid_combinations -= syntenic_total_gene_count_remove

    filtered_df = df[df[['num_syntenic_genes', 'total_num_genes']].apply(tuple, axis=1).isin(valid_combinations)]
    removed_ids = set(df["genome_id"]) - set(filtered_df["genome_id"])

    filtered_df.to_csv(output_csv, index=False)

    for genome_id in removed_ids:
        for ext, dir_path in [("gff", gff_dir), ("gbk", gbk_dir), ("pdf", pdf_dir), ("", metadata_dir)]:
            file_path = os.path.join(dir_path, f"{genome_id}.{ext}" if ext else genome_id)
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                print(f"Deleted: {file_path}")
