#!/usr/bin/env python3
"""
Nucleotide Data Pipeline (No Wildtype Version)

This script processes virulence dataset CSV files and converts protein sequences to nucleotide sequences.
Since we don't have wildtype sequences, we directly convert each protein sequence using seeded codon selection.

Usage:
    python nucleotide_data_pipeline_no_wildtype.py --input_folder /path/to/virulence_dataset --output_dir /path/to/output --seed 42

Features:
- Reads CSV files from virulence dataset directory
- Converts protein sequences to nucleotide sequences using codon tables
- Uses seeded random selection for reproducible results
- Processes all CSV files in the specified directory

"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from Bio.Seq import Seq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard genetic code mapping amino acids to codons
AMINO_ACID_TO_CODONS = {
    'A': ['GCA', 'GCC', 'GCG', 'GCT'],
    'R': ['AGA', 'AGG', 'CGA', 'CGC', 'CGG', 'CGT'],
    'N': ['AAC', 'AAT'],
    'D': ['GAC', 'GAT'],
    'C': ['TGC', 'TGT'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGA', 'GGC', 'GGG', 'GGT'],
    'H': ['CAC', 'CAT'],
    'I': ['ATA', 'ATC', 'ATT'],
    'L': ['CTA', 'CTC', 'CTG', 'CTT', 'TTA', 'TTG'],
    'K': ['AAA', 'AAG'],
    'M': ['ATG'],
    'F': ['TTC', 'TTT'],
    'P': ['CCA', 'CCC', 'CCG', 'CCT'],
    'S': ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT'],
    'T': ['ACA', 'ACC', 'ACG', 'ACT'],
    'W': ['TGG'],
    'Y': ['TAC', 'TAT'],
    'V': ['GTA', 'GTC', 'GTG', 'GTT'],
    '*': ['TAA', 'TAG', 'TGA']  # Stop codons
}

def select_codon_for_amino_acid(amino_acid: str, seed: int = None) -> str:
    """
    Select a codon for an amino acid using seeded random selection.
    
    Args:
        amino_acid: Single letter amino acid code
        seed: Random seed for reproducible selection
        
    Returns:
        Selected codon
    """
    if seed is not None:
        # Use seed to get deterministic selection
        random.seed(seed)
    
    if amino_acid in AMINO_ACID_TO_CODONS:
        codons = AMINO_ACID_TO_CODONS[amino_acid]
        selected_codon = random.choice(codons)
        return selected_codon
    else:
        logger.warning(f"Unknown amino acid: {amino_acid}")
        raise ValueError(f"Unknown amino acid: {amino_acid}")

def convert_protein_to_nucleotide(protein_sequence: str, seed: int = 42) -> str:
    """
    Convert protein sequence to nucleotide sequence using seeded codon selection.
    
    Args:
        protein_sequence: Protein sequence to convert
        seed: Random seed for reproducible codon selection
        
    Returns:
        Nucleotide sequence
    """
    logger.debug(f"Converting protein sequence of length {len(protein_sequence)}")
    
    # Store original random state
    original_seed = random.getstate()
    
    try:
        nucleotide_sequence = ""
        
        for i, amino_acid in enumerate(protein_sequence):
            # Use a different seed for each position to ensure variety
            position_seed = seed + i
            codon = select_codon_for_amino_acid(amino_acid, position_seed)
            nucleotide_sequence += codon
            
        # Validate conversion by translating back
        translated_back = str(Seq(nucleotide_sequence).translate())
        
        if translated_back != protein_sequence:
            raise ValueError(f"Conversion validation failed: {protein_sequence[:50]}... != {translated_back[:50]}...")
        
        logger.debug(f"Successfully converted protein to nucleotide sequence of length {len(nucleotide_sequence)}")
        return nucleotide_sequence
        
    except Exception as e:
        logger.error(f"Error converting protein sequence: {str(e)}")
        raise
    finally:
        # Restore original random state
        random.setstate(original_seed)

def process_virulence_csv(input_file: str, output_file: str, seed: int = 42) -> None:
    """
    Process a single virulence dataset CSV file and convert protein sequences to nucleotide.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file  
        seed: Random seed for reproducible codon selection
    """
    logger.info(f"Processing CSV file: {input_file}")
    
    # Read CSV file
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logger.error(f"Failed to read CSV file {input_file}: {str(e)}")
        raise
    
    # Check if protein column exists
    if 'protein' not in df.columns:
        raise ValueError(f"No 'protein' column found in {input_file}")
    
    # Convert protein sequences to nucleotide sequences
    nucleotide_sequences = []
    successful_conversions = 0
    
    for idx, row in df.iterrows():
        try:
            protein_seq = row['protein']
            
            if pd.isna(protein_seq) or protein_seq == '':
                logger.warning(f"Empty protein sequence at row {idx}")
                nucleotide_sequences.append(None)
                continue
            
            # Convert protein to nucleotide
            nucleotide_seq = convert_protein_to_nucleotide(str(protein_seq), seed)
            nucleotide_sequences.append(nucleotide_seq)
            successful_conversions += 1
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} sequences")
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            nucleotide_sequences.append(None)
    
    # Add nucleotide sequences to dataframe
    df['nucleotide_sequence'] = nucleotide_sequences
    
    # Save results
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to: {output_file}")
        logger.info(f"Successful conversions: {successful_conversions}/{len(df)}")
        
        if successful_conversions > 0:
            # Calculate some statistics
            valid_sequences = df[df['nucleotide_sequence'].notna()]
            avg_protein_length = valid_sequences['protein'].str.len().mean()
            avg_nucleotide_length = valid_sequences['nucleotide_sequence'].str.len().mean()
            logger.info(f"Average protein length: {avg_protein_length:.1f}")
            logger.info(f"Average nucleotide length: {avg_nucleotide_length:.1f}")
            
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {str(e)}")
        raise

def process_virulence_dataset(input_folder: str, output_dir: str, seed: int = 42) -> None:
    """
    Process all CSV files in the virulence dataset folder.
    
    Args:
        input_folder: Path to folder containing virulence dataset CSV files
        output_dir: Output directory for processed files
        seed: Random seed for reproducible codon selection
    """
    logger.info(f"Processing virulence dataset from: {input_folder}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using seed: {seed}")
    
    # Find all CSV files in input folder
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_folder}")
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each CSV file
    successful_files = 0
    failed_files = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"\n=== Processing file {i}/{len(csv_files)}: {csv_file.name} ===")
        
        try:
            # Create output filename
            output_file = output_path / f"{csv_file.stem}_nucleotide.csv"
            
            # Process the file
            process_virulence_csv(str(csv_file), str(output_file), seed)
            successful_files += 1
            logger.info(f"✓ Successfully processed: {csv_file.name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {csv_file.name}: {str(e)}")
            failed_files += 1
            continue
    
    logger.info(f"\n=== Processing Complete ===")
    logger.info(f"Total files: {len(csv_files)}")
    logger.info(f"Successful: {successful_files}")
    logger.info(f"Failed: {failed_files}")
    
    if failed_files > 0:
        logger.warning(f"Some files failed to process. Check the logs above for details.")
    else:
        logger.info("All files processed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Nucleotide Data Pipeline (No Wildtype Version)')
    
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing virulence dataset CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files with nucleotide sequences')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible codon selection (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        process_virulence_dataset(args.input_folder, args.output_dir, args.seed)
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
