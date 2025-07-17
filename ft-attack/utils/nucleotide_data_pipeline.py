#!/usr/bin/env python3
"""
Nucleotide Data Pipeline

This script processes DMS protein mutation files and converts them to nucleotide sequences.
There's 3 main steps:
1. Get the target protein sequence from the DMS file, and use it to find the wild type nucleotide sequence
2. Use the BLAST API to search for the wild type nucleotide sequence based on the target protein sequence. 
   Sequentially we try:
   a) 100% identity + organism filter (exact match preferred)
   b) 100% identity + no organism filter (exact match any organism)
   c) 99% identity + organism filter + seed-based mismatch correction
3. Based on the wild type nucleotide sequence, we convert the protein mutations to nucleotide mutations using codon tables with seeded random selection.

Usage modes:
1. Process single file: --input_file path/to/file.csv
2. Process entire folder: --input_folder path/to/folder/
3. Process specific files from CSV list: --input_folder path/to/folder/ --file_list path/to/list.csv

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
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tempfile
import requests
import time
from Bio.Align import PairwiseAligner


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

def blast_protein_to_nucleotide(protein_sequence: str, organism: str = None, min_identity: float = 1.0) -> Optional[str]:
    """
    Use BLAST to find nucleotide sequence from protein sequence.
    
    Args:
        protein_sequence: Protein sequence to search
        organism: Optional organism name to help with search
        min_identity: Minimum identity threshold (default 1.0 = 100%)
        
    Returns:
        Nucleotide sequence or None if not found
    """
    logger.info(f"BLASTing protein sequence (length: {len(protein_sequence)}) with min_identity: {min_identity:.1%}")
    
    try:
        # Create a temporary protein sequence record
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
            tmp_file.write(f">query\n{protein_sequence}\n")
            tmp_file.flush()
            
            # Read the sequence back
            with open(tmp_file.name, 'r') as f:
                seq_record = SeqIO.read(f, "fasta")
        
        # Clean up temp file
        os.unlink(tmp_file.name)
        
    except Exception as e:
        logger.error(f"Failed to create sequence record: {str(e)}")
        return None
        
    # Try BLAST search with organism filter first (if provided), then without
    search_attempts = []
    if organism:
        search_attempts.append(("with organism filter", f'"{organism}"[Organism]'))
        search_attempts.append(("without organism filter", None))
    else:
        search_attempts.append(("without organism filter", None))
    
    for attempt_name, entrez_query in search_attempts:
        try:
            logger.info(f"Performing BLAST search {attempt_name}...")
            
            if entrez_query:
                result_handle = NCBIWWW.qblast(
                    "tblastn",  # protein query against nucleotide database
                    "nt",       # nucleotide database
                    seq_record.seq,
                    expect=10,
                    hitlist_size=20,  # Increased to get more candidates
                    entrez_query=entrez_query
                )
            else:
                result_handle = NCBIWWW.qblast(
                    "tblastn",  # protein query against nucleotide database
                    "nt",       # nucleotide database
                    seq_record.seq,
                    expect=10,
                    hitlist_size=20  # Increased to get more candidates
                )
            
            # Parse results
            blast_records = NCBIXML.parse(result_handle)
            blast_record = next(blast_records)
            result_handle.close()
            
            # Find best hit that meets identity threshold
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    # Calculate identity percentage
                    identity = hsp.identities / hsp.align_length
                    
                    logger.info(f"Evaluating hit: {alignment.title}")
                    logger.info(f"Score: {hsp.score}, E-value: {hsp.expect}, Identity: {identity:.2%}")
                    
                    if identity >= min_identity:
                        logger.info(f"Found acceptable match {attempt_name}: {alignment.title}")
                        logger.info(f"Identity: {identity:.2%} (threshold: {min_identity:.2%})")
                        
                        # Extract accession number
                        accession = alignment.accession
                        
                        # Get alignment coordinates
                        subject_start = hsp.sbjct_start
                        subject_end = hsp.sbjct_end
                        
                        logger.info(f"Subject coordinates: {subject_start}-{subject_end}")
                        
                        # Fetch nucleotide sequence for the specific region
                        nucleotide_seq = fetch_nucleotide_sequence_region(accession, subject_start, subject_end)
                        return nucleotide_seq
                    else:
                        logger.info(f"Identity {identity:.2%} below threshold {min_identity:.2%}")
                        
        except Exception as e:
            logger.warning(f"BLAST search {attempt_name} failed: {str(e)}")
            logger.info(f"Continuing to next BLAST attempt...")
            continue
    
    # If we get here, all attempts failed
    logger.warning("No BLAST hits found meeting identity threshold in any search attempt")
    return None



def find_wild_type_nucleotide_sequence(target_seq: str, source_organism: str, identity_threshold: float) -> tuple[Optional[str], str]:
    """
    Find wild type nucleotide sequence using an enhanced three-step strategy:
    1. BLAST with 100% identity + organism filter
    2. BLAST with 100% identity + no organism filter  
    3. BLAST with 99% identity + organism filter + seed-based mismatch correction
    
    Args:
        target_seq: Target protein sequence
        source_organism: Source organism name
        identity_threshold: Minimum identity threshold for validation
        
    Returns:
        Tuple of (wild type nucleotide sequence or None if not found, method used)
    """
    wild_type_nucleotide = None
    
    # Step 1: Try 100% identity with organism filter
    logger.info("Step 1: Attempting BLAST with 100% identity + organism filter")
    wild_type_nucleotide = blast_protein_to_nucleotide(target_seq, source_organism, 1.0)
    
    if wild_type_nucleotide:
        logger.info("✅ Step 1 SUCCESS: Found exact match with organism filter")
        return wild_type_nucleotide, "exact_match"
    
    # Step 2: Try 100% identity without organism filter
    logger.info("Step 2: Attempting BLAST with 100% identity + no organism filter")
    wild_type_nucleotide = blast_protein_to_nucleotide(target_seq, None, 1.0)
    
    if wild_type_nucleotide:
        logger.info("✅ Step 2 SUCCESS: Found exact match without organism filter")
        return wild_type_nucleotide, "exact_match"
    
    # Step 3: Try 99% identity with organism filter + seed-based correction
    logger.info("Step 3: Attempting BLAST with 99% identity + organism filter + seed-based correction")
    blast_nucleotide = blast_protein_to_nucleotide(target_seq, source_organism, 0.99)
    
    if blast_nucleotide:
        logger.info("Found 99% identity match - applying seed-based mismatch correction...")
        corrected_nucleotide = correct_blast_mismatches_with_seed(blast_nucleotide, target_seq, seed=42)
        
        if corrected_nucleotide:
            logger.info("✅ Step 3 SUCCESS: Corrected 99% match using seed-based codon selection")
            return corrected_nucleotide, "99%_match_corrected"
        else:
            logger.warning("❌ Step 3 FAILED: Could not correct 99% match")
    
    # All steps failed
    logger.error("❌ ALL STEPS FAILED: No suitable nucleotide sequence found with BLAST")
    return None, "failed"

def correct_blast_mismatches_with_seed(blast_nucleotide: str, target_protein: str, seed: int = 42) -> Optional[str]:
    """
    Correct amino acid mismatches in BLAST nucleotide sequence using seed-based codon selection.
    
    Args:
        blast_nucleotide: Nucleotide sequence from BLAST
        target_protein: Expected protein sequence
        seed: Random seed for reproducible codon selection
        
    Returns:
        Corrected nucleotide sequence or None if correction fails
    """
    logger.info("Correcting BLAST mismatches using seed-based codon selection...")
    
    # Set seed for reproducible results
    original_seed = random.getstate()
    random.seed(seed)
    
    try:
        # Translate BLAST sequence to see what we got
        blast_protein = str(Seq(blast_nucleotide).translate())
        
        # Find mismatches
        mismatches = []
        min_len = min(len(blast_protein), len(target_protein))
        
        for i in range(min_len):
            if blast_protein[i] != target_protein[i]:
                mismatches.append((i, blast_protein[i], target_protein[i]))
        
        # Check for length mismatches
        if len(blast_protein) != len(target_protein):
            logger.warning(f"Length mismatch: BLAST={len(blast_protein)}, Target={len(target_protein)}")
            if abs(len(blast_protein) - len(target_protein)) > 5:
                logger.error("Length difference too large (>5 amino acids) - cannot correct")
                return None
        
        logger.info(f"Found {len(mismatches)} amino acid mismatches to correct")
        
        if len(mismatches) == 0:
            logger.info("No mismatches found - returning original sequence")
            return blast_nucleotide
        
        if len(mismatches) > 10:
            logger.warning(f"Too many mismatches ({len(mismatches)}) - correction may be unreliable")
        
        # Correct each mismatch
        corrected_nucleotide = blast_nucleotide
        
        for position, blast_aa, target_aa in mismatches:
            # Calculate nucleotide position
            nt_pos = position * 3
            
            # Check bounds
            if nt_pos + 3 > len(corrected_nucleotide):
                logger.error(f"Mismatch at position {position+1} exceeds sequence length")
                continue
            
            # Get original codon
            original_codon = corrected_nucleotide[nt_pos:nt_pos+3]
            
            # Select new codon for target amino acid using seed
            new_codon = select_codon_for_amino_acid(target_aa, seed + position)  # Different seed per position
            
            # Replace the codon
            corrected_nucleotide = (corrected_nucleotide[:nt_pos] + 
                                  new_codon + 
                                  corrected_nucleotide[nt_pos + 3:])
            
            logger.info(f"Corrected position {position+1}: {blast_aa} -> {target_aa} "
                       f"(codon: {original_codon} -> {new_codon})")
        
        # Validate the correction worked
        final_protein = str(Seq(corrected_nucleotide).translate())
        
        # Check if correction was successful
        if final_protein == target_protein:
            logger.info("✅ Mismatch correction successful!")
            logger.info(f"Corrected {len(mismatches)} mismatches using seed-based codon selection")
            return corrected_nucleotide
        elif final_protein[:min_len] == target_protein[:min_len]:
            # Handle length differences
            if len(final_protein) > len(target_protein):
                # Truncate extra nucleotides
                target_nt_len = len(target_protein) * 3
                corrected_nucleotide = corrected_nucleotide[:target_nt_len]
                final_protein = str(Seq(corrected_nucleotide).translate())
                
                if final_protein == target_protein:
                    logger.info("✅ Mismatch correction successful after truncation!")
                    return corrected_nucleotide
        
        # If we get here, correction failed
        logger.error("❌ Mismatch correction failed validation")
        logger.error(f"Expected: {target_protein[:50]}...")
        logger.error(f"Got:      {final_protein[:50]}...")
        return None
        
    except Exception as e:
        logger.error(f"Error during mismatch correction: {str(e)}")
        return None
    
    finally:
        # Restore original random state
        random.setstate(original_seed)

def fetch_nucleotide_sequence(accession: str) -> Optional[str]:
    """
    Fetch nucleotide sequence from NCBI using accession number.
    
    Args:
        accession: NCBI accession number
        
    Returns:
        Nucleotide sequence or None if not found
    """
    try:
        logger.info(f"Fetching nucleotide sequence for {accession}")
        
        # Use Entrez API to fetch sequence
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'nuccore',
            'id': accession,
            'rettype': 'fasta',
            'retmode': 'text'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse FASTA response
        lines = response.text.strip().split('\n')
        if len(lines) > 1:
            sequence = ''.join(lines[1:])  # Skip header line
            logger.info(f"Retrieved sequence of length {len(sequence)}")
            return sequence
        else:
            logger.warning(f"No sequence found for {accession}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch sequence for {accession}: {str(e)}")
        return None

def fetch_nucleotide_sequence_region(accession: str, start: int, end: int) -> Optional[str]:
    """
    Fetch specific region of nucleotide sequence from NCBI.
    
    Args:
        accession: NCBI accession number
        start: Start position (1-based, BLAST coordinates)
        end: End position (1-based, BLAST coordinates)
        
    Returns:
        Nucleotide sequence for the specified region or None if not found
    """
    try:
        logger.info(f"Fetching nucleotide sequence region {start}-{end} for {accession}")
        
        # Determine if this is reverse strand (start > end in BLAST coordinates)
        is_reverse = start > end
        
        # For Entrez API, always use smaller coordinate as seq_start
        seq_start = min(start, end)
        seq_stop = max(start, end)
        
        # Use Entrez API to fetch sequence region
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'nuccore',
            'id': accession,
            'rettype': 'fasta',
            'retmode': 'text',
            'seq_start': seq_start,
            'seq_stop': seq_stop
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse FASTA response
        lines = response.text.strip().split('\n')
        if len(lines) > 1:
            sequence = ''.join(lines[1:])  # Skip header line
            
            # Handle reverse complement if needed
            if is_reverse:
                # Reverse complement the sequence
                from Bio.Seq import Seq
                bio_seq = Seq(sequence)
                sequence = str(bio_seq.reverse_complement())
                logger.info(f"Applied reverse complement for reverse strand coordinates {start}-{end}")
            
            logger.info(f"Retrieved sequence region of length {len(sequence)} ({'reverse' if is_reverse else 'forward'} strand)")
            return sequence
        else:
            logger.warning(f"No sequence found for {accession} region {start}-{end}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch sequence region for {accession}: {str(e)}")
        return None

def validate_nucleotide_protein_correspondence(nucleotide_seq: str, protein_seq: str, identity_threshold: float = 0.8) -> bool:
    """
    Validate that nucleotide sequence corresponds to protein sequence.
    
    Args:
        nucleotide_seq: Nucleotide sequence
        protein_seq: Protein sequence
        identity_threshold: Minimum identity threshold (default 0.8 = 80%)
        
    Returns:
        True if sequences correspond, False otherwise
    """
    try:
        logger.info(f"Validating nucleotide sequence of length {len(nucleotide_seq)} against protein sequence of length {len(protein_seq)}")
        
        # Create Bio.Seq object for easier manipulation
        bio_seq = Seq(nucleotide_seq)
        
        # Try all 6 reading frames (3 forward + 3 reverse complement)
        sequences_to_test = [
            ("Forward", nucleotide_seq),
            ("Reverse", str(bio_seq.reverse_complement()))
        ]
        
        # FIRST ATTEMPT: Standard validation (original behavior)
        logger.info("Attempting standard validation (stopping at stop codons)...")
        
        for strand_name, seq in sequences_to_test:
            for frame in range(3):
                frame_seq = seq[frame:]
                # Only translate if we have at least some complete codons
                if len(frame_seq) >= 3:
                    # Translate and stop at first stop codon (original behavior)
                    translated_protein = str(Seq(frame_seq).translate(to_stop=True))
                    
                    logger.info(f"{strand_name} frame {frame+1}: {translated_protein[:50]}...")
                    
                    # Check for exact match
                    if translated_protein == protein_seq:
                        logger.info(f"Exact match found in {strand_name.lower()} frame {frame+1}")
                        return True
                    
                    # Check for partial matches (common with BLAST hits)
                    if len(translated_protein) > 10:
                        # Check if protein is contained in translation or vice versa
                        if translated_protein in protein_seq:
                            logger.info(f"Translated protein is contained in expected protein sequence ({strand_name.lower()} frame {frame+1})")
                            return True
                        elif protein_seq in translated_protein:
                            logger.info(f"Expected protein sequence is contained in translated protein ({strand_name.lower()} frame {frame+1})")
                            return True
                        
                        # Check for significant overlap (at least threshold% of the shorter sequence)
                        min_len = min(len(translated_protein), len(protein_seq))
                        if min_len > 0:
                            # Simple character-by-character comparison
                            matches = sum(1 for i in range(min_len) if i < len(translated_protein) and i < len(protein_seq) and translated_protein[i] == protein_seq[i])
                            identity = matches / min_len
                            if identity >= identity_threshold:
                                logger.info(f"High identity match ({identity:.2%}) in {strand_name.lower()} frame {frame+1} (threshold: {identity_threshold:.2%})")
                                return True
        
        # FALLBACK: Enhanced validation for fusion constructs
        logger.warning("Standard validation failed. Attempting fallback validation for fusion constructs...")
        logger.info("Fallback: Translating entire sequences without stopping at stop codons...")
        
        for strand_name, seq in sequences_to_test:
            for frame in range(3):
                frame_seq = seq[frame:]
                if len(frame_seq) >= 3:
                    # Translate entire sequence without stopping (for fusion constructs)
                    translated_protein_full = str(Seq(frame_seq).translate(to_stop=False))
                    
                    logger.info(f"Fallback {strand_name} frame {frame+1}: {translated_protein_full[:50]}... (length: {len(translated_protein_full)})")
                    
                    # Check for exact match with full translation
                    if translated_protein_full == protein_seq:
                        logger.info(f"FALLBACK SUCCESS: Exact match found in full translation ({strand_name.lower()} frame {frame+1})")
                        return True
                    
                    # Check for target protein as substring in full translation
                    if len(translated_protein_full) > 10 and protein_seq in translated_protein_full:
                        logger.info(f"FALLBACK SUCCESS: Target protein found as substring in full translation ({strand_name.lower()} frame {frame+1})")
                        
                        # Find the position for additional validation
                        start_pos = translated_protein_full.find(protein_seq)
                        end_pos = start_pos + len(protein_seq)
                        logger.info(f"Target protein found at positions {start_pos}-{end_pos} in translation")
                        logger.info(f"Context: ...{translated_protein_full[max(0, start_pos-10):start_pos]}[{protein_seq[:20]}...]{translated_protein_full[end_pos:end_pos+10]}...")
                        return True
                    
                    # Check if full translation is contained in target
                    elif len(translated_protein_full) > 10 and translated_protein_full in protein_seq:
                        logger.info(f"FALLBACK SUCCESS: Full translation is contained in target protein ({strand_name.lower()} frame {frame+1})")
                        return True
        
        # FINAL FALLBACK: Try different starting positions (coordinate adjustment)
        logger.warning("Fusion construct validation failed. Attempting coordinate adjustment...")
        logger.info("Trying different starting positions to account for potential BLAST coordinate misalignment...")
        
        # Try shifting the sequence by 1-2 nucleotides in case BLAST coordinates aren't aligned to codon boundaries
        for shift in [-2, -1, 1, 2]:
            logger.info(f"Trying coordinate shift: {shift} nucleotides")
            
            for strand_name, seq in sequences_to_test:
                # Apply shift - but ensure we don't go out of bounds
                if shift < 0:
                    shifted_seq = seq[abs(shift):]  # Remove nucleotides from start
                else:
                    shifted_seq = seq[:-shift] if shift < len(seq) else ""  # Remove nucleotides from end
                
                if len(shifted_seq) < 3:
                    continue
                    
                for frame in range(3):
                    frame_seq = shifted_seq[frame:]
                    if len(frame_seq) >= 3:
                        # Try both translation approaches
                        translated_with_stop = str(Seq(frame_seq).translate(to_stop=True))
                        translated_full = str(Seq(frame_seq).translate(to_stop=False))
                        
                        # Check exact matches
                        if translated_with_stop == protein_seq:
                            logger.info(f"COORDINATE ADJUSTMENT SUCCESS: Exact match with shift {shift}, {strand_name.lower()} frame {frame+1} (to_stop=True)")
                            return True
                        if translated_full == protein_seq:
                            logger.info(f"COORDINATE ADJUSTMENT SUCCESS: Exact match with shift {shift}, {strand_name.lower()} frame {frame+1} (to_stop=False)")
                            return True
                            
                        # Check substring matches
                        if protein_seq in translated_full:
                            logger.info(f"COORDINATE ADJUSTMENT SUCCESS: Substring match with shift {shift}, {strand_name.lower()} frame {frame+1}")
                            start_pos = translated_full.find(protein_seq)
                            logger.info(f"Target protein found at position {start_pos} in shifted translation")
                            return True
        
        # If we get here, all attempts failed
        logger.warning(f"All validation attempts failed (standard, fallback, and coordinate adjustment)")
        logger.warning(f"Expected: {protein_seq[:50]}...")
        logger.warning(f"Nucleotide: {nucleotide_seq[:150]}...")
        return False
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False

def load_or_create_reference_file(reference_file_path: str) -> pd.DataFrame:
    """
    Load DMS reference file or create if it doesn't exist.
    
    Args:
        reference_file_path: Path to reference file
        
    Returns:
        DataFrame with reference data
    """
    if os.path.exists(reference_file_path):
        logger.info(f"Loading existing reference file: {reference_file_path}")
        df = pd.read_csv(reference_file_path)
        
        # Backward compatibility: add sequence_method column if missing
        if 'sequence_method' not in df.columns:
            df['sequence_method'] = 'exact_match'  # Default for existing entries
            logger.info("Added sequence_method column for backward compatibility")
        
        return df
    else:
        logger.info(f"Creating new reference file: {reference_file_path}")
        # Create basic structure based on expected columns
        df = pd.DataFrame(columns=[
            'DMS_filename', 'target_seq', 'source_organism', 'wild_type_nucleotide', 'sequence_method'
        ])
        return df

def update_reference_file(reference_file_path: str, dms_filename: str, 
                         target_seq: str, source_organism: str, 
                         wild_type_nucleotide: str, sequence_method: str = "exact_match") -> None:
    """
    Update reference file with wild type nucleotide sequence.
    
    Args:
        reference_file_path: Path to reference file
        dms_filename: DMS filename
        target_seq: Target protein sequence
        source_organism: Source organism
        wild_type_nucleotide: Wild type nucleotide sequence
        sequence_method: Method used to obtain sequence (exact_match, close_match_corrected, etc.)
    """
    # Load or create reference file
    df = load_or_create_reference_file(reference_file_path)
    
    # Check if entry already exists
    existing_row = df[df['DMS_filename'] == dms_filename]
    
    if len(existing_row) > 0:
        # Update existing row
        df.loc[df['DMS_filename'] == dms_filename, 'wild_type_nucleotide'] = wild_type_nucleotide
        df.loc[df['DMS_filename'] == dms_filename, 'sequence_method'] = sequence_method
        logger.info(f"Updated nucleotide sequence for {dms_filename} (method: {sequence_method})")
    else:
        # Add new row
        new_row = pd.DataFrame({
            'DMS_filename': [dms_filename],
            'target_seq': [target_seq],
            'source_organism': [source_organism],
            'wild_type_nucleotide': [wild_type_nucleotide],
            'sequence_method': [sequence_method]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        logger.info(f"Added new entry for {dms_filename}")
    
    # Save updated file
    os.makedirs(os.path.dirname(reference_file_path), exist_ok=True)
    df.to_csv(reference_file_path, index=False)
    logger.info(f"Saved reference file: {reference_file_path}")

def select_codon_for_amino_acid(amino_acid: str, seed: int = None, organism: str = None) -> str:
    """
    Select a codon for an amino acid using organism-appropriate codon selection.
    
    Args:
        amino_acid: Single letter amino acid code
        seed: Random seed for reproducible selection
        organism: Source organism (if viral, will use viral codon tables)
        
    Returns:
        Selected codon
    """
    if seed is not None:
        random.seed(seed)
    
    # Note: Future enhancement could include organism-specific codon usage tables
    
    # Fall back to generic codon table
    if amino_acid in AMINO_ACID_TO_CODONS:
        codons = AMINO_ACID_TO_CODONS[amino_acid]
        selected_codon = random.choice(codons)
        return selected_codon
    else:
        logger.warning(f"Unknown amino acid: {amino_acid}")
        raise ValueError(f"Unknown amino acid: {amino_acid}")

def parse_mutation(mutant_str: str) -> Tuple[str, int, str]:
    """
    Parse single mutation string like 'A58C' into original AA, position, new AA.
    
    Args:
        mutant_str: Single mutation string
        
    Returns:
        Tuple of (original_aa, position, new_aa)
    """
    if len(mutant_str) < 3:
        raise ValueError(f"Invalid mutation string: {mutant_str}")
    
    original_aa = mutant_str[0]
    new_aa = mutant_str[-1]
    position_str = mutant_str[1:-1]
    
    try:
        position = int(position_str)
    except ValueError:
        raise ValueError(f"Invalid position in mutation string: {mutant_str}")
    
    return original_aa, position, new_aa

def parse_multiple_mutations(mutant_str: str) -> List[Tuple[str, int, str]]:
    """
    Parse multiple mutations string like 'D27V:S46W' into list of mutations.
    
    Args:
        mutant_str: Multiple mutations string (colon-separated)
        
    Returns:
        List of tuples (original_aa, position, new_aa)
    """
    if ':' not in mutant_str:
        # Single mutation
        return [parse_mutation(mutant_str)]
    
    mutations = []
    mutation_parts = mutant_str.split(':')
    
    for part in mutation_parts:
        part = part.strip()
        if part:  # Skip empty parts
            mutations.append(parse_mutation(part))
    
    if not mutations:
        raise ValueError(f"No valid mutations found in string: {mutant_str}")
    
    return mutations

def convert_protein_mutation_to_nucleotide(wild_type_nucleotide: str, 
                                         mutation_str: str, 
                                         seed: int = None) -> str:
    """
    Convert protein mutation(s) to nucleotide mutation.
    
    Args:
        wild_type_nucleotide: Wild type nucleotide sequence
        mutation_str: Mutation string (e.g., 'A58C' or 'D27V:S46W')
        seed: Random seed for codon selection
        
    Returns:
        Mutated nucleotide sequence
    """
    # Parse multiple mutations
    mutations = parse_multiple_mutations(mutation_str)
    
    # Sort mutations by position to avoid conflicts when applying them
    mutations.sort(key=lambda x: x[1])  # Sort by position
    
    # Start with wild type sequence
    current_nucleotide = wild_type_nucleotide
    wild_type_protein = str(Seq(wild_type_nucleotide).translate())
    
    # Apply mutations in order
    for original_aa, position, new_aa in mutations:
        # Convert to 0-based indexing
        position_0_based = position - 1
        
        # Calculate nucleotide position (each amino acid = 3 nucleotides)
        nucleotide_position = position_0_based * 3
        
        # Check bounds
        if nucleotide_position + 3 > len(current_nucleotide):
            raise ValueError(f"Mutation position {position} exceeds sequence length")
        
        # Get new codon for mutated amino acid
        new_codon = select_codon_for_amino_acid(new_aa, seed)
        
        # Apply mutation to current sequence
        current_nucleotide = (current_nucleotide[:nucleotide_position] + 
                             new_codon + 
                             current_nucleotide[nucleotide_position + 3:])
        
        logger.debug(f"Applied mutation {original_aa}{position}{new_aa}")
        logger.debug(f"Nucleotide change at position {nucleotide_position}: {current_nucleotide[nucleotide_position:nucleotide_position+3]} -> {new_codon}")
    
    # ADDED: End-to-end validation - check that the result translates correctly
    mutated_protein = str(Seq(current_nucleotide).translate())
    
    # Create expected mutated protein sequence by applying all mutations
    expected_mutated_protein = wild_type_protein
    for original_aa, position, new_aa in mutations:
        position_0_based = position - 1
        expected_mutated_protein = (expected_mutated_protein[:position_0_based] + 
                                   new_aa + 
                                   expected_mutated_protein[position_0_based + 1:])
    
    # Validate that the mutation was applied correctly
    if mutated_protein != expected_mutated_protein:
        raise ValueError(f"Mutation validation failed: {mutation_str}\n"
                        f"Expected protein: {expected_mutated_protein}\n"
                        f"Actual protein:   {mutated_protein}\n"
                        f"Wild-type protein: {wild_type_protein}")
    
    logger.debug(f"Multiple mutations {mutation_str}: {len(mutations)} mutations applied")
    logger.debug(f"Final protein validation: {wild_type_protein} -> {mutated_protein}")
    
    return current_nucleotide

def process_dms_file(input_file_path: str, reference_file_path: str, seed: int = 42) -> pd.DataFrame:
    """
    Process DMS file and convert protein mutations to nucleotide mutations.
    
    Args:
        input_file_path: Path to input DMS file
        reference_file_path: Path to reference file with wild type sequences
        seed: Random seed for codon selection
        
    Returns:
        DataFrame with nucleotide sequences added
    """
    logger.info(f"Processing DMS file: {input_file_path}")
    
    # Load input file
    df = pd.read_csv(input_file_path)
    logger.info(f"Loaded {len(df)} mutations from {input_file_path}")
    
    # Get DMS filename
    dms_filename = os.path.basename(input_file_path)
    
    # Load reference file to get wild type nucleotide sequence
    reference_df = load_or_create_reference_file(reference_file_path)
    
    # Find wild type nucleotide sequence
    wild_type_row = reference_df[reference_df['DMS_filename'] == dms_filename]
    
    if len(wild_type_row) == 0:
        raise ValueError(f"No wild type nucleotide sequence found for {dms_filename}")
    
    wild_type_nucleotide = wild_type_row['wild_type_nucleotide'].iloc[0]
    
    if pd.isna(wild_type_nucleotide):
        raise ValueError(f"Wild type nucleotide sequence is missing for {dms_filename}")
    
    logger.info(f"Using wild type nucleotide sequence of length {len(wild_type_nucleotide)}")
    
    # Convert mutations to nucleotide sequences
    nucleotide_sequences = []
    
    for idx, row in df.iterrows():
        try:
            mutant_str = row['mutant']
            
            # Convert protein mutation to nucleotide mutation
            mutated_nucleotide = convert_protein_mutation_to_nucleotide(
                wild_type_nucleotide, mutant_str, seed
            )
            
            nucleotide_sequences.append(mutated_nucleotide)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} mutations")
                
        except Exception as e:
            logger.error(f"Error processing mutation {row['mutant']}: {str(e)}")
            nucleotide_sequences.append(None)
    
    # Add nucleotide sequences to dataframe
    df['nucleotide_sequence'] = nucleotide_sequences
    
    logger.info(f"Generated nucleotide sequences for {len(df)} mutations")
    return df


def process_single_file(input_file: str, reference_file: str, wild_type_reference_file: str, 
                       output_dir: str, seed: int, force_blast: bool, identity_threshold: float):
    """
    Process a single DMS file and convert protein mutations to nucleotide mutations.
    
    Args:
        input_file: Path to input DMS file
        reference_file: Path to DMS reference file
        wild_type_reference_file: Path to wild type nucleotide reference file
        output_dir: Output directory for nucleotide sequences
        seed: Random seed for reproducible codon selection
        force_blast: Force BLAST search even if nucleotide sequence exists
        identity_threshold: Minimum identity threshold for nucleotide validation
    """
    # Get input filename
    input_filename = os.path.basename(input_file)
    
    # Check if output already exists
    output_file = os.path.join(output_dir, input_filename)
    if os.path.exists(output_file) and not force_blast:
        logger.info(f"Output file already exists: {output_file}")
        logger.info("Skipping (use --force_blast to override)")
        return
    
    # Step 1: Get wild type nucleotide sequence
    need_to_blast = False
    
    # Check if we need to BLAST
    if force_blast:
        logger.info("Step 1: Force BLAST enabled - will perform BLAST search")
        need_to_blast = True
    else:
        # Check if nucleotide sequence already exists in reference file
        if os.path.exists(wild_type_reference_file):
            wild_type_df = load_or_create_reference_file(wild_type_reference_file)
            existing_row = wild_type_df[wild_type_df['DMS_filename'] == input_filename]
            
            if len(existing_row) > 0 and not pd.isna(existing_row['wild_type_nucleotide'].iloc[0]):
                logger.info(f"Step 1: Found existing nucleotide sequence for {input_filename}")
                logger.info("Skipping BLAST search (use --force_blast to override)")
                need_to_blast = False
            else:
                logger.info(f"Step 1: No existing nucleotide sequence found for {input_filename}")
                need_to_blast = True
        else:
            logger.info(f"Step 1: Wild type reference file doesn't exist: {wild_type_reference_file}")
            need_to_blast = True
    
    if need_to_blast:
        logger.info("Step 1: Finding wild type nucleotide sequence via BLAST")
        
        # Load reference file to get wild type protein sequence
        if os.path.exists(reference_file):
            reference_df = pd.read_csv(reference_file)
            
            # Find matching row
            matching_row = reference_df[reference_df['DMS_filename'] == input_filename]
            
            if len(matching_row) > 0:
                target_seq = matching_row['target_seq'].iloc[0]
                source_organism = matching_row.get('source_organism', [None]).iloc[0]
                
                logger.info(f"Found target sequence: {target_seq[:50]}...")
                logger.info(f"Source organism: {source_organism}")
                
                # Use enhanced three-step strategy to find nucleotide sequence
                wild_type_nucleotide, sequence_method = find_wild_type_nucleotide_sequence(target_seq, source_organism, identity_threshold)
                
                if wild_type_nucleotide:
                    # Step 2: Validate nucleotide sequence
                    logger.info("Step 2: Validating nucleotide sequence")
                    
                    # Use 99% threshold for validation since we may have corrected sequences
                    validation_threshold = 0.99 if identity_threshold >= 1.0 else identity_threshold
                    if validate_nucleotide_protein_correspondence(wild_type_nucleotide, target_seq, validation_threshold):
                        # Step 3: Update reference file
                        logger.info("Step 3: Updating reference file")
                        
                        update_reference_file(
                            wild_type_reference_file,
                            input_filename,
                            target_seq,
                            source_organism,
                            wild_type_nucleotide,
                            sequence_method
                        )
                    else:
                        logger.error("Nucleotide sequence validation failed")
                        logger.error("SUGGESTION: Consider checking the original DMS paper for exact strain/sequence used")
                        raise RuntimeError("Nucleotide sequence validation failed")
                else:
                    logger.error("Could not find wild type nucleotide sequence via BLAST")
                    
                    # Suggest next steps
                    logger.error("SUGGESTIONS:")
                    logger.error("1. Check the original DMS paper for exact strain/sequence used")
                    logger.error("2. For viral sequences: consider manual sequence generation")
                    logger.error("3. Try lowering identity threshold (--identity_threshold 0.95)")
                    
                    raise RuntimeError("Could not find wild type nucleotide sequence")
            else:
                logger.error(f"No reference data found for {input_filename}")
                raise RuntimeError(f"No reference data found for {input_filename}")
        else:
            logger.error(f"Reference file not found: {reference_file}")
            raise RuntimeError(f"Reference file not found: {reference_file}")
    
    # Step 4: Process mutations
    logger.info("Step 4: Processing mutations")
    
    result_df = process_dms_file(
        input_file,
        wild_type_reference_file,
        seed
    )
    
    # Step 5: Save results
    logger.info("Step 5: Saving results")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df.to_csv(output_file, index=False)
    
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Total mutations processed: {len(result_df)}")
    
    # Print some statistics
    non_null_sequences = result_df['nucleotide_sequence'].notna().sum()
    logger.info(f"Successful nucleotide conversions: {non_null_sequences}/{len(result_df)}")
    
    if non_null_sequences > 0:
        avg_length = result_df['nucleotide_sequence'].str.len().mean()
        logger.info(f"Average nucleotide sequence length: {avg_length:.1f}")


def process_folder(input_folder: str, reference_file: str, wild_type_reference_file: str, 
                  output_dir: str, seed: int, force_blast: bool, identity_threshold: float):
    """
    Process all CSV files in a folder and convert protein mutations to nucleotide mutations.
    
    Args:
        input_folder: Path to input folder containing DMS files
        reference_file: Path to DMS reference file
        wild_type_reference_file: Path to wild type nucleotide reference file
        output_dir: Output directory for nucleotide sequences
        seed: Random seed for reproducible codon selection
        force_blast: Force BLAST search even if nucleotide sequence exists
        identity_threshold: Minimum identity threshold for nucleotide validation
    """
    logger.info(f"Processing folder: {input_folder}")
    
    # Find all CSV files in the input folder
    csv_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(input_folder, file))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_folder}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    for i, input_file in enumerate(csv_files, 1):
        logger.info(f"\n=== Processing file {i}/{len(csv_files)}: {os.path.basename(input_file)} ===")
        
        try:
            process_single_file(
                input_file, 
                reference_file, 
                wild_type_reference_file,
                output_dir, 
                seed, 
                force_blast, 
                identity_threshold
            )
            logger.info(f"Successfully processed {os.path.basename(input_file)}")
        except Exception as e:
            logger.error(f"Failed to process {os.path.basename(input_file)}: {str(e)}")
            continue
    
    logger.info(f"\nFolder processing completed! Processed {len(csv_files)} files.")


def validate_file_list(file_list_path, input_folder):
    """
    Validate that the file list CSV exists and contains valid file names.
    
    Args:
        file_list_path: Path to CSV file containing list of files to process
        input_folder: Path to folder containing DMS files
    
    Returns:
        List of validated file paths
    """
    if not os.path.exists(file_list_path):
        logger.error(f"File list CSV does not exist: {file_list_path}")
        sys.exit(1)
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_list_path)
        
        # Check if the CSV has a 'filename' column or just use the first column
        if 'filename' in df.columns:
            filenames = df['filename'].tolist()
        elif 'file' in df.columns:
            filenames = df['file'].tolist()
        else:
            # Use the first column
            filenames = df.iloc[:, 0].tolist()
        
        # Validate that files exist in the input folder
        valid_files = []
        for filename in filenames:
            if pd.isna(filename):
                continue
            
            # Remove any extensions if present and add .csv
            filename_str = str(filename).strip()
            if not filename_str.endswith('.csv'):
                filename_str += '.csv'
            
            full_path = os.path.join(input_folder, filename_str)
            if os.path.exists(full_path):
                valid_files.append(full_path)
            else:
                logger.warning(f"File not found in input folder: {filename_str}")
        
        if not valid_files:
            logger.error("No valid files found from the file list!")
            sys.exit(1)
        
        logger.info(f"Found {len(valid_files)} valid files to process")
        return valid_files
        
    except Exception as e:
        logger.error(f"Error reading file list CSV: {str(e)}")
        sys.exit(1)


def process_file_list(file_list_path, input_folder, reference_file, wild_type_reference_file, output_dir, seed, force_blast, identity_threshold):
    """
    Process a list of specific files from a CSV file list.
    
    Args:
        file_list_path: Path to CSV file containing list of files to process
        input_folder: Path to folder containing DMS files
        reference_file: Path to DMS reference file
        wild_type_reference_file: Path to wild type nucleotide reference file
        output_dir: Output directory for nucleotide sequences
        seed: Random seed for reproducible codon selection
        force_blast: Force BLAST search even if nucleotide sequence exists
        identity_threshold: Minimum identity threshold for nucleotide validation
    """
    # Validate and get file list
    file_list = validate_file_list(file_list_path, input_folder)
    
    logger.info(f"Processing {len(file_list)} files from CSV list...")
    
    success_count = 0
    fail_count = 0
    
    for file_path in file_list:
        logger.info(f"Processing file: {os.path.basename(file_path)}")
        
        try:
            # Process individual file
            process_single_file(
                file_path,
                reference_file,
                wild_type_reference_file,
                output_dir,
                seed,
                force_blast,
                identity_threshold
            )
            logger.info(f"✓ Successfully processed: {os.path.basename(file_path)}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to process {os.path.basename(file_path)}: {str(e)}")
            fail_count += 1
    
    logger.info(f"File list processing completed! Success: {success_count}, Failed: {fail_count}")
    
    if fail_count > 0:
        logger.warning(f"Some files failed to process. Check the logs above for details.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Nucleotide Data Pipeline')
    
    # Make input_file and input_folder mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_file', type=str,
                            help='Path to input DMS file')
    input_group.add_argument('--input_folder', type=str,
                            help='Path to folder containing DMS files')
    
    parser.add_argument('--file_list', type=str,
                        help='Path to CSV file containing list of specific files to process (requires --input_folder)')
    parser.add_argument('--reference_file', type=str, 
                        default='data/eval_dataset/DMS_ProteinGym_substitutions/DMS_substitutions.csv',
                        help='Path to DMS reference file')
    parser.add_argument('--wild_type_reference_file', type=str,
                        default='data/eval_dataset/DMS_ProteinGym_substitutions/DMS_substitutions_wild_type_nucleotides.csv',
                        help='Path to wild type nucleotide reference file')
    parser.add_argument('--output_dir', type=str, default='data/eval_dataset/DMS_ProteinGym_substitutions/nucleotides',
                        help='Output directory for nucleotide sequences')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible codon selection')
    parser.add_argument('--force_blast', action='store_true',
                        help='Force BLAST search even if nucleotide sequence exists in reference file')
    parser.add_argument('--identity_threshold', type=float, default=1.0,
                        help='Minimum identity threshold for nucleotide validation (default: 1.0 = 100%)')
    
    args = parser.parse_args()
    
    # Validate file_list argument
    if args.file_list and not args.input_folder:
        parser.error("--file_list requires --input_folder to be specified")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        if args.input_file:
            # Process single file
            logger.info("Processing single file...")
            process_single_file(
                args.input_file,
                args.reference_file,
                args.wild_type_reference_file,
                args.output_dir,
                args.seed,
                args.force_blast,
                args.identity_threshold
            )
        elif args.input_folder:
            if args.file_list:
                # Process specific files from CSV list
                logger.info("Processing specific files from CSV list...")
                success = process_file_list(
                    args.file_list,
                    args.input_folder,
                    args.reference_file,
                    args.wild_type_reference_file,
                    args.output_dir,
                    args.seed,
                    args.force_blast,
                    args.identity_threshold
                )
                if not success:
                    logger.error("Some files failed to process")
                    sys.exit(1)
            else:
                # Process entire folder
                logger.info("Processing entire folder...")
                process_folder(
                    args.input_folder,
                    args.reference_file,
                    args.wild_type_reference_file,
                    args.output_dir,
                    args.seed,
                    args.force_blast,
                    args.identity_threshold
                )
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
