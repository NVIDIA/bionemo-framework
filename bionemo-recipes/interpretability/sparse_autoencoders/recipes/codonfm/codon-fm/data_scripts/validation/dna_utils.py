import polars as pl
import pandas as pd
import numpy as np
import pyfaidx
import ast

dna_code = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGC": "C",
    "TGT": "C",
    "TGA": "*",
    "TGG": "W",
}

# Define the genetic code as a dictionary mapping codons to amino acids
rna_code = {
    "UUU": "F",
    "UUC": "F",  # Phenylalanine
    "UUA": "L",
    "UUG": "L",  # Leucine
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",  # Serine
    "UAU": "Y",
    "UAC": "Y",  # Tyrosine
    "UAA": "*",
    "UAG": "*",  # Stop codons
    "UGA": "*",  # Stop codon
    "UGU": "C",
    "UGC": "C",  # Cysteine
    "UGG": "W",  # Tryptophan
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAU": "H",
    "CAC": "H",  # Histidine
    "CAA": "Q",
    "CAG": "Q",  # Glutamine
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",  # Methionine (start codon)
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def translate(seq, is_rna=False):
    """
    Translate an RNA sequence into a protein sequence.
    Stops translation when a stop codon ('*') is encountered.
    """
    protein = ""
    # Process the RNA sequence three nucleotides (codon) at a time.
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        # Look up the codon in the genetic code dictionary.
        if is_rna:
            amino_acid = rna_code.get(codon, "?")
        else:
            amino_acid = dna_code.get(codon, "?")

        protein += amino_acid
    return protein

def reverse_complement_dna(seq):
    """
    Return the reverse complement of a DNA sequence.
    
    Parameters:
        seq (str): A DNA sequence with uppercase letters only (e.g., "ATCG").
    
    Returns:
        str: The reverse complement DNA sequence.
        
    Raises:
        KeyError: If the sequence contains lowercase letters or invalid characters.
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement[base] for base in seq[::-1])


def process_gtf(gtf_path, fasta_path):
    gtf = pd.read_csv(gtf_path, sep='\t')
    if '#name' in gtf.columns:
        gtf = gtf.rename({'#name':'name'}, axis=1)
    gtf =gtf.loc[gtf['cdsStart']!=gtf['cdsEnd']].reset_index(drop=True).copy()

    gtf['exonStarts_arr'] = gtf['exonStarts'].map(lambda x:ast.literal_eval(x))
    gtf['exonEnds_arr'] = gtf['exonEnds'].map(lambda x:ast.literal_eval(x))
    fasta = {}
    with pyfaidx.Fasta(fasta_path) as f:
        for chrom in gtf['chrom'].unique():
            fasta[chrom] = f[chrom][:].seq.upper()
    cds_starts = []
    cds_ends = []
    lengths = []
    seqs = []
    for i in range(gtf.shape[0]):
        t = gtf.iloc[i]
        chrom = t['chrom']
        cds_s = []
        cds_e = []
        cs,ce = t[['cdsStart','cdsEnd']]
        length = 0
        curr_seq = []
        for a,b in zip(t['exonStarts_arr'],t['exonEnds_arr']):
            v1 = max(a,cs) 
            v2 = min(b,ce)
            if v1 < v2:
                cds_s.append(v1)
                cds_e.append(v2)
                length += v2-v1
                curr_seq.append(fasta[chrom][v1:v2])
        cds_starts.append(tuple(cds_s))
        cds_ends.append( tuple(cds_e))
        lengths.append(length)
        curr_seq = ''.join(curr_seq)
        if t['strand'] == '-':
            curr_seq = reverse_complement_dna(curr_seq)
        seqs.append(curr_seq)

    gtf['cds_starts'] = cds_starts
    gtf['cds_ends'] = cds_ends
    gtf['cds_length'] = lengths
    gtf['cds'] = seqs

    gtf_s = gtf[['name','chrom','strand','cdsStart','cdsEnd','cds_starts','cds_ends','cds_length','cds']].copy()
    gtf_s['name'] = gtf_s['name'].str.split('.').str[0]
    gtf_s = gtf_s.sort_values(by=['chrom','cdsStart','cdsEnd']).reset_index(drop=True).copy()
    
    return gtf_s, fasta



def process_a_chrom(chrom_variants, chrom_refseq, return_alt_cds=False):
    var_ids = chrom_variants['variant_id'].values
    var_pos = chrom_variants['pos'].values - 1  # Convert to 0-based
    var_ref = chrom_variants['ref'].values
    var_alt = chrom_variants['alt'].values
    chrom = chrom_refseq.iloc[0]['chrom']
    
    cds_strands = chrom_refseq['strand'].values
    cds_starts = chrom_refseq['cdsStart'].values  # 0-based
    cds_ends = chrom_refseq['cdsEnd'].values  # 0-based
    cds_lengths = chrom_refseq['cds_length'].values
    rec_cds_starts = chrom_refseq['cds_starts'].values  # List of exon starts
    rec_cds_ends = chrom_refseq['cds_ends'].values  # List of exon ends
    rec_cds = chrom_refseq['cds'].values  # CDS sequence
    rec_names = chrom_refseq['name'].values

    # Find transcripts that overlap each variant position
    # s1 = np.searchsorted(cds_starts, var_pos, side='right')  # First transcript starting after pos
    # s2 = np.searchsorted(cds_ends, var_pos, side='left')  # Last transcript ending before pos
    s1 = np.searchsorted(var_pos, cds_starts, side='left')
    s2 = np.searchsorted(var_pos, cds_ends, side='right')
    results = []
    for j, (ss1, ss2) in enumerate(zip(s1, s2)):
        
        curr_starts = rec_cds_starts[j]  # Exon starts for this transcript
        curr_ends = rec_cds_ends[j]  # Exon ends for this transcript
        # Check transcripts that contain this position
        if ss1 < ss2:
            for i in range(ss1, ss2):
                
                if var_ids[i] == 'chr1_1339609_G_A_hg38':
                    print(var_ids[i], ss1, ss2)
                pos = var_pos[i]
                curr_ref = var_ref[i]
                curr_alt = var_alt[i]
                # Calculate offset in CDS sequence
                offset = 0
                bound = False
                for a,b in zip(curr_starts, curr_ends):
                    if pos >= b:
                        offset += b-a  # Add length of complete exon
                    elif a <= pos < b:
                        offset += pos-a  # Add partial exon length
                        bound = True
                        break
                if var_ids[i] == 'chr1_1339609_G_A_hg38':
                    print(var_ids[i], j, bound)
                if bound:
                    if cds_strands[j] == '-':
                        # Handle reverse strand
                        offset = cds_lengths[j]-1-offset  # Convert to reverse strand position
                        ref_codon = rec_cds[j][offset//3*3:offset//3*3+3]
                        assert rec_cds[j][offset] == reverse_complement_dna(curr_ref), f'-, {ref_codon} {var_ids[i]}'
                        alt_codon = ref_codon[:offset%3] + reverse_complement_dna(curr_alt) + ref_codon[offset%3+1:]
                        
                        results.append([chrom, pos, f'{chrom}_{pos+1}_{curr_ref}_{curr_alt}', curr_ref, curr_alt,
                                     rec_names[j], cds_starts[j], cds_ends[j], cds_strands[j], offset, rec_cds[j],
                                     ref_codon, alt_codon, translate(ref_codon), translate(alt_codon)])
                        
                        if return_alt_cds:
                            alt_cds = rec_cds[j][:offset] + reverse_complement_dna(curr_alt) + rec_cds[j][offset+1:]
                            results[-1].append(alt_cds)
                    else:
                        # Handle forward strand
                        ref_codon = rec_cds[j][offset//3*3:offset//3*3+3]
                        assert rec_cds[j][offset] == curr_ref, f'+, {ref_codon} {var_ids[i]}'
                        alt_codon = ref_codon[:offset%3] + curr_alt + ref_codon[offset%3+1:]
                        
                        results.append([chrom, pos, f'{chrom}_{pos+1}_{curr_ref}_{curr_alt}', curr_ref, curr_alt,
                                     rec_names[j], cds_starts[j], cds_ends[j], cds_strands[j], offset, rec_cds[j],
                                     ref_codon, alt_codon, translate(ref_codon), translate(alt_codon)])
                        
                        if return_alt_cds:
                            alt_cds = rec_cds[j][:offset] + curr_alt + rec_cds[j][offset+1:]
                            results[-1].append(alt_cds)

    columns = ['chrom','pos','variant_id','ref','alt','tx_name', 'cdsStart','cdsEnd','tx_strand',
              'var_rel_dist_in_cds', 'ref_seq','ref_codon','alt_codon','ref_aa','alt_aa']
    if return_alt_cds:
        columns.append('alt_seq')
    
    if results:
        results = pd.DataFrame(results)
        results.columns = columns
        results['pos'] += 1  # Convert back to 1-based
        results['codon_position'] = results['var_rel_dist_in_cds']//3
    else:
        # Create empty DataFrame with correct columns
        results = pd.DataFrame(columns=columns)
        results['codon_position'] = pd.Series(dtype='int64')
    
    return results