from enum import Enum

class MetadataFields(str, Enum):
    ID = 'id'
    INPUT_IDS = 'input_ids'
    ATTENTION_MASK = 'attention_mask'
    LABELS = 'labels'
    INPUT_MASK = 'mask'
    REF_CODON_TOKS = 'ref_codon_toks'
    ALT_CODON_TOKS = 'alt_codon_toks'
    MUTATION_TOKEN_IDX = 'mutation_token_idx'
    REF_SYNOM_MASK = 'ref_synom_mask'
    ALT_SYNOM_MASK = 'alt_synom_mask'

class MetadataConstants:
    CODON_LENGTH = 3
    MLM_TOK_ADJUST = 2
    CLM_TOK_ADJUST = 0

class TrainerModes(str, Enum):
    PRETRAIN = 'pretrain'
    FINETUNE = 'finetune'
    PREDICT = 'predict'

class SplitNames(str, Enum):
    ALL = 'all'
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
class SynomCodons:
    """Class to store and retrieve synonymous codons based on the standard genetic code."""
    
    def __init__(self):
        # Standard genetic code mapping: codon -> amino acid
        self.codon_to_aa = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        
        # Build reverse mapping: amino acid -> list of codons
        self.aa_to_codons = {}
        for codon, aa in self.codon_to_aa.items():
            if aa not in self.aa_to_codons:
                self.aa_to_codons[aa] = []
            self.aa_to_codons[aa].append(codon)
    
    def get_synonymous_codons(self, codon: str) -> list[str]:
        """
        Get all synonymous codons for a given codon.
        
        Args:
            codon: Three-letter codon string (e.g., 'ATG')
            
        Returns:
            List of all codons that code for the same amino acid,
            including the input codon itself.
            Returns empty list if codon is invalid.
        """
        codon = codon.upper()
        
        if codon not in self.codon_to_aa:
            return []
        
        amino_acid = self.codon_to_aa[codon]
        return self.aa_to_codons[amino_acid].copy()
    
    def get_amino_acid(self, codon: str) -> str:
        """
        Get the amino acid coded by a given codon.
        
        Args:
            codon: Three-letter codon string (e.g., 'ATG')
            
        Returns:
            Single letter amino acid code, or empty string if codon is invalid.
        """
        codon = codon.upper()
        return self.codon_to_aa.get(codon, '')
    
    def get_codons_for_aa(self, amino_acid: str) -> list[str]:
        """
        Get all codons that code for a specific amino acid.
        
        Args:
            amino_acid: Single letter amino acid code (e.g., 'M')
            
        Returns:
            List of all codons that code for the amino acid.
            Returns empty list if amino acid is invalid.
        """
        amino_acid = amino_acid.upper()
        return self.aa_to_codons.get(amino_acid, []).copy()
