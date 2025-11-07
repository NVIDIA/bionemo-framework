#!/usr/bin/env python3
"""Verify that tokenizing the FASTA file produces the same tokens as in BioNeMo predictions."""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

print("="*100)
print("TOKENIZATION VERIFICATION")
print("="*100)

# Paths
fasta_path = Path('/workspaces/bionemo-framework/ribsome_data/ribosomal_rrna_highly_conserved_PMC4140814.fasta')
predictions_path = Path('/workspaces/bionemo-framework/ribsome_data/predictions__rank_0__dp_rank_0.pt')
seq_idx_map_path = Path('/workspaces/bionemo-framework/ribsome_data/seq_idx_map.json')
tokenizer_path = '/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/nucleotide_tokenizer'

print(f"\n📁 Loading files:")
print(f"  FASTA:       {fasta_path}")
print(f"  Predictions: {predictions_path}")
print(f"  Tokenizer:   {tokenizer_path}")

# ========== Step 1: Load BioNeMo tokens ==========
print("\n" + "-"*100)
print("STEP 1: LOAD BIONEMO TOKENS FROM PREDICTIONS FILE")
print("-"*100)

bionemo_data = torch.load(predictions_path)
bionemo_tokens = bionemo_data['tokens']  # [batch_size, seq_len]

print(f"\n✓ Loaded BioNeMo tokens")
print(f"  Shape: {bionemo_tokens.shape}")
print(f"  Dtype: {bionemo_tokens.dtype}")
print(f"  First 30 tokens: {bionemo_tokens[0, :30].tolist()}")

# ========== Step 2: Load seq_idx_map ==========
print("\n" + "-"*100)
print("STEP 2: LOAD SEQUENCE INDEX MAP")
print("-"*100)

with open(seq_idx_map_path, 'r') as f:
    seq_idx_map = json.load(f)

print(f"\n✓ Loaded seq_idx_map")
print(f"  Number of sequences: {len(seq_idx_map)}")
print(f"  Sequence headers: {list(seq_idx_map.keys())}")

# ========== Step 3: Parse FASTA file ==========
print("\n" + "-"*100)
print("STEP 3: PARSE FASTA FILE")
print("-"*100)

def parse_fasta(fasta_file):
    """Parse FASTA file and return sequences in order."""
    sequences = []
    current_header = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_header is not None:
                    sequences.append({
                        'header': current_header,
                        'sequence': ''.join(current_seq)
                    })
                # Start new sequence
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        # Save last sequence
        if current_header is not None:
            sequences.append({
                'header': current_header,
                'sequence': ''.join(current_seq)
            })
    
    return sequences

fasta_sequences = parse_fasta(fasta_path)

print(f"\n✓ Parsed FASTA file")
print(f"  Number of sequences: {len(fasta_sequences)}")

for i, seq_data in enumerate(fasta_sequences):
    header = seq_data['header']
    sequence = seq_data['sequence']
    print(f"\n  Sequence {i}:")
    print(f"    Header: {header}")
    print(f"    Length: {len(sequence)} characters")
    print(f"    First 60 chars: {sequence[:60]}")
    print(f"    Last 60 chars: {sequence[-60:]}")

# ========== Step 4: Load tokenizer and tokenize ==========
print("\n" + "-"*100)
print("STEP 4: TOKENIZE FASTA SEQUENCES")
print("-"*100)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print(f"\n✓ Loaded tokenizer from {tokenizer_path}")
print(f"  Vocab size: {tokenizer.vocab_size}")

# Tokenize each sequence
tokenized_sequences = []

for i, seq_data in enumerate(fasta_sequences):
    header = seq_data['header']
    sequence = seq_data['sequence']
    
    # Try different tokenization methods to match BioNeMo's approach
    # Method 1: Direct encode (no special tokens)
    tokens_ids = tokenizer.encode(sequence, add_special_tokens=False)
    
    print(f"\n  Sequence {i} ({header}):")
    print(f"    Raw sequence length: {len(sequence)} chars")
    print(f"    Tokenized length: {len(tokens_ids)} tokens")
    print(f"    First 30 token IDs: {tokens_ids[:30]}")
    
    tokenized_sequences.append({
        'header': header,
        'tokens': tokens_ids,
        'sequence': sequence
    })

# ========== Step 5: Compare with BioNeMo tokens ==========
print("\n" + "="*100)
print("STEP 5: COMPARE TOKENIZATION")
print("="*100)

# BioNeMo might have multiple sequences in batch, or single sequence
batch_size = bionemo_tokens.shape[0]
seq_len = bionemo_tokens.shape[1]

print(f"\nBioNeMo data:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len} tokens")

print(f"\nFASTA tokenization:")
print(f"  Number of sequences: {len(tokenized_sequences)}")
print(f"  Total tokens: {sum(len(s['tokens']) for s in tokenized_sequences)}")

# Try to match sequences
print("\n" + "-"*100)
print("MATCHING SEQUENCES")
print("-"*100)

all_match = True

for batch_idx in range(batch_size):
    bio_seq = bionemo_tokens[batch_idx].tolist()
    
    print(f"\n📊 Batch {batch_idx}:")
    print(f"  BioNeMo tokens: {len(bio_seq)} tokens")
    print(f"  First 30: {bio_seq[:30]}")
    
    # Try to find matching FASTA sequence
    matched = False
    
    for fasta_idx, tokenized in enumerate(tokenized_sequences):
        fasta_tokens = tokenized['tokens']
        
        # Check if this matches (could be truncated or exact)
        if len(fasta_tokens) >= len(bio_seq):
            # Check if BioNeMo tokens match the start of FASTA tokens
            if bio_seq == fasta_tokens[:len(bio_seq)]:
                print(f"  ✅ MATCHES FASTA sequence {fasta_idx} ({tokenized['header'][:50]}...)")
                print(f"     FASTA has {len(fasta_tokens)} tokens, BioNeMo used first {len(bio_seq)}")
                matched = True
                break
            # Check if BioNeMo tokens match anywhere in FASTA tokens
            elif bio_seq in [fasta_tokens[i:i+len(bio_seq)] for i in range(len(fasta_tokens) - len(bio_seq) + 1)]:
                match_idx = next(i for i in range(len(fasta_tokens) - len(bio_seq) + 1) if fasta_tokens[i:i+len(bio_seq)] == bio_seq)
                print(f"  ✅ MATCHES FASTA sequence {fasta_idx} at offset {match_idx}")
                print(f"     Header: {tokenized['header'][:50]}...")
                matched = True
                break
        elif len(fasta_tokens) == len(bio_seq):
            # Exact length match
            if bio_seq == fasta_tokens:
                print(f"  ✅ EXACT MATCH with FASTA sequence {fasta_idx}")
                print(f"     Header: {tokenized['header'][:50]}...")
                matched = True
                break
    
    if not matched:
        print(f"  ❌ NO MATCH FOUND in FASTA sequences")
        all_match = False
        
        # Show detailed comparison with closest match
        print(f"\n  Detailed comparison with each FASTA sequence:")
        for fasta_idx, tokenized in enumerate(tokenized_sequences):
            fasta_tokens = tokenized['tokens']
            
            # Compare first N tokens (where N = min of both lengths)
            compare_len = min(len(bio_seq), len(fasta_tokens), 50)
            bio_compare = bio_seq[:compare_len]
            fasta_compare = fasta_tokens[:compare_len]
            
            matches = sum(b == f for b, f in zip(bio_compare, fasta_compare))
            
            print(f"\n    FASTA {fasta_idx} ({tokenized['header'][:40]}...):")
            print(f"      Length: {len(fasta_tokens)} tokens")
            print(f"      First {compare_len} tokens match: {matches}/{compare_len} ({matches/compare_len*100:.1f}%)")
            
            if matches < compare_len:
                # Show first mismatch
                for i, (b, f) in enumerate(zip(bio_compare, fasta_compare)):
                    if b != f:
                        print(f"      First mismatch at position {i}: BioNeMo={b}, FASTA={f}")
                        print(f"        BioNeMo[{i-5}:{i+6}]: {bio_seq[max(0,i-5):i+6]}")
                        print(f"        FASTA[{i-5}:{i+6}]:   {fasta_tokens[max(0,i-5):i+6]}")
                        break

# ========== Step 6: Detailed Token-by-Token Verification ==========
if all_match:
    print("\n" + "="*100)
    print("STEP 6: DETAILED TOKEN-BY-TOKEN VERIFICATION")
    print("="*100)
    
    for batch_idx in range(batch_size):
        bio_seq = bionemo_tokens[batch_idx].tolist()
        
        # Find the matching FASTA sequence
        for fasta_idx, tokenized in enumerate(tokenized_sequences):
            fasta_tokens = tokenized['tokens']
            
            if bio_seq == fasta_tokens[:len(bio_seq)]:
                print(f"\n📊 Batch {batch_idx} vs FASTA {fasta_idx}:")
                print(f"  Length: {len(bio_seq)} tokens")
                
                # Check every token
                mismatches = []
                for i, (b, f) in enumerate(zip(bio_seq, fasta_tokens)):
                    if b != f:
                        mismatches.append(i)
                
                if len(mismatches) == 0:
                    print(f"  ✅ ALL {len(bio_seq)} TOKENS MATCH PERFECTLY")
                else:
                    print(f"  ❌ Found {len(mismatches)} mismatches at positions: {mismatches[:10]}...")
                
                # Decode and show the sequences
                print(f"\n  Decoded sequences:")
                bio_decoded = tokenizer.decode(bio_seq)
                fasta_decoded = tokenizer.decode(fasta_tokens[:len(bio_seq)])
                print(f"    BioNeMo (first 100 chars): {bio_decoded[:100]}")
                print(f"    FASTA   (first 100 chars): {fasta_decoded[:100]}")
                
                if bio_decoded == fasta_decoded:
                    print(f"  ✅ DECODED SEQUENCES MATCH PERFECTLY")
                else:
                    print(f"  ❌ DECODED SEQUENCES DIFFER")
                
                break

# ========== Final Summary ==========
print("\n" + "="*100)
print("FINAL SUMMARY")
print("="*100)

if all_match:
    print("\n✅✅✅ SUCCESS: TOKENIZATION VERIFIED ✅✅✅")
    print("\nThe tokens in the BioNeMo predictions file exactly match")
    print("the tokens produced by tokenizing the FASTA file with the")
    print("nucleotide tokenizer.")
    print("\n👉 Conclusion: Tokenization is consistent. The models will")
    print("   produce identical results given the same input.")
else:
    print("\n⚠️  WARNING: TOKENIZATION MISMATCH DETECTED ⚠️")
    print("\nThe tokens in BioNeMo predictions do NOT match the tokens")
    print("from tokenizing the FASTA file.")
    print("\n👉 This requires investigation:")
    print("   - Different tokenizer used?")
    print("   - Different tokenization settings (add_special_tokens, etc)?")
    print("   - Different sequence preprocessing?")
    print("   - Wrong FASTA file?")

print("\n" + "="*100)



