#!/usr/bin/env python
"""Step 1: Save a specific input sequence to disk (configurable sequence index).

This ensures both models get EXACTLY the same input.

Usage:
    python 01_save_hardcoded_input_next.py --seq_idx 1  # Second sequence
    python 01_save_hardcoded_input_next.py --seq_idx 2  # Third sequence
"""

import argparse

import torch
from transformers import AutoTokenizer

print("=" * 80)
print("STEP 1: SAVE HARDCODED INPUT SEQUENCE (CONFIGURABLE)")
print("=" * 80)

# Parse arguments
parser = argparse.ArgumentParser(description='Save a specific sequence from FASTA')
parser.add_argument('--seq_idx', type=int, default=1,
                    help='Sequence index to save (0-based, default=1 for second sequence)')
parser.add_argument('--output', type=str, default='hardcoded_input_next.pt',
                    help='Output filename (default: hardcoded_input_next.pt)')
parser.add_argument('--fasta', type=str, default='/workspaces/bionemo-framework/test_sequences.fasta',
                    help='Path to FASTA file')
parser.add_argument('--tokenizer', type=str,
                    default='/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/nucleotide_tokenizer',
                    help='Path to tokenizer directory')
args = parser.parse_args()

# Load FASTA
with open(args.fasta, 'r') as f:
    lines = f.readlines()

# Parse FASTA to find sequences
sequences = []
current_header = None
current_seq = []

for raw_line in lines:
    line = raw_line.strip()
    if line.startswith('>'):
        if current_header is not None:
            sequences.append({
                'header': current_header,
                'sequence': ''.join(current_seq)
            })
        current_header = line[1:]  # Remove '>'
        current_seq = []
    else:
        current_seq.append(line)

# Add last sequence
if current_header is not None:
    sequences.append({
        'header': current_header,
        'sequence': ''.join(current_seq)
    })

print(f"\nTotal sequences in FASTA: {len(sequences)}")
print(f"Available indices: 0-{len(sequences)-1}")

# Validate index
if args.seq_idx >= len(sequences) or args.seq_idx < 0:
    raise ValueError(f"Sequence index {args.seq_idx} out of range (0-{len(sequences)-1})")

# Use selected sequence
selected = sequences[args.seq_idx]
header = selected['header']
sequence = selected['sequence']

print(f"\n{'='*80}")
print(f"SELECTED SEQUENCE {args.seq_idx}:")
print(f"{'='*80}")
print(f"Header: {header}")
print(f"Length: {len(sequence)}")
print(f"First 100 chars: {sequence[:100]}")

# Tokenize (no BOS, matching John's setup)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokens = tokenizer.encode(sequence, add_special_tokens=False)

# Use full sequence length (8192 tokens)
# tokens = tokens[:100]  # Uncomment to use shorter sequence if OOM
# sequence = sequence[:100]

print(f"\nTokenized length: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")

# Create the exact input that will go to both models
input_data = {
    'sequence': sequence,
    'header': header,
    'tokens': tokens,
    'tokens_tensor': torch.tensor([tokens], dtype=torch.long),  # [1, seq_len]
    'sequence_index': args.seq_idx,  # Track which sequence this is
}

# Save to disk
torch.save(input_data, args.output)

print(f"\n✅ Saved sequence {args.seq_idx} to: {args.output}")
print(f"   Shape: [1, {len(tokens)}]")
print(f"   Sequence: {header[:60]}...")
print("=" * 80)

