#!/usr/bin/env python3
"""Evaluate YOUR OLD model.py using the GOLDEN VALUES method.

This script uses YOUR original model.py implementation (NVLlamaForCausalLM)
with the same golden values loss calculation method.

This should be comparable to batch_evaluate_loss_golden_method.py (Peter's model).
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Add your model.py to path
sys.path.insert(0, '/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp')

# Apply noop_cat patch for your model
import transformer_engine.pytorch.module._common as te_common
te_common.noop_cat = lambda tensors, dim=0: torch.cat(tensors, dim=dim).contiguous()
from transformer_engine.pytorch.module import layernorm_linear
layernorm_linear.noop_cat = lambda tensors, dim=0: torch.cat(tensors, dim=dim).contiguous()

from model import NVLlamaForCausalLM  # noqa: E402

print("=" * 80)
print("BATCH LOSS EVALUATION - GOLDEN VALUES METHOD (YOUR OLD MODEL.PY)")
print("=" * 80)

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate YOUR old model.py using golden values method')
parser.add_argument('--fasta', type=str,
                    default='/workspaces/bionemo-framework/test_sequences.fasta',
                    help='Path to FASTA file')
parser.add_argument('--tokenizer', type=str,
                    default='/workspaces/bionemo-framework/bionemo-recipes/recipes/llama_native_te_nvfsdp/nucleotide_tokenizer',
                    help='Path to tokenizer directory')
parser.add_argument('--checkpoint', type=str,
                    default='/workspaces/bionemo-framework/checkpoints/bcr_eden_checkpoint_hf',
                    help='Path to HF checkpoint')
parser.add_argument('--max_length', type=int, default=8192,
                    help='Maximum sequence length (default: 8192)')
parser.add_argument('--max_sequences', type=int, default=None,
                    help='Maximum number of sequences to process (default: all)')
parser.add_argument('--prepend_bos', action='store_true',
                    help='Prepend BOS token (default: False, matching golden values)')
args = parser.parse_args()

# ========== Step 1: Load and Parse FASTA ==========
print("\n" + "-"*80)
print("STEP 1: LOADING FASTA SEQUENCES")
print("-"*80)

with open(args.fasta, 'r') as f:
    lines = f.readlines()

# Parse FASTA
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

total_sequences = len(sequences)
if args.max_sequences:
    sequences = sequences[:args.max_sequences]
    print(f"Limiting to first {args.max_sequences} sequences")

print(f"✓ Loaded {len(sequences)} sequences (total: {total_sequences})")

# ========== Step 2: Tokenize All Sequences ==========
print("\n" + "-"*80)
print("STEP 2: TOKENIZING SEQUENCES")
print("-"*80)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

tokenized_sequences = []
skipped = 0

for i, seq_data in enumerate(sequences):
    sequence = seq_data['sequence']
    tokens = tokenizer.encode(sequence, add_special_tokens=False)
    
    # Prepend BOS if requested (matching golden values default: False)
    if args.prepend_bos:
        tokens = [2] + tokens  # Token 2 is EOS/BOS in Eden tokenizer
    
    # Truncate to max_length
    if len(tokens) > args.max_length:
        tokens = tokens[:args.max_length]
    
    if len(tokens) == 0:
        skipped += 1
        continue
    
    tokenized_sequences.append({
        'header': seq_data['header'],
        'tokens': tokens,
        'token_length': len(tokens)
    })

print(f"✓ Tokenized {len(tokenized_sequences)} sequences")
if skipped > 0:
    print(f"  ⚠️ Skipped {skipped} empty sequences")

# ========== Step 3: Load Model (YOUR OLD MODEL.PY) ==========
print("\n" + "-"*80)
print("STEP 3: LOADING YOUR OLD MODEL.PY")
print("-"*80)

print("Loading your model.py implementation...")
model = NVLlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
model = model.to("cuda")
model.eval()
model.config.use_cache = False

torch.cuda.empty_cache()

print("✓ Model loaded and ready (YOUR original model.py)")

# ========== Step 4: Generate Golden Values (Log Probabilities) ==========
print("\n" + "-"*80)
print("STEP 4: GENERATING LOG PROBABILITIES (GOLDEN VALUES METHOD)")
print("-"*80)

all_log_probs = []
all_loss_masks = []
total_tokens = 0

with torch.no_grad():
    for i, seq_data in enumerate(tokenized_sequences):
        # Create tensor
        tokens = torch.tensor([seq_data['tokens']], dtype=torch.long).to('cuda')
        
        # Create loss mask
        loss_mask = torch.ones_like(tokens)
        if args.prepend_bos:
            loss_mask[:, 0] = 0  # Mask the prepended BOS token
        
        try:
            # Forward pass to get logits
            outputs = model(input_ids=tokens, attention_mask=None)
            logits = outputs.logits  # [1, seq_len, vocab_size]
            
            # Apply log_softmax to get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
            
            # Align predictions with targets
            # Predictions at [:, :-1], targets at [:, 1:]
            log_probs_aligned = log_probs[:, :-1]  # [1, seq_len-1, vocab_size]
            target_tokens = tokens[:, 1:]  # [1, seq_len-1]
            
            # Gather log probabilities for the actual tokens
            token_log_probs = torch.gather(
                log_probs_aligned,
                dim=2,
                index=target_tokens.unsqueeze(-1),
            ).squeeze(-1)  # [1, seq_len-1]
            
            # Apply loss mask (aligned)
            loss_mask_aligned = loss_mask[:, 1:]  # [1, seq_len-1]
            masked_log_probs = token_log_probs * loss_mask_aligned.float()
            
            # Store results
            all_log_probs.append(masked_log_probs.cpu())
            all_loss_masks.append(loss_mask_aligned.cpu())
            total_tokens += loss_mask_aligned.sum().item()
            
            # Progress reporting
            if (i + 1) % 10 == 0 or (i + 1) == len(tokenized_sequences):
                print(f"  Processed {i+1}/{len(tokenized_sequences)}")
            
            # Clear GPU memory
            del outputs, logits, log_probs, tokens, loss_mask
            torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ OOM on sequence {i}")
                torch.cuda.empty_cache()
                continue
            else:
                raise

print("\n✓ Generated log probabilities for all sequences")

# ========== Step 5: Calculate Global LM Loss ==========
print("\n" + "="*80)
print("RESULTS - GOLDEN VALUES METHOD (YOUR OLD MODEL.PY)")
print("="*80)

if len(all_log_probs) > 0:
    # Concatenate all log probs and masks
    log_probs_all = torch.cat(all_log_probs, dim=1).squeeze(0)  # [total_tokens]
    loss_mask_all = torch.cat(all_loss_masks, dim=1).squeeze(0)  # [total_tokens]
    
    # Calculate global LM loss (SAME as golden values comparison)
    # LM Loss = -log_prob, averaged over all valid tokens
    lm_loss = (-log_probs_all * loss_mask_all).sum() / loss_mask_all.sum()
    
    print(f"\nLM Loss (Golden Values Method): {lm_loss:.8f}")
    print(f"Total valid tokens: {int(loss_mask_all.sum().item()):,}")
    print(f"Total sequences: {len(tokenized_sequences)}")
    print(f"Avg tokens/sequence: {loss_mask_all.sum().item() / len(tokenized_sequences):.1f}")
    
    # Calculate perplexity
    perplexity = torch.exp(lm_loss).item()
    print(f"\nPerplexity: {perplexity:.4f}")
    
    # Calculate per-sequence stats for comparison
    per_seq_results = []
    start_idx = 0
    for i, mask in enumerate(all_loss_masks):
        seq_len = mask.shape[1]
        seq_log_probs = all_log_probs[i]  # [1, seq_len]
        seq_mask = mask  # [1, seq_len]
        
        # Per-sequence AVERAGE TOKEN LOSS
        # This is the average negative log probability per valid token in this sequence
        num_valid_tokens = seq_mask.sum().item()
        seq_loss = (-seq_log_probs * seq_mask).sum() / num_valid_tokens
        
        per_seq_results.append({
            'header': tokenized_sequences[i]['header'],
            'loss': seq_loss.item(),
            'num_tokens': int(num_valid_tokens),
            'seq_index': i,
        })
        start_idx += seq_len
    
    per_seq_losses = [r['loss'] for r in per_seq_results]
    avg_per_seq = sum(per_seq_losses) / len(per_seq_losses)
    
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"\nGlobal LM Loss (Golden Values):  {lm_loss:.8f}")
    print(f"Per-sequence averaged:            {avg_per_seq:.8f}")
    print(f"Difference:                       {abs(lm_loss - avg_per_seq):.8f}")
    
    print("\n✓ This should match your OLD golden values results!")
    
    # Show per-sequence statistics
    print(f"\n{'='*80}")
    print("PER-SEQUENCE AVERAGE TOKEN LOSS")
    print(f"{'='*80}")
    
    # Sort by loss
    sorted_results = sorted(per_seq_results, key=lambda x: x['loss'])
    
    print(f"\n📊 Distribution:")
    print(f"   Min loss:    {sorted_results[0]['loss']:.6f}")
    print(f"   25th %ile:   {sorted_results[len(sorted_results)//4]['loss']:.6f}")
    print(f"   Median:      {sorted_results[len(sorted_results)//2]['loss']:.6f}")
    print(f"   75th %ile:   {sorted_results[3*len(sorted_results)//4]['loss']:.6f}")
    print(f"   Max loss:    {sorted_results[-1]['loss']:.6f}")
    print(f"   Mean:        {avg_per_seq:.6f}")
    
    print(f"\n🏆 Top 5 BEST sequences (lowest loss):")
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"   {i}. Loss={result['loss']:.6f} ({result['num_tokens']} tokens): {result['header'][:60]}")
    
    print(f"\n⚠️  Top 5 WORST sequences (highest loss):")
    for i, result in enumerate(sorted_results[-5:][::-1], 1):
        print(f"   {i}. Loss={result['loss']:.6f} ({result['num_tokens']} tokens): {result['header'][:60]}")
    
    # Prepare data in BioNeMo-compatible format for comparison
    # Stack all log_probs into 2D tensor [num_seqs, seq_len]
    log_probs_stacked = torch.stack([lp.squeeze(0) for lp in all_log_probs], dim=0)  # [N, seq_len]
    loss_mask_stacked = torch.stack([lm.squeeze(0) for lm in all_loss_masks], dim=0)  # [N, seq_len]
    
    print(f"\nStacked log_probs shape: {log_probs_stacked.shape}")
    print(f"Stacked loss_mask shape: {loss_mask_stacked.shape}")
    
    # Create seq_idx_map (header -> index)
    seq_idx_map = {seq['header']: i for i, seq in enumerate(tokenized_sequences)}
    
    # Save results
    results = {
        'lm_loss': lm_loss.item(),
        'perplexity': perplexity,
        'total_tokens': int(loss_mask_all.sum().item()),
        'num_sequences': len(tokenized_sequences),
        'per_sequence_results': per_seq_results,  # Detailed per-sequence info
        'per_sequence_losses': per_seq_losses,  # Just the loss values
        'method': 'golden_values',
        'model': 'old_model_py',
        # BioNeMo-compatible format for comparison
        'log_probs_seqs': log_probs_stacked,  # [N, seq_len] format
        'loss_mask': loss_mask_stacked,  # [N, seq_len] format
        'seq_idx_map': seq_idx_map,  # header -> index mapping
    }
    
    output_file = 'batch_evaluation_golden_method_old_model.pt'
    torch.save(results, output_file)
    print(f"\n✓ Results saved to: {output_file}")
    print(f"  Includes BioNeMo-compatible format for comparison")
    
else:
    print("\n❌ No sequences were successfully evaluated")

print("\n" + "="*80)
print("MODEL: YOUR OLD model.py (NVLlamaForCausalLM)")
print("Compare with Peter's results from batch_evaluate_loss_golden_method.py")
print("="*80)

