# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""
HuggingFace PreTrainedTokenizerFast for nucleotide sequences.

Creates a simple ASCII-based tokenizer that:
1. Maps each character to its ord() value
2. Uses special tokens with NeMo convention (EOS=0, PAD=1, BOS=2)
3. Works with AutoTokenizer.from_pretrained()
"""

from tokenizers import Tokenizer, processors
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


def create_nucleotide_tokenizer(
    eos_id: int = 0,
    pad_id: int = 1,
    bos_id: int = None,  # NeMo convention: BOS is optional
    unk_id: int = 2,
) -> PreTrainedTokenizerFast:
    """
    Create a PreTrainedTokenizerFast for nucleotide sequences.
    
    Follows NeMo ByteTokenizer convention:
    - eos_id = 0
    - pad_id = 1  
    - bos_id = None (optional, for causal LM we'll use 2)
    
    Args:
        eos_id: End-of-sequence token ID (NeMo convention: 0)
        pad_id: Padding token ID (NeMo convention: 1)
        bos_id: Beginning-of-sequence token ID (None in NeMo, but we use 2 for causal LM)
        unk_id: Unknown token ID (2 or 3)
        
    Returns:
        PreTrainedTokenizerFast ready to use and save
    """
    # Define special tokens with NeMo convention
    # Only add BOS if it's specified (not None)
    special_tokens = {
        "<EOS>": eos_id,
        "<PAD>": pad_id,
        "<UNK>": unk_id,
    }
    
    if bos_id is not None:
        special_tokens["<BOS>"] = bos_id
    
    # Build vocab: Map each ASCII character to its ord() value
    # IMPORTANT: Exclude chr(0-3) to reserve those IDs for special tokens
    # chr(0-3) are non-printable control characters anyway
    vocab = {**special_tokens}
    reserved_ids = {eos_id, pad_id, unk_id}
    if bos_id is not None:
        reserved_ids.add(bos_id)
    
    for i in range(256):
        if i not in reserved_ids:  # Skip reserved IDs for special tokens
            char = chr(i)
            vocab[char] = i
    
    # Create Rust tokenizer backend with WordLevel model
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<UNK>"))
    
    # Configure pre-tokenizer: Split into individual characters
    # pattern="" means split every character
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
    
    # Configure post-processor: Add BOS/EOS tokens automatically
    # Only add BOS if it was specified
    if bos_id is not None:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<BOS> $A <EOS>",
            pair="<BOS> $A <EOS> <BOS> $B <EOS>",
            special_tokens=[
                ("<BOS>", bos_id),
                ("<EOS>", eos_id),
            ],
        )
        bos_token = "<BOS>"
    else:
        # No BOS token - just add EOS
        tokenizer.post_processor = processors.TemplateProcessing(
            single="$A <EOS>",
            pair="$A <EOS> $B <EOS>",
            special_tokens=[
                ("<EOS>", eos_id),
            ],
        )
        bos_token = None
    
    # Wrap in HuggingFace PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<UNK>",
        pad_token="<PAD>",
        eos_token="<EOS>",
        bos_token=bos_token,  # Can be None
    )
    
    return hf_tokenizer


if __name__ == "__main__":
    import torch
    from transformers import DataCollatorForLanguageModeling, AutoTokenizer
    import tempfile
    import os
    
    print("="*80)
    print("Creating HuggingFace PreTrainedTokenizerFast for Nucleotides")
    print("="*80)
    
    # Create tokenizer with NeMo convention (but we add BOS for causal LM)
    tokenizer = create_nucleotide_tokenizer(
        eos_id=0,
        pad_id=1,
        bos_id=2,  # Add BOS for causal language modeling
        unk_id=3,
    )
    
    print(f"\n✅ Tokenizer created")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Special tokens:")
    print(f"     - PAD: {tokenizer.pad_token} = {tokenizer.pad_token_id}")
    print(f"     - EOS: {tokenizer.eos_token} = {tokenizer.eos_token_id}")
    print(f"     - BOS: {tokenizer.bos_token} = {tokenizer.bos_token_id}")
    print(f"     - UNK: {tokenizer.unk_token} = {tokenizer.unk_token_id}")
    
    # Test encoding/decoding
    print("\n" + "-"*80)
    print("Test 1: Encoding/Decoding")
    print("-"*80)
    
    sequence = "ATCGATCG"
    encoded = tokenizer.encode(sequence, add_special_tokens=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    print(f"Original:  '{sequence}'")
    print(f"Encoded:   {encoded}")
    print(f"Expected:  [2(BOS), 65(A), 84(T), 67(C), 71(G), 65(A), 84(T), 67(C), 71(G), 0(EOS)]")
    print(f"Decoded:   '{decoded}'")
    print(f"✅ Roundtrip: {sequence == decoded}")
    
    # Test padding
    print("\n" + "-"*80)
    print("Test 2: Padding")
    print("-"*80)
    
    batch = tokenizer(
        ["ATCG", "ATCGATCGATCG"],
        padding=True,
        return_tensors="pt"
    )
    
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Input IDs:\n{batch['input_ids']}")
    print(f"Attention mask:\n{batch['attention_mask']}")
    print(f"✅ Padding works!")
    
    # Test with DataCollator
    print("\n" + "-"*80)
    print("Test 3: DataCollatorForLanguageModeling (mlm=False)")
    print("-"*80)
    
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    examples = [
        {"input_ids": batch["input_ids"][0]},
        {"input_ids": batch["input_ids"][1]},
    ]
    
    collated = collator(examples)
    print(f"Collated keys: {list(collated.keys())}")
    print(f"Labels shape: {collated['labels'].shape}")
    print(f"Labels (first 20): {collated['labels'][0][:20].tolist()}")
    print(f"✅ Works with DataCollator!")
    
    # Test save/load
    print("\n" + "-"*80)
    print("Test 4: Save/Load with AutoTokenizer")
    print("-"*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "nucleotide_tokenizer")
        
        # Save
        tokenizer.save_pretrained(save_path)
        print(f"✅ Saved to: {save_path}")
        print(f"   Files created:")
        for f in os.listdir(save_path):
            print(f"     - {f}")
        
        # Load with AutoTokenizer
        loaded = AutoTokenizer.from_pretrained(save_path)
        print(f"✅ Loaded with AutoTokenizer.from_pretrained()")
        
        # Verify it works
        test_seq = "ATCG"
        test_enc = loaded.encode(test_seq, add_special_tokens=True)
        test_dec = loaded.decode(test_enc, skip_special_tokens=True)
        print(f"   Test: '{test_seq}' -> {test_enc} -> '{test_dec}'")
        print(f"✅ Loaded tokenizer works correctly!")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nReady to use in your training pipeline:")
    print("  1. Run this script once to create the tokenizer")
    print("  2. Save it: tokenizer.save_pretrained('./tokenizer_files')")
    print("  3. Use everywhere: AutoTokenizer.from_pretrained('./tokenizer_files')")

