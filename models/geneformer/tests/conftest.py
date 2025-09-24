"""Pytest configuration for geneformer tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import requests
import torch


# Add the src directory to the Python path so tests can import geneformer modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def input_data():
    """Create realistic geneformer input data using actual gene token dictionary.

    Following the actual Geneformer implementation:
    - input_ids: Gene token IDs from actual gene vocabulary (truncated to max length)
    - attention_mask: Mask for valid tokens (1 for genes, 0 for unused positions)
    - labels: Masked language modeling labels (-100 for non-masked tokens)

    Note: Geneformer truncates sequences rather than padding them, as seen in the tokenizer.
    """
    import pickle

    # Download the token dictionary from Hugging Face
    token_dict_url = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary_gc104M.pkl"

    # Create a temporary file to store the downloaded dictionary
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
        tmp_file_path = tmp_file.name

    try:
        print(f"Downloading geneformer token dictionary from {token_dict_url}")
        response = requests.get(token_dict_url, stream=True)
        response.raise_for_status()

        # Write the downloaded content to the temporary file
        with open(tmp_file_path, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

        print("Successfully downloaded token dictionary")

        # Load the token dictionary from the temporary file
        with open(tmp_file_path, "rb") as f:
            token_dictionary = pickle.load(f)

        vocab_size = len(token_dictionary)
        pad_token_id = token_dictionary.get("<pad>", 0)
        mask_token_id = token_dictionary.get("<mask>", 1)

        print(f"Loaded geneformer token dictionary with {vocab_size} tokens")
        print(f"Pad token ID: {pad_token_id}, Mask token ID: {mask_token_id}")

        # Get some actual gene token IDs (excluding special tokens)
        gene_tokens = [
            token_id for token_id in token_dictionary.values() if isinstance(token_id, int) and token_id > 2
        ]  # Exclude special tokens

        assert len(gene_tokens) > 0, "No valid gene tokens found in token dictionary"

        # Use actual gene tokens for realistic testing
        batch_size = 2
        seq_length = 2048  # taken from examples/pretraining_new_model/pretrain_geneformer_w_deepspeed.py

        # Create input_ids using actual gene tokens (based on Geneformer/geneformer/tokenizer.py)
        # No padding needed - Geneformer truncates sequences like in tokenizer.py line 742
        input_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
        for i in range(batch_size):
            # example["input_ids"][0 : self.model_input_size]
            num_genes = min(seq_length, len(gene_tokens))  # Use full sequence length
            gene_indices = torch.randint(0, len(gene_tokens), (num_genes,))
            input_ids[i, :num_genes] = torch.tensor([gene_tokens[idx] for idx in gene_indices])

        # Create attention mask - 1 for real tokens, 0 for unused positions
        # based on models/geneformer/Geneformer/geneformer/evaluation_utils.py line 45
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bfloat16)
        for i in range(batch_size):
            num_genes = min(seq_length, len(gene_tokens))
            attention_mask[i, num_genes:] = 0  # Mask out unused positions

        # Create labels for masked language modeling
        # -100 for non-masked tokens, actual token ID for masked tokens
        labels = torch.full((batch_size, seq_length), -100, dtype=torch.long)

        # Mask some tokens randomly (15% masking rate, typical for MLM)
        mask_indices = torch.rand(batch_size, seq_length) < 0.15
        mask_indices = mask_indices & (attention_mask.bool())  # Only mask real tokens
        labels[mask_indices] = input_ids[mask_indices]

        print("Created realistic geneformer input data:")
        print(f"  - Using {len(gene_tokens)} actual gene tokens")
        print(f"  - Sequence length: {seq_length}")
        print(f"  - Actual genes per sequence: {num_genes}")
        print(f"  - Masked tokens: {(labels != -100).sum().item()}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
