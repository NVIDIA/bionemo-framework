"""
Dataset for sequence generation prompts.

This dataset provides batches of prompts (organism tokens) for use with
DecodonInference._predict_step when task_type is SEQUENCE_GENERATION.
"""

from typing import Callable, List, Optional
import torch
from torch.utils.data import Dataset

from src.data.metadata import MetadataFields


class GenerationPromptDataset(Dataset):
    """
    Simple dataset that yields batches of organism token prompts for generation.
    
    Can be used with the standard predict pipeline:
    DataLoader → _predict_step → PredWriter
    
    Args:
        organism_id: Single organism ID to generate sequences for (e.g., "9606" for human)
        tokenizer: Tokenizer instance with organism token support
        num_sequences_per_organism: Number of sequences to generate for this organism
        
    Example:
        # Generate 100 sequences for human (9606)
        dataset = GenerationPromptDataset(
            organism_id="9606",
            tokenizer=tokenizer,
            num_sequences_per_organism=100
        )
    """
    
    def __init__(
        self,
        organism_id: str,
        tokenizer,
        num_sequences_per_organism: int = 1,
        seed: int = 123,
    ):
        self.tokenizer = tokenizer
        self.organism_id = organism_id
        
        # Create list of organism IDs (all the same) for the requested number of sequences
        self.organism_ids = [organism_id] * num_sequences_per_organism
        
        # Validate organism ID exists in tokenizer
        org_token = f"<{organism_id}>"
        if org_token not in tokenizer.encoder:
            raise KeyError(
                f"Organism ID '{organism_id}' not found in tokenizer. "
                f"Make sure organism_tokens_file contains this organism."
            )
    
    def __len__(self):
        return len(self.organism_ids)
    
    def __getitem__(self, idx):
        """
        Returns a prompt for generation.
        
        Returns dict with:
            - INPUT_IDS: tensor of shape [1] containing organism token ID
            - ATTENTION_MASK: tensor of shape [1] with all 1s
            - ID: string identifier like "9606_42"
        """
        org_id = self.organism_ids[idx]
        org_token = f"<{org_id}>"
        org_token_id = self.tokenizer.encoder[org_token]
        
        return {
            MetadataFields.INPUT_IDS: torch.tensor([org_token_id], dtype=torch.long),
            MetadataFields.ATTENTION_MASK: torch.tensor([1], dtype=torch.long),
            MetadataFields.ID: f"{org_id}_{idx}",
        }
    
    # Standard dataset interface methods for compatibility with CodonFMDataModule
    def get_train(self, process_item: Callable = None) -> "GenerationPromptDataset":
        """Not used for generation, but required for interface compatibility."""
        return self
    
    def get_validation(self, process_item: Callable = None) -> "GenerationPromptDataset":
        """Not used for generation, but required for interface compatibility."""
        return self
    
    def get_test(self, process_item: Callable = None) -> "GenerationPromptDataset":
        """Not used for generation, but required for interface compatibility."""
        return self
    
    def get_predict(self, process_item: Callable = None) -> "GenerationPromptDataset":
        """Returns self for predict mode."""
        return self

