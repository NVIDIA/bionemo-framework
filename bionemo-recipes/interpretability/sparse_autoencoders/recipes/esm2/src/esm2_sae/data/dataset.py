from torch.utils.data import Dataset
from typing import List, Optional, Dict, Union
import torch


class ProteinDataset(Dataset):
    """Dataset for protein sequences.

    Args:
        sequences: List of protein sequences (amino acid strings)
        tokenizer: ESM2 tokenizer
        max_length: Maximum sequence length
        ids: Optional list of sequence identifiers
    """

    def __init__(
        self,
        sequences: List[str],
        tokenizer,
        max_length: int = 1024,
        ids: Optional[List[str]] = None
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ids = ids if ids is not None else [str(i) for i in range(len(sequences))]

    def __len__(self) -> int:
        return len(self.sequences)

    def preprocess_sequence(self, sequence: str) -> str:
        """Clean protein sequence."""
        return ''.join(sequence.split()).upper()

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        sequence = self.preprocess_sequence(self.sequences[index])

        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sequence': sequence,
            'id': self.ids[index]
        }
