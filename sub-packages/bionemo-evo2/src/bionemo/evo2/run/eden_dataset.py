from typing import TYPE_CHECKING, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, Subset
from pyfaidx import Fasta

from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils.import_utils import safe_import

_, HAVE_TE = safe_import("transformer_engine")

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class FastaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fasta_file: str,
        seq_length: int = 8192,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 1,
        global_batch_size: int = 4,
        rampup_batch_size: Optional[List[int]] = None,
        train_val_test_split: List[float] = [0.8, 0.1, 0.1],
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        create_attention_mask: bool = False,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        rc_aug: bool = False,
        stride: int = None,
        num_train_samples: int = None,
        num_val_samples: int = None,
        num_test_samples: int = None,
    ):
        super().__init__()
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.create_attention_mask = create_attention_mask or not HAVE_TE
        self.rc_aug = rc_aug
        self.stride = stride if stride is not None else seq_length // 2
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        if tokenizer is None:
            from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

            self.tokenizer = get_nmt_tokenizer("megatron", "GPT2BPETokenizer", vocab_file=vocab_file, merges_file=merges_file)
        else:
            self.tokenizer = tokenizer

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        # Create a single dataset and split it for train/val/test
        self._full_dataset = FastaDataset(self.tokenizer, self.fasta_file, self.seq_length, self.create_attention_mask, rc_aug=self.rc_aug, stride=self.stride)

        # Calculate split sizes
        total_size = len(self._full_dataset)
        train_size = int(total_size * self.train_val_test_split[0])
        val_size = int(total_size * self.train_val_test_split[1])
        test_size = total_size - train_size - val_size

        # Create train/val/test datasets
        indices = list(range(total_size))
        np.random.shuffle(indices)  # Shuffle to ensure random distribution

        self._train_ds = Subset(self._full_dataset, indices[:train_size])
        self._validation_ds = Subset(self._full_dataset, indices[train_size : train_size + val_size])
        self._test_ds = Subset(self._full_dataset, indices[train_size + val_size :])

        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._full_dataset.collate_fn if hasattr(self, "_full_dataset") else None,
            **kwargs,
        )


class FastaDataset(Dataset):
    def __init__(
        self,
        tokenizer: "TokenizerSpec",
        fasta_file: str,
        seq_length: int,
        create_attention_mask: bool = False,
        rc_aug: bool = False,
        stride: int = None,
    ) -> None:
        super().__init__()
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.create_attention_mask = create_attention_mask
        self.rc_aug = rc_aug
        self.stride = stride if stride is not None else seq_length // 2

        # Load sequences from FASTA file
        self.sequences = Fasta(fasta_file)

        # Create a list of (sequence_name, start_pos) tuples for all possible windows
        self.sequence_windows = []
        for seq_name in self.sequences.keys():
            seq_len = len(self.sequences[seq_name])
            if seq_len < self.seq_length:
                continue  # Skip sequences shorter than seq_length

            # Create windows with specified stride
            for start_pos in range(0, seq_len - self.seq_length + 1, self.stride):
                self.sequence_windows.append((seq_name, start_pos))

        self.length = len(self.sequence_windows)
        print(f"Created dataset with {self.length} sequence windows from {len(self.sequences.keys())} sequences")

        if create_attention_mask:
            self.attention_mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device="cpu")).unsqueeze(0)
            self.attention_mask = self.attention_mask < 0.5

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

        # Complement mapping for reverse complement
        self.complement_map = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N", "a": "t", "c": "g", "g": "c", "t": "a", "n": "n"}

    def __len__(self) -> int:
        return self.length

    def reverse_complement(self, seq):
        """Generate the reverse complement of a DNA sequence"""
        return "".join(self.complement_map.get(base, base) for base in reversed(seq))

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        seq_name, start_pos = self.sequence_windows[idx]
        seq = str(self.sequences[seq_name][start_pos : start_pos + self.seq_length])

        # Apply reverse complement augmentation if enabled
        if self.rc_aug and np.random.random() > 0.5:
            seq = self.reverse_complement(seq)

        # Tokenize using provided tokenizer
        tokenized = self.tokenizer.text_to_ids(seq)

        # Ensure tokenized sequence matches the desired seq_length
        if len(tokenized) < self.seq_length:
            tokenized += [self.tokenizer.eos_id] * (self.seq_length - len(tokenized))
        else:
            tokenized = tokenized[: self.seq_length]

        tokens = torch.tensor(tokenized, dtype=torch.int64)

        # For language modeling, labels are tokens shifted by one
        labels = torch.roll(tokens, shifts=-1, dims=0)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }

        if self.create_attention_mask:
            batch["attention_mask"] = self.attention_mask

        return batch


    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        """
        return data.dataloader.default_collate(batch)

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader."""
        return self._collate_fn(batch)

