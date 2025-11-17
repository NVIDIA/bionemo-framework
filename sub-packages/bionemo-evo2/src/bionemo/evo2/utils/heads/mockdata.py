# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from torch.utils.data import DataLoader, Dataset


if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.bytelevel_tokenizers import ByteLevelTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class _ParallelHeadMockDataset(Dataset):
    """ParallelHead Mock Dataset.

    Synthetic dataset designed to support training of models with *parallel heads*, such as:
    - DNA language modeling (standard token prediction)
    - RNA-seq expression prediction (per-token regression)

    This is a drop-in replacement for `_MockGPTDataset` with:
    - identical token/label structure
    - additional `rna_seq_targets` field for each sample

    Expression targets can follow different patterns:
    - "realistic" (based on GC content, promoter regions, and positional decay)
    - "periodic"
    - "constant"
    - "random"
    """

    def __init__(
        self,
        tokenizer: "ByteLevelTokenizer",
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
        create_attention_mask: bool = False,
        expression_pattern: str = "realistic",
        expression_noise_level: float = 0.1,
        expression_range: tuple = (0.0, 10.0),
        rna_seq: bool = True,
        pep_map: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length
        self.vocab_size = tokenizer.vocab_size
        self.length = num_samples
        self.seed = seed
        self.create_attention_mask = create_attention_mask

        # Expression generation configuration
        self.expression_pattern = expression_pattern
        self.expression_noise_level = expression_noise_level
        self.expression_range = expression_range

        # Mock data types
        self.rna_seq = rna_seq
        self.pep_map = pep_map

        # Precomputed masks and IDs
        if create_attention_mask:
            # Causal attention mask (lower triangle)
            self.attention_mask = torch.tril(torch.ones((self.seq_length, self.seq_length), device="cpu")).unsqueeze(0)
            self.attention_mask = self.attention_mask < 0.5  # Convert to bool

        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.long)

        logging.info("🔧 Created _ParallelHeadMockDataset:")
        logging.info(f"   Name: {name}")
        logging.info(f"   Samples: {num_samples}")
        logging.info(f"   Seq length: {seq_length}")
        logging.info(f"   Expression pattern: {expression_pattern}")
        logging.info(f"   RNA-seq: {rna_seq}")
        logging.info(f"   PEP-map: {pep_map}")

    def __len__(self) -> int:
        return self.length

    def _generate_rna_seq_targets(self, tokens: torch.Tensor) -> torch.Tensor:
        """RNA-seq Target Generation.

        Generate simulated RNA-seq expression values per token based on a pattern.

        Args:
            tokens (torch.Tensor): Token sequence (shape: [seq_length]).

        Returns:
            torch.Tensor: Expression values (shape: [seq_length], dtype: bfloat16).
        """
        seq_len = len(tokens)

        try:
            if self.expression_pattern == "realistic":
                expression = self._realistic_expression_pattern(tokens)
            elif self.expression_pattern == "periodic":
                x = np.linspace(0, 4 * np.pi, seq_len)
                expression = 2.0 + 1.5 * np.sin(x) + 0.5 * np.cos(2 * x)
            elif self.expression_pattern == "constant":
                expression = np.full(seq_len, 3.0)
            else:  # "random"
                expression = np.random.uniform(self.expression_range[0], self.expression_range[1], seq_len)

            # Add Gaussian noise
            if self.expression_noise_level > 0:
                noise = np.random.normal(0, self.expression_noise_level, seq_len)
                expression += noise

            # Clip to valid range
            expression = np.clip(expression, self.expression_range[0], self.expression_range[1])

            return torch.tensor(expression, dtype=torch.bfloat16)

        except Exception as e:
            logging.error(f"❌ Error generating expression targets: {e}")
            return torch.zeros(seq_len, dtype=torch.bfloat16)

    def _realistic_expression_pattern(self, tokens: torch.Tensor) -> np.ndarray:
        """Biologically inspired mock data.

        Generate biologically-inspired expression levels based on
        nucleotide composition and positional effects.

        Args:
            tokens (torch.Tensor): Sequence of tokens (shape: [seq_length]).

        Returns:
            np.ndarray: Simulated expression values (float32).
        """
        seq_len = len(tokens)
        expression = np.zeros(seq_len, dtype=np.float32)

        # Map token IDs to mock nucleotide classes (1='A', 2='T', 3='C', 4='G')
        mock_nucleotides = (tokens % 4) + 1

        # Additive contribution from GC content
        gc_content = self._calculate_sliding_gc_content(mock_nucleotides)
        expression += gc_content * 2.0

        # Add contribution from promoter-like regions
        promoter_positions = self._find_mock_promoters(mock_nucleotides)
        for pos in promoter_positions:
            start = max(0, pos - 50)
            end = min(seq_len, pos + 200)
            distances = np.arange(start, end) - pos
            signal = 3.0 * np.exp(-np.abs(distances) / 100.0)
            expression[start:end] += signal

        # Add positional decay (stronger expression near sequence start)
        distance_decay = np.exp(-np.arange(seq_len) / (seq_len * 0.3))
        expression += distance_decay

        return expression

    def _calculate_sliding_gc_content(self, nucleotides: torch.Tensor, window_size: int = 50) -> np.ndarray:
        """Compute GC content.

        Compute GC content across the sequence using a sliding window.

        Args:
            nucleotides (torch.Tensor): Values in range 1-4 representing nucleotides.
            window_size (int): Size of window to compute GC frequency.

        Returns:
            np.ndarray: GC content (0-1) for each position.
        """
        seq_len = len(nucleotides)
        gc_content = np.zeros(seq_len, dtype=np.float32)

        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2)
            window = nucleotides[start:end]
            gc_count = torch.sum((window == 3) | (window == 4)).float()  # C or G
            gc_content[i] = gc_count / len(window)

        return gc_content

    def _find_mock_promoters(self, nucleotides: torch.Tensor) -> list:
        """Mock promoter finder.

        Heuristically identify "promoter-like" regions based on token pattern.

        Returns:
            list[int]: List of center positions for promoter-like signals.
        """
        promoters = []
        seq_len = len(nucleotides)

        for i in range(0, seq_len - 10, 100):
            window = nucleotides[i : i + 10]
            if len(window) == 10:
                pattern_score = torch.sum(window[::2] == 2).float()
                if pattern_score >= 3:
                    promoters.append(i + 5)

        return promoters

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Generate a synthetic training example.

        Returns a synthetic training example with:
        - tokens: input tokens (seq_length)
        - labels: target tokens (next-token)
        - position_ids, loss_mask, optional attention_mask
        - rna_seq_targets: expression values per token

        Returns:
            Dict[str, torch.Tensor]: A training example with multiple targets.
        """
        # Sample-specific RNG for reproducibility
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length + 1], dtype=np.int64))

        # Create batch exactly like original MockDataModule
        batch = {
            "tokens": tokens[:-1],  # ✅ Input tokens (original field name)
            "labels": tokens[1:],  # ✅ Shifted tokens for next-token prediction (original logic)
            "loss_mask": self.loss_mask,  # ✅ Original field name
            "position_ids": self.position_ids,  # ✅ Original field name
        }

        # Add attention mask if needed (original logic)
        if self.create_attention_mask:
            batch["attention_mask"] = self.attention_mask

        # Generate expression targets (same length as tokens[:-1])
        if self.expression_pattern is not None:
            input_tokens = tokens[:-1]  # Use input tokens for expression generation
            rna_seq_targets = self._generate_rna_seq_targets(input_tokens)
            batch["rna_seq_targets"] = rna_seq_targets

        # Generate expression targets (same length as tokens[:-1])
        if self.expression_pattern is not None:
            input_tokens = tokens[:-1]  # Use input tokens for expression generation
            rna_seq_targets = self._generate_rna_seq_targets(input_tokens)
            batch["pep_map_targets"] = rna_seq_targets

        # Basic sanity check
        for key, value in batch.items():
            if value is None:
                logging.error(f"❌ Batch key '{key}' is None!")
                raise ValueError(f"Batch key '{key}' is None!")

        return batch

    def _collate_fn(self, batch):
        """Default PyTorch collation (batched tensor stacking)."""
        from torch.utils.data import dataloader

        return dataloader.default_collate(batch)

    def collate_fn(self, batch):
        """Returns the callable to be used by DataLoader."""
        return self._collate_fn(batch)


class ParallelHeadMockDataModule(pl.LightningDataModule):
    """ParallelHeadMockDataModule.

    A minimal extension of a GPT-style `MockDataModule` to support dual-task training with
    both language modeling (DNA) and RNA expression prediction targets.

    ✅ Keeps the original structure and behavior of MockDataModule.
    ✅ Adds `rna_seq_targets` to each batch via `_ParallelHeadMockDataset`.
    ✅ Designed to be easy to integrate into existing pipelines with no structural changes.

    Inherits:
        pl.LightningDataModule - PyTorch Lightning standard for modular dataloading.

    Args:
        seq_length (int): Length of input sequences.
        tokenizer (Optional[TokenizerSpec]): Tokenizer object or None to auto-create GPT2 BPE.
        micro_batch_size (int): Per-GPU micro-batch size.
        global_batch_size (int): Total batch size across all GPUs.
        rampup_batch_size (Optional[List[int]]): Optional ramp-up schedule for batch sizes.
        num_train_samples (int): Number of synthetic training samples.
        num_val_samples (int): Number of synthetic validation samples.
        num_test_samples (int): Number of synthetic test samples.
        num_workers (int): Dataloader workers.
        pin_memory (bool): Use CUDA pinned memory.
        persistent_workers (bool): Persist worker threads across epochs.
        create_attention_mask (bool): Whether to generate attention masks for sequences.
        vocab_file (Optional[str]): Optional tokenizer vocab path.
        merges_file (Optional[str]): Optional tokenizer merges file path.
        expression_pattern (str): Pattern of RNA expression to simulate (e.g. "realistic").
        expression_noise_level (float): Level of noise added to RNA signal.
        expression_range (tuple): Min and max values for simulated RNA targets.
    """

    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,  # type: ignore
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        create_attention_mask: bool = False,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        expression_pattern: str = "realistic",
        expression_noise_level: float = 0.1,
        expression_range: tuple = (0.0, 10.0),
        rna_seq: bool = True,
        pep_map: bool = False,
    ):
        """Initialize SimpleDualHeadMockDataModule.

        Exact copy of MockDataModule + expression params.

        Args:
            seq_length (int): Sequence length.
            tokenizer (Optional["TokenizerSpec"]): An instance of a TokenizerSpec object.
            micro_batch_size (int): Batch size per GPU.
            global_batch_size (int): Global batch size.
            rampup_batch_size (Optional[List[int]]): Rampup batch size, should be in format of
                [start_global_batch_size, batch_size_increment, ramup_samples].
            num_train_samples (int): The number of samples to use for training, defaults to total
                train steps times global batch size.
            num_val_samples (int): The number of samples to use for validation, defaults to total
                validation steps times global batch size.
            num_test_samples (int): The number of samples to use for testing, defaults to total
                test steps times global batch size.
            num_workers (int): See `torch.utils.data.DataLoader` documentation.
            pin_memory (bool): See `torch.utils.data.DataLoader` documentation.
            persistent_workers (bool): See `torch.utils.data.DataLoader` documentation.
            create_attention_mask (bool): Whether to create attention masks for the sequences.
            vocab_file (Optional[str]): Optional vocabulary file for tokenizer.
            merges_file (Optional[str]): Optional merges file for tokenizer.
            expression_pattern (str): Pattern of RNA expression to simulate (e.g. "realistic").
            expression_noise_level (float): Level of noise added to RNA signal.
            expression_range (tuple): Min and max values for simulated RNA targets.
            rna_seq (bool): Whether to simulate RNA sequences.
            pep_map (bool): Whether to simulate peptide maps.
        """
        super().__init__()

        # Core parameters
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.create_attention_mask = create_attention_mask

        # Expression parameters
        self.expression_pattern = expression_pattern
        self.expression_noise_level = expression_noise_level
        self.expression_range = expression_range

        # Mock data types
        self.rna_seq = rna_seq
        self.pep_map = pep_map

        # Setup tokenizer exactly like original
        if tokenizer is None:
            from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

            self.tokenizer = get_nmt_tokenizer(
                "megatron", "GPT2BPETokenizer", vocab_file=vocab_file, merges_file=merges_file
            )
        else:
            self.tokenizer = tokenizer

        # Data sampler for batch ramp-up and sample generation
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

        # Logging configuration summary
        logging.info("🚀 SimpleParallelHeadMockDataModule initialized:")
        logging.info(f"   Seq length: {seq_length}")
        logging.info(f"   Micro batch size: {micro_batch_size}")
        logging.info(f"   Global batch size: {global_batch_size}")
        logging.info(f"   Expression pattern: {expression_pattern}")
        logging.info(f"   Has MegatronDataSampler: {hasattr(self, 'data_sampler')}")

    def setup(self, stage: str = "") -> None:
        """Setup datasets.

        Exact copy of original MockDataModule setup.

        Args:
            stage (str): Stage of setup (train/val/test).
        """
        # Create datasets using our dual-head version instead of _MockGPTDataset
        self._train_ds = _ParallelHeadMockDataset(
            self.tokenizer,
            "train",
            self.num_train_samples,
            self.seq_length,  # type: ignore
            create_attention_mask=self.create_attention_mask,
            expression_pattern=self.expression_pattern,
            expression_noise_level=self.expression_noise_level,
            expression_range=self.expression_range,
            rna_seq=self.rna_seq,
            pep_map=self.pep_map,
        )
        self._validation_ds = _ParallelHeadMockDataset(
            self.tokenizer,
            "valid",
            self.num_val_samples,
            self.seq_length,  # type: ignore
            create_attention_mask=self.create_attention_mask,
            expression_pattern=self.expression_pattern,
            expression_noise_level=self.expression_noise_level,
            expression_range=self.expression_range,
            rna_seq=self.rna_seq,
            pep_map=self.pep_map,
        )
        self._test_ds = _ParallelHeadMockDataset(
            self.tokenizer,
            "test",
            self.num_test_samples,
            self.seq_length,  # type: ignore
            create_attention_mask=self.create_attention_mask,
            expression_pattern=self.expression_pattern,
            expression_noise_level=self.expression_noise_level,
            expression_range=self.expression_range,
            rna_seq=self.rna_seq,
            pep_map=self.pep_map,
        )

        logging.info("✅ SimpleParallelHeadMockDataModule setup completed")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Train Dataloader.

        Returns:
            DataLoader: Training dataloader using synthetic dual-head data.
        """
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Validation Dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Test Dataloader.

        Returns:
            DataLoader: Test dataloader.
        """
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        """Internal helper for constructing a dataloader.

        Given a dataset, constructs a DataLoader with appropriate settings.

        Args:
            dataset: The dataset to wrap in a DataLoader.
            **kwargs: Additional arguments for DataLoader.

        Returns:
            DataLoader: Configured dataloader with correct collate function.
        """
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )
