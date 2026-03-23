# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Activation Store for SAE training.

Provides disk-based storage for model activations when they don't fit in memory.
Activations are saved as sharded Parquet files and can be loaded, shuffled,
and served in batches during SAE training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


@dataclass
class ActivationStoreConfig:
    """Configuration for ActivationStore.

    Attributes:
        shard_size: Number of activations per shard file
        compression: Parquet compression ('snappy', 'gzip', 'zstd', None)
    """

    shard_size: int = 100_000
    compression: Optional[str] = "snappy"


class ActivationStore:
    """Store and serve activations from disk for SAE training.

    When working with large datasets, activations from the base model can be
    extracted once and saved to disk. This store handles:
    - Saving activations as sharded Parquet files
    - Loading and shuffling activations across shards
    - Serving batches via a DataLoader-compatible interface

    Example:
        >>> # Save all at once (if fits in memory)
        >>> store = ActivationStore("./activations")
        >>> store.save(embeddings_tensor)

        >>> # Or append incrementally (for large datasets)
        >>> store = ActivationStore("./activations")
        >>> for chunk in chunks:
        ...     store.append(chunk)
        >>> store.finalize()

        >>> # Load for training
        >>> store = ActivationStore("./activations")
        >>> dataloader = store.get_dataloader(batch_size=4096, shuffle=True)
        >>> for batch in dataloader:
        ...     loss = sae.loss(batch)
    """

    def __init__(
        self,
        path: Union[str, Path],
        config: Optional[ActivationStoreConfig] = None,
    ):
        """Initialize the activation store.

        Args:
            path: Directory to store/load activation shards
            config: Store configuration (uses defaults if None)
        """
        if not HAS_PARQUET:
            raise ImportError("pyarrow required. Install with: pip install pyarrow")

        self.path = Path(path)
        self.config = config or ActivationStoreConfig()
        self._metadata: Optional[dict] = None

        # State for incremental appending
        self._append_buffer: Optional[np.ndarray] = None
        self._append_n_samples: int = 0
        self._append_n_shards: int = 0
        self._append_hidden_dim: Optional[int] = None

    def save(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        metadata: Optional[dict] = None,
    ) -> int:
        """Save activations to sharded Parquet files.

        Args:
            activations: Activation tensor of shape [n_samples, hidden_dim]
            metadata: Optional metadata dict (saved with first shard)

        Returns:
            Number of shards created
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # Convert to numpy
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()

        n_samples, hidden_dim = activations.shape
        shard_size = self.config.shard_size
        n_shards = (n_samples + shard_size - 1) // shard_size

        # Save metadata
        self._metadata = {
            "n_samples": n_samples,
            "hidden_dim": hidden_dim,
            "n_shards": n_shards,
            "shard_size": shard_size,
            **(metadata or {}),
        }
        self._save_metadata()

        # Save shards
        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, n_samples)
            shard_data = activations[start:end]

            self._save_shard(shard_idx, shard_data)

        print(f"Saved {n_samples} activations to {n_shards} shards at {self.path}")
        return n_shards

    def append(
        self,
        activations: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Append activations incrementally (for large datasets).

        Use this when you can't fit all activations in memory at once.
        Call finalize() after appending all chunks.

        Args:
            activations: Activation tensor of shape [n_samples, hidden_dim]

        Example:
            >>> store = ActivationStore("./activations")
            >>> for i in range(0, len(sequences), chunk_size):
            ...     embeddings = extract_embeddings(sequences[i:i+chunk_size])
            ...     store.append(embeddings)
            >>> store.finalize(metadata={'model': 'esm2'})
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # Convert to numpy
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()

        n_samples, hidden_dim = activations.shape

        # Validate hidden_dim consistency
        if self._append_hidden_dim is None:
            self._append_hidden_dim = hidden_dim
        elif self._append_hidden_dim != hidden_dim:
            raise ValueError(f"Hidden dim mismatch: expected {self._append_hidden_dim}, got {hidden_dim}")

        # Add to buffer
        if self._append_buffer is None:
            self._append_buffer = activations
        else:
            self._append_buffer = np.concatenate([self._append_buffer, activations], axis=0)

        self._append_n_samples += n_samples

        # Flush full shards
        shard_size = self.config.shard_size
        while len(self._append_buffer) >= shard_size:
            shard_data = self._append_buffer[:shard_size]
            self._save_shard(self._append_n_shards, shard_data)
            self._append_buffer = self._append_buffer[shard_size:]
            self._append_n_shards += 1

    def finalize(self, metadata: Optional[dict] = None) -> int:
        """Finalize the store after appending all chunks.

        Writes any remaining buffered data and saves metadata.

        Args:
            metadata: Optional metadata dict to save

        Returns:
            Total number of shards created
        """
        if self._append_hidden_dim is None:
            raise RuntimeError("No data appended. Call append() before finalize().")

        # Flush remaining buffer
        if self._append_buffer is not None and len(self._append_buffer) > 0:
            self._save_shard(self._append_n_shards, self._append_buffer)
            self._append_n_shards += 1

        # Save metadata
        self._metadata = {
            "n_samples": self._append_n_samples,
            "hidden_dim": self._append_hidden_dim,
            "n_shards": self._append_n_shards,
            "shard_size": self.config.shard_size,
            **(metadata or {}),
        }
        self._save_metadata()

        n_shards = self._append_n_shards
        n_samples = self._append_n_samples

        # Reset append state
        self._append_buffer = None
        self._append_n_samples = 0
        self._append_n_shards = 0
        self._append_hidden_dim = None

        print(f"Finalized {n_samples} activations in {n_shards} shards at {self.path}")
        return n_shards

    def _save_shard(self, shard_idx: int, data: np.ndarray) -> None:
        """Save a single shard to Parquet."""
        # Store as a table with one column per dimension
        # This is more efficient for columnar reads
        columns = {f"dim_{i}": data[:, i] for i in range(data.shape[1])}
        table = pa.table(columns)

        shard_path = self.path / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(
            table,
            shard_path,
            compression=self.config.compression,
        )

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        import json

        metadata_path = self.path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def _load_metadata(self) -> dict:
        """Load metadata from JSON file."""
        import json

        metadata_path = self.path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found at {metadata_path}")
        with open(metadata_path, "r") as f:
            return json.load(f)

    @property
    def metadata(self) -> dict:
        """Get store metadata (loads from disk if needed)."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata

    @property
    def n_samples(self) -> int:
        """Total number of samples in the store."""
        return self.metadata["n_samples"]

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of stored activations."""
        return self.metadata["hidden_dim"]

    @property
    def n_shards(self) -> int:
        """Number of shard files."""
        return self.metadata["n_shards"]

    def _load_shard(self, shard_idx: int) -> np.ndarray:
        """Load a single shard from Parquet."""
        shard_path = self.path / f"shard_{shard_idx:05d}.parquet"
        table = pq.read_table(shard_path)

        # Reconstruct array from columns
        hidden_dim = self.hidden_dim
        arrays = [table.column(f"dim_{i}").to_numpy() for i in range(hidden_dim)]
        return np.stack(arrays, axis=1)

    def iter_shards(
        self,
        shuffle_shards: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """Iterate over shards, optionally shuffled.

        Args:
            shuffle_shards: Whether to randomize shard order
            seed: Random seed for reproducibility

        Yields:
            Shard data as numpy arrays
        """
        shard_indices = list(range(self.n_shards))

        if shuffle_shards:
            rng = np.random.default_rng(seed)
            rng.shuffle(shard_indices)

        for shard_idx in shard_indices:
            yield self._load_shard(shard_idx)

    def get_streaming_dataloader(
        self,
        batch_size: int = 4096,
        shuffle: bool = True,
        seed: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        max_shards: Optional[int] = None,
    ) -> DataLoader:
        """Get a streaming DataLoader that reads one shard at a time from disk.

        Each rank gets a disjoint slice of shards. Peak RAM per rank is ~1 shard.

        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle shard order and within-shard data
            seed: Random seed for reproducibility
            rank: This rank's index (0-indexed)
            world_size: Total number of ranks
            max_shards: Limit total shards used (for subsampling). None = all.

        Returns:
            DataLoader yielding [batch_size, hidden_dim] tensors
        """
        n_total = max_shards if max_shards else self.n_shards
        n_total = min(n_total, self.n_shards)

        # Drop the last shard if it's shorter than shard_size (avoids DDP batch desync)
        shard_size = self.metadata.get("shard_size", 100_000)
        last_shard_path = self.path / f"shard_{n_total - 1:05d}.parquet"
        if n_total > 1 and pq.read_metadata(last_shard_path).num_rows < shard_size:
            n_total -= 1

        # Assign equal shards to each rank (drop remainder to keep DDP in sync)
        all_indices = list(range(n_total))
        per_rank = n_total // world_size
        my_indices = all_indices[rank * per_rank : (rank + 1) * per_rank]

        dataset = _StreamingBatchDataset(
            store=self,
            shard_indices=my_indices,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

        # batch_size=None: dataset already yields pre-formed batches
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def get_dataloader(
        self,
        batch_size: int = 4096,
        shuffle: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        num_workers: int = 0,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> DataLoader:
        """Get a DataLoader for training.

        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            shuffle_buffer_size: Size of shuffle buffer (default: 2x batch_size)
            num_workers: Number of DataLoader workers
            seed: Random seed for reproducibility
            device: Device to move batches to (None = keep on CPU)

        Returns:
            PyTorch DataLoader yielding activation batches
        """
        if shuffle_buffer_size is None:
            shuffle_buffer_size = batch_size * 2

        dataset = _ActivationIterableDataset(
            store=self,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            device=device,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            # IterableDataset handles its own shuffling
        )


class _ActivationIterableDataset(IterableDataset):
    """IterableDataset that streams activations from disk."""

    def __init__(
        self,
        store: ActivationStore,
        shuffle: bool = True,
        shuffle_buffer_size: int = 8192,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.store = store
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.device = device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over activations with optional shuffling."""
        rng = np.random.default_rng(self.seed)

        if self.shuffle:
            # Shuffle within a buffer as we stream
            buffer = []

            for shard in self.store.iter_shards(shuffle_shards=True, seed=self.seed):
                # Shuffle within shard
                rng.shuffle(shard)

                for row in shard:
                    buffer.append(row)

                    if len(buffer) >= self.shuffle_buffer_size:
                        # Shuffle and yield from buffer
                        rng.shuffle(buffer)
                        for item in buffer:
                            tensor = torch.from_numpy(item).float()
                            if self.device:
                                tensor = tensor.to(self.device)
                            yield tensor
                        buffer = []

            # Yield remaining
            if buffer:
                rng.shuffle(buffer)
                for item in buffer:
                    tensor = torch.from_numpy(item).float()
                    if self.device:
                        tensor = tensor.to(self.device)
                    yield tensor
        else:
            # Stream directly without shuffling
            for shard in self.store.iter_shards(shuffle_shards=False):
                for row in shard:
                    tensor = torch.from_numpy(row).float()
                    if self.device:
                        tensor = tensor.to(self.device)
                    yield tensor

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.store.n_samples


class _StreamingBatchDataset(IterableDataset):
    """IterableDataset that streams pre-formed batches from assigned shards.

    Each __next__ returns a [batch_size, hidden_dim] tensor. One shard in RAM at a time.
    """

    def __init__(
        self,
        store: ActivationStore,
        shard_indices: List[int],
        batch_size: int = 4096,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.store = store
        self.shard_indices = shard_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.max_batches = None  # Set externally to cap iteration (for DDP sync)

        # Approximate length: total tokens in assigned shards / batch_size
        shard_size = store.metadata.get("shard_size", 100_000)
        self._approx_tokens = len(shard_indices) * shard_size
        self._approx_len = self._approx_tokens // batch_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        rng = np.random.default_rng(self.seed)
        indices = self.shard_indices.copy()
        if self.shuffle:
            rng.shuffle(indices)

        buffer = None
        n_yielded = 0
        for shard_idx in indices:
            shard = torch.from_numpy(self.store._load_shard(shard_idx)).float()
            if self.shuffle:
                shard = shard[torch.randperm(len(shard))]

            buffer = torch.cat([buffer, shard]) if buffer is not None else shard

            while len(buffer) >= self.batch_size:
                if self.max_batches is not None and n_yielded >= self.max_batches:
                    return
                yield buffer[: self.batch_size]
                buffer = buffer[self.batch_size :]
                n_yielded += 1

        # Yield remainder as a partial batch (skip if capped)
        if self.max_batches is None and buffer is not None and len(buffer) > 0:
            yield buffer

    def __len__(self) -> int:
        if self.max_batches is not None:
            return self.max_batches
        return self._approx_len


def save_activations(
    activations: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    shard_size: int = 100_000,
    compression: Optional[str] = "snappy",
    metadata: Optional[dict] = None,
) -> ActivationStore:
    """Convenience function to save activations to disk.

    Args:
        activations: Tensor of shape [n_samples, hidden_dim]
        path: Directory to save shards
        shard_size: Number of samples per shard
        compression: Parquet compression type
        metadata: Optional metadata dict

    Returns:
        ActivationStore instance for the saved data

    Example:
        >>> store = save_activations(embeddings, "./activations")
        >>> dataloader = store.get_dataloader(batch_size=4096)
    """
    config = ActivationStoreConfig(shard_size=shard_size, compression=compression)
    store = ActivationStore(path, config)
    store.save(activations, metadata=metadata)
    return store


def load_activations(path: Union[str, Path]) -> ActivationStore:
    """Convenience function to load an existing activation store.

    Args:
        path: Directory containing activation shards

    Returns:
        ActivationStore instance

    Example:
        >>> store = load_activations("./activations")
        >>> print(f"Loaded {store.n_samples} activations")
        >>> dataloader = store.get_dataloader(batch_size=4096)
    """
    return ActivationStore(path)
