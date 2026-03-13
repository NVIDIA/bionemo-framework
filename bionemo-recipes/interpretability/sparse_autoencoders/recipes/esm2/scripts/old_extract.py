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

"""Step 1: Extract activations from ESM2 and save to disk.

Extracts layer activations from an ESM2 model for a set of protein sequences
and writes them as sharded Parquet files via ActivationStore.

Supports multi-GPU extraction via torchrun for Nx speedup:
    torchrun --nproc_per_node=4 scripts/step1_extract.py ...

Single-GPU usage:
    python scripts/step1_extract.py \
        data.source=uniref50 data.data_dir=./data data.num_proteins=10000 \
        activations.model_name=facebook/esm2_t33_650M_UR50D activations.layer=25 \
        activations.cache_dir=.cache/activations/esm2_650m_layer25
"""

import json
import os
import shutil
import time
from pathlib import Path

import hydra
import torch
from esm2_sae.data import download_swissprot, download_uniref50, read_fasta
from esm2_sae.models import ESM2Model
from omegaconf import DictConfig, OmegaConf
from sae.utils import get_device, set_seed


def resolve_data_path(cfg: DictConfig, data_dir: Path, rank: int) -> Path:
    """Resolve protein FASTA path, downloading if needed. Only rank 0 downloads."""
    source = str(cfg.data.get("source", "swissprot")).lower()
    num_proteins = cfg.data.get("num_proteins", None)

    if source == "swissprot":
        fasta_path = data_dir / "uniprot_sprot.fasta.gz"
        if not fasta_path.exists():
            if rank == 0:
                print(f"Downloading SwissProt to {fasta_path}")
                download_swissprot(data_dir)
            else:
                _wait_for_file(fasta_path)
        return fasta_path

    if source == "uniref50":
        download_max = cfg.data.get("download_max_proteins", num_proteins)
        if download_max is None:
            fasta_path = data_dir / "uniref50.fasta.gz"
        else:
            fasta_path = data_dir / f"uniref50_first_{download_max}.fasta"

        if not fasta_path.exists():
            if rank == 0:
                print(f"Downloading UniRef50 to {fasta_path}")
                download_uniref50(data_dir, max_proteins=download_max)
            else:
                _wait_for_file(fasta_path)
        return fasta_path

    raise ValueError(f"Unknown data.source='{source}'. Use 'swissprot' or 'uniref50'.")


def _wait_for_file(path: Path, timeout_sec: int = 7200, poll_sec: float = 2.0) -> None:
    """Wait for a file to appear (non-rank-0 waits for rank 0 to download)."""
    start = time.time()
    while not path.exists():
        if (time.time() - start) > timeout_sec:
            raise TimeoutError(f"Timed out waiting for: {path}")
        time.sleep(poll_sec)


def _merge_rank_stores(cache_path: Path, world_size: int, metadata: dict) -> None:
    """Merge per-rank temp stores into a single store by moving shard files."""
    cache_path.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    total_samples = 0
    hidden_dim = None
    shard_size = None

    for r in range(world_size):
        tmp_dir = cache_path / f".tmp_rank_{r}"
        with open(tmp_dir / "metadata.json") as f:
            tmp_meta = json.load(f)

        hidden_dim = tmp_meta["hidden_dim"]
        shard_size = tmp_meta["shard_size"]

        for i in range(tmp_meta["n_shards"]):
            src = tmp_dir / f"shard_{i:05d}.parquet"
            dst = cache_path / f"shard_{shard_idx:05d}.parquet"
            shutil.move(str(src), str(dst))
            shard_idx += 1

        total_samples += tmp_meta["n_samples"]
        shutil.rmtree(tmp_dir)

    metadata.update(
        n_samples=total_samples,
        n_shards=shard_idx,
        hidden_dim=hidden_dim,
        shard_size=shard_size,
    )
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Merged {world_size} rank stores: {total_samples:,} tokens, {shard_idx} shards")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract ESM2 layer activations using Hydra configuration."""
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    # Distributed setup
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        from datetime import timedelta

        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(hours=48))
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = get_device()

    print(f"[Rank {rank}/{world_size}] Device: {device}")

    # Resolve cache path
    cache_dir = cfg.activations.get("cache_dir", None)
    if not cache_dir:
        raise ValueError("activations.cache_dir is required for extraction.")
    cache_path = Path(hydra.utils.get_original_cwd()) / cache_dir

    # Check if cache already exists
    if (cache_path / "metadata.json").exists():
        if rank == 0:
            print(f"Cache already exists at {cache_path}. Skipping extraction.")
            with open(cache_path / "metadata.json") as f:
                meta = json.load(f)
            print(f"  {meta['n_samples']:,} tokens, {meta['n_shards']} shards, dim={meta['hidden_dim']}")
        if world_size > 1:
            import torch.distributed as dist

            dist.barrier()
            dist.destroy_process_group()
        return

    # Clean up stale temp dirs from a previous failed multi-GPU run
    if rank == 0 and cache_path.exists():
        for tmp in cache_path.glob(".tmp_rank_*"):
            shutil.rmtree(tmp)

    # Load sequences
    data_dir = Path(hydra.utils.get_original_cwd()) / cfg.data.data_dir
    fasta_path = resolve_data_path(cfg, data_dir, rank)

    # Wait for download to finish on all ranks
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

    num_proteins = cfg.data.get("num_proteins", None)
    records = read_fasta(
        fasta_path,
        max_sequences=num_proteins,
        max_length=cfg.data.max_seq_length,
    )
    sequences = [rec.sequence for rec in records]
    total_sequences = len(sequences)

    if rank == 0:
        print(f"Loaded {total_sequences} sequences from {fasta_path}")

    # Split sequences across ranks
    if world_size > 1:
        chunk = total_sequences // world_size
        my_start = rank * chunk
        my_end = total_sequences if rank == world_size - 1 else (rank + 1) * chunk
        my_sequences = sequences[my_start:my_end]
        print(f"[Rank {rank}] Extracting sequences {my_start}-{my_end} ({len(my_sequences)} proteins)")
    else:
        my_sequences = sequences

    # Create ESM2 model
    esm2 = ESM2Model(
        model_name=cfg.activations.model_name,
        layer=cfg.activations.layer,
        device=device,
    )

    # Extract activations and write to store
    from sae.activation_store import ActivationStore
    from tqdm import tqdm

    if world_size > 1:
        store_path = cache_path / f".tmp_rank_{rank}"
    else:
        store_path = cache_path

    store = ActivationStore(store_path)
    batch_size = cfg.activations.batch_size
    remove_special = cfg.activations.remove_special_tokens
    padding = cfg.activations.get("tokenizer_padding", "longest")

    n_batches = (len(my_sequences) + batch_size - 1) // batch_size
    show_progress = rank == 0

    iterator = range(0, len(my_sequences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting activations")

    t0 = time.time()
    for i in iterator:
        batch_seqs = my_sequences[i : i + batch_size]
        batch_emb, batch_masks = esm2.generate_activations(
            sequences=batch_seqs,
            batch_size=len(batch_seqs),
            remove_special_tokens=remove_special,
            show_progress=False,
            padding=padding,
        )
        batch_flat = batch_emb[batch_masks.bool()]
        store.append(batch_flat)

    store.finalize(
        metadata={
            "model_name": cfg.activations.model_name,
            "layer": cfg.activations.layer,
            "n_sequences": len(my_sequences),
        }
    )

    elapsed = time.time() - t0
    print(
        f"[Rank {rank}] Extracted {store.metadata['n_samples']:,} tokens "
        f"from {len(my_sequences)} proteins in {elapsed:.1f}s"
    )

    # Free GPU memory
    del esm2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Multi-GPU: merge rank stores
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

        if rank == 0:
            _merge_rank_stores(
                cache_path,
                world_size,
                metadata={
                    "model_name": cfg.activations.model_name,
                    "layer": cfg.activations.layer,
                    "n_sequences": total_sequences,
                },
            )

        dist.barrier()
        dist.destroy_process_group()

    # Print final summary
    if rank == 0:
        with open(cache_path / "metadata.json") as f:
            meta = json.load(f)
        print("\nExtraction complete:")
        print(f"  Cache: {cache_path}")
        print(f"  Sequences: {meta.get('n_sequences', '?')}")
        print(f"  Tokens: {meta['n_samples']:,}")
        print(f"  Hidden dim: {meta['hidden_dim']}")
        print(f"  Shards: {meta['n_shards']}")


if __name__ == "__main__":
    main()
