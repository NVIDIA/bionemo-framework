#!/usr/bin/env python

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

"""Test script for remote chunked SCDL loading with ChunkAwareSampler.

Usage:
    python test_remote_loading.py s3://my-bucket/chunked_scdl
    python test_remote_loading.py gs://my-bucket/chunked_scdl
    python test_remote_loading.py --cache-dir /tmp/cache --max-chunks 3 s3://bucket/path
"""

import argparse

from torch.utils.data import DataLoader

from bionemo.scdl.io.chunk_sampler import ChunkAwareSampler
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch


def main():
    parser = argparse.ArgumentParser(description="Test remote chunked SCDL loading")
    parser.add_argument(
        "--remote_path",
        default="s3://general-purpose/polina/chunked",
        help="Remote path (s3://..., gs://..., https://...)",
    )
    parser.add_argument("--endpoint-url", default="https://pbss.s8k.io", help="S3 endpoint URL (for non-AWS S3)")
    parser.add_argument("--cache-dir", default="/tmp/scdl_cache", help="Local cache directory")
    parser.add_argument("--max-chunks", type=int, default=3, help="Max chunks to cache")
    parser.add_argument("--chunks-per-window", type=int, default=2, help="Chunks per sampling window")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to iterate")
    args = parser.parse_args()

    print(f"Loading remote dataset: {args.remote_path}")
    print(f"  Endpoint: {args.endpoint_url}")
    print(f"  Cache dir: {args.cache_dir}")
    print(f"  Max cached chunks: {args.max_chunks}")

    # Build storage_options for S3-compatible storage
    # For s3fs, endpoint_url must be in client_kwargs
    storage_options = {}
    if args.endpoint_url:
        storage_options["client_kwargs"] = {"endpoint_url": args.endpoint_url}

    # 1. Load from remote
    ds = SingleCellMemMapDataset.from_remote(
        args.remote_path,
        cache_dir=args.cache_dir,
        max_cached_chunks=args.max_chunks,
        storage_options=storage_options if storage_options else None,
    )
    print(f"  Rows: {len(ds)}")
    print(f"  Chunks: {ds.header.chunked_info.num_chunks}")
    print(f"  Chunk size: {ds.header.chunked_info.chunk_size}")

    # 2. Create sampler
    print(f"\nCreating ChunkAwareSampler (chunks_per_window={args.chunks_per_window})...")
    sampler = ChunkAwareSampler(
        ds,
        shuffle_chunks=True,
        shuffle_within_window=True,
        chunks_per_window=args.chunks_per_window,
        seed=42,
    )

    # 3. Create DataLoader
    print(f"Creating DataLoader (batch_size={args.batch_size})...")
    loader = DataLoader(ds, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_sparse_matrix_batch)

    # 4. Iterate batches
    print(f"\nIterating {args.num_batches} batches...")
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        print(f"  Batch {i}: shape={batch.shape}")

    print("\nSuccess! Remote chunked loading works.")

    # 5. Cleanup (optional)
    if ds._chunk_loader:
        print(f"\nCleaning up cache at {args.cache_dir}...")
        ds._chunk_loader.cleanup()


if __name__ == "__main__":
    main()
