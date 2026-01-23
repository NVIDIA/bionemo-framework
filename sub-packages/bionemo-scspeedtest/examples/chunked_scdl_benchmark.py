# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""Benchmark comparing Regular SCDL vs Chunked SCDL vs Remote Chunked SCDL.

Usage (with defaults):
    python chunked_scdl_benchmark.py

Custom paths:
    python chunked_scdl_benchmark.py \
        --scdl-path /path/to/scdl/ \
        --chunked-path /path/to/chunked/ \
        --remote-path s3://bucket/chunked \
        --endpoint-url https://your-s3-endpoint.com

This script benchmarks:
1. Regular SCDL - Standard DataLoader with shuffle (baseline)
2. Chunked SCDL (local) - Pre-converted chunked dataset with ChunkAwareSampler
3. Remote Chunked SCDL - S3/GCS with LRU caching and ChunkAwareSampler
"""

import argparse
import os
import time
from datetime import datetime

from torch.utils.data import DataLoader

from bionemo.scdl.io.chunk_sampler import ChunkAwareSampler
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch
from bionemo.scspeedtest import benchmark_dataloaders_with_configs, print_comparison


def create_regular_scdl_factory(
    batch_size: int = 64, shuffle: bool = True, data_path: str | None = None, num_workers: int = 0
):
    """Create a regular SCDL dataloader factory (baseline)."""

    def factory():
        dataset = SingleCellMemMapDataset(data_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_sparse_matrix_batch,
            num_workers=num_workers,
        )

    return factory


def create_chunked_scdl_preconverted_factory(
    batch_size: int = 64,
    chunked_path: str | None = None,
    num_workers: int = 0,
    shuffle_chunks: bool = True,
    shuffle_within_window: bool = True,
    chunks_per_window: int = 2,
):
    """Create a chunked SCDL dataloader factory from pre-converted chunked dataset."""

    def factory():
        dataset = SingleCellMemMapDataset(chunked_path)

        sampler = ChunkAwareSampler(
            dataset,
            shuffle_chunks=shuffle_chunks,
            shuffle_within_window=shuffle_within_window,
            chunks_per_window=chunks_per_window,
        )

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_sparse_matrix_batch,
            num_workers=num_workers,
        )

    return factory


def create_chunked_scdl_random_factory(
    batch_size: int = 64,
    chunked_path: str | None = None,
    num_workers: int = 0,
):
    """Create a chunked SCDL dataloader with random shuffle (no ChunkAwareSampler)."""
    start_time = time.perf_counter()

    def factory():
        dataset = SingleCellMemMapDataset(chunked_path)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # Standard random shuffle - worst case for I/O locality
            collate_fn=collate_sparse_matrix_batch,
            num_workers=num_workers,
        )

    end_time = time.perf_counter()
    print(f"Time taken to instantiate chunked SCDL dataset: {end_time - start_time:.2f} seconds")
    return factory


class RemoteDataloaderFactory:
    """Factory that tracks the last created dataset for stats access."""

    def __init__(
        self,
        batch_size: int = 64,
        remote_path: str | None = None,
        cache_dir: str | None = None,
        max_cached_chunks: int = 3,
        storage_options: dict | None = None,
        num_workers: int = 0,
        shuffle_chunks: bool = True,
        shuffle_within_window: bool = True,
        chunks_per_window: int = 2,
        batch_download_size: int = 30,
    ):
        """Initialize the remote dataloader factory with configuration."""
        self.batch_size = batch_size
        self.remote_path = remote_path
        self.cache_dir = cache_dir
        self.max_cached_chunks = max_cached_chunks
        self.storage_options = storage_options
        self.num_workers = num_workers
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_window = shuffle_within_window
        self.chunks_per_window = chunks_per_window
        self.batch_download_size = batch_download_size
        self.last_dataset = None  # Store reference for stats access
        # Timing breakdown
        self.cache_clear_time = 0.0
        self.dataset_init_time = 0.0
        self.sampler_init_time = 0.0
        self.dataloader_init_time = 0.0
        self.total_init_time = 0.0

    def __call__(self):
        """Create a new dataloader."""
        import shutil

        # Clear cache before each run to measure streaming performance
        import time

        total_start = time.perf_counter()

        # 1. Clear cache
        if self.cache_dir:
            print(f"Clearing cache directory: {self.cache_dir}")
            t0 = time.perf_counter()
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            t1 = time.perf_counter()
            self.cache_clear_time = t1 - t0
            print(f"Cache clear time: {self.cache_clear_time:.3f} sec")

        # 2. Create dataset
        print("Instantiating SingleCellMemMapDataset.from_remote ...")
        t2 = time.perf_counter()
        self.last_dataset = SingleCellMemMapDataset.from_remote(
            self.remote_path,
            cache_dir=self.cache_dir,
            max_cached_chunks=self.max_cached_chunks,
            storage_options=self.storage_options,
            batch_download_size=self.batch_download_size,
            use_async_downloads=True,
        )
        t3 = time.perf_counter()
        self.dataset_init_time = t3 - t2
        print(f"SingleCellMemMapDataset.from_remote time: {self.dataset_init_time:.3f} sec")

        # 3. Create sampler
        print("Instantiating ChunkAwareSampler ...")
        t4 = time.perf_counter()
        sampler = ChunkAwareSampler(
            self.last_dataset,
            shuffle_chunks=self.shuffle_chunks,
            shuffle_within_window=self.shuffle_within_window,
            chunks_per_window=self.chunks_per_window,
        )
        t5 = time.perf_counter()
        self.sampler_init_time = t5 - t4
        print(f"ChunkAwareSampler instantiation time: {self.sampler_init_time:.3f} sec")

        # 4. Create DataLoader
        print("Instantiating DataLoader ...")
        t6 = time.perf_counter()
        dataloader = DataLoader(
            self.last_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=collate_sparse_matrix_batch,
            num_workers=self.num_workers,
        )
        t7 = time.perf_counter()
        self.dataloader_init_time = t7 - t6
        self.total_init_time = t7 - total_start
        print(f"DataLoader instantiation time: {self.dataloader_init_time:.3f} sec")
        print(f"Total init time: {self.total_init_time:.3f} sec")

        return dataloader

    def get_download_stats(self) -> dict | None:
        """Get download statistics from the last created dataset."""
        if self.last_dataset and hasattr(self.last_dataset, "_chunk_loader"):
            return self.last_dataset._chunk_loader.stats.summary()
        return None

    def print_download_stats(self):
        """Print download statistics."""
        stats = self.get_download_stats()
        if stats:
            iteration_time = stats["wall_clock_time_sec"]
            total_time = self.total_init_time + iteration_time

            print("\n" + "=" * 50)
            print("REMOTE DOWNLOAD STATS")
            print("=" * 50)
            print(f"Init: {self.total_init_time:.1f}s | Iteration: {iteration_time:.1f}s | Total: {total_time:.1f}s")
            print(f"Cold start: {stats['cold_start_time_sec']:.1f}s | Wait time: {stats['total_wait_time_sec']:.1f}s")
            print(
                f"Downloaded: {stats['total_bytes_downloaded_mb']:.0f} MB ({stats['download_count']} chunks, {stats['cache_hits']} cache hits)"
            )
            print(
                f"Throughput: {stats['throughput_mbps']:.1f} MB/s effective, {stats['per_thread_throughput_mbps']:.1f} MB/s per-thread"
            )


def create_remote_chunked_scdl_factory(
    batch_size: int = 64,
    remote_path: str | None = None,
    cache_dir: str | None = None,
    max_cached_chunks: int = 3,
    storage_options: dict | None = None,
    num_workers: int = 0,
    shuffle_chunks: bool = True,
    shuffle_within_window: bool = True,
    chunks_per_window: int = 2,
    batch_download_size: int = 30,
) -> RemoteDataloaderFactory:
    """Create a remote chunked SCDL dataloader factory with ChunkAwareSampler."""
    return RemoteDataloaderFactory(
        batch_size=batch_size,
        remote_path=remote_path,
        cache_dir=cache_dir,
        max_cached_chunks=max_cached_chunks,
        storage_options=storage_options,
        num_workers=num_workers,
        shuffle_chunks=shuffle_chunks,
        shuffle_within_window=shuffle_within_window,
        chunks_per_window=chunks_per_window,
        batch_download_size=batch_download_size,
    )


def chunked_scdl_benchmarking(
    num_epochs: int = 1,
    num_runs: int = 1,
    scdl_path: str | None = None,
    chunked_path: str | None = None,
    remote_path: str | None = None,
    endpoint_url: str | None = None,
    cache_dir: str = "/tmp/scdl_cache",
    max_cached_chunks: int = 3,
    chunks_per_window: int = 2,
    max_time_seconds: float = 120.0,
    warmup_time_seconds: float = 30.0,
    batch_size: int = 64,
):
    """Run benchmarks comparing regular SCDL vs chunked SCDL.

    Args:
        num_epochs: Number of epochs per configuration
        num_runs: Number of runs per configuration
        scdl_path: Path to regular (non-chunked) SCDL dataset
        chunked_path: Path to pre-converted chunked SCDL dataset (optional)
        remote_path: Remote path to chunked dataset (s3://, gs://, etc.)
        endpoint_url: Custom S3 endpoint URL (for non-AWS S3)
        cache_dir: Local cache directory for remote chunks
        max_cached_chunks: Max chunks to keep in LRU cache
        chunks_per_window: Chunks per sampling window
        max_time_seconds: Max time per configuration
        warmup_time_seconds: Warmup time per configuration
        batch_size: Batch size for dataloaders
    """
    print("=" * 80)
    print("CHUNKED SCDL BENCHMARKING")
    print("=" * 80)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configurations = []

    # Configuration 1: Regular SCDL baseline
    if scdl_path:
        print(f"Adding Regular SCDL baseline: {scdl_path}")
        configurations.append(
            {
                "name": "Regular_SCDL_Baseline",
                "dataloader_factory": create_regular_scdl_factory(
                    batch_size=batch_size, shuffle=True, data_path=scdl_path, num_workers=0
                ),
                "num_epochs": num_epochs,
                "max_time_seconds": max_time_seconds,
                "warmup_time_seconds": warmup_time_seconds,
                "data_path": scdl_path,
                "num_runs": num_runs,
            }
        )

    # Configuration 2: Pre-converted chunked SCDL with ChunkAwareSampler
    if chunked_path:
        print(f"Adding Chunked SCDL + ChunkAwareSampler: {chunked_path}")
        configurations.append(
            {
                "name": f"Chunked_SCDL_ChunkAware_window{chunks_per_window}",
                "dataloader_factory": create_chunked_scdl_preconverted_factory(
                    batch_size=batch_size,
                    chunked_path=chunked_path,
                    num_workers=0,
                    shuffle_chunks=True,
                    shuffle_within_window=True,
                    chunks_per_window=chunks_per_window,
                ),
                "num_epochs": num_epochs,
                "max_time_seconds": max_time_seconds,
                "warmup_time_seconds": warmup_time_seconds,
                "data_path": chunked_path,
                "num_runs": num_runs,
            }
        )

        # Configuration 3: Chunked SCDL with random shuffle (no ChunkAwareSampler)
        print(f"Adding Chunked SCDL + Random Shuffle: {chunked_path}")
        configurations.append(
            {
                "name": "Chunked_SCDL_RandomShuffle",
                "dataloader_factory": create_chunked_scdl_random_factory(
                    batch_size=batch_size,
                    chunked_path=chunked_path,
                    num_workers=0,
                ),
                "num_epochs": num_epochs,
                "max_time_seconds": max_time_seconds,
                "warmup_time_seconds": warmup_time_seconds,
                "data_path": chunked_path,
                "num_runs": num_runs,
            }
        )

    # Configuration 4: Remote chunked SCDL
    remote_factory = None
    if remote_path:
        storage_options = {
            "default_fill_cache": False,  # Don't cache file contents in memory
            "default_cache_type": "none",  # No block caching
            "config_kwargs": {"max_pool_connections": 100},  # More parallel connections
        }
        if endpoint_url:
            storage_options["client_kwargs"] = {"endpoint_url": endpoint_url}

        print(f"Adding Remote Chunked SCDL: {remote_path}")
        if endpoint_url:
            print(f"  Endpoint: {endpoint_url}")
        print(f"  Cache dir: {cache_dir}")
        print(f"  Max cached chunks: {max_cached_chunks}")
        remote_factory = create_remote_chunked_scdl_factory(
            batch_size=batch_size,
            remote_path=remote_path,
            cache_dir=cache_dir,
            max_cached_chunks=max_cached_chunks,
            storage_options=storage_options,
            num_workers=0,
            shuffle_chunks=True,
            shuffle_within_window=True,
            chunks_per_window=chunks_per_window,
            batch_download_size=max_cached_chunks,  # Download batch = cache size
        )

        configurations.append(
            {
                "name": f"Chunked_SCDL_Remote_cache{max_cached_chunks}_window{chunks_per_window}",
                "dataloader_factory": remote_factory,
                "num_epochs": num_epochs,
                "max_time_seconds": max_time_seconds,
                "warmup_time_seconds": warmup_time_seconds,
                "data_path": remote_path,
                "num_runs": num_runs,
            }
        )

    if not configurations:
        print("ERROR: No configurations to benchmark. Provide --scdl-path, --chunked-path, or --remote-path")
        return

    print()
    print(f"Running {len(configurations)} configuration(s)...")
    print()

    results = benchmark_dataloaders_with_configs(
        dataloader_configs=configurations,
        shared_dataset_factory=None,  # Each config loads its own dataset
        output_prefix=f"chunked_scdl_benchmark_{timestamp}",
    )

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print_comparison(results)

    # Print remote download statistics if available
    if remote_factory:
        # Record wall_clock_time from benchmark results (since sampler may not complete naturally)
        for result in results:
            if "Remote" in result.name and remote_factory.last_dataset and remote_factory.last_dataset._chunk_loader:
                remote_factory.last_dataset._chunk_loader.stats.record_wall_clock(result.total_time_seconds)
                break
        remote_factory.print_download_stats()

    print()
    print(f"Results saved to: chunked_scdl_benchmark_{timestamp}_detailed_breakdown.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Regular SCDL vs Chunked SCDL with ChunkAwareSampler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark local regular vs local chunked
  %(prog)s --scdl-path /data/scdl/ --chunked-path /data/chunked_scdl/

  # Benchmark remote chunked dataset
  %(prog)s --remote-path s3://bucket/chunked --endpoint-url https://s3.example.com

  # Full comparison
  %(prog)s --scdl-path /data/scdl/ --chunked-path /data/chunked/ --remote-path s3://bucket/chunked
        """,
    )

    # Data paths - with sensible defaults
    parser.add_argument(
        "--scdl-path",
        type=str,
        default="",  # "/home/pbinder/bionemo-framework/small_tahoe_format",
        help="Path to regular SCDL dataset (baseline)",
    )
    parser.add_argument(
        "--chunked-path",
        type=str,
        default="",  # "/home/pbinder/bionemo-framework/sub-packages/bionemo-scspeedtest/example_data/tahoe_chunked",
        help="Path to pre-converted chunked SCDL dataset",
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        default="s3://general-purpose/polina/tahoe_chunked",
        help="Remote path to chunked dataset (s3://, gs://)",
    )
    parser.add_argument("--endpoint-url", type=str, default="https://pbss.s8k.io", help="Custom S3 endpoint URL")

    # Cache settings
    parser.add_argument("--cache-dir", type=str, default="/tmp/scdl_cache", help="Local cache for remote chunks")
    parser.add_argument(
        "--max-cached-chunks",
        type=int,
        default=2,
        help="Max chunks in LRU cache (>= 2x chunks-per-window for prefetching)",
    )

    # Chunking settings
    parser.add_argument("--chunks-per-window", type=int, default=1, help="Chunks per sampling window")

    # Benchmark settings
    parser.add_argument("--num-epochs", type=int, default=1, help="Epochs per configuration")
    parser.add_argument("--num-runs", type=int, default=1, help="Runs per configuration")
    parser.add_argument("--max-time", type=float, default=120, help="Max time per config (seconds)")
    parser.add_argument("--warmup-time", type=float, default=0, help="Warmup time per config (seconds)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    # Validate paths exist (for local paths)
    if args.scdl_path and not os.path.exists(args.scdl_path):
        print(f"Warning: SCDL path not found: {args.scdl_path}, skipping...")
        args.scdl_path = None
    if args.chunked_path and not os.path.exists(args.chunked_path):
        print(f"Warning: Chunked path not found: {args.chunked_path}, skipping...")
        args.chunked_path = None

    # Check at least one valid config
    if not any([args.scdl_path, args.chunked_path, args.remote_path]):
        parser.error("No valid data paths found. Provide --scdl-path, --chunked-path, or --remote-path")

    chunked_scdl_benchmarking(
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        scdl_path=args.scdl_path,
        chunked_path=args.chunked_path,
        remote_path=args.remote_path,
        endpoint_url=args.endpoint_url,
        cache_dir=args.cache_dir,
        max_cached_chunks=args.max_cached_chunks,
        chunks_per_window=args.chunks_per_window,
        max_time_seconds=args.max_time,
        warmup_time_seconds=args.warmup_time,
        batch_size=args.batch_size,
    )
