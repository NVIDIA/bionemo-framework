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


#!/usr/bin/env python3

import argparse
from datetime import datetime
import os
from pathlib import Path

import anndata as ad
import scipy.sparse as sp
import torch
import zarr
import psutil
from annbatch import ZarrSparseDataset, create_anndata_collection
from torch.utils.data import DataLoader

from bionemo.scspeedtest import benchmark_dataloaders_with_configs


# TODO: Should num_workers control threading? If scdataset were to wrap zarr, it would still be multithreaded under the processes, so my inclination is "no".
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

def create_dataset_factory(adata_path: Path | str, num_workers: int) -> ad.AnnData:
    """Generate an `anndata.AnnData` object for use in `annbatch` i.e., zarr v3 on disk, loaded using `anndata.io.sparse_dataset`."""
    def dataset_factory():
        # TODO: Where to dump this data?
        collection_path = Path(adata_path)
        output_zarr_collection = collection_path.parent / "collection"
        def load_adata(adata_path):
            adata = ad.experimental.read_lazy(adata_path)
            # TODO: File anndata issue
            import dask.array as da
            adata.uns = { k: v.compute() if isinstance(v, da.Array) else v for k, v in adata.uns.items() }
            return adata
        create_anndata_collection([adata_path], output_zarr_collection, zarr_sparse_chunk_size=32768//2, load_adata=load_adata, zarr_compressor=None)
        # Allocate each worker an even number of threads plus a little
        # TODO: How big is the data? Presumably if it is not big enough to fit in memory, we would want O_DIRECT reading to be on.
        # TODO: There are probably faster ways to get a directory size? If you're using this data loader, presumably your data doesn't fit in memory?
        # TODO: Does X always contain the genes of interest? Layers? Raw?
        use_direct_io = psutil.virtual_memory().total < sum(f.stat().st_size for f in output_zarr_collection.glob("**/*") if f.is_file())
        with zarr.config.set(
            {
                "threading.max_workers": (os.cpu_count() // max(num_workers, 1)) + 1,
                "codec_pipeline.direct_io": use_direct_io,
            }
        ):
            return [ad.AnnData(X=ad.io.sparse_dataset(zarr.open(p)["X"])) for p in output_zarr_collection.iterdir()]
    return dataset_factory

def to_annbatch(adatas: list[ad.AnnData], *, batch_size: int = 64, shuffle: bool = True, block_size: int = 1, fetch_factor: int = 2, num_workers: int = 0) -> ZarrSparseDataset:
    """Generate an `annbatch.ZarrSparseDataset` based on configuration and a list of input `anndata.AnnData` objects backed by zarr v3 on disk."""
    if num_workers > 0:
        ds = ZarrSparseDataset(
            batch_size=batch_size,
            chunk_size=block_size,
            preload_nchunks=((fetch_factor * batch_size) // block_size),
            preload_to_gpu=False,
            shuffle=shuffle,
            to_torch=True
        ).add_anndatas(adatas)
        loader = DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,
            multiprocessing_context="spawn",
        )
        return (v[0] for v in iter(loader))
    ds = ZarrSparseDataset(
        batch_size=batch_size,
        chunk_size=block_size,
        preload_nchunks=((fetch_factor * batch_size) // block_size),
        preload_to_gpu=torch.cuda.is_available()
    ).add_anndatas(adatas)
    # Internally `get_batch_size` in the benchmarking will just take `len(v)` for benchmarking, but we return a tuple (for now) whose first element is the matrix
    return (v[0] for v in iter(ds))

def create_annbatch_from_preloaded_anndata_factory(batch_size=64, shuffle=True, block_size=1, fetch_factor=2, num_workers=0):
    """Factory creator for an `annbatch.ZarrSparseDataset` given a list of anndatas (backed by zarr v3 on-disk) and based on configuration in the arguments to this function."""
    def factory(adatas: list[ad.AnnData]):
        return to_annbatch(adatas, batch_size=batch_size ,shuffle=shuffle ,block_size=block_size ,fetch_factor=fetch_factor ,num_workers=num_workers)

    return factory

def create_annbatch_factory(
    batch_size=64, block_size=1, shuffle=True, adata_path=None, num_workers=0, fetch_factor=1
):
    """Factory creator for on-disk zarr v3 sharded anndata file __and__ a `annbatch.ZarrSparseDataset` based on configuration in the arguments to this function as well as that on-disk."""

    def factory():
        adatas = create_dataset_factory(adata_path, num_workers)()
        return to_annbatch(
            adatas,
            batch_size=batch_size,
            shuffle=shuffle,
            block_size=block_size,
            fetch_factor=fetch_factor,
            num_workers=num_workers,
        )

    return factory


def comprehensive_benchmarking_example(
    num_epochs=1,
    num_runs=1,
    adata_path=None,
    fetch_factors=None,
    block_sizes=None,
    max_time_seconds=120.0,
    warmup_time_seconds=30.0,
):
    """Benchmarking exampe for `annbatch.ZarrSparseDataset`.

    Args:
        num_epochs: Number of epochs to run for each configuration
        num_runs: Number of runs to perform for each configuration
        adata_path: Path to the AnnData file (.h5ad)
        fetch_factors: List of fetch factors to test (default: [1])
        block_sizes: List of block sizes to test (default: [1, 2, 4, 8, 16, 32, 64])
        max_time_seconds: Maximum time to run each configuration (default: 120.0)
        warmup_time_seconds: Time to warmup before benchmarking (default: 30.0)
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARKING EXAMPLE")
    print("=" * 80)
    print()

    print(f"Using AnnData path: {adata_path}")
    print(f"Fetch factors: {fetch_factors}")
    print(f"Block sizes: {block_sizes}")
    print()

    # Create timestamped prefix for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"All results will be saved to: scdataset_benchmark_{timestamp}_detailed_breakdown.csv")
    print()


    print(f"Benchmarking {num_runs} run(s) each")
    print()
    print("Running annbatch...")
    num_workers = 0  # TODO: why 0? the other scripts seem to have this hardcoded though
    annbatch_configurations = []
    for fetch_factor in fetch_factors:
        for block_size in block_sizes:
            annbatch_configurations.append(
                {
                    "name": f"annbatch_{block_size}_{fetch_factor}",
                    "dataloader_factory": create_annbatch_from_preloaded_anndata_factory(
                        batch_size=64,
                        shuffle=True,
                        num_workers=num_workers,
                        block_size=block_size,
                        fetch_factor=fetch_factor,
                    ),
                    "num_epochs": num_epochs,
                    "max_time_seconds": max_time_seconds,
                    "warmup_time_seconds": warmup_time_seconds,
                    "data_path": adata_path,
                    "num_runs": 1,
                }
            )

    benchmark_dataloaders_with_configs(
        dataloader_configs=annbatch_configurations,
        shared_dataset_factory=create_dataset_factory(adata_path, num_workers),
        output_prefix=f"annbatch_benchmark_{timestamp}",
    )

    print("Benchmarking completed!")
    print(f"All results saved to: annbatch_benchmark_{timestamp}_detailed_breakdown.csv")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioNeMo Benchmarking Framework - annbatch Test")
    parser.add_argument(
        "--adata-path",
        type=str,
        default="/home/pbinder/bionemo-framework/tahoe_data",
        help="Path to the AnnData file (.h5ad). Default: %(default)s",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of epochs to run for each configuration. Default: %(default)s",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs to perform for each configuration. Default: %(default)s",
    )
    parser.add_argument(
        "--fetch-factors",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="List of fetch factors to test. Default: %(default)s",
    )
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="List of block sizes to test. Default: %(default)s",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=120.0,
        help="Maximum time to run each configuration in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=1.0,
        help="Time to warmup before benchmarking in seconds. Default: %(default)s",
    )

    args = parser.parse_args()

    print("BioNeMo Benchmarking Framework - annbatch Test")
    print("=" * 80)
    comprehensive_benchmarking_example(
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        adata_path=args.adata_path,
        fetch_factors=args.fetch_factors,
        block_sizes=args.block_sizes,
        max_time_seconds=args.max_time,
        warmup_time_seconds=args.warmup_time,
    )
