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


# NBVAL_CHECK_OUTPUT
import argparse
import random
from contextlib import contextmanager

import cellxgene_census


@contextmanager
def random_seed(seed: int):
    """Context manager to set the random seed for reproducibility."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(description="Download and prepare cell type benchmark dataset")

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/ubuntu/data/20250501-bench/notebook_tutorials/geneformer_celltype_classification/celltype-bench-dataset-input",
        help="Base directory for data downloads",
    )

    parser.add_argument("--census-version", type=str, default="2023-12-15", help="Cellxgene census version to use")

    parser.add_argument(
        "--dataset-id",
        type=str,
        default="8e47ed12-c658-4252-b126-381df8d52a3d",
        help="Dataset ID to download from census. Don't change this unless you have a good reason.",
    )

    parser.add_argument("--micro-batch-size", type=int, default=16, help="Micro batch size for processing")

    parser.add_argument("--num-steps", type=int, default=512, help="Number of steps to process")

    parser.add_argument("--random-seed", type=int, default=32, help="Random seed for reproducibility")

    parser.add_argument("--cleanup", action="store_true", help="Clean up existing output directory if it exists")

    return parser.parse_args()


def main():  # noqa: D103
    args = parse_args()

    # Setup paths
    h5ad_outfile = args.base_dir / "hs-celltype-bench.h5ad"

    # Download data from census
    print(f"Downloading data from census version {args.census_version}")
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            "Homo sapiens",
            obs_value_filter=f'dataset_id=="{args.dataset_id}"',
        )

    # Print unique cell types
    uq_cells = sorted(adata.obs["cell_type"].unique().tolist())
    print(f"Found {len(uq_cells)} unique cell types")

    # Handle subsampling
    selection = list(range(len(adata)))

    print(f"Selected {len(selection)} cells")

    # Subset and save data
    adata = adata[selection].copy()
    adata.write_h5ad(h5ad_outfile)
    print(f"Saved data to {h5ad_outfile}")


if __name__ == "__main__":
    main()
