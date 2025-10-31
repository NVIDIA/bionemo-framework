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

import argparse
import tempfile

from bionemo.scdl.io.single_cell_collection import SingleCellCollection


def main():
    """Parse the arguments to process the single cell collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers", type=int, default=4, help="The number of AnnData loaders to run in parallel [4]."
    )
    parser.add_argument(
        "--use-mp",
        action="store_true",
        default=False,
        help="Use a subprocess for each worker rather than a lightweight OS thread [False].",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="A path containing AnnData files. Note: These will all be concatenated.",
    )
    parser.add_argument(
        "--save-path", required=True, type=str, help="An output path where an SCDataset will be stored."
    )

    parser.add_argument(
        "--data-dtype",
        type=str,
        default=None,
        help="The data type to use for the SCDataset. Must be one of 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64'.",
    )
    parser.add_argument(
        "--paginated-load-cutoff",
        type=int,
        default=10_000,
        help="The cutoff in MB for paginated loading of the AnnData files.",
    )
    parser.add_argument(
        "--load-block-row-size",
        type=int,
        default=1_000_000,
        help="The number of rows to load into memory at a time for paginated loading of the AnnData files.",
    )
    parser.add_argument(
        "--use-X-not-raw",
        action="store_true",
        default=False,
        help="Use .X instead of raw.X from the anndata file [False].",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        coll = SingleCellCollection(temp_dir)
        coll.load_h5ad_multi(
            args.data_path,
            max_workers=args.num_workers,
            use_processes=args.use_mp,
            data_dtype=args.data_dtype,
            paginated_load_cutoff=args.paginated_load_cutoff,
            load_block_row_size=args.load_block_row_size,
            use_X_not_raw=args.use_X_not_raw,
        )
        coll.flatten(args.save_path, destroy_on_copy=True)


if __name__ == "__main__":
    main()
