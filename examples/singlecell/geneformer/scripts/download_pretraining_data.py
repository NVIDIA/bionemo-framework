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
import math
import os
import re
from functools import partial
from pathlib import Path, PosixPath
from typing import List, Tuple, Union

import cellxgene_census
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


CENSUS_VERSION = "2023-12-15"  # This is the LTS version of the cellxgene-census data as of 2024-05-02
parser = argparse.ArgumentParser("Download CellxGene Dataset")
parser.add_argument(
    "--num-workers", "--nw", type=int, default=12, help="number of workers to use for multi-processing"
)
parser.add_argument(
    "--cells-per-query", "--cpq", type=int, default=400_000, help="Maximum number of cells to query at a time"
)
parser.add_argument(
    "--cells-per-file", "--cpf", type=int, default=5_000, help="Maximum number of cells to put in each file"
)
parser.add_argument(
    "--save-path",
    "--sp",
    type=str,
    default=f"data/cellxgene_{CENSUS_VERSION}/input_data",
    help="path to save the downloaded data",
)
parser.add_argument(
    "--census-version",
    "--cv",
    type=str,
    default=CENSUS_VERSION,
    help="version of the cellxgene-census data to download",
)


def sanitize_string(s):
    # Replace spaces with underscores
    s = s.replace(" ", "_")

    # Remove special characters except for hyphens and underscores
    s = re.sub(r"[^\w\-]", "", s)

    # Remove single quotes
    s = s.replace("'", "")

    return s


def sanitize_list(lst):
    return [sanitize_string(item) for item in lst]


def download_anndata(
    folder_to_soma_indices: Tuple[List[str], List],
    save_path: Union[str, PosixPath],
    census_version=CENSUS_VERSION,
    cells_per_file=5_000,
):
    """Filters `tissue` from the cellxgene census and downloads into `save_path`"""

    folder_names, idxs = folder_to_soma_indices
    if len(idxs) == 0:
        return 0
    folder_name = os.path.join(*folder_names)
    folder_path = save_path / f"{folder_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    file_name0 = f"sidx_{idxs[0]}_{len(idxs)}_0"
    file_path0 = folder_path / f"{file_name0}.h5ad"
    if os.path.exists(file_path0):
        # Unsafe but check if only the first exists for now, since we don't yet know how many
        #  there will be, and the query is the slow part.
        return 0
    else:
        try:
            with cellxgene_census.open_soma(census_version=census_version) as census:
                adata = cellxgene_census.get_anndata(
                    census,
                    organism="Homo sapiens",
                    obs_coords=idxs,
                )
            n_cells = adata.shape[0]
            for i in range(int(math.ceil(n_cells / cells_per_file))):
                file_name = f"sidx_{idxs[0]}_{len(idxs)}_{i}"
                file_path = folder_path / f"{file_name}.h5ad"
                start = i * cells_per_file
                stop = min((i + 1) * cells_per_file, n_cells)
                adata[start:stop].copy().write_h5ad(file_path)
            return 0
        except Exception as e:
            print(f"Could not download: {folder_name} - {file_name}. Caught exception {e}")
            return -1


if __name__ == "__main__":
    args = parser.parse_args()
    save_path = Path(args.save_path)
    partition_size = args.cells_per_query

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fname_to_somaidx: List[Tuple[Tuple[str, str], List[int]]] = []

    # - for each tissue extract the corresponding soma partitions
    # https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/5.0.0/schema.md#is_primary_data
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        query = census["census_data"]["homo_sapiens"].obs.read(
            value_filter='suspension_type != "na" and is_primary_data == True and disease == "normal"',
            column_names=[
                "soma_joinid",
                "assay",
                "sex",
                "development_stage",
                "n_measured_vars",
                "tissue_general",
                "self_reported_ethnicity",
                "dataset_id",
            ],
        )
    df: pd.DataFrame = query.concat().to_pandas()

    print(f"Found {len(df)} cells matching your query in soma.")

    grp_keys = ["assay", "sex", "development_stage", "self_reported_ethnicity", "tissue_general", "dataset_id"]
    add_metadata = ["n_measured_vars"]

    dataset_metadata = []
    for grp, gdf in df.groupby(grp_keys):
        other_info = gdf.iloc[0][add_metadata]
        dset_meta = dict(zip(list(grp_keys) + list(add_metadata), list(grp) + list(other_info)))
        n_cells = len(gdf)
        dset_meta["num_cells"] = n_cells
        dataset_metadata.append(dset_meta)
        # grp is a tuple of tissue_general, dataset_id
        fname_parts = tuple(f"{d}__{sanitize_string(s)}" for d, s in zip(grp_keys, grp))
        num_parts = int(math.ceil(len(gdf) / partition_size))
        idxs: List[List[int]] = list(
            map(list, np.array_split(gdf["soma_joinid"].values, num_parts))
        )  # Will split into num_parts files.

        fname_idxs = list(zip([fname_parts] * len(idxs), idxs))
        fname_to_somaidx.extend(fname_idxs)

    metadata_df = pd.DataFrame(dataset_metadata)
    metadata_df.to_csv(save_path / "dataset_metadata.csv", index=False)

    download_fn = partial(
        download_anndata, save_path=save_path, census_version=args.census_version, cells_per_file=args.cells_per_file
    )

    result = Parallel(n_jobs=args.num_workers)(
        delayed(download_fn)(tsid) for tsid in tqdm(fname_to_somaidx, desc="Downloading Soma Partitions ...")
    )
    print(
        f"Attempted to download {len(result)} files, succeeded in downloading {sum([1 for r in result if r == 0])} files."
    )
