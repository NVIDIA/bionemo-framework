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
import json
import os
from functools import partial
from multiprocessing import Lock, Manager, Pool
from pathlib import Path, PosixPath
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy
from nemo.utils import logging
from scanpy import AnnData
from scipy.sparse import csr_matrix
from tqdm import tqdm


parser = argparse.ArgumentParser("Converts a series of AnnData objects into a memmap format")
parser.add_argument("--save-path", "--sp", type=str, default="./", help="save path to save memmap files")
parser.add_argument("--data-path", "--dp", type=str, default="./data", help="path to the data")
parser.add_argument(
    "--override_metadata",
    "--om",
    default=False,
    action="store_true",
    help="always overwrite metadata to prevent caching outdated files",
)
parser.add_argument("--use-mp", "-mp", action="store_true", help="use multiprocessing")
parser.add_argument(
    "--next-cell", "--ncp", default=False, action="store_true", help="enable next cell prediction tasks"
)
parser.add_argument(
    "--strict-metadata",
    "-strict",
    dest="strict",
    action="store_true",
    help="Fails if any of the columns in obs_cols are not present in the AnnData object.",
)
parser.add_argument(
    "--num-workers", "--nw", type=int, default=12, help="number of workers to use for multi-processing"
)
parser.add_argument(
    "--obs-cols",
    nargs="+",
    default=[
        "suspension_type",
        "is_primary_data",
        "cell_type",
        "assay",
        "disease",
        "tissue_general",
        "sex",
        "tissue",
        "self_reported_ethnicity",
        "development_stage",
    ],
    help="series of columns to extract from each AnnData `obs` dataframe",
)
# - XXX: obs-cols argument can be turned into a txt file input if the list is long

GLOBAL_LOCK = Lock()


def create_metadata(file_path: PosixPath, shared_dict: Dict[str, Dict[str, object]]) -> None:
    """Extract a series of metadata values from `AnnData` required to process all files into memmaps.

    Note: it assumes var.feature_id contains the gene symbols for each dataset and corresponds to the same order as the data.X columns.

    Args:
        file_path (PosixPath):
            Path to `AnnData` stored as *.h5ad.
        shared_dict (Dict[str, Dict[str, object]]):
            Dictionary to store the extracted metadata.

    Returns:
        None:
            If the file cannot be read or if the `data` object is None.

    """

    try:
        data = scanpy.read_h5ad(file_path)
    except Exception:
        print(f"Could not read {file_path}")
        return

    if data is None:
        return

    shape = data.shape
    feature_ids = list(data.var.feature_id)

    if data.raw is not None:
        X = data.raw.X
    else:
        X = data.X

    num_el = X.count_nonzero()  # Count the number of non-zero elements in the sparse array, in total
    # - metadata associated with each file
    d = {
        "shape": shape,
        "feature_ids": feature_ids,
        "num_el": num_el,
        "file_path": str(file_path),
        "next_cell_prediction": has_pt_neighbors(data),
    }

    # update metadata with total number of next-cell-to-predict indices
    if has_pt_neighbors(data):
        PT_neighbors = data.obsp["next_cell_ids"]
        num_neighbors = PT_neighbors.count_nonzero()
        d["num_neighbors_el"] = num_neighbors
    else:
        d["num_neighbors_el"] = 0

    shared_dict[str(file_path)] = d


def calculate_running_sums(metadata):
    num_el = 0
    num_neighbors_el = 0
    cur_count = 0
    for k in metadata:
        metadata[k]["running_el"] = num_el
        metadata[k]["pt_running_el"] = num_neighbors_el
        metadata[k]["cur_count"] = cur_count
        num_el += metadata[k]["num_el"]
        num_neighbors_el += metadata[k]["num_neighbors_el"]
        cur_count += metadata[k]["shape"][0]
    return metadata


def has_pt_neighbors(
    anndata: AnnData,
) -> bool:
    """Wrapper for checking if the provided `AnnData` has neighbors for next
    cell prediction tasks. By default it is stored in `next_cell_ids` attribute.

    Args:
        anndata (AnnData): Input AnnData

    Returns:
        bool:
    """
    return True if "next_cell_ids" in anndata.obsp.keys() else False


def write_data(
    file_path: PosixPath,
    obs_cols: list,
    metadata: Dict[str, Dict[str, object]],
    gene_data: np.ndarray,
    gene_data_indices: np.ndarray,
    gene_data_ptr: np.ndarray,
    strict: bool = False,
    next_cell_indices: Optional[np.ndarray] = None,
    next_cell_ptr: Optional[np.ndarray] = None,
) -> List[pd.DataFrame]:
    """
    Writes `AnnData` into memmap.

    Args:
        file_path (PosixPath): The path to the file.
        obs_cols (List[str]): A list of columns to extract from each AnnData `obs` dataframe.
        metadata (Dict[str, Dict[str, object]]): A dictionary containing metadata information
            on number of elements, shape, and feature names.
        gene_data (np.ndarray): The array to store gene data.
        gene_data_indices (np.ndarray): The array to store gene data indices.
        gene_data_ptr (np.ndarray): The array to store gene data pointers.
        next_cell_indices (np.ndarray, optional): The array to store neighbor cell data indices.
        next_cell_ptr (np.ndarray, optional): The array to store neighbor cell data pointers.
        strict (bool): If True, only extract the columns specified in `obs_cols`.
    Returns:
        List[pd.DataFrame]: The features extracted from the data.
    """

    # - check if the file name exists in the metadata dictionary
    if str(file_path) not in metadata:
        return []

    # Get the metadata for the file
    meta = metadata[str(file_path)]
    num_el = meta["num_el"]
    num_neighbors_el = meta["num_neighbors_el"]
    running_el = meta["running_el"]
    pt_running_el = meta["pt_running_el"]
    num_obs = meta["shape"][0]
    cur_count = meta["cur_count"]

    try:
        # - read the data from the file using scanpy
        data = scanpy.read_h5ad(file_path)
    except Exception:
        print(f"couldn't read {file_path}")
        return []

    # - get the gene data from the data object
    X = data.X if data.raw is None else data.raw.X  # Use X if raw is not None, otherwise use raw
    assert isinstance(X, csr_matrix), "Anndata .X data must be in Compressed Sparse Row format"
    # - store the gene data, indices, and pointers in the respective arrays
    gene_data[running_el : running_el + num_el] = X.data  # This is a flattened array with everything in it.
    gene_data_indices[running_el : running_el + num_el] = X.indices.astype(
        int
    )  # these are flattened column indices eg [0, 1, 2, 0, 1, 3] for a 2x4 sparse matrix
    gene_data_ptr[cur_count : cur_count + num_obs + 1] = X.indptr.astype(int) + int(
        running_el
    )  # These are mappings between row indices and ranges. eg [0, 3, 6] for a 2x4 sparse matrix

    logging.info(f"pt_running_el: {pt_running_el}, cur_count: {cur_count}")

    # if args.ncp store the pt neighbors indices, and pointers in the respective arrays
    if args.next_cell:
        if meta["next_cell_prediction"]:
            # - get pseudotime neighbors for each sample in CSR format
            assert (
                "next_cell_ids" in data.obsp.keys()
            ), "Missing .obsp['next_cell_ids'] sample neighbor information required for the next cell prediction"
            PT_neighbors = data.obsp.get("next_cell_ids")
            next_cell_indices[pt_running_el : pt_running_el + num_neighbors_el] = PT_neighbors.indices.astype(
                int
            ) + int(cur_count)  # these are flattened neighbor cell indices + anndata offset
            print(PT_neighbors.indptr.astype(int) + int(pt_running_el))
            next_cell_ptr[cur_count : cur_count + num_obs + 1] = PT_neighbors.indptr.astype(int) + int(pt_running_el)
        else:
            next_cell_ptr[cur_count : cur_count + num_obs + 1] = int(
                pt_running_el
            )  # no pt neighbors, so the pointer stays the same
            print(next_cell_ptr[cur_count : cur_count + num_obs + 1])
    # - extract the features from the data
    # TODO: this doesnt work if obs_column doesnt have the right things in it.
    if not strict:
        new_obs_cols = list(set(data.obs.columns.tolist()) & set(obs_cols))
        features = data.obs[new_obs_cols]
    else:
        features = data.obs[obs_cols]

    # - flush the data arrays to disk
    GLOBAL_LOCK.acquire()
    gene_data.flush()
    gene_data_ptr.flush()
    gene_data_indices.flush()
    GLOBAL_LOCK.release()

    return features


def find_ann_data_files(data_path: Path) -> List[Path]:
    """Find all AnnData files with the extension '.h5ad' in the given data path and its subdirectories.

    Args:
        data_path (str): The path to the directory containing the AnnData files.

    Returns:
        List[str]: A list of file paths to the AnnData files.
    """
    return sorted(data_path.rglob("*.h5ad"))


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = Path(args.data_path)
    save_path = Path(args.save_path)
    strict = args.strict

    if not save_path.exists():
        os.makedirs(save_path)

    file_paths = find_ann_data_files(data_path)
    if len(file_paths) == 0:
        print(f"No files ending in .h5ad found in {data_path}, check your argument for data_path.")
        exit()

    print(f"Found {len(file_paths)} files")
    print("Starting to create memmap files...")
    # - create metadata required to store data into memmap
    num_proc = args.num_workers
    manager = Manager()
    shared_dict = manager.dict()
    metadata_path = save_path / "metadata.json"
    if metadata_path.exists() and not args.override_metadata:
        logging.warning(
            "Data path found, not writing over it. If you want to override the data, re-run with --override-metadata"
        )
        print("Metadata already exists, loading...")
        with open(metadata_path, "r") as fp:
            metadata = json.load(fp)
    else:
        if args.use_mp:
            with Pool(num_proc) as pool:
                _ = list(
                    tqdm(
                        pool.imap(partial(create_metadata, shared_dict=shared_dict), file_paths),
                        desc="Creating metadata...",
                        total=len(file_paths),
                    )
                )
        else:
            for file_path in tqdm(file_paths, desc="Creating metadata..."):
                create_metadata(file_path, shared_dict)

        metadata = dict(shared_dict)
        for k, v in metadata.items():
            assert v["shape"][1] == len(v["feature_ids"]), f"feature names and shape mismatch for file {k}"

        with open(metadata_path, "w") as fp:
            json.dump(metadata, fp)

        print("Done creating `metadata.json`")

    print(f"Writing data into memmaps to {save_path}...")

    # - calculate totals to initalize array sizes
    total_el = sum([v["num_el"] for k, v in metadata.items()])
    num_samples = sum([v["shape"][0] for k, v in metadata.items()])
    gene_path = save_path

    # - init or append mode memmap files
    gene_data = np.memmap(
        gene_path / "gene_expression_data.npy",
        dtype="float32",
        mode="w+" if not os.path.exists(gene_path / "gene_expression_data.npy") else "r+",
        shape=(total_el,),
    )

    gene_data_indices = np.memmap(
        gene_path / "gene_expression_ind.npy",
        dtype="int32",
        mode="w+" if not os.path.exists(gene_path / "gene_expression_ind.npy") else "r+",
        shape=(total_el,),
    )

    gene_data_ptr = np.memmap(
        gene_path / "gene_expression_ptr.npy",
        dtype="int64",
        mode="w+" if not os.path.exists(gene_path / "gene_expression_ptr.npy") else "r+",
        shape=(num_samples + 1,),
    )

    if args.next_cell:
        total_pt_el = sum([v["num_neighbors_el"] for k, v in metadata.items()])
        assert (
            total_pt_el > 0
        ), "There are no samples with neighbors. When next_cell_prediction is enabled, input data must contain samples with neighbor information."
        next_cell_indices = np.memmap(
            gene_path / "pseudotime_neighbors_ind.npy",
            dtype="int32",
            mode="w+" if not os.path.exists(gene_path / "pseudotime_neighbors_ind.npy") else "r+",
            shape=(total_pt_el,),
        )

        next_cell_ptr = np.memmap(
            gene_path / "pseudotime_neighbors_ptr.npy",
            dtype="int64",
            mode="w+" if not os.path.exists(gene_path / "pseudotime_neighbors_ptr.npy") else "r+",
            shape=(num_samples + 1,),
        )
    else:
        next_cell_indices = None
        next_cell_ptr = None

    metadata = json.load(open(save_path / "metadata.json", "r"))
    # - start processing all files
    metadata = calculate_running_sums(metadata)
    files = list(metadata.keys())
    obs_cols = args.obs_cols

    features = []
    for fp in tqdm(file_paths, desc="Merging AnnData into numpy memaps..."):
        feature = write_data(
            fp,
            obs_cols=obs_cols,
            metadata=metadata,
            gene_data=gene_data,
            gene_data_indices=gene_data_indices,
            gene_data_ptr=gene_data_ptr,
            strict=strict,
            next_cell_indices=next_cell_indices,
            next_cell_ptr=next_cell_ptr,
        )
        features.append(feature)

    print("Saving dataframe ...")
    df = pd.concat(features)
    df.to_csv(save_path / "features.csv", index=False)
    print("Done creating dataset ...")
