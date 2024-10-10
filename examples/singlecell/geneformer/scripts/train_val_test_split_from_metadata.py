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
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main(metadata_file: Path, data_base_path: Path, test_val_fraction: float, seed: int = 42, move=False):
    """
    Split the dataset into training, validation, and test sets based on metadata.

    Args:
        metadata_file (Path): Path to the metadata file.
        test_val_fraction (float): Fraction of the dataset to be used for validation and testing.

    Returns:
        None
    """

    # Load the metadata file
    data_base_path = data_base_path.absolute()
    assert 0 < test_val_fraction < 1, "Test-Val fraction must be between 0 and 1"
    d_m = pd.read_csv(metadata_file)

    # Calculate dataset summary statistics
    md = []
    for ds, ddf in d_m.groupby("dataset_id"):
        n_cells = ddf.num_cells.sum()
        m_vars = ddf.n_measured_vars.mean()
        assays = tuple(sorted(ddf.assay.unique()))
        sexes = tuple(sorted(ddf.sex.unique()))
        ethnicities = tuple(sorted(ddf.self_reported_ethnicity.unique()))
        stages = tuple(sorted(ddf.development_stage.unique()))
        tissues = tuple(sorted(ddf.tissue_general.unique()))
        md.append(
            {
                "dataset_id": ds,
                "n_cells": n_cells,
                "mu_n_vars": m_vars,
                "sexes": sexes,
                "assays": assays,
                "ethnicities": ethnicities,
                "stages": stages,
                "tissues": tissues,
            }
        )
    dset_summary = pd.DataFrame(md).sort_values(by="n_cells").reset_index(drop=True)
    dset_summary["n_cells_cumulative"] = dset_summary.n_cells.cumsum()

    # Determine tissues and assays for validation and training sets
    n_test_val = int(math.ceil(test_val_fraction * dset_summary.n_cells.sum()))
    tissues_ho = {t for tissues in dset_summary[dset_summary.n_cells_cumulative < n_test_val].tissues for t in tissues}
    tissues_train = {
        t for tissues in dset_summary[dset_summary.n_cells_cumulative >= n_test_val].tissues for t in tissues
    }
    assays_ho = {t for assays in dset_summary[dset_summary.n_cells_cumulative < n_test_val].assays for t in assays}
    assays_train = {t for assays in dset_summary[dset_summary.n_cells_cumulative >= n_test_val].assays for t in assays}
    missing_tissues = tissues_ho - tissues_train
    missing_assays = assays_ho - assays_train

    # Get indices for validation and test sets
    val_test_indices = dset_summary[
        (dset_summary.n_cells_cumulative < n_test_val)
        & (~dset_summary.assays.isin(missing_assays))
        & (~dset_summary.tissues.isin(missing_tissues))
    ].index
    train_indices = dset_summary.index[~dset_summary.index.isin(val_test_indices)]
    print(
        f"After splitting we have {dset_summary.loc[val_test_indices].n_cells.sum()/dset_summary.n_cells.sum()*100:.2f}% of the data in the validation and test sets with all assays and tissues present in training."
    )
    all_files = list(data_base_path.rglob("*.h5ad"))
    train_datasets = set(dset_summary.loc[train_indices].dataset_id.unique())
    val_test_datasets = set(dset_summary.loc[val_test_indices].dataset_id.unique())
    assert train_datasets.isdisjoint(
        val_test_datasets
    ), f"Training and validation/test datasets should be disjoint. Got {train_datasets.intersection(val_test_datasets)}."
    # assert len(all_files) >= len(d_m), f"We should have at least one file per row in the metadata file. Saw {len(all_files)} files and {len(d_m)} rows in the metadata file."
    train_paths = []
    val_test_paths = []
    for file in all_files:
        if any(ds in str(file) for ds in train_datasets):
            train_paths.append(file)
        elif any(ds in str(file) for ds in val_test_datasets):
            val_test_paths.append(file)
        else:
            raise ValueError(f"File {file} does not match any dataset")

    # Identify datasets that are redundant in terms of tissues and assays
    assays_tissues_to_datasets = defaultdict(set)
    for at, ddf in dset_summary.loc[val_test_indices].reset_index().groupby(["assays", "tissues"]):
        if len(ddf) > 0:
            assays_tissues_to_datasets[at].update(set(ddf.dataset_id.unique()))
    most_redundant_ds_grp = sorted(max(assays_tissues_to_datasets.items(), key=lambda x: len(x[1]))[1])
    assert (
        len(most_redundant_ds_grp) > 1
    ), f"Expected the most redundant group of datasets to have more than 1 element. Got {most_redundant_ds_grp}."

    random.seed(seed)
    # get the smallest by number of files in most_redundant_ds_grp
    test_heldout_dataset = random.choice(most_redundant_ds_grp)
    for ds in most_redundant_ds_grp:
        if len([file for file in val_test_paths if ds in str(file)]) < len(
            [file for file in val_test_paths if test_heldout_dataset in str(file)]
        ):
            test_heldout_dataset = ds

    test_paths = [file for file in val_test_paths if test_heldout_dataset in str(file)]
    print(f"Holding {len(test_paths)} files out that are part of a random redundant dataset: {test_heldout_dataset}.")
    frac_files = len(test_paths) / len(val_test_paths)
    val_frac = 0.5 + frac_files
    print(f"Pick random {val_frac} of files for validation and the rest are additional for testing.")

    val_test_paths = [file for file in val_test_paths if test_heldout_dataset not in str(file)]

    random.seed(seed)
    random.shuffle(val_test_paths)
    n_val = int(val_frac * len(val_test_paths))  # Additionally hold
    val_paths = val_test_paths[:n_val]
    test_paths.extend(val_test_paths[n_val:])

    # Move files to test directory
    for i, file in enumerate(train_paths):
        remainder_of_path = file.relative_to(data_base_path)
        destination = data_base_path / "train" / remainder_of_path
        if move:
            if i < 5:
                print(f"Moving {file} to {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(file, destination)
        elif i < 10:
            print(f"Would move {file} to {destination}")

    for i, file in enumerate(val_paths):
        remainder_of_path = file.relative_to(data_base_path)
        destination = data_base_path / "val" / remainder_of_path
        if move:
            if i < 5:
                print(f"Moving {file} to {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(file, destination)
        elif i < 10:
            print(f"Would move {file} to {destination}")
    for i, file in enumerate(test_paths):
        remainder_of_path = file.relative_to(data_base_path)
        destination = data_base_path / "test" / remainder_of_path
        if move:
            if i < 5:
                print(f"Moving {file} to {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(file, destination)
        elif i < 10:
            print(f"Would move {file} to {destination}")


def existing_file(path: str, parser: argparse.ArgumentParser):
    fp = Path(path)
    if not fp.exists():
        parser.error("The file %s does not exist!" % path)
    return fp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train-Val-Test Split from Metadata")
    parser.add_argument(
        "metadata_file", type=lambda x: existing_file(x, parser), help="Path to the metadata file", metavar="FILE"
    )
    parser.add_argument(
        "data_base_path",
        type=lambda x: existing_file(x, parser),
        help="Path to the directory that contains the data files we want to split up.",
        metavar="DIR",
    )
    parser.add_argument(
        "--test-val-fraction", type=float, default=0.01, help="Fraction of data to place in test and validation"
    )
    parser.add_argument("--move", action="store_true", default=False, help="Actually do it!")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args.metadata_file, args.data_base_path, args.test_val_fraction, args.seed, args.move)
