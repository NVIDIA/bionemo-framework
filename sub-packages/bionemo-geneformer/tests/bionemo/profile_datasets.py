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


import logging
import random
import time
from functools import wraps
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from bionemo.core.data.multi_epoch_dataset import EpochIndex, MultiEpochDatasetResampler
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.geneformer.data.singlecell.dataset_old import SingleCellDataset as OldSingleCellDataset
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.lightning import batch_collator


def timeit(method):
    """Wrapper to time functions."""

    @wraps(method)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Method {method.__name__} took {run_time:.4f} seconds")
        return result, run_time

    return timed


def time_all_methods(cls):
    """Time all methods in class."""
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and attr_name != "__init__":  # Check if the attribute is a method
            setattr(cls, attr_name, timeit(attr_value))
    return cls


@time_all_methods
class GeneformerDatasetMetrics:
    """SCDL Metrics."""

    def __init__(self, data_dir, tokenizer, median_dict, old=False):
        """Instantiate class."""
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.median_dict = median_dict
        self.old = old

    def create_from_memmap(self):
        """Create from memmap dir."""
        self.ds = SingleCellDataset(
            self.data_dir,
            tokenizer=self.tokenizer,
            median_dict=self.median_dict,
            seed=42,
        )

    def get_length(self):
        """Length."""
        self.length = len(self.ds)
        return self.length

    def get_first_item(self):
        """Get first item."""
        index = EpochIndex(epoch=0, idx=0)
        return self.ds.__getitem__(index)
        # return self.ds.scdl.get_row(0, return_features=True, feature_vars="feature_id")

    def get_last_item(self):
        """Get last item."""
        index = EpochIndex(epoch=0, idx=self.length - 1)
        return self.ds.__getitem__(index)

    def get_middle_item(self):
        """Get middle item."""
        index = EpochIndex(epoch=0, idx=(self.length - 1) // 2)
        return self.ds.__getitem__(index)

    def stress_test_item(self):
        """Stress test get item."""
        random.seed(42)
        random_integers = [random.randint(0, self.length - 1) for _ in range(500)]
        for i in random_integers:
            index = EpochIndex(idx=i, epoch=0)
            self.ds.__getitem__(index)
        return 0

    def stress_test_get_indices(self, num_indices):
        """Stress test get indices."""
        random.seed(42)
        _ = [random.randint(0, self.length - 1) for _ in range(num_indices)]
        for i in range(self.length):
            index = EpochIndex(idx=i, epoch=0)
            vals, _ = self.ds.scdl.get_row(
                index.idx, return_features=True, feature_vars=["feature_id"]
            )  # , feature_vars=["feature_id"])
            # print(" NEW Vals:, ", vals[0].dtype, type(vals[0]))
            # print("NEW Indices: ", vals[1].dtype, type(vals[1]))
        return 0

    def iterate_train(self):
        """Call get item on each item in training set."""
        print(self.length)
        for i in range(self.length):
            index = EpochIndex(idx=i, epoch=0)
            self.ds.__getitem__(index)
        return 0

    def iterate_train_dataloader(self, num_workers=64, num_indices=30_000):
        """Call get item on each item in training set."""
        # print(self.length)
        shuffled_dataset = MultiEpochDatasetResampler(
            self.ds,
            shuffle=True,
            seed=42,
        )
        print("HERE 1")

        dataloader = DataLoader(
            shuffled_dataset,  # NewWrapperDataset(self.ds),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            batch_size=256,
            collate_fn=batch_collator,
            pin_memory=True,
            persistent_workers=True,
        )
        print("HERE 2")

        for i, thing in enumerate(dataloader):
            print("STEP", i)
            if i == num_indices:
                break
            pass
        return 0


class NewWrapperDataset(Dataset):
    """Wrapper for Single Cell Dataset."""

    def __init__(self, ds: SingleCellDataset):
        """Initialize class."""
        self.ds = ds

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.ds)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features (X) and, if available, the label (y).
        """
        # Extract data for the given index
        index = EpochIndex(idx=idx, epoch=0)
        return self.ds.__getitem__(index)


class OldWrapperDataset(Dataset):
    """Wrapper for Old Single Cell Dataset."""

    def __init__(self, ds: OldSingleCellDataset):
        """Initialize class."""
        self.ds = ds

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.ds)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features (X) and, if available, the label (y).
        """
        # Extract data for the given index
        index = EpochIndex(idx=idx, epoch=0)
        return self.ds.__getitem__(index)


@time_all_methods
class OldGeneformerDatasetMetrics:
    """Old dataset Metrics."""

    def __init__(self, data_dir, tokenizer, median_dict):
        """Instantiate class."""
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.median_dict = median_dict

    def create_from_memmap(self):
        """Create from memmap dir."""
        self.ds = OldSingleCellDataset(
            self.data_dir,
            tokenizer=self.tokenizer,
            median_dict=self.median_dict,
            seed=42,
        )

    def get_length(self):
        """Length."""
        self.length = len(self.ds)
        return self.length

    def get_first_item(self):
        """Get first item."""
        index = EpochIndex(epoch=0, idx=0)
        return self.ds.__getitem__(index)

    def get_last_item(self):
        """Get last item."""
        index = EpochIndex(epoch=0, idx=self.length - 1)
        return self.ds.__getitem__(index)

    def get_middle_item(self):
        """Get middle item."""
        index = EpochIndex(epoch=0, idx=(self.length - 1) // 2)
        return self.ds.__getitem__(index)

    def stress_test_item(self):
        """Stress test get item."""
        random.seed(42)
        random_integers = [random.randint(0, self.length - 1) for _ in range(500)]
        for i in random_integers:
            index = EpochIndex(idx=i, epoch=0)
            self.ds.__getitem__(index)
        return 0

    def stress_test_get_indices(self, num_indices):
        """Stress test get indices."""
        random.seed(42)
        _ = [random.randint(0, self.length - 1) for _ in range(num_indices)]
        for i in range(self.length):
            vals, indices, _ = self.ds.lookup_cell_by_idx(i)
            # print("old Vals:, ",  vals.dtype, type(vals))
            # print("OLD Indices: ", indices.dtype, type(indices))
        return

    def iterate_train(self):
        """Call get item on each item in training set."""
        print(self.length)
        for i in range(self.length):
            index = EpochIndex(idx=i, epoch=0)
            self.ds.__getitem__(index)
        return 0

    def iterate_train_dataloader(self, num_workers=64, num_indices=30_000):
        """Call get item on each item in training set."""
        # print(self.length)

        shuffled_dataset = MultiEpochDatasetResampler(
            self.ds,
            shuffle=True,
            seed=42,
        )
        print("HERE 1")
        dataloader = DataLoader(
            shuffled_dataset,  # OldWrapperDataset(self.ds),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            batch_size=256,
            collate_fn=batch_collator,
            pin_memory=True,
            persistent_workers=True,
        )
        print("HERE 2")

        for i, _ in enumerate(dataloader):
            print("HERE!!!", i)
            if i == num_indices:
                break
            pass
        return 0


if __name__ == "__main__":
    results_dict = {}
    # memap_data_path = load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl" / "train"
    # old_data_path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data" / "train"

    # old_data_path = Path("/workspace/bionemo2/sub-packages/data/merged_30GB_old_geneformer")
    # memap_data_path = Path("/workspace/bionemo2/sub-packages/data/test_30GB_merged_test_subset")
    old_data_path = Path("/workspace/bionemo2/sub-packages/data/train_old")
    memap_data_path = Path("/workspace/bionemo2/sub-packages/data/train_new")
    # memap_data_path = Path("/workspace/bionemo2/sub-packages/data/revised_large_memmap_dataset")
    # old_data_path = Path("/workspace/bionemo2/sub-packages/data/revised_large_memmap_dataset")
    preprocessor = GeneformerPreprocess(
        download_directory=memap_data_path,
        medians_file_path=memap_data_path / "medians.json",
        tokenizer_vocab_path=memap_data_path / "geneformer.vocab",
    )
    print(memap_data_path)
    print(old_data_path)
    num_indices = 1000
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    geneformer_metrics_new = GeneformerDatasetMetrics(
        data_dir=memap_data_path,
        tokenizer=tokenizer,
        median_dict=median_dict,  # type: ignore
    )  # type: ignore
    print("STARTING")
    print("NEW STUFF")

    results_dict["Create Geneformer Dataset"] = geneformer_metrics_new.create_from_memmap()[1]  # type: ignore
    # results_dict["Geneformer Dataset Get Length (s)"] = geneformer_metrics_new.get_length()[1]
    # results_dict["Geneformer Dataset Get First Item (s)"] = geneformer_metrics_new.get_first_item()[1]
    # results_dict["Geneformer Dataset Get Middle Item (s)"] = geneformer_metrics_new.get_middle_item()[1]
    # results_dict["Geneformer Dataset Get Last Item (s)"] = geneformer_metrics_new.get_last_item()[1]
    # results_dict["Geneformer Dataset Get Indices (s)"] = geneformer_metrics_new.stress_test_get_indices(num_indices)[1]
    # results_dict["Geneformer Dataset Get Items (s)"] = geneformer_metrics_new.stress_test_item()[1]
    # results_dict["Geneformer Dataset Get Items (s)"] = geneformer_metrics_new.iterate_train()[1]
    print("ITERATE TRAIN DATA LOADER NEW: ")
    for num_workers in [16, 32, 64, 128]:
        print(
            "Numworkers: ",
            num_workers,
            geneformer_metrics_new.iterate_train_dataloader(num_workers=num_workers, num_indices=num_indices)[1],
        )

    geneformer_metrics_old = OldGeneformerDatasetMetrics(
        data_dir=old_data_path,
        tokenizer=tokenizer,
        median_dict=median_dict,  # type: ignore
    )  # type: ignore
    print("OLD STUFF")
    results_dict["Old Create Geneformer Dataset"] = geneformer_metrics_old.create_from_memmap()[1]  # type: ignore
    # results_dict["Old Geneformer Dataset Get Length (s)"] = geneformer_metrics_old.get_length()[1]
    # results_dict["Old Geneformer Dataset Get First Item (s)"] = geneformer_metrics_old.get_first_item()[1]
    # results_dict["Old Geneformer Dataset Get Middle Item (s)"] = geneformer_metrics_old.get_middle_item()[1]
    # results_dict["Old Geneformer Dataset Get Last Item (s)"] = geneformer_metrics_old.get_last_item()[1]
    # results_dict["Old Geneformer Dataset Get Indices (s)"] = geneformer_metrics_old.stress_test_get_indices(
    #    num_indices
    # )[1]
    # results_dict["Old Geneformer Dataset Get Items (s)"] = geneformer_metrics_old.stress_test_item()[1]
    # results_dict["Geneformer Dataset Get Items (s)"] = geneformer_metrics_old.iterate_train()[1]
    # results_dict["Geneformer Dataset Get Items Dataloader (s)"] = geneformer_metrics_old.iterate_train_dataloader()[1]

    print("ITERATE TRAIN DATA LOADER OLD: ")
    for num_workers in [16, 32, 64, 128]:
        print(
            "Numworkers: ",
            num_workers,
            geneformer_metrics_old.iterate_train_dataloader(num_workers=num_workers, num_indices=num_indices)[1],
        )
    df = pd.DataFrame([results_dict])
    df.to_csv("full_runtime.csv")
