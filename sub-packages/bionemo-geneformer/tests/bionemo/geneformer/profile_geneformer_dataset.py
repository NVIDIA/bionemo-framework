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

import pandas as pd

from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.geneformer.data.singlecell.dataset_old import SingleCellDataset as OldSingleCellDataset
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.testing.data.load import load


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
            self.data_dir, tokenizer=self.tokenizer, median_dict=self.median_dict, bypass_tokenizer_vocab=True
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
        random_integers = [random.randint(0, self.length) for _ in range(500)]
        for i in random_integers:
            index = EpochIndex(idx=i, epoch=0)
            self.ds.__getitem__(index)
        return 0

    def stress_test_get_indices(self, num_indices):
        """Stress test get indices."""
        random.seed(42)
        random_integers = [random.randint(0, self.length - 1) for _ in range(num_indices)]
        for i in random_integers:
            index = EpochIndex(idx=i, epoch=0)
            self.ds.scdl.get_row(index.idx, return_features=True)  # , feature_vars=["feature_id"])
        return 0


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
        self.ds = OldSingleCellDataset(self.data_dir, tokenizer=self.tokenizer, median_dict=self.median_dict)

    def get_length(self):
        """Length."""
        self.length = len(self.ds)
        return self.length

    def get_first_item(self):
        """Get first item."""
        return self.ds.__getitem__(0)

    def get_last_item(self):
        """Get last item."""
        return self.ds.__getitem__(self.length - 1)

    def get_middle_item(self):
        """Get middle item."""
        return self.ds.__getitem__((self.length - 1) // 2)

    def stress_test_item(self):
        """Stress test get item."""
        random.seed(42)
        random_integers = [random.randint(0, self.length) for _ in range(5_000)]
        for i in random_integers:
            self.ds.__getitem__(i)
        return 0

    def stress_test_get_indices(self, num_indices):
        """Stress test get indices."""
        random.seed(42)
        random_integers = [random.randint(0, self.length - 1) for _ in range(num_indices)]
        for i in random_integers:
            self.ds.lookup_cell_by_idx(i)
        return 0


if __name__ == "__main__":
    results_dict = {}
    memap_data_path = load("single_cell/testdata-memmap-format") / "cellxgene_2023-12-15_small_mmap" / "train"
    old_data_path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data" / "train"
    preprocessor = GeneformerPreprocess(
        download_directory=memap_data_path,
        medians_file_path=memap_data_path / "medians.json",
        tokenizer_vocab_path=memap_data_path / "geneformer.vocab",
    )
    print(memap_data_path)
    print(old_data_path)
    num_indices = 30_000
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
    print("NEW STUFF")
    results_dict["Create Geneformer Dataset"] = geneformer_metrics_new.create_from_memmap()[1]  # type: ignore
    results_dict["Geneformer Dataset Get Length (s)"] = geneformer_metrics_new.get_length()[1]
    results_dict["Geneformer Dataset Get First Item (s)"] = geneformer_metrics_new.get_first_item()[1]
    results_dict["Geneformer Dataset Get Middle Item (s)"] = geneformer_metrics_new.get_middle_item()[1]
    results_dict["Geneformer Dataset Get Last Item (s)"] = geneformer_metrics_new.get_last_item()[1]
    results_dict["Geneformer Dataset Get Indices (s)"] = geneformer_metrics_new.stress_test_get_indices(num_indices)[1]

    # results_dict["Geneformer Dataset Get Items (s)"] = geneformer_metrics_new.stress_test_item()[1]

    geneformer_metrics_old = OldGeneformerDatasetMetrics(
        data_dir=old_data_path,
        tokenizer=tokenizer,
        median_dict=median_dict,  # type: ignore
    )  # type: ignore
    print("OLD STUFF")
    results_dict["Old Create Geneformer Dataset"] = geneformer_metrics_old.create_from_memmap()[1]  # type: ignore
    results_dict["Old Geneformer Dataset Get Length (s)"] = geneformer_metrics_old.get_length()[1]
    results_dict["Old Geneformer Dataset Get First Item (s)"] = geneformer_metrics_old.get_first_item()[1]
    results_dict["Old Geneformer Dataset Get Middle Item (s)"] = geneformer_metrics_old.get_middle_item()[1]
    results_dict["Old Geneformer Dataset Get Last Item (s)"] = geneformer_metrics_old.get_last_item()[1]
    results_dict["Old Geneformer Dataset Get Indices (s)"] = geneformer_metrics_old.stress_test_get_indices(
        num_indices
    )[1]
    # results_dict["Old Geneformer Dataset Get Items (s)"] = geneformer_metrics_old.stress_test_item()[1]
    df = pd.DataFrame([results_dict])
    df.to_csv("full_runtime.csv")
