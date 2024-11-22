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


import subprocess
import sys
import time
from enum import Enum
from functools import wraps
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch


class FileNames(str, Enum):
    """Names of files that are generated in SingleCellCollection."""

    DATA = "data.npy"
    COLPTR = "col_ptr.npy"
    ROWPTR = "row_ptr.npy"
    METADATA = "metadata.json"
    DTYPE = "dtypes.json"
    FEATURES = "features"
    VERSION = "version.json"


def get_disk_size(directory):
    """Size of directory on disk."""
    result = subprocess.run(["du", "-sb", directory], stdout=subprocess.PIPE, text=True)
    size_in_bytes = int(result.stdout.split()[0])
    return size_in_bytes


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
class SCDLMetrics:
    """SCDL Metrics."""

    def __init__(
        self,
        memmap_dir,
        adatapath=None,
    ):
        """Instantiate class."""
        self.adatapath = adatapath
        self.memmap_dir = memmap_dir

    def create_from_memmap(self):
        """Create from anndataset."""
        self.first_ds = SingleCellMemMapDataset(self.memmap_dir)
        # self.first_ds = SingleCellCollection(
        #     self.memmap_dir,
        #     self.adatapath,
        # )

    def save(self):
        """Save."""
        self.first_ds.save()
        del self.first_ds

    def load_backed(self):
        """Load Scdl from disk."""
        self.ds = SingleCellMemMapDataset(self.memmap_dir)

    def num_values(self):
        """Number of values."""
        return self.ds.number_of_values()

    def get_row_for_all_indices(self):
        """Stress test get indices."""
        for i in range(len(self.ds)):
            vals, _ = self.ds.get_row(i, return_features=True, feature_vars=["feature_id"])
        return 0

    def sparsity_stats(self):
        """Sparsity of dataset."""
        return self.ds.sparsity()

    def size_disk_bytes(self):
        """Size of scdl on disk."""
        return get_disk_size(self.memmap_dir)

    def anndata_size_disk_bytes(self):
        """Size of anndata on disk."""
        return get_disk_size(self.adatapath)

    def size_mem_dataset_bytes(self):
        """Size of dataset in memory."""
        return sys.getsizeof(self.ds)

    def iterate_dl(self, batch_size=128, num_workers=8):
        """Iterate over the dataset."""
        dataloader = DataLoader(
            self.ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_sparse_matrix_batch,
        )
        n_epochs = 1
        for _ in range(n_epochs):
            for _ in dataloader:
                pass


if __name__ == "__main__":
    dicts = []
    # path = Path("../../samples/")
    path = Path("/workspace/bionemo2/cellxgene_2023-12-15_small_mmap/train")

    scdl_m = SCDLMetrics(memmap_dir=path)
    results_dict = {}
    # results_dict["AnnData Dataset Size on Disk (MB)"] = scdl_m.anndata_size_disk_bytes()[0] / (1_024**2)

    results_dict["SCDL Dataset from AnnData Time (s)"] = scdl_m.create_from_memmap()[1]
    results_dict["SCDL Dataset save time (s)"] = scdl_m.save()[1]
    results_dict["SCDL Dataset Load Time (s)"] = scdl_m.load_backed()[1]
    results_dict["SCDL Dataset Fetch Indices Time (s)"] = scdl_m.load_backed()[1]
    results_dict["SCDL Time to iterate over all indices in datatset "] = scdl_m.get_row_for_all_indices()[1]

    results_dict["SCDL Time to iterate over Dataset 0 workers (s)"] = scdl_m.iterate_dl(num_workers=0)[1]
    results_dict["SCDL Time to iterate over Dataset 8 workers (s)"] = scdl_m.iterate_dl(num_workers=8)[1]

    results_dict["SCDL Dataset Size on Disk (MB)"] = scdl_m.size_disk_bytes()[0] / (1_024**2)

    dicts.append(results_dict)
    combined = {key: [d[key] for d in dicts] for key in dicts[0]}
    df = pd.DataFrame(combined)
    print(df)
    df.to_csv("all_runtime.csv", index=False)
