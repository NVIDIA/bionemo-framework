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

import anndata as ad
import numpy as np
import pandas as pd
import torch
from memory_profiler import profile
from torch.utils.data import DataLoader, Dataset

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


class AnnDataset(Dataset):
    """Ann Data Dataset."""

    def __init__(self, anndata_obj: ad.AnnData):
        """Custom Dataset for AnnData objects compatible with PyTorch's DataLoader.

        Args:
            anndata_obj (ad.AnnData): The AnnData object to load data from.
        """
        self.anndata_obj = anndata_obj

    def __len__(self):
        """Returns the total number of samples."""
        return self.anndata_obj.shape[0]

    def __getitem__(self, idx):
        """Returns a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features (X) and, if available, the label (y).
        """
        # Extract data for the given index
        row = self.anndata_obj.X[idx]
        return torch.from_numpy(np.stack([row.indices, row.data]))


class AnnDataRandomDataset(Dataset):
    """Random Blocks of the Dataset."""

    def __init__(self, adata, sample_size=128):
        """Instantiate class."""
        self.adata = adata
        self.sample_size = sample_size
        self.total_rows = self.adata.shape[0]

    def __len__(self):
        """Length."""
        return len(self.adata) // self.sample_size

    def __getitem__(self, idx):
        """Randomly sample indices for the requested rows."""
        random_indices = np.random.choice(self.total_rows, self.sample_size, replace=False)

        # Extract the data for the requested rows
        data = self.adata.X[random_indices]

        # Convert to torch tensor
        data_tensor = torch.tensor(data.toarray(), dtype=torch.float32)

        return data_tensor


class AnnDataContinuousBlockDataset(Dataset):
    """Continous Blocks of the Dataset."""

    def __init__(self, adata, sample_size=128):
        """Instantiate class."""
        self.adata = adata
        self.sample_size = sample_size
        self.total_rows = self.adata.shape[0]

    def __len__(self):
        """Length."""
        return len(self.adata) // self.sample_size

    def __getitem__(self, idx):
        """Randomly sample indices for the requested rows."""
        start_idx = idx * self.sample_size
        end_idx = min(start_idx + self.sample_size, self.total_rows)
        data = self.adata.X[start_idx:end_idx]
        data_tensor = torch.tensor(data.toarray(), dtype=torch.float32)
        return data_tensor


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
class AnnDataMetrics:
    """AnnData Metrics."""

    def __init__(self, adatapath):
        """Instantiate class."""
        self.adatapath = adatapath

    @profile
    def load(self):
        """Create from anndataset."""
        self.ad = ad.read_h5ad(self.adatapath)

    def load_backed(self):
        """Create from anndataset."""
        self.ad_backed = ad.read_h5ad(self.adatapath, backed="r")

    def size_disk_bytes(self):
        """Size of scdl on disk."""
        return get_disk_size(self.adatapath)

    @profile
    def iterate_dl(self, batch_size=128, num_workers=8):
        """Iterate over the dataset."""
        dataloader = DataLoader(
            AnnDataset(self.ad),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_sparse_matrix_batch,
        )
        n_epochs = 1
        for _ in range(n_epochs):
            for _ in dataloader:
                pass

    def iterate_dl_backed_random(self, batch_size=128, num_workers=8):
        """Get random chunks at once."""
        dataloader = DataLoader(
            AnnDataRandomDataset(self.ad_backed, sample_size=batch_size),
            batch_size=1,
            num_workers=num_workers,
            shuffle=True,
        )
        n_epochs = 1
        for _ in range(n_epochs):
            for _ in dataloader:
                pass

    def iterate_dl_backed_continuous(self, batch_size=128, num_workers=8):
        """Get continous chunks at once."""
        dataloader = DataLoader(
            AnnDataContinuousBlockDataset(self.ad_backed, sample_size=batch_size),
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
        )
        n_epochs = 1
        for _ in range(n_epochs):
            for _ in dataloader:
                pass


@time_all_methods
class SCDLMetrics:
    """SCDL Metrics."""

    def __init__(self, adatapath, memmap_dir):
        """Instantiate class."""
        self.adatapath = adatapath
        self.memmap_dir = memmap_dir

    def create_from_adata(self):
        """Create from anndataset."""
        self.first_ds = SingleCellMemMapDataset(
            self.memmap_dir,
            self.adatapath,
        )

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
    path = Path("../../samples/")
    for fn in path.rglob("*"):
        if get_disk_size(fn) > 10 * (1_024**2):
            continue
        print(fn, get_disk_size(fn))
        results_dict = {}

        anndatapath = fn
        results_dict["anndata file"] = fn
        anndata_m = AnnDataMetrics(anndatapath)
        results_dict["AnnData Dataset Load Time (s)"] = anndata_m.load()[1]
        results_dict["AnnData Dataset Backed Load Time (s)"] = anndata_m.load_backed()[1]

        results_dict["AnnData Time to iterate over Dataset 0 workers (s)"] = anndata_m.iterate_dl(num_workers=0)[1]
        results_dict["AnnData Time to iterate over Dataset 8 workers (s)"] = anndata_m.iterate_dl(num_workers=8)[1]
        """
        results_dict["AnnData Time to iterate over Dataset Block Batches 0 workers (s)"] = (
            anndata_m.iterate_dl_backed_continuous(num_workers=0)[1]
        )
        results_dict["AnnData Time to iterate over Dataset Block Batches 8 workers (s)"] = (
            anndata_m.iterate_dl_backed_continuous(num_workers=8)[1]
        )

        results_dict["AnnData Time to iterate over Dataset Random Batches 0 workers (s)"] = (
            anndata_m.iterate_dl_backed_random(num_workers=0)[1]
        )
        results_dict["AnnData Time to iterate over Dataset Random Batches 8 workers (s)"] = (
            anndata_m.iterate_dl_backed_random(num_workers=8)[1]
        )

        del anndata_m
        scdl_m = SCDLMetrics(memmap_dir="memmap_" + Path(fn).stem, adatapath=anndatapath)
        results_dict["AnnData Dataset Size on Disk (MB)"] = scdl_m.anndata_size_disk_bytes()[0] / (1_024**2)

        results_dict["SCDL Dataset from AnnData Time (s)"] = scdl_m.create_from_adata()[1]
        results_dict["SCDL Dataset save time (s)"] = scdl_m.save()[1]
        results_dict["SCDL Dataset Load Time (s)"] = scdl_m.load_backed()[1]

        results_dict["SCDL Time to iterate over Dataset 0 workers (s)"] = scdl_m.iterate_dl(num_workers=0)[1]
        results_dict["SCDL Time to iterate over Dataset 8 workers (s)"] = scdl_m.iterate_dl(num_workers=8)[1]

        results_dict["SCDL Dataset Size on Disk (MB)"] = scdl_m.size_disk_bytes()[0] / (1_024**2)
        """
        dicts.append(results_dict)
        combined = {key: [d[key] for d in dicts] for key in dicts[0]}
        df = pd.DataFrame(combined)
        df.to_csv("rt.csv", index=False)
        break