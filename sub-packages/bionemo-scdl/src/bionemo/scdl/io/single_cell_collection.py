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

import importlib.metadata
import json
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bionemo.scdl.api.single_cell_row_dataset import SingleCellRowDatasetCore
from bionemo.scdl.index.row_feature_index import VariableFeatureIndex
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.async_worker_queue import AsyncWorkQueue


__all__: Sequence[str] = (
    "FileNames",
    "SingleCellCollection",
)

logger = logging.getLogger("sc_collection")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s")


def _create_single_cell_memmap_dataset_from_h5ad(
    h5ad_path: str,
    base_directory_path: str,
    data_dtype: str | None = None,
    paginated_load_cutoff: int = 10_000,
    load_block_row_size: int = 1_000_000,
    use_X_not_raw: bool = False,
) -> SingleCellMemMapDataset:
    """The SingleCellMemMapDataset is loaded from h5ad_path.

    The data is stored in the base_data_path directory.

    Args:
        h5ad_path: the path to the dataset
        base_directory_path: the base directory path where the dataset will be stored
        data dtype: Optional dtype string for `data.npy` (e.g., 'uint8','float16','float32','float64')
        paginated_load_cutoff: The cutoff in MB for paginated loading of the AnnData files.
        load_block_row_size: The number of rows to load into memory at a time for paginated loading of the AnnData files.
        use_X_not_raw: If True, prefer `adata.X`; otherwise use `adata.raw.X`.

    Returns:
        The created SingleCellMemMapDataset

    Args:
        data_dtype: Optional dtype string for `data.npy` (e.g., 'uint8','float16','float32','float64')
    """
    fname = Path(h5ad_path).stem
    obj = SingleCellMemMapDataset(
        data_path=Path(base_directory_path) / fname,
        h5ad_path=h5ad_path,
        data_dtype=data_dtype,
        paginated_load_cutoff=paginated_load_cutoff,
        load_block_row_size=load_block_row_size,
        use_X_not_raw=use_X_not_raw,
    )
    return obj


class FileNames(str, Enum):
    """Names of files that are generated in SingleCellCollection."""

    VERSION = "version.json"
    METADATA = "metadata.json"
    FEATURES = "features"


class SingleCellCollection(SingleCellRowDatasetCore):
    """A collection of one or more SingleCellMemMapDatasets.

    SingleCellCollection support most of the functionality of the
    SingleCellDataSet API. An SingleCellCollection can be converted
    to a single SingleCellMemMapDataset. A SingleCellCollection
    enables the use of heterogeneous datasets, such as those composed of many
    AnnData files.

    Attributes:
        _version: The version of the dataset
        data_path: The directory where the colleection of datasets is stored.
        _feature_index: The corresponding VariableFeatureIndex where features are
        stored.
        fname_to_mmap:  dictionary to hold each SingleCellMemMapDataset object.
        This maps from the path to the dataset.
        ragged dataset is an dataset of arrays where the arrays have different
        lengths
        False: not ragged; all SingleCellMemMapDataset have same column dimemsion
        True: ragged; scmmap column dimemsions vary
    """

    def __init__(self, data_path: str) -> None:
        """Instantiate the class.

        Args:
            data_path: Where the class will be stored.
        """
        self.data_path: str = data_path
        self._version: str = importlib.metadata.version("bionemo.scdl")
        self.metadata: Dict[str, int] = {}
        self._var_feature_index: VariableFeatureIndex = VariableFeatureIndex()
        self.fname_to_mmap: Dict[str, SingleCellMemMapDataset] = {}
        Path(self.data_path).mkdir(parents=True, exist_ok=True)

        # Write the version
        if not os.path.exists(f"{self.data_path}/{FileNames.VERSION.value}"):
            with open(f"{self.data_path}/{FileNames.VERSION.value}", "w") as vfi:
                json.dump(self.version(), vfi)

    def version(self) -> str:
        """Returns a version number.

        (following <major>.<minor>.<point> convention).
        """
        return self._version

    def load_h5ad(
        self,
        h5ad_path: str,
        paginated_load_cutoff: int = 10_000,
        load_block_row_size: int = 1_000_000,
        use_X_not_raw: bool = False,
    ) -> None:
        """Loads data from an existing AnnData archive.

        This creates and saves a new backing data structure.
        Then, the location and the data and the dataset are stored.

        Args:
            h5ad_path: the path to AnnData archive
            paginated_load_cutoff: The cutoff in MB for paginated loading of the AnnData files.
            load_block_row_size: The number of rows to load into memory at a time for paginated loading of the AnnData files.
            use_X_not_raw: If True, prefer `adata.X`; otherwise use `adata.raw.X`.
        """
        mmap_path = Path(self.data_path) / Path(h5ad_path).stem
        self.fname_to_mmap[mmap_path] = _create_single_cell_memmap_dataset_from_h5ad(
            h5ad_path=h5ad_path,
            base_directory_path=self.data_path,
            paginated_load_cutoff=paginated_load_cutoff,
            load_block_row_size=load_block_row_size,
            use_X_not_raw=use_X_not_raw,
        )
        self._var_feature_index.concat(self.fname_to_mmap[mmap_path]._var_feature_index)

    def load_h5ad_multi(
        self,
        directory_path: str,
        max_workers: int = 5,
        use_processes: bool = False,
        data_dtype: str | None = None,
        paginated_load_cutoff: int = 10_000,
        load_block_row_size: int = 1_000_000,
        use_X_not_raw: bool = False,
    ) -> None:
        """Loads one or more AnnData files and adds them to the collection.

        Args:
            directory_path: The path to the directory with the AnnData files
            max_workers: the maximal number of workers to use
            use_processes: If True, use ProcessPoolExecutor; otherwise, use
                ThreadPoolExecutor
            paginated_load_cutoff: The cutoff in MB for paginated loading of the AnnData files.
            load_block_row_size: The number of rows to load into memory at a time for paginated loading of the AnnData files.
            data_dtype: Optional dtype string propagated to dataset creation
            use_X_not_raw: If True, prefer `adata.X`; otherwise use `adata.raw.X`.

        Raises:
            FileNotFoundError: If no h5ad files are found in the directory.
            RuntimeError: If an error occurs in the loading of any of the h5ad files.

        Args:
            data_dtype: Optional dtype string propagated to dataset creation
        """
        directory_path = Path(directory_path)
        ann_data_paths = sorted(directory_path.rglob("*.h5ad"))
        if len(ann_data_paths) == 0:
            raise FileNotFoundError(f"There a no h5ad files in {directory_path}.")
        mmap_paths = [Path(self.data_path) / Path(ann_datapath).stem for ann_datapath in ann_data_paths]
        queue = AsyncWorkQueue(max_workers=max_workers, use_processes=use_processes)
        for ann in ann_data_paths:
            queue.submit_task(
                _create_single_cell_memmap_dataset_from_h5ad,
                ann,
                base_directory_path=self.data_path,
                data_dtype=data_dtype,
                paginated_load_cutoff=paginated_load_cutoff,
                load_block_row_size=load_block_row_size,
                use_X_not_raw=use_X_not_raw,
            )
        queue.wait()
        mmaps = queue.get_task_results()

        for result_path, result in zip(ann_data_paths, mmaps):
            if isinstance(result, Exception):
                raise RuntimeError(f"Error in processing file {result_path}: {result}") from result

        for mmap_path, mmap in zip(mmap_paths, mmaps):
            if isinstance(mmap, Exception):
                raise RuntimeError(f"Error in processing file {mmap_path}: {mmap}") from mmap

            self.fname_to_mmap[mmap_path] = mmap
            self._var_feature_index.concat(self.fname_to_mmap[mmap_path]._var_feature_index)

    def number_nonzero_values(self) -> int:
        """Sum of the number of non zero entries in each dataset."""
        return sum([self.fname_to_mmap[mmap_path].number_nonzero_values() for mmap_path in self.fname_to_mmap])

    def number_of_values(self) -> int:
        """Sum of the number of values in each dataset."""
        return sum([self.fname_to_mmap[mmap_path].number_of_values() for mmap_path in self.fname_to_mmap])

    def number_of_rows(self) -> int:
        """The number of rows in the dataset.

        Returns:
            The number of rows in the dataset
        Raises:
            ValueError if the length of the number of rows in the feature
            index does not correspond to the number of stored rows.
        """
        row_sum_from_datasets = sum(
            [self.fname_to_mmap[mmap_path].number_of_rows() for mmap_path in self.fname_to_mmap]
        )
        if len(self._var_feature_index) > 0 and self._var_feature_index.number_of_rows() != row_sum_from_datasets:
            raise ValueError(
                f"""The nuber of rows in the feature index {self._var_feature_index.number_of_rows()}
                             does not correspond to the number of rows in the datasets {row_sum_from_datasets}"""
            )

        return row_sum_from_datasets

    def number_of_variables(self) -> List[int]:
        """If ragged, returns a list of variable lengths.

        If not ragged, returns a list with one entry. A ragged
        collection is one where the datasets have different lengths.
        """
        if len(self._var_feature_index) == 0:
            return [0]
        else:
            num_vars = self._var_feature_index.column_dims()
            return num_vars

    def shape(self) -> Tuple[int, List[int]]:
        """Get the shape of the dataset.

        This is the number of entries by the the length of the feature index
        corresponding to that variable.

        Returns:
            The total number of elements across dataset
            A list containing the number of variables for each entry in the
                VariableFeatureIndex.
        """
        return self.number_of_rows(), self.number_of_variables()

    def flatten(
        self,
        output_path: str,
        destroy_on_copy: bool = False,
        extend_copy_size: int = 10 * 1_024 * 1_024,
    ) -> None:
        """Flattens the collection into a single SingleCellMemMapDataset.

        This is done by concatenating all of the SingleCellMemMapDatasets together. This can be used
        to create a single dataset from h5ad files that are in a directory:
                coll = SingleCellCollection(temp_dir)
                coll.load_h5ad_multi(data_path)
                coll.flatten(output_path, destroy_on_copy=True)
        Then, there would be one SingleCellMemMapDataset dataset in output_path. read in with
        SingleCellMemmpDataset(output_path).

        Args:
            output_path: location to store new dataset
            destroy_on_copy: Whether to remove the current data_path
            extend_copy_size: how much to copy in memory at once

        """
        output = next(iter(self.fname_to_mmap.values()))

        single_cell_list = list(self.fname_to_mmap.values())[1:]
        if len(single_cell_list) > 0:
            output.concat(
                single_cell_list,
                extend_copy_size=extend_copy_size,
                output_path=output_path,
                destroy_on_copy=destroy_on_copy,
            )
        else:
            shutil.move(output.data_path, output_path)
            output.data_path = output_path
        # Hit save!
        output.save()

        if destroy_on_copy:
            shutil.rmtree(self.data_path)
