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


import os
import pickle
from typing import Any, Callable, Dict, List, Optional

import webdataset as wds
from nemo.utils import logging


def pickles_to_tars(
    dir_input: str,
    input_suffix: str,
    input_prefix_subset: List[str],
    dir_output: str,
    output_prefix: str,
    func_output_data: Callable[[str, str, Any], Dict[str, Any]] = lambda prefix, suffix, data: {
        "__key__": prefix,
        suffix: pickle.dumps(data),
    },
    min_num_shards: Optional[int] = None,
) -> None:
    """Convert a subset of pickle files from a directory to Webdataset tar files
    Input path and name pattern:
    f"{dir_input}/{input_prefix_subset}.{input_suffix}"
    Output path and name pattern:
    f"{dir_output}/{output_prefix}-%06d.tar"

    The webdataset tar archive is specified by the dictionary:
    {
        "__key__" : sample_filename_preifx,
        sample_filename_suffix_1 : data_1,
        sample_filename_suffix_2 : data_2,
        ...
    }
    so that parsing the tar archive is equivalent of reading
    {sample_filename_preifx}.{sample_filename_suffix_1} etc.

    Here, the assumption is that there is only one sample data file, whose name
    prefix is given in each of the elements of `input_prefix_subset` and whose
    name suffix is given by `input_suffix`. Per the webdataset file format
    specification, the `sample_filename_preifx` can't contain dots '.' so this
    function removes it for the user by calling .replace(".", "-") on the
    elements of `input_prefix_subset`

    Args:
        dir_input (str): Input directory
        input_suffix (str): Input pickle file name suffix
        input_prefix_subset (List[str]): Input subset of pickle files' prefix
        dir_output (str): Output directory
        output_prefix (str): Output tar file name prefix
        func_output_data (Callable[[str, str, Any], Dict[str, Any]]) : function
            that maps the name prefix, name suffix and data object to a
            webdataset tar archive dictionary. Refer to the webdataset github
            repo for the archive file format specification.
        min_num_shards (int) : create at least this number of tar files.
            WebDataset has bugs when reading small number of tar files in a
            multi-node lightening + DDP setting so this option can be used to
            guarantee the tar file counts

    Returns: None

    """
    os.makedirs(dir_output, exist_ok=True)
    wd_subset_pattern = os.path.join(dir_output, f"{output_prefix}-%06d.tar")
    maxsize = 1e8
    # Due to a Webdataset bug, number of shards should be >= number of workers
    # (num. of gpus * num. of workers per gpu)
    # TODO: this algorithm is not accurate enough because it doesn't take into
    # account the block structure so I have to multiply the total_size with a
    # small prefactor to purposely underestimate the size so that it ends up
    # creating more tar files than min_num_shards
    if min_num_shards is not None and min_num_shards > 1:
        total_size = 0
        for name in input_prefix_subset:
            try:
                total_size += os.stat(os.path.join(dir_input, f"{name}.{input_suffix}")).st_size
            except Exception:
                continue
        maxsize = min(total_size * 0.6 // min_num_shards, maxsize)
    with wds.ShardWriter(wd_subset_pattern, encoder=False, maxsize=maxsize, compress=False, mode=0o777) as sink:
        for name in input_prefix_subset:
            try:
                data = pickle.load(open(os.path.join(dir_input, f"{name}.{input_suffix}"), "rb"))
                # the prefix name shouldn't contain any "." per webdataset's
                # specification
                sample = func_output_data(name.replace(".", "-"), input_suffix, data)
            except ModuleNotFoundError as e:
                logging.error(f"Dependency for parsing input pickle data not " f"found: {e}")
                raise e
            except Exception as e:
                logging.error(f"Failed to write {name} into tar files due to error {e}")
                raise e

            sink.write(sample)
