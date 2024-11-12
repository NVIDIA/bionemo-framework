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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, get_args

import webdataset as wds


def pickles_to_tars(
    dir_input: str,
    input_prefix_subset: List[str],
    input_suffix: Union[str, Iterable[str]],
    dir_output: str,
    output_prefix: str,
    func_output_data: Callable[[str, Dict[str, Any]], Dict[str, Any]] = lambda prefix, suffix_to_data: {
        "__key__": prefix,
        **suffix_to_data,
    },
    min_num_shards: Optional[int] = None,
) -> None:
    """Convert a subset of pickle files from a directory to Webdataset tar files.

    Input path and name pattern for sample 0:
    f"{dir_input}/{input_prefix_subset[0]}.{input_suffix[0]}"
    f"{dir_input}/{input_prefix_subset[0]}.{input_suffix[1]}"
    Input path and name pattern for sample 1:
    f"{dir_input}/{input_prefix_subset[1]}.{input_suffix[0]}"
    f"{dir_input}/{input_prefix_subset[1]}.{input_suffix[1]}"
    ...
    Output path and name pattern:
    f"{dir_output}/{output_prefix}-%06d.tar".

    The webdataset tar archive is specified by the dictionary:
    {
        "__key__" : sample_filename_preifx,
        sample_filename_suffix_1 : data_1,
        sample_filename_suffix_2 : data_2,
        ...
    }
    so that parsing the tar archive is equivalent of reading
    {sample_filename_preifx}.{sample_filename_suffix_1} etc.

    Here, each sample data get its name prefix from one element of
    `input_prefix_subset` and its name suffixes from the list `input_suffix`.
    Per the webdataset file format specification, the `sample_filename_preifx`
    can't contain dots '.' so this function removes it for the user by calling
    .replace(".", "-") on the elements of `input_prefix_subset`

    Args:
        dir_input: Input directory
        input_prefix_subset: Input subset of pickle files' prefix
        input_suffix: Input pickle file name
            suffixes, each for one type of data object, for all the samples
        dir_output: Output directory
        output_prefix: Output tar file name prefix
        func_output_data: function that maps the name prefix, name suffix and
            data object to a webdataset tar archive dictionary. Refer to the webdataset
            github repo for the archive file format specification.
        min_num_shards : create at least this number of tar files.
            WebDataset has bugs when reading small number of tar files in a
            multi-node lightening + DDP setting so this option can be used to
            guarantee the tar file counts
    """
    if not isinstance(input_suffix, get_args(Union[str, Iterable])):
        raise TypeError("input_suffix can only be str or Iterable[str]")
    os.makedirs(dir_output, exist_ok=True)
    wd_subset_pattern = os.path.join(dir_output, f"{output_prefix}-%06d.tar")
    n_samples_per_shard_max = 100000
    if min_num_shards is not None:
        if min_num_shards <= 0:
            raise ValueError(f"Invalid min_num_shards = {min_num_shards} <= 0")
        n_samples_per_shard_max = len(input_prefix_subset) // min_num_shards
    with wds.ShardWriter(
        wd_subset_pattern,
        encoder=False,
        maxcount=n_samples_per_shard_max,
        compress=False,
        mode=0o777,
    ) as sink:
        for name in input_prefix_subset:
            try:
                if isinstance(input_suffix, str):
                    suffix_to_data = {
                        input_suffix: pickle.dumps(
                            pickle.loads((Path(dir_input) / f"{name}.{input_suffix}").read_bytes())
                        )
                    }
                else:
                    suffix_to_data = {
                        suffix: pickle.dumps(pickle.loads((Path(dir_input) / f"{name}.{suffix}").read_bytes()))
                        for suffix in input_suffix
                    }
                # the prefix name shouldn't contain any "." per webdataset's
                # specification
                sample = func_output_data(name.replace(".", "-"), suffix_to_data)
                sink.write(sample)
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Can't process pickle file due to\
                                   missing dependencies"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to write {name} into tar files.") from e
