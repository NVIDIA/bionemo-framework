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
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class DataFrameTransformDataset(Dataset):
    def __init__(
        self,
        data_file: Union[str, bytes, os.PathLike],
        functions: List[Callable[[pd.Series], Dict[str, torch.TensorType]]],
        read_csv_args: Optional[dict] = None,
    ):
        """Dataset that is fundamentally a dataset but allows transforms on rows

        Args:
            data_file (Union[str, bytes, os.PathLike]): csv file for pandas
            functions (List[Callable[[pd.Series], Dict[str, torch.TensorType]]]):
                Each entry transforms a dataframe row to a dict of tensor to use.
                Union of all function results on an entry will be used.
                These can be loosely coupled to the structure of `data_file`.
        """

        if read_csv_args is None:
            read_csv_args = {}
        self.df = pd.read_csv(data_file, **read_csv_args)
        self.length = self.df.shape[0]
        self.functions = functions
        self.do_transforms = True

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        df_entry = self.df.iloc[idx]
        if self.do_transforms:
            result = {}
            for fn in self.functions:
                result.update(fn(df_entry))
        else:
            result = df_entry.to_dict()
        return result
