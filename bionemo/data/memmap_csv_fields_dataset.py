# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

import re

from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import TextMemMapDataset
from nemo.utils import logging


__all__ = ["CSVFieldsMemmapDataset"]


class CSVFieldsMemmapDataset(TextMemMapDataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    Returns a dictionary with multiple fields.

    WARNING: This class has been migrated to NeMo and will be removed from BioNeMo when NeMo container 1.21 is used.
    Every change to this class should be added to the class in NeMo.
    https://github.com/NVIDIA/NeMo/blob/83d6614fbf29cf885f3bc36233f6e3758ba8f1e3/nemo/collections/nlp/data/language_modeling/text_memmap_dataset.py#L336

    """

    def __init__(
        self,
        dataset_paths,
        newline_int=10,
        header_lines=1,
        workers=None,
        tokenizer=None,
        sort_dataset_paths=True,
        # data_fields - dict of field names and their corresponding indices
        data_sep=',',
        data_fields={"data": 0},
        index_mapping_dir=None,
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=newline_int,
            header_lines=header_lines,
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )

        self._data_fields = data_fields
        self._data_sep = data_sep
        logging.warning("CSVFieldsMemmapDataset will be available in NeMo 1.21")

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # convert text into data
        _build_data_from_text = super()._build_data_from_text
        # extract id and sequence and tokenize (if needed)
        data = {}
        rule = self._data_sep + r'\s*(?=(?:[^"]*"[^"]*")*[^"]*$)'
        text_fields = re.split(r'{}'.format(rule), text)
        for field_name, field_idx in self._data_fields.items():
            data[field_name] = _build_data_from_text(text_fields[field_idx].strip('\"').strip())

        return data
